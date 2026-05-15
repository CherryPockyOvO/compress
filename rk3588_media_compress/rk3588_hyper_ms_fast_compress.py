#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import batch_compress as media_utils  # noqa: E402


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def default_hyper_ms_encode_cli() -> Path | None:
    repo_root = SCRIPT_DIR.parent
    candidates = [
        SCRIPT_DIR / "hyper_ms_encode_cli",
        SCRIPT_DIR / "bin" / "hyper_ms_encode_cli",
        repo_root / "cpp" / "build" / "hyper_ms_encode_cli",
        repo_root / "cpp" / "bin" / "hyper_ms_encode_cli",
        Path.cwd() / "hyper_ms_encode_cli",
    ]
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    return None


def run_command(cmd: list[str], timeout: float | None = None) -> tuple[str, str]:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {proc.returncode}: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout, proc.stderr


def load_image_rgb(
    image_path: Path,
    target_h: int | None,
    target_w: int | None,
    padding_multiple: int,
    resize_input: bool,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    source_w, source_h = image.size
    if resize_input:
        if target_h is None or target_w is None:
            raise ValueError("--resize-input requires --height and --width")
        image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)

    orig_w, orig_h = image.size
    array = np.asarray(image).astype(np.float32) / 255.0
    pad_h = (padding_multiple - orig_h % padding_multiple) % padding_multiple
    pad_w = (padding_multiple - orig_w % padding_multiple) % padding_multiple
    if pad_h > 0 or pad_w > 0:
        array = np.pad(array, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

    padded_h, padded_w = array.shape[:2]
    if target_h is not None and target_w is not None and (padded_h, padded_w) != (target_h, target_w):
        raise RuntimeError(
            f"padded input is {padded_w}x{padded_h}, but RKNN expects {target_w}x{target_h}. "
            "Use --resize-input or export a matching RKNN shape."
        )

    metadata: dict[str, Any] = {
        "format": "compressai-nano-hyper-ms-npu-metadata-v1",
        "image": str(image_path),
        "source_h": int(source_h),
        "source_w": int(source_w),
        "orig_h": int(orig_h),
        "orig_w": int(orig_w),
        "padded_h": int(padded_h),
        "padded_w": int(padded_w),
        "padding_multiple": int(padding_multiple),
        "resize_input": bool(resize_input),
        "input_dtype": "float32",
    }
    return np.expand_dims(array, axis=0).astype(np.float32), metadata


def resolve_core_mask(rknn_lite: Any, name: str) -> int:
    normalized = name.lower().replace("-", "_")
    attr_by_name = {
        "auto": "NPU_CORE_AUTO",
        "any": "NPU_CORE_AUTO",
        "0": "NPU_CORE_0",
        "core0": "NPU_CORE_0",
        "1": "NPU_CORE_1",
        "core1": "NPU_CORE_1",
        "2": "NPU_CORE_2",
        "core2": "NPU_CORE_2",
        "0_1": "NPU_CORE_0_1",
        "01": "NPU_CORE_0_1",
        "1_2": "NPU_CORE_1_2",
        "12": "NPU_CORE_1_2",
        "0_1_2": "NPU_CORE_0_1_2",
        "012": "NPU_CORE_0_1_2",
        "all": "NPU_CORE_0_1_2",
    }
    attr = attr_by_name.get(normalized)
    if attr is None:
        raise ValueError(f"unknown core mask: {name}")
    if hasattr(rknn_lite, attr):
        return int(getattr(rknn_lite, attr))
    if normalized in {"auto", "any", "all", "0_1_2", "012"}:
        return int(rknn_lite.NPU_CORE_AUTO)
    raise ValueError(f"RKNNLite on this board does not expose {attr}")


def to_nchw(output: Any, channels: int, name: str):
    import numpy as np

    array = np.asarray(output)
    if array.ndim != 4:
        raise RuntimeError(f"{name} must be 4D, got {array.shape}")
    if array.shape[1] == channels:
        return array.astype("<f4", copy=False)
    if array.shape[-1] == channels:
        return np.transpose(array, (0, 3, 1, 2)).astype("<f4", copy=False)
    raise RuntimeError(f"{name} has unexpected shape {array.shape}; expected channel count {channels}")


def load_hyper_params(path: Path) -> dict[str, Any]:
    params = json.loads(path.read_text(encoding="utf-8"))
    if not params.get("has_means_y", False):
        raise RuntimeError("expected mean-scale hyperprior params with has_means_y=true")
    for key in ("channels_y", "channels_z", "quant_step_y", "quant_step_z"):
        if key not in params:
            raise RuntimeError(f"params missing {key}")
    return params


def run_entropy_command(job: dict[str, Any], config: dict[str, Any]) -> tuple[str, str]:
    cmd = [
        str(config["hyper_ms_encode_cli"]),
        "--y",
        str(job["y_path"]),
        "--z",
        str(job["z_path"]),
        "--means",
        str(job["means_path"]),
        "--params",
        str(config["params"]),
        "--metadata",
        str(job["meta_path"]),
        "--output",
        str(job["out_path"]),
        "--codec",
        str(config["codec"]),
        "--zlib-level",
        str(config["zlib_level"]),
    ]
    return run_command(cmd, timeout=float(config["entropy_timeout"]))


def entropy_worker(
    entropy_worker_id: int,
    entropy_queue: mp.Queue,
    result_queue: mp.Queue,
    config: dict[str, Any],
) -> None:
    try:
        while True:
            job = entropy_queue.get()
            if job is None:
                break
            entropy_wait_sec = time.perf_counter() - float(job["queued_for_entropy_at"])
            entropy_t0 = time.perf_counter()
            try:
                stdout, _stderr = run_entropy_command(job, config)
                entropy_sec = time.perf_counter() - entropy_t0
                out_path = Path(job["out_path"])

                if not config["keep_intermediate"]:
                    Path(job["y_path"]).unlink(missing_ok=True)
                    Path(job["z_path"]).unlink(missing_ok=True)
                    Path(job["means_path"]).unlink(missing_ok=True)
                    Path(job["meta_path"]).unlink(missing_ok=True)

                total_sec = time.perf_counter() - float(job["total_t0"])
                result_queue.put(
                    {
                        "type": "frame",
                        "index": int(job["index"]),
                        "source": str(job["source"]),
                        "output": str(job["out_rel"]),
                        "bytes": out_path.stat().st_size,
                        "worker_id": int(job["worker_id"]),
                        "core_mask": str(job["core_mask"]),
                        "entropy_worker_id": entropy_worker_id,
                        "elapsed_sec": round(total_sec, 4),
                        "preprocess_sec": round(float(job["preprocess_sec"]), 4),
                        "npu_inference_sec": round(float(job["npu_inference_sec"]), 4),
                        "save_intermediate_sec": round(float(job["save_intermediate_sec"]), 4),
                        "entropy_wait_sec": round(entropy_wait_sec, 4),
                        "entropy_sec": round(entropy_sec, 4),
                        "frame_fps": round(1.0 / total_sec, 4) if total_sec > 0 else 0.0,
                        "encoder_stdout": stdout[-2000:],
                    }
                )
            except Exception as exc:
                result_queue.put(
                    {
                        "type": "error",
                        "index": int(job.get("index", -1)),
                        "worker_id": int(job.get("worker_id", -1)),
                        "core_mask": str(job.get("core_mask", "")),
                        "entropy_worker_id": entropy_worker_id,
                        "error": f"entropy worker failed: {exc}",
                    }
                )
    finally:
        result_queue.put({"type": "entropy_stopped", "entropy_worker_id": entropy_worker_id})


def npu_worker(
    worker_id: int,
    core_mask_name: str,
    task_queue: mp.Queue,
    entropy_queue: mp.Queue,
    result_queue: mp.Queue,
    config: dict[str, Any],
) -> None:
    import numpy as np
    from rknnlite.api import RKNNLite

    params = config["hyper_params"]
    channels_y = int(params["channels_y"])
    channels_z = int(params["channels_z"])

    rknn = RKNNLite()
    init_t0 = time.perf_counter()
    ret = rknn.load_rknn(str(config["rknn"]))
    if ret != 0:
        result_queue.put({"type": "fatal", "worker_id": worker_id, "error": f"load_rknn failed: {ret}"})
        rknn.release()
        return
    ret = rknn.init_runtime(core_mask=resolve_core_mask(RKNNLite, core_mask_name))
    if ret != 0:
        result_queue.put({"type": "fatal", "worker_id": worker_id, "error": f"init_runtime failed: {ret}"})
        rknn.release()
        return
    result_queue.put(
        {
            "type": "ready",
            "worker_id": worker_id,
            "core_mask": core_mask_name,
            "init_sec": round(time.perf_counter() - init_t0, 4),
        }
    )

    try:
        while True:
            task = task_queue.get()
            if task is None:
                break
            index, source_text = task
            source = Path(source_text)
            base = f"frame_{int(index):08d}"
            y_path = Path(config["work_dir"]) / f"{base}.y.bin"
            z_path = Path(config["work_dir"]) / f"{base}.z.bin"
            means_path = Path(config["work_dir"]) / f"{base}.means_y.bin"
            meta_path = Path(config["work_dir"]) / f"{base}.json"
            out_path = Path(config["frames_dir"]) / f"{base}.hms"
            total_t0 = time.perf_counter()

            try:
                pre_t0 = time.perf_counter()
                input_data, metadata = load_image_rgb(
                    source,
                    target_h=config["height"],
                    target_w=config["width"],
                    padding_multiple=config["padding_multiple"],
                    resize_input=config["resize_input"],
                )
                if config["input_layout"] == "nchw":
                    input_data = np.transpose(input_data, (0, 3, 1, 2)).copy()
                pre_sec = time.perf_counter() - pre_t0

                npu_t0 = time.perf_counter()
                outputs = rknn.inference(inputs=[input_data])
                npu_sec = time.perf_counter() - npu_t0
                if len(outputs) < 4:
                    raise RuntimeError(f"expected RKNN outputs [y,z,scales,means], got {len(outputs)}")

                save_t0 = time.perf_counter()
                y = to_nchw(outputs[0], channels_y, "y")
                z = to_nchw(outputs[1], channels_z, "z")
                means = to_nchw(outputs[3], channels_y, "means_y")
                if y.shape != means.shape:
                    raise RuntimeError(f"y/means shape mismatch: {y.shape} vs {means.shape}")

                metadata.update(
                    {
                        "model_variant": str(params.get("model_variant", "nano_hyper_ms_q_nano")),
                        "model_type": str(params.get("model_type", "mean_scale_hyperprior")),
                        "core_mask": core_mask_name,
                        "worker_id": worker_id,
                        "y_shape": [int(v) for v in y.shape],
                        "z_shape": [int(v) for v in z.shape],
                        "means_shape": [int(v) for v in means.shape],
                    }
                )
                y_path.parent.mkdir(parents=True, exist_ok=True)
                y.tofile(y_path)
                z.tofile(z_path)
                means.tofile(means_path)
                meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
                save_sec = time.perf_counter() - save_t0

                entropy_queue.put(
                    {
                        "index": int(index),
                        "source": str(source),
                        "y_path": str(y_path),
                        "z_path": str(z_path),
                        "means_path": str(means_path),
                        "meta_path": str(meta_path),
                        "out_path": str(out_path),
                        "out_rel": str(out_path.relative_to(config["output"])),
                        "worker_id": worker_id,
                        "core_mask": core_mask_name,
                        "total_t0": total_t0,
                        "preprocess_sec": pre_sec,
                        "npu_inference_sec": npu_sec,
                        "save_intermediate_sec": save_sec,
                        "queued_for_entropy_at": time.perf_counter(),
                    }
                )
                result_queue.put(
                    {
                        "type": "npu_frame",
                        "index": int(index),
                        "worker_id": worker_id,
                        "core_mask": core_mask_name,
                        "preprocess_sec": round(pre_sec, 4),
                        "npu_inference_sec": round(npu_sec, 4),
                        "save_intermediate_sec": round(save_sec, 4),
                    }
                )
            except Exception as exc:
                result_queue.put(
                    {
                        "type": "error",
                        "index": int(index),
                        "worker_id": worker_id,
                        "core_mask": core_mask_name,
                        "error": str(exc),
                    }
                )
    finally:
        rknn.release()
        result_queue.put({"type": "stopped", "worker_id": worker_id, "core_mask": core_mask_name})


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fast RK3588 hyper_ms_nano compressor: persistent RKNN workers feed "
            "3 NPU cores, CPU workers run C++ symbol/zlib entropy coding."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rknn", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--hyper-ms-encode-cli", type=Path, default=None)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--padding-multiple", type=int, default=64)
    parser.add_argument("--resize-input", action="store_true")
    parser.add_argument("--input-layout", choices=("nhwc", "nchw"), default="nhwc")
    parser.add_argument("--core-masks", default="0,1,2")
    parser.add_argument("--workers-per-core", type=int, default=1)
    parser.add_argument("--entropy-workers", type=int, default=3)
    parser.add_argument("--entropy-timeout", type=float, default=30.0)
    parser.add_argument("--stall-timeout", type=float, default=45.0)
    parser.add_argument("--codec", choices=("zlib", "none"), default="zlib")
    parser.add_argument("--zlib-level", type=int, default=1)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--frame-format", choices=("png", "jpg"), default="png")
    parser.add_argument("--start-time", default=None)
    parser.add_argument("--duration", default=None)
    parser.add_argument("--fps", default=None)
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.rknn = args.rknn.resolve()
    args.params = args.params.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    if args.hyper_ms_encode_cli is None:
        detected = default_hyper_ms_encode_cli()
        if detected is None:
            raise FileNotFoundError(
                "hyper_ms_encode_cli not found. Build cpp/ or pass --hyper-ms-encode-cli PATH."
            )
        args.hyper_ms_encode_cli = detected
    else:
        args.hyper_ms_encode_cli = args.hyper_ms_encode_cli.resolve()

    for label, path in (("--rknn", args.rknn), ("--params", args.params), ("--hyper-ms-encode-cli", args.hyper_ms_encode_cli)):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    hyper_params = load_hyper_params(args.params)
    temp_context = None
    if args.work_dir is None:
        if args.keep_intermediate:
            work_root = args.output / "_work"
            work_root.mkdir(parents=True, exist_ok=True)
        else:
            temp_context = tempfile.TemporaryDirectory(prefix="rk3588_hyper_ms_")
            work_root = Path(temp_context.name)
    else:
        work_root = args.work_dir.resolve()
        work_root.mkdir(parents=True, exist_ok=True)
    work_dir = work_root / "npu_outputs"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        source_type, frames, video_info = media_utils.collect_inputs(args, work_root)
        if args.frame_limit is not None and source_type != "video":
            frames = frames[: args.frame_limit]
        if not frames:
            raise RuntimeError("no frames to process")

        core_masks = media_utils.parse_csv(args.core_masks)
        worker_core_masks = [core for _ in range(args.workers_per_core) for core in core_masks]
        worker_count = len(worker_core_masks)
        manifest = {
            "format": "rk3588-hyper-ms-frame-sequence-v1",
            "created_at": now_iso(),
            "input": str(args.input),
            "source_type": source_type,
            "output": str(args.output),
            "rknn": str(args.rknn),
            "params": str(args.params),
            "hyper_ms_encode_cli": str(args.hyper_ms_encode_cli),
            "height": args.height,
            "width": args.width,
            "padding_multiple": args.padding_multiple,
            "resize_input": args.resize_input,
            "input_layout": args.input_layout,
            "core_masks": core_masks,
            "workers_per_core": args.workers_per_core,
            "entropy_workers": args.entropy_workers,
            "codec": args.codec,
            "zlib_level": args.zlib_level,
            "video_info": video_info,
            "frame_count": len(frames),
            "frames": [],
            "npu_frames": [],
            "errors": [],
        }
        manifest_path = args.output / "manifest.json"
        write_manifest(manifest_path, manifest)

        print(f"input_type: {source_type}")
        print(f"frames: {len(frames)}")
        print(f"output: {args.output}")
        print(f"npu_workers: {worker_count}, core_masks: {','.join(worker_core_masks)}")
        print(f"entropy_workers: {args.entropy_workers}")
        print("tip: default workers bind one persistent process to each NPU core: 0,1,2.")

        context = mp.get_context("spawn")
        task_queue: mp.Queue = context.Queue(maxsize=worker_count * 4)
        entropy_queue: mp.Queue = context.Queue(maxsize=max(4, args.entropy_workers * 4))
        result_queue: mp.Queue = context.Queue()

        config = {
            "rknn": str(args.rknn),
            "params": str(args.params),
            "hyper_ms_encode_cli": str(args.hyper_ms_encode_cli),
            "output": str(args.output),
            "frames_dir": str(frames_dir),
            "work_dir": str(work_dir),
            "height": args.height,
            "width": args.width,
            "padding_multiple": args.padding_multiple,
            "resize_input": args.resize_input,
            "input_layout": args.input_layout,
            "hyper_params": hyper_params,
            "codec": args.codec,
            "zlib_level": args.zlib_level,
            "entropy_timeout": args.entropy_timeout,
            "keep_intermediate": args.keep_intermediate,
        }

        npu_workers = [
            context.Process(
                target=npu_worker,
                args=(worker_id, core_mask, task_queue, entropy_queue, result_queue, config),
            )
            for worker_id, core_mask in enumerate(worker_core_masks)
        ]
        entropy_workers = [
            context.Process(
                target=entropy_worker,
                args=(worker_id, entropy_queue, result_queue, config),
            )
            for worker_id in range(args.entropy_workers)
        ]
        for proc in entropy_workers + npu_workers:
            proc.start()

        for index, frame in enumerate(frames):
            task_queue.put((index, str(frame)))
        for _ in npu_workers:
            task_queue.put(None)

        completed = 0
        npu_stopped = 0
        entropy_stopped = 0
        fatal = False
        last_result_time = time.perf_counter()
        while entropy_stopped < len(entropy_workers):
            try:
                item = result_queue.get(timeout=1.0)
            except queue.Empty:
                if time.perf_counter() - last_result_time > args.stall_timeout:
                    fatal = True
                    print(f"ERROR: no worker result for {args.stall_timeout:.1f}s; stopping")
                    break
                continue
            last_result_time = time.perf_counter()
            kind = item.get("type")
            if kind == "ready":
                print(f"worker ready: id={item['worker_id']} core={item['core_mask']} init={item['init_sec']}s")
            elif kind == "npu_frame":
                manifest["npu_frames"].append(item)
                print(
                    f"npu {item['index'] + 1}/{len(frames)} "
                    f"core={item['core_mask']} infer={item['npu_inference_sec']:.4f}s"
                )
            elif kind == "frame":
                completed += 1
                manifest["frames"].append(item)
                print(
                    f"done {completed}/{len(frames)} idx={item['index']} "
                    f"core={item['core_mask']} npu={item['npu_inference_sec']:.4f}s "
                    f"cpu={item['entropy_sec']:.4f}s bytes={item['bytes']}"
                )
                write_manifest(manifest_path, manifest)
            elif kind == "error":
                manifest["errors"].append(item)
                print(f"ERROR frame={item.get('index')} worker={item.get('worker_id')}: {item.get('error')}")
                write_manifest(manifest_path, manifest)
                if not args.continue_on_error:
                    fatal = True
                    break
            elif kind == "fatal":
                manifest["errors"].append(item)
                print(f"FATAL worker={item.get('worker_id')}: {item.get('error')}")
                fatal = True
                break
            elif kind == "stopped":
                npu_stopped += 1
                if npu_stopped == len(npu_workers):
                    for _ in entropy_workers:
                        entropy_queue.put(None)
            elif kind == "entropy_stopped":
                entropy_stopped += 1

        if fatal:
            for proc in npu_workers + entropy_workers:
                if proc.is_alive():
                    proc.terminate()
        for proc in npu_workers + entropy_workers:
            proc.join(timeout=5.0)

        manifest["completed_frames"] = completed
        manifest["finished_at"] = now_iso()
        write_manifest(manifest_path, manifest)
        if fatal or (completed != len(frames) and not args.continue_on_error):
            raise RuntimeError(f"pipeline failed: completed {completed}/{len(frames)} frames")
        print(f"manifest: {manifest_path}")
    finally:
        if temp_context is not None:
            temp_context.cleanup()


if __name__ == "__main__":
    main()
