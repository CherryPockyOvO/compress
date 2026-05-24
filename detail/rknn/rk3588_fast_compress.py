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
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import batch_compress as media_utils  # noqa: E402


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def run_command(cmd: Sequence[str], timeout: Optional[float] = None) -> None:
    proc = subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        rendered = " ".join(str(part) for part in cmd)
        raise RuntimeError(
            f"command failed with exit code {proc.returncode}: {rendered}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def default_cnz_encode_cli() -> Optional[Path]:
    repo_root = SCRIPT_DIR.parent
    candidates = [
        SCRIPT_DIR / "cnz_encode_cli",
        SCRIPT_DIR / "bin" / "cnz_encode_cli",
        repo_root / "cpp" / "build" / "cnz_encode_cli",
        repo_root / "cpp" / "bin" / "cnz_encode_cli",
        Path.cwd() / "cnz_encode_cli",
    ]
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    return None


def load_image_rgb(
    image_path: Path,
    height: Optional[int],
    width: Optional[int],
    downsampling_factor: int,
) -> Tuple[Any, Dict[str, Any]]:
    import numpy as np
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    source_w, source_h = image.size

    if height is not None or width is not None:
        if height is None or width is None:
            raise ValueError("--height and --width must be provided together")
        image = image.resize((width, height), Image.Resampling.BICUBIC)

    orig_w, orig_h = image.size
    array = np.asarray(image).astype(np.float32) / 255.0

    pad_h = (downsampling_factor - orig_h % downsampling_factor) % downsampling_factor
    pad_w = (downsampling_factor - orig_w % downsampling_factor) % downsampling_factor
    if pad_h > 0 or pad_w > 0:
        array = np.pad(array, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

    padded_h, padded_w = array.shape[:2]
    input_nhwc = np.expand_dims(array, axis=0).astype(np.float32)

    return input_nhwc, {
        "format": "compressai-nano-latent-metadata-v1",
        "image": str(image_path),
        "dtype": "float32",
        "layout": "NCHW",
        "source_h": int(source_h),
        "source_w": int(source_w),
        "orig_h": int(orig_h),
        "orig_w": int(orig_w),
        "padded_h": int(padded_h),
        "padded_w": int(padded_w),
        "downsampling_factor": int(downsampling_factor),
    }


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


def run_entropy_command(
    bin_path: Path,
    meta_path: Path,
    cnz_path: Path,
    config: Dict[str, Any],
) -> None:
    run_command(
        [
            str(config["cnz_encode_cli"]),
            "--latent",
            str(bin_path),
            "--params",
            str(config["params"]),
            "--output",
            str(cnz_path),
            "--metadata",
            str(meta_path),
            "--codec",
            str(config["codec"]),
            "--zlib-level",
            str(config["zlib_level"]),
        ],
        timeout=float(config["entropy_timeout"]),
    )


def entropy_worker(
    entropy_worker_id: int,
    entropy_queue: mp.Queue,
    result_queue: mp.Queue,
    config: Dict[str, Any],
) -> None:
    try:
        while True:
            job = entropy_queue.get()
            if job is None:
                break

            bin_path = Path(job["bin_path"])
            meta_path = Path(job["meta_path"])
            cnz_path = Path(job["cnz_path"])
            entropy_wait_sec = time.perf_counter() - float(job["queued_for_entropy_at"])
            entropy_t0 = time.perf_counter()
            try:
                run_entropy_command(
                    bin_path=bin_path,
                    meta_path=meta_path,
                    cnz_path=cnz_path,
                    config=config,
                )
                entropy_sec = time.perf_counter() - entropy_t0

                if not config["keep_bin"]:
                    bin_path.unlink(missing_ok=True)
                if not config["keep_latent_metadata"]:
                    meta_path.unlink(missing_ok=True)

                total_sec = time.perf_counter() - float(job["total_t0"])
                result_queue.put(
                    {
                        "type": "frame",
                        "index": int(job["index"]),
                        "source": str(job["source"]),
                        "cnz": str(job["cnz_rel"]),
                        "core_mask": str(job["core_mask"]),
                        "worker_id": int(job["worker_id"]),
                        "entropy_worker_id": entropy_worker_id,
                        "bytes": cnz_path.stat().st_size,
                        "elapsed_sec": round(total_sec, 4),
                        "preprocess_sec": round(float(job["preprocess_sec"]), 4),
                        "npu_inference_sec": round(float(job["npu_inference_sec"]), 4),
                        "save_bin_sec": round(float(job["save_bin_sec"]), 4),
                        "entropy_wait_sec": round(entropy_wait_sec, 4),
                        "entropy_sec": round(entropy_sec, 4),
                        "frame_fps": round(1.0 / total_sec, 4) if total_sec > 0 else 0.0,
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


def encode_worker(
    worker_id: int,
    core_mask_name: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    entropy_queue: Optional[mp.Queue],
    entropy_semaphore: mp.Semaphore,
    config: Dict[str, Any],
) -> None:
    import numpy as np
    from rknnlite.api import RKNNLite

    rknn = RKNNLite()
    init_t0 = time.perf_counter()
    ret = rknn.load_rknn(str(config["rknn"]))
    if ret != 0:
        result_queue.put({"type": "fatal", "worker_id": worker_id, "error": f"load_rknn failed: {ret}"})
        rknn.release()
        return
    ret = rknn.init_runtime(core_mask=resolve_core_mask(RKNNLite, core_mask_name))
    if ret != 0:
        result_queue.put(
            {"type": "fatal", "worker_id": worker_id, "error": f"init_runtime failed: {ret}"}
        )
        rknn.release()
        return
    init_sec = time.perf_counter() - init_t0
    result_queue.put(
        {
            "type": "ready",
            "worker_id": worker_id,
            "core_mask": core_mask_name,
            "init_sec": round(init_sec, 4),
        }
    )

    try:
        while True:
            task = task_queue.get()
            if task is None:
                break

            index, frame_path_text = task
            frame_path = Path(frame_path_text)
            base = f"frame_{index:08d}"
            bin_path = Path(config["bin_dir"]) / f"{base}.bin"
            meta_path = Path(config["bin_dir"]) / f"{base}.bin.json"
            cnz_path = Path(config["cnz_dir"]) / f"{base}.cnz"
            total_t0 = time.perf_counter()

            try:
                pre_t0 = time.perf_counter()
                input_data, metadata = load_image_rgb(
                    frame_path,
                    height=config["height"],
                    width=config["width"],
                    downsampling_factor=config["downsampling_factor"],
                )
                if config["input_layout"] == "nchw":
                    input_data = np.transpose(input_data, (0, 3, 1, 2)).copy()
                pre_sec = time.perf_counter() - pre_t0

                npu_t0 = time.perf_counter()
                outputs = rknn.inference(inputs=[input_data])
                npu_sec = time.perf_counter() - npu_t0
                if not outputs:
                    raise RuntimeError("RKNN inference returned no outputs")

                save_t0 = time.perf_counter()
                latent = np.asarray(outputs[0])
                if latent.ndim != 4:
                    raise RuntimeError(f"unexpected latent shape: {latent.shape}")
                if latent.shape[1] != 128 and latent.shape[-1] == 128:
                    latent = np.transpose(latent, (0, 3, 1, 2)).copy()
                latent = latent.astype("<f4", copy=False)
                metadata["latent_c"] = int(latent.shape[1])
                metadata["latent_h"] = int(latent.shape[2])
                metadata["latent_w"] = int(latent.shape[3])
                metadata["core_mask"] = core_mask_name
                metadata["worker_id"] = worker_id
                bin_path.parent.mkdir(parents=True, exist_ok=True)
                latent.tofile(bin_path)
                meta_path.write_text(
                    json.dumps(metadata, indent=2) + "\n",
                    encoding="utf-8",
                )
                save_sec = time.perf_counter() - save_t0

                if config["separate_entropy"]:
                    if entropy_queue is None:
                        raise RuntimeError("separate entropy is enabled but entropy_queue is missing")
                    entropy_queue.put(
                        {
                            "index": index,
                            "source": str(frame_path),
                            "bin_path": str(bin_path),
                            "meta_path": str(meta_path),
                            "cnz_path": str(cnz_path),
                            "cnz_rel": str(cnz_path.relative_to(config["output"])),
                            "core_mask": core_mask_name,
                            "worker_id": worker_id,
                            "total_t0": total_t0,
                            "preprocess_sec": pre_sec,
                            "npu_inference_sec": npu_sec,
                            "save_bin_sec": save_sec,
                            "queued_for_entropy_at": time.perf_counter(),
                        }
                    )
                    result_queue.put(
                        {
                            "type": "npu_frame",
                            "index": index,
                            "core_mask": core_mask_name,
                            "worker_id": worker_id,
                            "preprocess_sec": round(pre_sec, 4),
                            "npu_inference_sec": round(npu_sec, 4),
                            "save_bin_sec": round(save_sec, 4),
                        }
                    )
                else:
                    entropy_wait_t0 = time.perf_counter()
                    entropy_semaphore.acquire()
                    entropy_wait_sec = time.perf_counter() - entropy_wait_t0
                    entropy_t0 = time.perf_counter()
                    try:
                        run_entropy_command(
                            bin_path=bin_path,
                            meta_path=meta_path,
                            cnz_path=cnz_path,
                            config=config,
                        )
                    finally:
                        entropy_semaphore.release()
                    entropy_sec = time.perf_counter() - entropy_t0

                    if not config["keep_bin"]:
                        bin_path.unlink(missing_ok=True)
                    if not config["keep_latent_metadata"]:
                        meta_path.unlink(missing_ok=True)

                    total_sec = time.perf_counter() - total_t0
                    result_queue.put(
                        {
                            "type": "frame",
                            "index": index,
                            "source": str(frame_path),
                            "cnz": str(cnz_path.relative_to(config["output"])),
                            "core_mask": core_mask_name,
                            "worker_id": worker_id,
                            "bytes": cnz_path.stat().st_size,
                            "elapsed_sec": round(total_sec, 4),
                            "preprocess_sec": round(pre_sec, 4),
                            "npu_inference_sec": round(npu_sec, 4),
                            "save_bin_sec": round(save_sec, 4),
                            "entropy_wait_sec": round(entropy_wait_sec, 4),
                            "entropy_sec": round(entropy_sec, 4),
                            "frame_fps": round(1.0 / total_sec, 4) if total_sec > 0 else 0.0,
                        }
                    )
            except Exception as exc:
                result_queue.put(
                    {
                        "type": "error",
                        "index": index,
                        "worker_id": worker_id,
                        "core_mask": core_mask_name,
                        "error": str(exc),
                    }
                )
    finally:
        rknn.release()
        result_queue.put({"type": "stopped", "worker_id": worker_id, "core_mask": core_mask_name})


def mean_stage(frames: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = [item[key] for item in frames if isinstance(item.get(key), (int, float))]
    return sum(values) / len(values) if values else None


def fmt_sec(value: Optional[float]) -> str:
    return f"{value:.3f}s" if isinstance(value, (int, float)) else "n/a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fast RK3588 media compressor. Keeps RKNN runtimes alive and feeds "
            "multiple NPU-core workers from one Python process."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rknn", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--cnz-encode-cli", type=Path, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--input-layout", choices=["nhwc", "nchw"], default="nhwc")
    parser.add_argument("--downsampling-factor", type=int, default=16)
    parser.add_argument("--core-masks", default="0,1,2")
    parser.add_argument(
        "--workers-per-core",
        type=int,
        default=2,
        help="Persistent workers per NPU core. Use 2 for stability; try 3 only if the board stays stable.",
    )
    parser.add_argument(
        "--entropy-workers",
        type=int,
        default=3,
        help="Maximum concurrent cnz_encode_cli processes. Lower this if the board stalls.",
    )
    parser.add_argument(
        "--separate-entropy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run CNZ entropy coding in separate CPU workers so RKNN workers can keep feeding NPU.",
    )
    parser.add_argument(
        "--entropy-timeout",
        type=float,
        default=30.0,
        help="Seconds before one cnz_encode_cli call is treated as stuck.",
    )
    parser.add_argument(
        "--stall-timeout",
        type=float,
        default=45.0,
        help="Abort if no worker returns a result for this many seconds.",
    )
    parser.add_argument(
        "--core-stall-timeout",
        type=float,
        default=25.0,
        help="Abort if one NPU core has assigned frames but returns no frame for this many seconds.",
    )
    parser.add_argument("--codec", choices=["zlib", "none"], default="zlib")
    parser.add_argument("--zlib-level", type=int, default=1)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--frame-format", choices=["png", "jpg"], default="png")
    parser.add_argument("--start-time", default=None)
    parser.add_argument("--duration", default=None)
    parser.add_argument("--fps", default=None)
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--keep-bin", action="store_true")
    parser.add_argument("--keep-latent-metadata", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.height is not None or args.width is not None:
        if args.height is None or args.width is None:
            raise ValueError("--height and --width must be provided together")
    if args.workers_per_core <= 0:
        raise ValueError("--workers-per-core must be > 0")
    if args.entropy_workers <= 0:
        raise ValueError("--entropy-workers must be > 0")
    if args.entropy_timeout <= 0:
        raise ValueError("--entropy-timeout must be > 0")
    if args.stall_timeout <= 0:
        raise ValueError("--stall-timeout must be > 0")
    if args.core_stall_timeout <= 0:
        raise ValueError("--core-stall-timeout must be > 0")

    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.rknn = args.rknn.resolve()
    args.params = args.params.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    cnz_dir = args.output / "frames"
    cnz_dir.mkdir(parents=True, exist_ok=True)

    if args.cnz_encode_cli is None:
        detected = default_cnz_encode_cli()
        if detected is None:
            raise FileNotFoundError(
                "cnz_encode_cli not found. Put the RK-built binary at "
                "rk3588_media_compress/cnz_encode_cli or pass --cnz-encode-cli PATH."
            )
        args.cnz_encode_cli = detected
    else:
        args.cnz_encode_cli = args.cnz_encode_cli.resolve()

    for label, path in (
        ("--rknn", args.rknn),
        ("--params", args.params),
        ("--cnz-encode-cli", args.cnz_encode_cli),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    temp_context = None
    if args.work_dir is None:
        if args.keep_bin or args.keep_latent_metadata:
            temp_root = args.output / "_work"
            temp_root.mkdir(parents=True, exist_ok=True)
        else:
            temp_context = tempfile.TemporaryDirectory(prefix="rk3588_fast_compress_")
            temp_root = Path(temp_context.name)
    else:
        temp_root = args.work_dir.resolve()
        temp_root.mkdir(parents=True, exist_ok=True)

    try:
        source_type, frames, video_info = media_utils.collect_inputs(args, temp_root)
        if args.frame_limit is not None and source_type != "video":
            frames = frames[: args.frame_limit]
        if not frames:
            raise RuntimeError("no frames to process")

        bin_dir = temp_root / "latent_bins"
        bin_dir.mkdir(parents=True, exist_ok=True)

        core_masks = media_utils.parse_csv(args.core_masks)
        worker_specs = [
            (core_index, core)
            for _replica in range(args.workers_per_core)
            for core_index, core in enumerate(core_masks)
        ]
        worker_core_masks = [core for _core_index, core in worker_specs]
        worker_count = len(worker_core_masks)

        manifest: Dict[str, Any] = {
            "format": "rk3588-fast-cnz-frame-sequence-v1",
            "created_at": now_iso(),
            "input": str(args.input),
            "source_type": source_type,
            "output": str(args.output),
            "rknn": str(args.rknn),
            "params": str(args.params),
            "cnz_encode_cli": str(args.cnz_encode_cli),
            "height": args.height,
            "width": args.width,
            "input_layout": args.input_layout,
            "downsampling_factor": args.downsampling_factor,
            "core_masks": core_masks,
            "workers_per_core": args.workers_per_core,
            "entropy_workers": args.entropy_workers,
            "separate_entropy": args.separate_entropy,
            "entropy_timeout": args.entropy_timeout,
            "stall_timeout": args.stall_timeout,
            "core_stall_timeout": args.core_stall_timeout,
            "worker_count": worker_count,
            "codec": args.codec,
            "zlib_level": args.zlib_level,
            "video_info": video_info,
            "frame_count": len(frames),
            "frames": [],
            "errors": [],
        }
        manifest_path = args.output / "manifest.json"
        media_utils.write_manifest(manifest_path, manifest)

        print(f"input_type: {source_type}")
        print(f"frames: {len(frames)}")
        print(f"output: {args.output}")
        print(f"workers: {worker_count}, core_masks: {','.join(worker_core_masks)}")
        print(
            f"entropy_workers: {args.entropy_workers}, "
            f"separate_entropy: {args.separate_entropy}, "
            f"stall_timeout: {args.stall_timeout:.1f}s, "
            f"core_stall_timeout: {args.core_stall_timeout:.1f}s"
        )
        print("tip: start with --workers-per-core 2. Try 3 only if the board stays stable.")

        context = mp.get_context("spawn")
        result_queue: mp.Queue = context.Queue()
        core_task_queues: List[mp.Queue] = [context.Queue(maxsize=0) for _ in core_masks]
        entropy_queue: Optional[mp.Queue] = context.Queue(maxsize=0) if args.separate_entropy else None
        entropy_semaphore = context.Semaphore(args.entropy_workers)
        config = {
            "rknn": str(args.rknn),
            "params": str(args.params),
            "cnz_encode_cli": str(args.cnz_encode_cli),
            "output": str(args.output),
            "cnz_dir": str(cnz_dir),
            "bin_dir": str(bin_dir),
            "height": args.height,
            "width": args.width,
            "input_layout": args.input_layout,
            "downsampling_factor": args.downsampling_factor,
            "codec": args.codec,
            "zlib_level": args.zlib_level,
            "entropy_timeout": args.entropy_timeout,
            "separate_entropy": args.separate_entropy,
            "keep_bin": args.keep_bin,
            "keep_latent_metadata": args.keep_latent_metadata,
        }

        workers = [
            context.Process(
                target=encode_worker,
                args=(
                    worker_id,
                    core_mask,
                    core_task_queues[core_index],
                    result_queue,
                    entropy_queue,
                    entropy_semaphore,
                    config,
                ),
                daemon=False,
            )
            for worker_id, (core_index, core_mask) in enumerate(worker_specs)
        ]
        entropy_workers = []
        if args.separate_entropy:
            assert entropy_queue is not None
            entropy_workers = [
                context.Process(
                    target=entropy_worker,
                    args=(entropy_worker_id, entropy_queue, result_queue, config),
                    daemon=False,
                )
                for entropy_worker_id in range(args.entropy_workers)
            ]
        for worker in workers:
            worker.start()
        for worker in entropy_workers:
            worker.start()

        core_assigned_counts = [0 for _ in core_masks]
        for index, frame_path in enumerate(frames):
            core_index = index % len(core_masks)
            core_assigned_counts[core_index] += 1
            core_task_queues[core_index].put((index, str(frame_path)))
        for core_index, task_queue in enumerate(core_task_queues):
            for _ in range(args.workers_per_core):
                task_queue.put(None)

        completed = 0
        stopped = 0
        entropy_stopped = 0
        entropy_stop_sent = False
        fatal_error: Optional[str] = None
        started = time.perf_counter()
        last_result_time = time.perf_counter()
        core_completed_counts = [0 for _ in core_masks]
        core_last_result = [started for _ in core_masks]
        core_index_by_mask = {core: index for index, core in enumerate(core_masks)}

        while stopped < worker_count or entropy_stopped < len(entropy_workers):
            try:
                item = result_queue.get(timeout=0.5)
            except queue.Empty:
                if (
                    args.separate_entropy
                    and stopped == worker_count
                    and not entropy_stop_sent
                    and entropy_queue is not None
                ):
                    for _ in entropy_workers:
                        entropy_queue.put(None)
                    entropy_stop_sent = True
                if fatal_error is not None:
                    break
                idle_sec = time.perf_counter() - last_result_time
                if idle_sec > args.stall_timeout:
                    alive = [
                        f"{idx}:pid={worker.pid}:alive={worker.is_alive()}"
                        for idx, worker in enumerate(workers)
                    ]
                    fatal_error = (
                        f"no worker result for {idle_sec:.1f}s; "
                        "RKNN/CNZ worker likely stalled. "
                        f"workers: {', '.join(alive)}"
                    )
                    print(f"[stall] {fatal_error}", file=sys.stderr)
                    break
                now = time.perf_counter()
                stalled_cores = []
                for core_index, core_mask in enumerate(core_masks):
                    if core_completed_counts[core_index] >= core_assigned_counts[core_index]:
                        continue
                    idle_core_sec = now - core_last_result[core_index]
                    if idle_core_sec > args.core_stall_timeout:
                        stalled_cores.append(
                            f"core={core_mask} idle={idle_core_sec:.1f}s "
                            f"completed={core_completed_counts[core_index]}/"
                            f"{core_assigned_counts[core_index]}"
                        )
                if stalled_cores:
                    alive = [
                        f"{idx}:pid={worker.pid}:alive={worker.is_alive()}"
                        for idx, worker in enumerate(workers)
                    ]
                    fatal_error = (
                        "one NPU core stopped returning frames: "
                        + "; ".join(stalled_cores)
                        + ". Try excluding that core with --core-masks, "
                        "or lower --workers-per-core. "
                        f"workers: {', '.join(alive)}"
                    )
                    print(f"[core-stall] {fatal_error}", file=sys.stderr)
                    break
                continue

            last_result_time = time.perf_counter()
            item_type = item.get("type")
            if item_type == "ready":
                print(
                    f"worker {item['worker_id']} ready: "
                    f"core={item['core_mask']} init={item['init_sec']:.3f}s"
                )
            elif item_type == "stopped":
                stopped += 1
                if (
                    args.separate_entropy
                    and stopped == worker_count
                    and not entropy_stop_sent
                    and entropy_queue is not None
                ):
                    for _ in entropy_workers:
                        entropy_queue.put(None)
                    entropy_stop_sent = True
            elif item_type == "entropy_stopped":
                entropy_stopped += 1
            elif item_type == "fatal":
                fatal_error = str(item.get("error"))
                manifest["errors"].append(item)
                print(f"[fatal] worker {item.get('worker_id')}: {fatal_error}", file=sys.stderr)
                if not args.continue_on_error:
                    break
            elif item_type == "error":
                manifest["errors"].append(item)
                print(
                    f"[error] frame {item.get('index')}: {item.get('error')}",
                    file=sys.stderr,
                )
                if not args.continue_on_error:
                    fatal_error = str(item.get("error"))
                    break
            elif item_type == "frame":
                completed += 1
                core_index = core_index_by_mask.get(str(item["core_mask"]))
                if core_index is not None and not args.separate_entropy:
                    core_completed_counts[core_index] += 1
                    core_last_result[core_index] = time.perf_counter()
                manifest["frames"].append(item)
                manifest["frames"].sort(key=lambda row: row["index"])
                elapsed = time.perf_counter() - started
                avg_fps = completed / elapsed if elapsed > 0 else 0.0
                print(
                    f"[{completed}/{len(frames)}] frame {item['index']:08d} -> {item['cnz']} "
                    f"total={item['elapsed_sec']:.3f}s "
                    f"pre={item['preprocess_sec']:.3f}s "
                    f"npu={item['npu_inference_sec']:.3f}s "
                    f"save_bin={item['save_bin_sec']:.3f}s "
                    f"entropy_wait={item.get('entropy_wait_sec', 0.0):.3f}s "
                    f"entropy={item['entropy_sec']:.3f}s "
                    f"worker={item['worker_id']} core={item['core_mask']} "
                    f"entropy_worker={item.get('entropy_worker_id', 'inline')} "
                    f"avg_fps={avg_fps:.2f}"
                )

                manifest["elapsed_sec"] = round(elapsed, 4)
                manifest["avg_fps"] = round(avg_fps, 4)
                media_utils.write_manifest(manifest_path, manifest)
            elif item_type == "npu_frame":
                core_index = core_index_by_mask.get(str(item["core_mask"]))
                if core_index is not None:
                    core_completed_counts[core_index] += 1
                    core_last_result[core_index] = time.perf_counter()

        if fatal_error is not None:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
            for worker in entropy_workers:
                if worker.is_alive():
                    worker.terminate()
            raise RuntimeError(fatal_error)

        for worker in workers:
            worker.join()
        for worker in entropy_workers:
            worker.join()

        elapsed = time.perf_counter() - started
        frame_records = manifest["frames"]
        manifest["elapsed_sec"] = round(elapsed, 4)
        manifest["avg_fps"] = round(completed / elapsed, 4) if elapsed > 0 else 0.0
        for key in ("preprocess_sec", "npu_inference_sec", "save_bin_sec", "entropy_wait_sec", "entropy_sec"):
            value = mean_stage(frame_records, key)
            manifest[f"avg_{key}"] = round(value, 4) if value is not None else None
        worker_stats: Dict[str, Dict[str, Any]] = {}
        for record in frame_records:
            worker_key = str(record["worker_id"])
            stats = worker_stats.setdefault(
                worker_key,
                {
                    "worker_id": record["worker_id"],
                    "core_mask": record["core_mask"],
                    "frames": 0,
                    "npu_total_sec": 0.0,
                    "elapsed_total_sec": 0.0,
                },
            )
            stats["frames"] += 1
            stats["npu_total_sec"] += float(record.get("npu_inference_sec", 0.0))
            stats["elapsed_total_sec"] += float(record.get("elapsed_sec", 0.0))
        for stats in worker_stats.values():
            frames_count = int(stats["frames"])
            stats["avg_npu_sec"] = round(stats["npu_total_sec"] / frames_count, 4) if frames_count else None
            stats["avg_elapsed_sec"] = (
                round(stats["elapsed_total_sec"] / frames_count, 4) if frames_count else None
            )
            stats["npu_total_sec"] = round(stats["npu_total_sec"], 4)
            stats["elapsed_total_sec"] = round(stats["elapsed_total_sec"], 4)
        manifest["worker_stats"] = [
            worker_stats[key]
            for key in sorted(worker_stats, key=lambda item: int(item))
        ]
        media_utils.write_manifest(manifest_path, manifest)

        print(
            "avg_stage_time: "
            f"pre={fmt_sec(manifest.get('avg_preprocess_sec'))}, "
            f"npu={fmt_sec(manifest.get('avg_npu_inference_sec'))}, "
            f"save_bin={fmt_sec(manifest.get('avg_save_bin_sec'))}, "
            f"entropy_wait={fmt_sec(manifest.get('avg_entropy_wait_sec'))}, "
            f"entropy={fmt_sec(manifest.get('avg_entropy_sec'))}"
        )
        print("worker_stats:")
        for stats in manifest["worker_stats"]:
            print(
                f"  worker={stats['worker_id']} core={stats['core_mask']} "
                f"frames={stats['frames']} avg_npu={fmt_sec(stats['avg_npu_sec'])} "
                f"avg_total={fmt_sec(stats['avg_elapsed_sec'])}"
            )
        print(f"done: {completed} frames, total_time={elapsed:.3f}s, avg_fps={manifest['avg_fps']:.2f}")
        print(f"manifest: {manifest_path}")
    finally:
        if temp_context is not None:
            temp_context.cleanup()


if __name__ == "__main__":
    main()
