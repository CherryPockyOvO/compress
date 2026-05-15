#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import multiprocessing as mp
import queue
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def run_command(cmd: Sequence[str]) -> tuple[str, str]:
    proc = subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        rendered = " ".join(str(part) for part in cmd)
        raise RuntimeError(
            f"command failed with exit code {proc.returncode}: {rendered}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc.stdout, proc.stderr


def rational_to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    text = str(value).strip()
    if not text:
        return None
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        try:
            num = float(numerator)
            den = float(denominator)
        except ValueError:
            return None
        if den == 0:
            return None
        fps = num / den
        return fps if fps > 0 else None
    try:
        fps = float(text)
    except ValueError:
        return None
    return fps if fps > 0 else None


def fps_from_manifest(manifest: dict[str, Any] | None) -> float | None:
    if not manifest:
        return None
    for key in ("fps", "output_fps"):
        parsed = rational_to_float(manifest.get(key))
        if parsed is not None:
            return parsed
    video_info = manifest.get("video_info")
    if not isinstance(video_info, dict):
        return None
    streams = video_info.get("streams")
    if not isinstance(streams, list) or not streams:
        return None
    stream = streams[0]
    if not isinstance(stream, dict):
        return None
    for key in ("avg_frame_rate", "r_frame_rate"):
        parsed = rational_to_float(stream.get(key))
        if parsed is not None:
            return parsed
    return None


def load_manifest(input_path: Path) -> dict[str, Any] | None:
    manifest_path = input_path / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def collect_hms_from_manifest(input_path: Path, manifest: dict[str, Any]) -> list[tuple[int, Path]]:
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        return []
    collected: list[tuple[int, Path]] = []
    for pos, item in enumerate(frames):
        if not isinstance(item, dict):
            continue
        rel = item.get("output") or item.get("hms") or item.get("path")
        if not rel:
            continue
        index = int(item.get("index", pos))
        collected.append((index, input_path / str(rel)))
    collected.sort(key=lambda pair: pair[0])
    return collected


def collect_hms_files(input_path: Path) -> tuple[list[tuple[int, Path]], dict[str, Any] | None]:
    input_path = input_path.resolve()
    if input_path.is_file() and input_path.suffix.lower() == ".hms":
        return [(0, input_path)], None
    if not input_path.is_dir():
        raise FileNotFoundError(f"input must be a .hms file or directory: {input_path}")

    manifest = load_manifest(input_path)
    if manifest is not None:
        files = collect_hms_from_manifest(input_path, manifest)
        missing = [path for _, path in files if not path.exists()]
        if missing:
            raise FileNotFoundError(f"manifest references missing HMS file: {missing[0]}")
        if files:
            return files, manifest

    search_roots = []
    frames_dir = input_path / "frames"
    if frames_dir.exists():
        search_roots.append(frames_dir)
    search_roots.append(input_path)
    seen: set[Path] = set()
    found: list[Path] = []
    for root in search_roots:
        for path in sorted(root.rglob("*.hms")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(resolved)
    if not found:
        raise FileNotFoundError(f"no .hms files found under: {input_path}")
    return [(idx, path) for idx, path in enumerate(found)], manifest


def parse_devices(value: str | None) -> list[str]:
    if value is None:
        return []
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if not devices:
        raise ValueError("--devices cannot be empty")
    return devices


def batched(items: Sequence[tuple[int, Path]], batch_size: int) -> Iterable[list[tuple[int, Path]]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def make_decode_jobs(hms_files: Sequence[tuple[int, Path]], batch_size: int) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    output_start = 0
    for batch_id, batch in enumerate(batched(hms_files, batch_size)):
        jobs.append(
            {
                "batch_id": batch_id,
                "output_start": output_start,
                "items": [(idx, str(path)) for idx, path in batch],
            }
        )
        output_start += len(batch)
    return jobs


def setup_model(config: dict[str, Any], device_name: str):
    import torch

    from compressai_nano import get_model, infer_model_variant_from_checkpoint
    from decode_hyper_ms_npz import load_checkpoint

    if device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if ":" in device_name:
            torch.cuda.set_device(int(device_name.split(":", 1)[1]))
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device(device_name)

    checkpoint = Path(config["checkpoint"])
    raw = torch.load(checkpoint, map_location="cpu")
    model_variant = infer_model_variant_from_checkpoint(raw)
    model = get_model(model_variant=model_variant).to(device).eval()
    if getattr(model, "supports_cnz_v4", False):
        raise RuntimeError("pc_decode_hms_video.py expects a hyper_ms checkpoint, not legacy CNZ4 nano")
    if device.type == "cuda" and config["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    if config["half"]:
        if device.type != "cuda":
            raise RuntimeError("--half requires CUDA")
        model = model.half()
    load_checkpoint(model, checkpoint)
    return torch, model, device, model_variant


def decode_hms_batch(job: dict[str, Any], config: dict[str, Any], torch_module: Any, model: Any, device: Any):
    torch = torch_module
    from decode_cnz import crop_to_size, resize_tensor_to_size, tensor_to_image
    from decode_hyper_ms_npz import load_package, make_autocast

    items = [(int(idx), Path(path)) for idx, path in job["items"]]
    t0 = time.perf_counter()
    load_t0 = time.perf_counter()
    y_symbols_list = []
    z_symbols_list = []
    metadata_list: list[dict[str, Any]] = []
    for _idx, hms_path in items:
        y_symbols, z_symbols, metadata = load_package(hms_path)
        y_symbols_list.append(y_symbols)
        z_symbols_list.append(z_symbols)
        metadata_list.append(metadata)
    load_t1 = time.perf_counter()

    y_shape = tuple(y_symbols_list[0].shape[1:])
    z_shape = tuple(z_symbols_list[0].shape[1:])
    if any(tuple(item.shape[1:]) != y_shape for item in y_symbols_list):
        raise ValueError("batch contains different y shapes; lower --batch-size")
    if any(tuple(item.shape[1:]) != z_shape for item in z_symbols_list):
        raise ValueError("batch contains different z shapes; lower --batch-size")

    move_t0 = time.perf_counter()
    y_symbols_batch = torch.cat(y_symbols_list, dim=0).to(device=device)
    z_symbols_batch = torch.cat(z_symbols_list, dim=0).to(device=device)
    move_t1 = time.perf_counter()

    model_t0 = time.perf_counter()
    with torch.inference_mode():
        z_hat = model.entropy_bottleneck_z.dequantize(
            z_symbols_batch,
            dtype=torch.float16 if config["half"] else torch.float32,
            device=device,
        )
        with make_autocast(device, config["half"]):
            hyper = model.hyper_decoder(z_hat)
            if not isinstance(hyper, tuple) or len(hyper) != 2:
                raise RuntimeError("expected hyper_decoder(z_hat) -> (scales_y, means_y)")
            _scales_y, means_y = hyper
            y_hat = model.conditional_entropy_y.dequantize(
                y_symbols_batch,
                means_y=means_y,
                dtype=torch.float16 if config["half"] else torch.float32,
                device=device,
            )
            x_hat_batch = model.decoder(y_hat)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model_t1 = time.perf_counter()

    save_t0 = time.perf_counter()
    records: list[dict[str, Any]] = []
    raw_frames: list[bytes] = []
    batch_size_hw: tuple[int, int] | None = None
    decoded_dir = Path(config["decoded_dir"])
    resize_to_source = bool(config["resize_to_source"])
    image_format = str(config["image_format"])
    pipe_video = bool(config["pipe_video"])
    output_start = int(job["output_start"])
    for offset, ((source_index, hms_path), metadata) in enumerate(zip(items, metadata_list)):
        tensor = x_hat_batch[offset : offset + 1]
        orig_h = int(metadata.get("orig_h", metadata.get("padded_h", tensor.shape[-2])))
        orig_w = int(metadata.get("orig_w", metadata.get("padded_w", tensor.shape[-1])))
        tensor = crop_to_size(tensor, (orig_h, orig_w))
        final_h, final_w = orig_h, orig_w
        if resize_to_source:
            if "source_h" not in metadata or "source_w" not in metadata:
                raise ValueError(f"{hms_path} metadata missing source_h/source_w")
            final_h = int(metadata["source_h"])
            final_w = int(metadata["source_w"])
            tensor = resize_tensor_to_size(tensor, (final_h, final_w), mode=str(config["resize_mode"]))
        output_index = output_start + offset
        if batch_size_hw is None:
            batch_size_hw = (final_h, final_w)
        elif batch_size_hw != (final_h, final_w):
            raise ValueError("--pipe-video requires all frames to have the same decoded size")
        image_path = None
        if pipe_video:
            raw_frames.append(tensor_to_rgb_bytes(torch, tensor))
        else:
            image_path = decoded_dir / f"frame_{output_index:08d}.{image_format}"
            tensor_to_image(tensor, image_path)
        records.append(
            {
                "output_index": output_index,
                "source_index": source_index,
                "hms": str(hms_path),
                "image": str(image_path) if image_path is not None else None,
                "height": final_h,
                "width": final_w,
            }
        )
    save_t1 = time.perf_counter()

    return {
        "batch_id": int(job["batch_id"]),
        "count": len(items),
        "records": records,
        "raw_frames": raw_frames,
        "size": batch_size_hw,
        "load_sec": load_t1 - load_t0,
        "move_sec": move_t1 - move_t0,
        "model_sec": model_t1 - model_t0,
        "save_sec": save_t1 - save_t0,
        "decode_sec": save_t1 - t0,
    }


def decode_worker(
    worker_id: int,
    device_name: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    config: dict[str, Any],
) -> None:
    try:
        torch, model, device, model_variant = setup_model(config, device_name)
        result_queue.put(
            {
                "type": "ready",
                "worker_id": worker_id,
                "device": device_name,
                "model_variant": model_variant,
            }
        )
        while True:
            job = task_queue.get()
            if job is None:
                break
            try:
                result = decode_hms_batch(job, config, torch, model, device)
                result.update({"type": "batch", "worker_id": worker_id, "device": device_name})
                result_queue.put(result)
            except Exception as exc:
                result_queue.put(
                    {
                        "type": "error",
                        "worker_id": worker_id,
                        "device": device_name,
                        "batch_id": int(job.get("batch_id", -1)),
                        "error": str(exc),
                    }
                )
    finally:
        result_queue.put({"type": "stopped", "worker_id": worker_id, "device": device_name})


def encode_video(decoded_dir: Path, output_video: Path, fps: float, args: argparse.Namespace) -> None:
    if not shutil.which(args.ffmpeg):
        raise FileNotFoundError(f"ffmpeg not found: {args.ffmpeg}")
    if output_video.exists() and not args.overwrite:
        raise FileExistsError(f"output video exists, pass --overwrite: {output_video}")
    output_video.parent.mkdir(parents=True, exist_ok=True)
    pattern = decoded_dir / f"frame_%08d.{args.image_format}"
    cmd = [
        args.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if args.overwrite else "-n",
        "-framerate",
        f"{fps:.6f}",
        "-i",
        str(pattern),
        "-c:v",
        args.video_codec,
    ]
    if args.video_codec in {"libx264", "libx265"}:
        cmd += ["-preset", args.preset, "-crf", str(args.crf), "-pix_fmt", args.pix_fmt]
    cmd += [str(output_video)]
    run_command(cmd)


def open_ffmpeg_raw_writer(
    output_video: Path,
    width: int,
    height: int,
    fps: float,
    args: argparse.Namespace,
) -> subprocess.Popen[bytes]:
    if not shutil.which(args.ffmpeg):
        raise FileNotFoundError(f"ffmpeg not found: {args.ffmpeg}")
    if output_video.exists() and not args.overwrite:
        raise FileExistsError(f"output video exists, pass --overwrite: {output_video}")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if args.overwrite else "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-framerate",
        f"{fps:.6f}",
        "-i",
        "-",
        "-c:v",
        args.video_codec,
    ]
    if args.video_codec in {"libx264", "libx265"}:
        cmd += ["-preset", args.preset, "-crf", str(args.crf), "-pix_fmt", args.pix_fmt]
    cmd += [str(output_video)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def tensor_to_rgb_bytes(torch_module: Any, tensor: Any) -> bytes:
    torch = torch_module
    tensor = tensor.detach().cpu().clamp(0, 1)
    tensor = (tensor * 255.0).round().to(torch.uint8)
    tensor = tensor.permute(0, 2, 3, 1).contiguous()
    return tensor.numpy().tobytes()


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode RK3588 hyper_ms .hms frame sequence with multiple GPUs and rebuild video."
    )
    parser.add_argument("--input", type=Path, required=True, help="out_hyper_ms directory, frames dir, or one .hms file")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, default=None)
    parser.add_argument("--devices", default="0,1,2", help="CUDA device ids, e.g. 0,1,2. Use cpu for CPU.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--resize-to-source", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resize-mode", choices=("nearest", "bilinear", "bicubic"), default="bicubic")
    parser.add_argument("--image-format", choices=("png", "jpg"), default="png")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--pipe-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pipe decoded RGB frames directly to ffmpeg instead of saving PNG/JPG frames first.",
    )
    parser.add_argument("--progress-interval", type=float, default=1.0)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--video-codec", default="libx264")
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--pix-fmt", default="yuv420p")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    hms_files, source_manifest = collect_hms_files(args.input)
    if not hms_files:
        raise RuntimeError("no HMS files to decode")
    fps = args.fps if args.fps is not None else fps_from_manifest(source_manifest)
    if fps is None:
        fps = 30.0
        print("warning: fps not found in manifest; using 30.0")

    devices = parse_devices(args.devices)
    if devices == ["cpu"]:
        device_names = ["cpu"]
    else:
        device_names = [f"cuda:{device}" if not device.startswith("cuda") else device for device in devices]

    decoded_dir = args.decoded_dir or (args.output_video.parent / f"{args.output_video.stem}_frames")
    if not args.pipe_video:
        decoded_dir.mkdir(parents=True, exist_ok=True)
    jobs = make_decode_jobs(hms_files, args.batch_size)
    manifest_path = args.output_video.with_suffix(args.output_video.suffix + ".decode_manifest.json")
    manifest: dict[str, Any] = {
        "format": "pc-hyper-ms-video-decode-v1",
        "created_at": now_iso(),
        "input": str(args.input),
        "checkpoint": str(args.checkpoint),
        "output_video": str(args.output_video),
        "decoded_dir": None if args.pipe_video else str(decoded_dir),
        "devices": device_names,
        "batch_size": args.batch_size,
        "fps": fps,
        "pipe_video": args.pipe_video,
        "frame_count": len(hms_files),
        "batches": [],
        "frames": [],
        "errors": [],
    }
    write_manifest(manifest_path, manifest)

    print(f"frames: {len(hms_files)}")
    print(f"batches: {len(jobs)} batch_size={args.batch_size}")
    print(f"devices: {','.join(device_names)}")
    if args.pipe_video:
        print("decoded_frames: pipe-to-ffmpeg")
    else:
        print(f"decoded_dir: {decoded_dir}")
    print(f"output_video: {args.output_video}")
    print(f"fps: {fps:.6f}")

    context = mp.get_context("spawn")
    task_queues: list[mp.Queue] = [context.Queue(maxsize=0) for _ in device_names]
    result_queue: mp.Queue = context.Queue()
    config = {
        "checkpoint": str(args.checkpoint.resolve()),
        "decoded_dir": str(decoded_dir.resolve()),
        "resize_to_source": args.resize_to_source,
        "resize_mode": args.resize_mode,
        "image_format": args.image_format,
        "pipe_video": args.pipe_video,
        "half": args.half,
        "channels_last": args.channels_last,
    }
    workers = [
        context.Process(target=decode_worker, args=(worker_id, device, task_queues[worker_id], result_queue, config))
        for worker_id, device in enumerate(device_names)
    ]
    for proc in workers:
        proc.start()
    for job_index, job in enumerate(jobs):
        task_queues[job_index % len(task_queues)].put(job)
    for task_queue in task_queues:
        task_queue.put(None)

    completed_batches = 0
    returned_batches = 0
    completed_frames = 0
    stopped = 0
    pending_batches: dict[int, dict[str, Any]] = {}
    next_batch_id = 0
    ffmpeg_proc: subprocess.Popen[bytes] | None = None
    expected_video_size: tuple[int, int] | None = None
    start_time = time.perf_counter()
    last_progress = start_time
    fatal = False

    def queue_size() -> str:
        total = 0
        for task_queue in task_queues:
            try:
                total += int(task_queue.qsize())
            except (AttributeError, NotImplementedError):
                return "?"
        return str(total)

    def print_progress(force: bool = False) -> None:
        nonlocal last_progress
        now = time.perf_counter()
        if not force and now - last_progress < args.progress_interval:
            return
        elapsed = max(1e-6, now - start_time)
        print(
            f"progress: frames={completed_frames}/{len(hms_files)} "
            f"batches={completed_batches}/{len(jobs)} returned={returned_batches} "
            f"fps={completed_frames / elapsed:.2f} queue={queue_size()}",
            flush=True,
        )
        last_progress = now

    def flush_ready_batches() -> None:
        nonlocal completed_batches, completed_frames, next_batch_id, ffmpeg_proc, expected_video_size
        while next_batch_id in pending_batches:
            item = pending_batches.pop(next_batch_id)
            records = item["records"]
            if args.pipe_video:
                size = item.get("size")
                if size is None:
                    raise RuntimeError("worker returned no decoded frame size")
                height, width = int(size[0]), int(size[1])
                if expected_video_size is None:
                    expected_video_size = (height, width)
                    ffmpeg_proc = open_ffmpeg_raw_writer(
                        args.output_video,
                        width=width,
                        height=height,
                        fps=fps,
                        args=args,
                    )
                elif expected_video_size != (height, width):
                    raise ValueError(
                        f"all frames must have the same size for --pipe-video, "
                        f"got {height}x{width} after {expected_video_size[0]}x{expected_video_size[1]}"
                    )
                if ffmpeg_proc is None or ffmpeg_proc.stdin is None:
                    raise RuntimeError("ffmpeg stdin is unavailable")
                for raw_frame in item["raw_frames"]:
                    ffmpeg_proc.stdin.write(raw_frame)

            completed_batches += 1
            completed_frames += int(item["count"])
            manifest["batches"].append(
                {key: value for key, value in item.items() if key not in {"records", "raw_frames"}}
            )
            manifest["frames"].extend(records)
            write_manifest(manifest_path, manifest)
            next_batch_id += 1

    while stopped < len(workers):
        try:
            item = result_queue.get(timeout=1.0)
        except queue.Empty:
            print_progress()
            continue
        kind = item.get("type")
        if kind == "ready":
            print(
                f"worker ready: id={item['worker_id']} device={item['device']} "
                f"model={item['model_variant']}",
                flush=True,
            )
        elif kind == "batch":
            returned_batches += 1
            pending_batches[int(item["batch_id"])] = item
            io_label = "pipe" if args.pipe_video else "save"
            print(
                f"decoded batch={item['batch_id']} worker={item['worker_id']} "
                f"device={item['device']} count={item['count']} "
                f"model={item['model_sec']:.3f}s {io_label}={item['save_sec']:.3f}s",
                flush=True,
            )
            flush_ready_batches()
            print_progress()
        elif kind == "error":
            manifest["errors"].append(item)
            write_manifest(manifest_path, manifest)
            print(f"ERROR worker={item.get('worker_id')} batch={item.get('batch_id')}: {item.get('error')}", flush=True)
            if not args.continue_on_error:
                fatal = True
                break
        elif kind == "stopped":
            stopped += 1

    if fatal:
        for proc in workers:
            if proc.is_alive():
                proc.terminate()
    for proc in workers:
        proc.join(timeout=5.0)
    if not fatal:
        flush_ready_batches()

    print_progress(force=True)
    manifest["completed_frames"] = completed_frames
    manifest["finished_at"] = now_iso()
    write_manifest(manifest_path, manifest)
    if fatal or (completed_frames != len(hms_files) and not args.continue_on_error):
        raise RuntimeError(f"decode failed: completed {completed_frames}/{len(hms_files)} frames")

    if args.pipe_video:
        if ffmpeg_proc is None:
            raise RuntimeError("no frames were decoded")
        if ffmpeg_proc.stdin is not None:
            ffmpeg_proc.stdin.close()
        ret = ffmpeg_proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {ret}")
    else:
        print("encoding video with ffmpeg...")
        encode_video(decoded_dir, args.output_video, fps, args)
    print(f"saved video: {args.output_video}")
    print(f"decode manifest: {manifest_path}")
    if not args.pipe_video and not args.keep_frames:
        shutil.rmtree(decoded_dir, ignore_errors=True)
        print(f"removed decoded frames: {decoded_dir}")


if __name__ == "__main__":
    main()
