#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def run_command(cmd: Sequence[str]) -> Tuple[str, str]:
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


def fmt_sec(value: Optional[object]) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}s"
    return "n/a"


def rational_to_float(value: Any) -> Optional[float]:
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
        result = num / den
        return result if result > 0 else None

    try:
        result = float(text)
    except ValueError:
        return None
    return result if result > 0 else None


def fps_from_manifest(manifest: Optional[Dict[str, Any]]) -> Optional[float]:
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


def load_manifest(input_path: Path) -> Optional[Dict[str, Any]]:
    manifest_path = input_path / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def collect_cnz_from_manifest(input_path: Path, manifest: Dict[str, Any]) -> List[Tuple[int, Path]]:
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        return []

    collected: List[Tuple[int, Path]] = []
    for pos, item in enumerate(frames):
        if not isinstance(item, dict) or "cnz" not in item:
            continue
        index = int(item.get("index", pos))
        cnz_path = input_path / str(item["cnz"])
        collected.append((index, cnz_path))

    collected.sort(key=lambda pair: pair[0])
    return collected


def collect_cnz_files(input_path: Path) -> Tuple[List[Tuple[int, Path]], Optional[Dict[str, Any]]]:
    if input_path.is_file() and input_path.suffix.lower() == ".cnz":
        return [(0, input_path.resolve())], None
    if not input_path.is_dir():
        raise FileNotFoundError(f"input must be a .cnz file or directory: {input_path}")

    manifest = load_manifest(input_path)
    if manifest is not None:
        files = collect_cnz_from_manifest(input_path, manifest)
        missing = [path for _, path in files if not path.exists()]
        if missing:
            raise FileNotFoundError(f"manifest references missing CNZ file: {missing[0]}")
        if files:
            return files, manifest

    search_roots = []
    frames_dir = input_path / "frames"
    if frames_dir.exists():
        search_roots.append(frames_dir)
    search_roots.append(input_path)

    seen: set[Path] = set()
    found: List[Path] = []
    for root in search_roots:
        for path in sorted(root.rglob("*.cnz")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(resolved)

    if not found:
        raise FileNotFoundError(f"no .cnz files found under: {input_path}")
    return [(idx, path) for idx, path in enumerate(found)], manifest


@contextlib.contextmanager
def maybe_suppress_stdout(verbose: bool):
    if verbose:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def load_decoder_modules() -> Tuple[Any, Any, Any]:
    import torch

    import decode_cnz as cnz_decoder
    from compressai_nano import FactorizedPriorNano

    return torch, cnz_decoder, FactorizedPriorNano


def setup_model(
    args: argparse.Namespace,
    torch_module: Any,
    cnz_decoder: Any,
    model_cls: Any,
) -> Tuple[Any, Any]:
    torch = torch_module
    if args.cpu:
        args.device = "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"device: cuda ({torch.cuda.get_device_name(device)})")
    else:
        print("device: cpu")

    model = model_cls().to(device).eval()
    if device.type == "cuda" and args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.half:
        if device.type != "cuda":
            raise RuntimeError("--half is only supported with CUDA")
        model = model.half()

    t0 = cnz_decoder.now_synced(device)
    cnz_decoder.load_checkpoint(model, args.checkpoint)
    t1 = cnz_decoder.now_synced(device)
    print(f"checkpoint_load_time={(t1 - t0):.3f}s")
    return model, device


def decode_one_cnz(
    cnz_path: Path,
    output_path: Path,
    model: Any,
    device: Any,
    use_half: bool,
    verbose: bool,
    torch_module: Any,
    cnz_decoder: Any,
) -> Tuple[float, Tuple[int, int], Dict[str, Optional[float]]]:
    torch = torch_module
    t0 = cnz_decoder.now_synced(device)
    with torch.inference_mode():
        magic_t0 = cnz_decoder.now_synced(device)
        with cnz_path.open("rb") as input_file:
            prefix = input_file.read(4)
        magic_t1 = cnz_decoder.now_synced(device)

        cnz_unpack_sec: Optional[float] = None
        model_sec: Optional[float] = None
        legacy_sec: Optional[float] = None

        if prefix == cnz_decoder.MAGIC:
            unpack_t0 = cnz_decoder.now_synced(device)
            cnz_file = cnz_decoder.read_cnz_file(cnz_path)
            y_hat = cnz_decoder.cnz_to_y_hat(cnz_file, device=device)
            if use_half:
                y_hat = y_hat.to(torch.float16)
            unpack_t1 = cnz_decoder.now_synced(device)

            model_t0 = cnz_decoder.now_synced(device)
            with cnz_decoder.make_autocast(device, use_half):
                x_hat = model.decoder(y_hat)
            model_t1 = cnz_decoder.now_synced(device)

            original_size = (cnz_file.header.orig_h, cnz_file.header.orig_w)
            cnz_unpack_sec = unpack_t1 - unpack_t0
            model_sec = model_t1 - model_t0
        else:
            legacy_t0 = cnz_decoder.now_synced(device)
            with maybe_suppress_stdout(verbose):
                x_hat, original_size = cnz_decoder.decode_legacy(cnz_path, model, device)
            legacy_t1 = cnz_decoder.now_synced(device)
            legacy_sec = legacy_t1 - legacy_t0

        crop_t0 = cnz_decoder.now_synced(device)
        x_hat = cnz_decoder.crop_to_size(x_hat, original_size)
        crop_t1 = cnz_decoder.now_synced(device)

    save_t0 = cnz_decoder.now_synced(device)
    cnz_decoder.tensor_to_image(x_hat, output_path)
    t1 = cnz_decoder.now_synced(device)
    timings: Dict[str, Optional[float]] = {
        "read_magic_sec": magic_t1 - magic_t0,
        "cnz_unpack_sec": cnz_unpack_sec,
        "torch_decoder_sec": model_sec,
        "legacy_decode_sec": legacy_sec,
        "crop_sec": crop_t1 - crop_t0,
        "save_image_sec": t1 - save_t0,
    }
    return t1 - t0, original_size, timings


def batched(items: Sequence[Tuple[int, Path]], batch_size: int) -> Iterable[List[Tuple[int, Path]]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


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
        raise FileExistsError(f"output video already exists, pass --overwrite: {output_video}")

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


def tensor_batch_to_uint8_nhwc(torch_module: Any, tensor: Any) -> Any:
    torch = torch_module
    tensor = tensor.detach().cpu().clamp(0, 1)
    tensor = (tensor * 255.0).round().to(torch.uint8)
    return tensor.permute(0, 2, 3, 1).contiguous()


def decode_cnz_batch(
    batch_items: Sequence[Tuple[int, Path]],
    output_start_index: int,
    args: argparse.Namespace,
    model: Any,
    device: Any,
    torch_module: Any,
    cnz_decoder: Any,
    ffmpeg_proc: Optional[subprocess.Popen[bytes]],
    expected_video_size: Optional[Tuple[int, int]],
) -> Tuple[List[Dict[str, Any]], Optional[subprocess.Popen[bytes]], Optional[Tuple[int, int]], Dict[str, float]]:
    torch = torch_module
    total_t0 = cnz_decoder.now_synced(device)

    unpack_t0 = cnz_decoder.now_synced(device)
    y_hats = []
    original_sizes: List[Tuple[int, int]] = []
    cnz_paths: List[Path] = []
    for _, cnz_path in batch_items:
        prefix = cnz_path.read_bytes()[:4]
        if prefix != cnz_decoder.MAGIC:
            raise ValueError(
                "batched decode only supports CNZ4 files. "
                "Use --batch-size 1 for legacy streams."
            )
        cnz_file = cnz_decoder.read_cnz_file(cnz_path)
        y_hat = cnz_decoder.cnz_to_y_hat(cnz_file, device=device)
        if args.half:
            y_hat = y_hat.to(torch.float16)
        y_hats.append(y_hat)
        original_sizes.append((int(cnz_file.header.orig_h), int(cnz_file.header.orig_w)))
        cnz_paths.append(cnz_path)

    first_shape = tuple(y_hats[0].shape[1:])
    if any(tuple(y_hat.shape[1:]) != first_shape for y_hat in y_hats):
        raise ValueError("batch contains different latent shapes; use --batch-size 1")

    y_hat_batch = torch.cat(y_hats, dim=0)
    unpack_t1 = cnz_decoder.now_synced(device)

    torch_t0 = cnz_decoder.now_synced(device)
    with torch.inference_mode():
        with cnz_decoder.make_autocast(device, args.half):
            x_hat_batch = model.decoder(y_hat_batch)
    torch_t1 = cnz_decoder.now_synced(device)

    crop_t0 = cnz_decoder.now_synced(device)
    cropped = []
    for i, original_size in enumerate(original_sizes):
        cropped.append(cnz_decoder.crop_to_size(x_hat_batch[i : i + 1], original_size))
    crop_t1 = cnz_decoder.now_synced(device)

    save_t0 = cnz_decoder.now_synced(device)
    batch_records: List[Dict[str, Any]] = []
    if args.pipe_video:
        height, width = original_sizes[0]
        if any(size != (height, width) for size in original_sizes):
            raise ValueError("--pipe-video requires all frames in a batch to have the same output size")
        if expected_video_size is None:
            expected_video_size = (height, width)
        elif expected_video_size != (height, width):
            raise ValueError("--pipe-video requires all decoded frames to have the same output size")
        if ffmpeg_proc is None:
            ffmpeg_proc = open_ffmpeg_raw_writer(
                args.output_video,
                width=width,
                height=height,
                fps=args.resolved_fps,
                args=args,
            )
        if ffmpeg_proc.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        for tensor in cropped:
            rgb = tensor_batch_to_uint8_nhwc(torch, tensor)
            ffmpeg_proc.stdin.write(rgb.numpy().tobytes())
    else:
        for i, tensor in enumerate(cropped):
            image_path = args.decoded_dir / f"frame_{output_start_index + i:08d}.{args.image_format}"
            cnz_decoder.tensor_to_image(tensor, image_path)
    save_t1 = cnz_decoder.now_synced(device)

    total_sec = save_t1 - total_t0
    timings = {
        "decode_sec": total_sec,
        "cnz_unpack_sec": unpack_t1 - unpack_t0,
        "torch_decoder_sec": torch_t1 - torch_t0,
        "crop_sec": crop_t1 - crop_t0,
        "save_or_pipe_sec": save_t1 - save_t0,
    }
    per_frame_total = total_sec / len(batch_items)

    for i, ((source_index, cnz_path), original_size) in enumerate(zip(batch_items, original_sizes)):
        image_path = None
        if not args.pipe_video:
            image_path = args.decoded_dir / f"frame_{output_start_index + i:08d}.{args.image_format}"
        batch_records.append(
            {
                "output_index": output_start_index + i,
                "source_index": source_index,
                "cnz": str(cnz_path),
                "image": str(image_path) if image_path is not None else None,
                "height": int(original_size[0]),
                "width": int(original_size[1]),
                "decode_sec": round(per_frame_total, 4),
                "decode_fps": round(1.0 / per_frame_total, 4) if per_frame_total > 0 else 0.0,
                "batch_size": len(batch_items),
                "batch_decode_sec": round(total_sec, 4),
                "cnz_unpack_sec": round(timings["cnz_unpack_sec"] / len(batch_items), 4),
                "torch_decoder_sec": round(timings["torch_decoder_sec"] / len(batch_items), 4),
                "crop_sec": round(timings["crop_sec"] / len(batch_items), 4),
                "save_or_pipe_sec": round(timings["save_or_pipe_sec"] / len(batch_items), 4),
            }
        )

    return batch_records, ffmpeg_proc, expected_video_size, timings


def build_video(
    decoded_dir: Path,
    image_format: str,
    output_video: Path,
    fps: float,
    args: argparse.Namespace,
) -> float:
    if not shutil.which(args.ffmpeg):
        raise FileNotFoundError(f"ffmpeg not found: {args.ffmpeg}")
    if output_video.exists() and not args.overwrite:
        raise FileExistsError(f"output video already exists, pass --overwrite: {output_video}")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    pattern = decoded_dir / f"frame_%08d.{image_format}"
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

    t0 = time.perf_counter()
    run_command(cmd)
    return time.perf_counter() - t0


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode a RK3588 CNZ frame folder on PC and combine frames into a video."
    )
    parser.add_argument("--input", type=Path, required=True, help="RK output folder or one .cnz file.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="PyTorch decoder checkpoint.")
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument(
        "--decoded-dir",
        type=Path,
        default=None,
        help="Where decoded frame images are written. Default: <output-video-stem>_frames.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Output video FPS. Default: manifest FPS or 30.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Decode this many CNZ frames per PyTorch decoder call. Use 1 for mixed shapes.",
    )
    parser.add_argument(
        "--pipe-video",
        action="store_true",
        help="Do not write intermediate PNG/JPG frames; pipe decoded RGB frames directly to ffmpeg.",
    )
    parser.add_argument("--image-format", choices=["png", "jpg"], default="png")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--half", action="store_true", help="Use CUDA FP16 decoder.")
    parser.add_argument(
        "--channels-last",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use channels-last memory format on CUDA.",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--video-codec", default="libx264")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--pix-fmt", default="yuv420p")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cleanup-frames", action="store_true")
    parser.add_argument("--verbose-decode", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input = args.input.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_video = args.output_video.resolve()
    if args.decoded_dir is None:
        args.decoded_dir = args.output_video.parent / f"{args.output_video.stem}_frames"
    else:
        args.decoded_dir = args.decoded_dir.resolve()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {args.checkpoint}")
    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    cnz_files, rk_manifest = collect_cnz_files(args.input)
    fps = args.fps or fps_from_manifest(rk_manifest) or 30.0
    args.resolved_fps = fps

    if not args.pipe_video:
        args.decoded_dir.mkdir(parents=True, exist_ok=True)
    decode_manifest_path = args.output_video.with_suffix(args.output_video.suffix + ".decode_manifest.json")

    print(f"input: {args.input}")
    print(f"cnz_frames: {len(cnz_files)}")
    if args.pipe_video:
        print("decoded_frames: pipe-to-ffmpeg")
    else:
        print(f"decoded_frames: {args.decoded_dir}")
    print(f"output_video: {args.output_video}")
    print(f"fps: {fps:.6f}")
    print(f"batch_size: {args.batch_size}")

    torch_module, cnz_decoder, model_cls = load_decoder_modules()
    model, device = setup_model(args, torch_module, cnz_decoder, model_cls)

    manifest: Dict[str, Any] = {
        "format": "pc-cnz-folder-video-decode-v1",
        "created_at": now_iso(),
        "input": str(args.input),
        "checkpoint": str(args.checkpoint),
        "decoded_dir": None if args.pipe_video else str(args.decoded_dir),
        "output_video": str(args.output_video),
        "fps": fps,
        "device": str(device),
        "batch_size": args.batch_size,
        "pipe_video": args.pipe_video,
        "frames": [],
    }

    total_t0 = time.perf_counter()
    ffmpeg_proc: Optional[subprocess.Popen[bytes]] = None
    expected_video_size: Optional[Tuple[int, int]] = None
    output_index = 0
    video_t0 = time.perf_counter()
    for batch_items in batched(cnz_files, args.batch_size):
        if args.batch_size == 1 and not args.pipe_video:
            source_index, cnz_path = batch_items[0]
            image_path = args.decoded_dir / f"frame_{output_index:08d}.{args.image_format}"
            decode_sec, original_size, timings = decode_one_cnz(
                cnz_path,
                image_path,
                model,
                device,
                use_half=args.half,
                verbose=args.verbose_decode,
                torch_module=torch_module,
                cnz_decoder=cnz_decoder,
            )
            record = {
                "output_index": output_index,
                "source_index": source_index,
                "cnz": str(cnz_path),
                "image": str(image_path),
                "height": int(original_size[0]),
                "width": int(original_size[1]),
                "decode_sec": round(decode_sec, 4),
                "decode_fps": round(1.0 / decode_sec, 4) if decode_sec > 0 else 0.0,
            }
            for key, value in timings.items():
                record[key] = round(value, 4) if isinstance(value, (int, float)) else None
            manifest["frames"].append(record)
            print(
                f"[{output_index + 1}/{len(cnz_files)}] "
                f"{cnz_path.name} -> {image_path.name} "
                f"decode={decode_sec:.3f}s "
                f"unpack={fmt_sec(timings.get('cnz_unpack_sec'))} "
                f"torch={fmt_sec(timings.get('torch_decoder_sec'))} "
                f"save={fmt_sec(timings.get('save_image_sec'))} "
                f"fps={record['decode_fps']:.2f}"
            )
            output_index += 1
        else:
            records, ffmpeg_proc, expected_video_size, timings = decode_cnz_batch(
                batch_items,
                output_index,
                args,
                model,
                device,
                torch_module,
                cnz_decoder,
                ffmpeg_proc,
                expected_video_size,
            )
            manifest["frames"].extend(records)
            batch_count = len(records)
            output_index += batch_count
            batch_fps = batch_count / timings["decode_sec"] if timings["decode_sec"] > 0 else 0.0
            first = records[0]["output_index"]
            last = records[-1]["output_index"]
            print(
                f"[{output_index}/{len(cnz_files)}] frames {first:08d}-{last:08d} "
                f"batch={batch_count} decode={timings['decode_sec']:.3f}s "
                f"unpack={timings['cnz_unpack_sec']:.3f}s "
                f"torch={timings['torch_decoder_sec']:.3f}s "
                f"save_pipe={timings['save_or_pipe_sec']:.3f}s "
                f"batch_fps={batch_fps:.2f}"
            )
        write_json(decode_manifest_path, manifest)

    if args.pipe_video:
        assert ffmpeg_proc is not None
        if ffmpeg_proc.stdin is not None:
            ffmpeg_proc.stdin.close()
        ret = ffmpeg_proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {ret}")
        video_sec = time.perf_counter() - video_t0
    else:
        video_sec = build_video(args.decoded_dir, args.image_format, args.output_video, fps, args)
    total_sec = time.perf_counter() - total_t0
    manifest["video_encode_sec"] = round(video_sec, 4)
    manifest["total_sec"] = round(total_sec, 4)
    manifest["avg_decode_fps"] = round(len(cnz_files) / total_sec, 4) if total_sec > 0 else 0.0
    frame_records = manifest["frames"]
    if frame_records:
        for key in ("cnz_unpack_sec", "torch_decoder_sec"):
            values = [
                item[key]
                for item in frame_records
                if isinstance(item.get(key), (int, float))
            ]
            manifest[f"avg_{key}"] = round(sum(values) / len(values), 4) if values else None
        save_values = [
            item.get("save_or_pipe_sec", item.get("save_image_sec"))
            for item in frame_records
        ]
        save_values = [value for value in save_values if isinstance(value, (int, float))]
        manifest["avg_save_or_pipe_sec"] = (
            round(sum(save_values) / len(save_values), 4) if save_values else None
        )
    write_json(decode_manifest_path, manifest)

    if args.cleanup_frames and not args.pipe_video:
        shutil.rmtree(args.decoded_dir)
        print(f"removed decoded frames: {args.decoded_dir}")

    print(f"video_encode_time={video_sec:.3f}s")
    if frame_records:
        print(
            "avg_decode_stage_time: "
            f"unpack={fmt_sec(manifest.get('avg_cnz_unpack_sec'))}, "
            f"torch={fmt_sec(manifest.get('avg_torch_decoder_sec'))}, "
            f"save_pipe={fmt_sec(manifest.get('avg_save_or_pipe_sec'))}"
        )
    print(f"done: {len(cnz_files)} frames, total_time={total_sec:.3f}s")
    print(f"saved_video: {args.output_video}")
    print(f"decode_manifest: {decode_manifest_path}")


if __name__ == "__main__":
    main()
