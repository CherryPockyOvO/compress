#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}
VIDEO_SUFFIXES = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".m4v",
    ".ts",
}


class CommandError(RuntimeError):
    pass


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def run_command(cmd: Sequence[str], cwd: Optional[Path] = None) -> Tuple[str, str]:
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        rendered = " ".join(str(part) for part in cmd)
        raise CommandError(
            f"command failed with exit code {proc.returncode}: {rendered}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc.stdout, proc.stderr


def parse_timing_json(stdout: str) -> Dict[str, float]:
    prefix = "timing_json:"
    for line in stdout.splitlines():
        if not line.startswith(prefix):
            continue
        try:
            data = json.loads(line[len(prefix) :].strip())
        except json.JSONDecodeError:
            return {}
        return {
            str(key): float(value)
            for key, value in data.items()
            if isinstance(value, (int, float))
        }
    return {}


def fmt_sec(value: Optional[object]) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}s"
    return "n/a"


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def default_cnz_tool() -> Optional[Path]:
    candidates = [
        script_dir() / "rk3588_encode_cnz.sh",
        script_dir() / "cnz_encode_cli",
        script_dir() / "bin" / "cnz_encode_cli",
        script_dir().parent / "cpp" / "scripts" / "rk3588_encode_cnz.sh",
        Path("cpp/scripts/rk3588_encode_cnz.sh"),
        Path("cnz_encode_cli"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def command_prefix_for_tool(path: Path) -> List[str]:
    if path.suffix == ".sh":
        return ["bash", str(path)]
    return [str(path)]


def safe_rel(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def parse_csv(value: str) -> List[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("empty comma-separated value")
    return items


def collect_image_dir(input_path: Path, recursive: bool) -> List[Path]:
    iterator: Iterable[Path]
    if recursive:
        iterator = input_path.rglob("*")
    else:
        iterator = input_path.iterdir()
    frames = sorted(path for path in iterator if path.is_file() and is_image(path))
    if not frames:
        raise FileNotFoundError(f"no image files found in: {input_path}")
    return frames


def ffprobe_video(ffprobe: str, input_path: Path) -> Dict[str, object]:
    if not shutil.which(ffprobe):
        return {}
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(input_path),
    ]
    try:
        stdout, _ = run_command(cmd)
        return json.loads(stdout)
    except Exception:
        return {}


def extract_video_frames(args: argparse.Namespace, temp_root: Path) -> Tuple[List[Path], Dict[str, object]]:
    ffmpeg = args.ffmpeg
    if not shutil.which(ffmpeg):
        raise FileNotFoundError(
            f"ffmpeg not found: {ffmpeg}. Install ffmpeg on RK3588 or pass --ffmpeg PATH."
        )

    frame_dir = temp_root / "video_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    pattern = frame_dir / f"frame_%08d.{args.frame_format}"

    cmd: List[str] = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    if args.start_time:
        cmd += ["-ss", args.start_time]
    cmd += ["-i", str(args.input), "-map", "0:v:0"]
    if args.duration:
        cmd += ["-t", args.duration]
    if args.fps:
        cmd += ["-vf", f"fps={args.fps}"]
    else:
        cmd += ["-vsync", "0"]
    cmd += ["-start_number", "0", str(pattern)]
    run_command(cmd)

    frames = sorted(frame_dir.glob(f"*.{args.frame_format}"))
    if args.frame_limit is not None:
        frames = frames[: args.frame_limit]
    if not frames:
        raise RuntimeError(f"ffmpeg extracted no frames from: {args.input}")
    return frames, ffprobe_video(args.ffprobe, args.input)


def collect_inputs(args: argparse.Namespace, temp_root: Path) -> Tuple[str, List[Path], Dict[str, object]]:
    input_path = args.input
    if input_path.is_dir():
        return "image_directory", collect_image_dir(input_path, args.recursive), {}
    if input_path.is_file() and is_image(input_path):
        return "image", [input_path], {}
    if input_path.is_file() and is_video(input_path):
        frames, video_info = extract_video_frames(args, temp_root)
        return "video", frames, video_info
    raise ValueError(f"unsupported input path or suffix: {input_path}")


def build_encoder_cmd(
    args: argparse.Namespace,
    frame_path: Path,
    bin_path: Path,
    meta_path: Path,
    core_mask: str,
) -> List[str]:
    cmd = [
        args.python,
        str(args.encoder_script),
        "--rknn",
        str(args.rknn),
        "--image",
        str(frame_path),
        "--output",
        str(bin_path),
        "--meta-output",
        str(meta_path),
        "--downsampling-factor",
        str(args.downsampling_factor),
        "--input-layout",
        args.input_layout,
    ]
    if args.height is not None or args.width is not None:
        if args.height is None or args.width is None:
            raise ValueError("--height and --width must be provided together")
        cmd += ["--height", str(args.height), "--width", str(args.width)]
    if not args.no_core_mask_arg:
        cmd += ["--core-mask", core_mask]
    return cmd


def build_cnz_cmd(
    args: argparse.Namespace,
    bin_path: Path,
    meta_path: Path,
    cnz_path: Path,
) -> List[str]:
    cmd = command_prefix_for_tool(args.cnz_tool)
    cmd += [
        "--latent",
        str(bin_path),
        "--params",
        str(args.params),
        "--output",
        str(cnz_path),
        "--metadata",
        str(meta_path),
        "--codec",
        args.codec,
        "--zlib-level",
        str(args.zlib_level),
    ]
    return cmd


def unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def compress_one_frame(
    args: argparse.Namespace,
    index: int,
    frame_path: Path,
    bin_dir: Path,
    cnz_dir: Path,
    core_masks: Sequence[str],
) -> Dict[str, object]:
    t0 = time.time()
    base = f"frame_{index:08d}"
    bin_path = bin_dir / f"{base}.bin"
    meta_path = bin_dir / f"{base}.bin.json"
    cnz_path = cnz_dir / f"{base}.cnz"
    core_mask = core_masks[index % len(core_masks)]

    encoder_cmd = build_encoder_cmd(args, frame_path, bin_path, meta_path, core_mask)
    encoder_t0 = time.time()
    encoder_stdout, _ = run_command(encoder_cmd)
    encoder_elapsed = time.time() - encoder_t0
    encoder_timing = parse_timing_json(encoder_stdout)

    cnz_cmd = build_cnz_cmd(args, bin_path, meta_path, cnz_path)
    entropy_t0 = time.time()
    run_command(cnz_cmd)
    entropy_elapsed = time.time() - entropy_t0

    if not args.keep_bin:
        unlink_if_exists(bin_path)
    if not args.keep_latent_metadata:
        unlink_if_exists(meta_path)

    elapsed = time.time() - t0
    return {
        "index": index,
        "source": str(frame_path),
        "cnz": safe_rel(cnz_path, args.output),
        "core_mask": core_mask,
        "bytes": cnz_path.stat().st_size,
        "elapsed_sec": round(elapsed, 4),
        "frame_fps": round(1.0 / elapsed, 4) if elapsed > 0 else 0.0,
        "bin_total_sec": round(encoder_elapsed, 4),
        "entropy_sec": round(entropy_elapsed, 4),
        "preprocess_sec": round(encoder_timing["preprocess_sec"], 4)
        if "preprocess_sec" in encoder_timing
        else None,
        "runtime_init_sec": round(encoder_timing["runtime_init_sec"], 4)
        if "runtime_init_sec" in encoder_timing
        else None,
        "npu_inference_sec": round(encoder_timing["npu_inference_sec"], 4)
        if "npu_inference_sec" in encoder_timing
        else None,
        "save_bin_sec": round(encoder_timing["save_bin_sec"], 4)
        if "save_bin_sec" in encoder_timing
        else None,
    }


def write_manifest(path: Path, data: Dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch compress an image, image directory, or video into per-frame CNZ files."
    )
    parser.add_argument("--input", type=Path, required=True, help="Image, image directory, or video path.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for CNZ frames.")
    parser.add_argument("--rknn", type=Path, required=True, help="RKNN encoder model.")
    parser.add_argument("--params", type=Path, required=True, help="entropy_params JSON.")
    parser.add_argument(
        "--cnz-tool",
        type=Path,
        default=None,
        help="Path to rk3588_encode_cnz.sh or cnz_encode_cli. Auto-detected by default.",
    )
    parser.add_argument(
        "--encoder-script",
        type=Path,
        default=script_dir() / "rknn_encode_frame.py",
        help="Single-frame RKNN encoder script.",
    )
    parser.add_argument("--python", default=sys.executable or "python3")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--input-layout", choices=["nhwc", "nchw"], default="nhwc")
    parser.add_argument("--downsampling-factor", type=positive_int, default=16)
    parser.add_argument("--workers", type=positive_int, default=3)
    parser.add_argument(
        "--core-masks",
        default="0,1,2",
        help="Comma-separated RKNN core masks assigned round-robin to workers.",
    )
    parser.add_argument(
        "--no-core-mask-arg",
        action="store_true",
        help="Use this if --encoder-script is your old run.py without --core-mask support.",
    )
    parser.add_argument("--codec", choices=["zlib", "none"], default="zlib")
    parser.add_argument("--zlib-level", type=int, default=1)
    parser.add_argument("--recursive", action="store_true", help="Recursively read images from a directory.")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--frame-format", choices=["png", "jpg"], default="png")
    parser.add_argument("--start-time", default=None, help="Optional video start time, e.g. 00:00:05.")
    parser.add_argument("--duration", default=None, help="Optional video duration, e.g. 10 or 00:00:10.")
    parser.add_argument("--fps", default=None, help="Optional ffmpeg fps filter. Default keeps every frame.")
    parser.add_argument("--frame-limit", type=positive_int, default=None)
    parser.add_argument("--work-dir", type=Path, default=None, help="Optional directory for temporary files.")
    parser.add_argument("--keep-bin", action="store_true", help="Keep intermediate latent .bin files.")
    parser.add_argument(
        "--keep-latent-metadata",
        action="store_true",
        help="Keep intermediate latent .bin.json files.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep processing other frames after one frame fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.rknn = args.rknn.resolve()
    args.params = args.params.resolve()
    args.encoder_script = args.encoder_script.resolve()

    if args.cnz_tool is None:
        detected = default_cnz_tool()
        if detected is None:
            raise FileNotFoundError(
                "cnz tool not found. Pass --cnz-tool /path/to/rk3588_encode_cnz.sh "
                "or /path/to/cnz_encode_cli."
            )
        args.cnz_tool = detected
    else:
        args.cnz_tool = args.cnz_tool.resolve()

    require_file(args.rknn, "--rknn")
    require_file(args.params, "--params")
    require_file(args.encoder_script, "--encoder-script")
    require_file(args.cnz_tool, "--cnz-tool")

    core_masks = parse_csv(args.core_masks)
    args.output.mkdir(parents=True, exist_ok=True)
    cnz_dir = args.output / "frames"
    cnz_dir.mkdir(parents=True, exist_ok=True)

    temp_context = None
    if args.work_dir is None:
        if args.keep_bin or args.keep_latent_metadata:
            temp_root = args.output / "_work"
            temp_root.mkdir(parents=True, exist_ok=True)
        else:
            temp_context = tempfile.TemporaryDirectory(prefix="rk3588_media_compress_")
            temp_root = Path(temp_context.name)
    else:
        temp_root = args.work_dir.resolve()
        temp_root.mkdir(parents=True, exist_ok=True)

    try:
        source_type, frames, video_info = collect_inputs(args, temp_root)
        if args.frame_limit is not None and source_type != "video":
            frames = frames[: args.frame_limit]

        bin_dir = temp_root / "latent_bins"
        bin_dir.mkdir(parents=True, exist_ok=True)

        manifest: Dict[str, object] = {
            "format": "rk3588-cnz-frame-sequence-v1",
            "created_at": now_iso(),
            "input": str(args.input),
            "source_type": source_type,
            "output": str(args.output),
            "rknn": str(args.rknn),
            "params": str(args.params),
            "cnz_tool": str(args.cnz_tool),
            "height": args.height,
            "width": args.width,
            "input_layout": args.input_layout,
            "downsampling_factor": args.downsampling_factor,
            "workers": args.workers,
            "core_masks": core_masks,
            "codec": args.codec,
            "zlib_level": args.zlib_level,
            "video_info": video_info,
            "frame_count": len(frames),
            "frames": [],
            "errors": [],
        }
        manifest_path = args.output / "manifest.json"
        write_manifest(manifest_path, manifest)

        print(f"input_type: {source_type}")
        print(f"frames: {len(frames)}")
        print(f"output: {args.output}")
        print(f"workers: {args.workers}, core_masks: {','.join(core_masks)}")

        completed = 0
        run_started = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_index = {
                executor.submit(
                    compress_one_frame,
                    args,
                    index,
                    frame_path,
                    bin_dir,
                    cnz_dir,
                    core_masks,
                ): index
                for index, frame_path in enumerate(frames)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    completed += 1
                    manifest["frames"].append(result)  # type: ignore[index]
                    total_elapsed = time.time() - run_started
                    avg_fps = completed / total_elapsed if total_elapsed > 0 else 0.0
                    print(
                        f"[{completed}/{len(frames)}] frame {index:08d} -> "
                        f"{result['cnz']} "
                        f"total={result['elapsed_sec']:.3f}s "
                        f"bin_total={result['bin_total_sec']:.3f}s "
                        f"pre={fmt_sec(result.get('preprocess_sec'))} "
                        f"rk_init={fmt_sec(result.get('runtime_init_sec'))} "
                        f"npu_infer={fmt_sec(result.get('npu_inference_sec'))} "
                        f"save_bin={fmt_sec(result.get('save_bin_sec'))} "
                        f"entropy={result['entropy_sec']:.3f}s "
                        f"frame_fps={result['frame_fps']:.2f} "
                        f"avg_fps={avg_fps:.2f} "
                        f"({result['bytes']} bytes)"
                    )
                except Exception as exc:
                    error = {"index": index, "error": str(exc)}
                    manifest["errors"].append(error)  # type: ignore[index]
                    print(f"[error] frame {index:08d}: {exc}", file=sys.stderr)
                    if not args.continue_on_error:
                        for pending in future_to_index:
                            pending.cancel()
                        raise
                finally:
                    frames_list = manifest["frames"]  # type: ignore[index]
                    frames_list.sort(key=lambda item: item["index"])
                    total_elapsed = time.time() - run_started
                    manifest["elapsed_sec"] = round(total_elapsed, 4)
                    manifest["avg_fps"] = (
                        round(completed / total_elapsed, 4) if total_elapsed > 0 else None
                    )
                    write_manifest(manifest_path, manifest)

        if manifest["errors"]:  # type: ignore[index]
            raise RuntimeError(f"finished with {len(manifest['errors'])} frame errors")
        total_elapsed = time.time() - run_started
        avg_fps = completed / total_elapsed if total_elapsed > 0 else 0.0
        frame_records = manifest["frames"]  # type: ignore[index]
        if frame_records:
            avg_bin_total = sum(item["bin_total_sec"] for item in frame_records) / len(frame_records)
            avg_entropy = sum(item["entropy_sec"] for item in frame_records) / len(frame_records)
            preprocess_values = [
                item["preprocess_sec"]
                for item in frame_records
                if isinstance(item.get("preprocess_sec"), (int, float))
            ]
            runtime_init_values = [
                item["runtime_init_sec"]
                for item in frame_records
                if isinstance(item.get("runtime_init_sec"), (int, float))
            ]
            npu_values = [
                item["npu_inference_sec"]
                for item in frame_records
                if isinstance(item.get("npu_inference_sec"), (int, float))
            ]
            save_values = [
                item["save_bin_sec"]
                for item in frame_records
                if isinstance(item.get("save_bin_sec"), (int, float))
            ]
            avg_preprocess = (
                sum(preprocess_values) / len(preprocess_values) if preprocess_values else None
            )
            avg_runtime_init = (
                sum(runtime_init_values) / len(runtime_init_values)
                if runtime_init_values
                else None
            )
            avg_npu = sum(npu_values) / len(npu_values) if npu_values else None
            avg_save = sum(save_values) / len(save_values) if save_values else None
            manifest["avg_bin_total_sec"] = round(avg_bin_total, 4)
            manifest["avg_entropy_sec"] = round(avg_entropy, 4)
            manifest["avg_preprocess_sec"] = (
                round(avg_preprocess, 4) if avg_preprocess is not None else None
            )
            manifest["avg_runtime_init_sec"] = (
                round(avg_runtime_init, 4) if avg_runtime_init is not None else None
            )
            manifest["avg_npu_inference_sec"] = round(avg_npu, 4) if avg_npu is not None else None
            manifest["avg_save_bin_sec"] = round(avg_save, 4) if avg_save is not None else None
            write_manifest(manifest_path, manifest)
            print(
                "avg_stage_time: "
                f"bin_total={avg_bin_total:.3f}s, "
                f"pre={fmt_sec(avg_preprocess)}, "
                f"rk_init={fmt_sec(avg_runtime_init)}, "
                f"npu_infer={fmt_sec(avg_npu)}, "
                f"save_bin={fmt_sec(avg_save)}, "
                f"entropy={avg_entropy:.3f}s"
            )
        print(f"done: {completed} frames, total_time={total_elapsed:.3f}s, avg_fps={avg_fps:.2f}")
        print(f"manifest: {manifest_path}")
    finally:
        if temp_context is not None:
            temp_context.cleanup()


if __name__ == "__main__":
    main()
