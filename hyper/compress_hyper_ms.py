from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from compressai_nano import get_model, infer_model_variant_from_checkpoint


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def now_synced(device: torch.device) -> float:
    sync_device(device)
    return time.perf_counter()


def make_autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> None:
    raw = torch.load(checkpoint, map_location="cpu")
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"missing_keys: {len(missing)}")
    print(f"unexpected_keys: {len(unexpected)}")


def load_image_rgb(
    path: Path,
    height: int | None,
    width: int | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    image = Image.open(path).convert("RGB")
    source_w, source_h = image.size
    if height is not None or width is not None:
        if height is None or width is None:
            raise ValueError("--height and --width must be provided together")
        image = image.resize((width, height), Image.Resampling.BICUBIC)
    orig_w, orig_h = image.size

    array = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    metadata: dict[str, Any] = {
        "format": "compressai-nano-hyper-ms-local-v1",
        "image": str(path),
        "source_h": int(source_h),
        "source_w": int(source_w),
        "orig_h": int(orig_h),
        "orig_w": int(orig_w),
        "input_dtype": "float32",
        "resize_input": bool((height is not None) or (width is not None)),
    }
    return tensor, metadata


def pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = x.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, (int(height), int(width))


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()


def symbols_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(device="cpu", dtype=torch.int32).contiguous().numpy()


def compact_symbols(symbols: np.ndarray) -> tuple[np.ndarray, str]:
    min_value = int(symbols.min())
    max_value = int(symbols.max())
    if -32768 <= min_value and max_value <= 32767:
        return symbols.astype("<i2", copy=False), "int16"
    return symbols.astype("<i4", copy=False), "int32"


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_npz(
    output: Path,
    y_symbols_i32: np.ndarray,
    z_symbols_i32: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    y_symbols, y_dtype = compact_symbols(y_symbols_i32)
    z_symbols, z_dtype = compact_symbols(z_symbols_i32)
    metadata = dict(metadata)
    metadata.update(
        {
            "format": "compressai-nano-hyper-ms-local-npz-v1",
            "y_dtype": y_dtype,
            "z_dtype": z_dtype,
            "y_symbol_min": int(y_symbols_i32.min()),
            "y_symbol_max": int(y_symbols_i32.max()),
            "z_symbol_min": int(z_symbols_i32.min()),
            "z_symbol_max": int(z_symbols_i32.max()),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez_compressed(
            handle,
            y_symbols=y_symbols,
            z_symbols=z_symbols,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )


def default_hms_cli() -> Path:
    return Path(__file__).resolve().parent / "cpp" / "build" / "hyper_ms_encode_cli"


def run_hms_encoder(
    cli: Path,
    output: Path,
    params: Path,
    metadata: Path,
    y_path: Path,
    z_path: Path,
    means_path: Path,
    codec: str,
    zlib_level: int,
) -> None:
    if not cli.exists():
        raise FileNotFoundError(
            f"missing hyper_ms_encode_cli: {cli}. Build it with: cd hyper/cpp && cmake -S . -B build && cmake --build build -j"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(cli),
        "--y",
        str(y_path),
        "--z",
        str(z_path),
        "--means",
        str(means_path),
        "--params",
        str(params),
        "--metadata",
        str(metadata),
        "--output",
        str(output),
        "--codec",
        codec,
        "--zlib-level",
        str(zlib_level),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"hyper_ms_encode_cli failed with exit code {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    print(proc.stdout, end="")


def infer_output_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if path.suffix.lower() == ".hms":
        return "hms"
    return "npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local hyper_ms_nano compression with PyTorch analysis and optional C++ HMS entropy packing."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--params", type=Path, default=Path("params/hyper_ms_nano_entropy_params.json"))
    parser.add_argument("--output", type=Path, default=Path("out/image.npz"))
    parser.add_argument("--format", choices=("auto", "npz", "hms"), default="auto")
    parser.add_argument("--hyper-ms-encode-cli", type=Path, default=None)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--padding-multiple", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--codec", choices=("zlib", "none"), default="zlib")
    parser.add_argument("--zlib-level", type=int, default=1)
    parser.add_argument("--keep-float-dir", type=Path, default=None)
    parser.add_argument("--timing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cpu:
        args.device = "cpu"
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.half and device.type != "cuda":
        raise RuntimeError("--half is only supported on CUDA")

    raw = torch.load(args.checkpoint, map_location="cpu")
    model_variant = infer_model_variant_from_checkpoint(raw)
    model = get_model(model_variant=model_variant).to(device).eval()
    if getattr(model, "supports_cnz_v4", False):
        raise RuntimeError("This is a detail/CNZ4 checkpoint. Use detail/encode_image.py instead.")
    if not hasattr(model, "analysis_transform"):
        raise RuntimeError(f"{model_variant} does not expose analysis_transform")
    if args.half:
        model = model.half()

    t0 = now_synced(device)
    load_checkpoint(model, args.checkpoint)
    t1 = now_synced(device)

    x, metadata = load_image_rgb(args.image, args.height, args.width)
    x, original_size = pad_to_multiple(x, args.padding_multiple)
    metadata.update(
        {
            "orig_h": int(original_size[0]),
            "orig_w": int(original_size[1]),
            "padded_h": int(x.shape[-2]),
            "padded_w": int(x.shape[-1]),
            "padding_multiple": int(args.padding_multiple),
            "model_variant": model_variant,
            "model_type": "mean_scale_hyperprior",
        }
    )
    x = x.to(device=device, dtype=torch.float16 if args.half else torch.float32)
    t2 = now_synced(device)

    with torch.inference_mode(), make_autocast(device, args.half):
        y, z, scales_y, means_y = model.analysis_transform(x)
        y_symbols = model.conditional_entropy_y.quantize(y, means_y)
        z_symbols = model.entropy_bottleneck_z.quantize(z)
    t3 = now_synced(device)

    y_np = tensor_to_numpy(y)
    z_np = tensor_to_numpy(z)
    means_np = tensor_to_numpy(means_y)
    scales_np = tensor_to_numpy(scales_y)
    y_symbols_np = symbols_to_numpy(y_symbols)
    z_symbols_np = symbols_to_numpy(z_symbols)
    metadata.update(
        {
            "channels_y": int(y.shape[1]),
            "channels_z": int(z.shape[1]),
            "quant_step_y": float(model.conditional_entropy_y.quant_step.detach().cpu()),
            "quant_step_z": float(model.entropy_bottleneck_z.quant_step.detach().cpu()),
            "y_shape": [int(v) for v in y.shape],
            "z_shape": [int(v) for v in z.shape],
            "scales_shape": [int(v) for v in scales_y.shape],
            "means_shape": [int(v) for v in means_y.shape],
            "scale_min": float(scales_np.min()),
            "scale_mean": float(scales_np.mean()),
            "scale_max": float(scales_np.max()),
        }
    )

    output_format = infer_output_format(args.output, args.format)
    metadata_path = args.metadata_output or Path(str(args.output) + ".json")
    write_metadata(metadata_path, metadata)

    t4 = now_synced(device)
    if output_format == "npz":
        save_npz(args.output, y_symbols_np, z_symbols_np, metadata)
    else:
        cli = args.hyper_ms_encode_cli or default_hms_cli()
        if args.keep_float_dir is not None:
            work_dir = args.keep_float_dir
            work_dir.mkdir(parents=True, exist_ok=True)
            y_path = work_dir / "y.bin"
            z_path = work_dir / "z.bin"
            means_path = work_dir / "means_y.bin"
            y_np.astype("<f4", copy=False).tofile(y_path)
            z_np.astype("<f4", copy=False).tofile(z_path)
            means_np.astype("<f4", copy=False).tofile(means_path)
            run_hms_encoder(
                cli,
                args.output,
                args.params,
                metadata_path,
                y_path,
                z_path,
                means_path,
                args.codec,
                args.zlib_level,
            )
        else:
            with tempfile.TemporaryDirectory(prefix="hyper_ms_local_") as tmp:
                work_dir = Path(tmp)
                y_path = work_dir / "y.bin"
                z_path = work_dir / "z.bin"
                means_path = work_dir / "means_y.bin"
                y_np.astype("<f4", copy=False).tofile(y_path)
                z_np.astype("<f4", copy=False).tofile(z_path)
                means_np.astype("<f4", copy=False).tofile(means_path)
                run_hms_encoder(
                    cli,
                    args.output,
                    args.params,
                    metadata_path,
                    y_path,
                    z_path,
                    means_path,
                    args.codec,
                    args.zlib_level,
                )
    t5 = now_synced(device)

    pixels = max(1, int(metadata["orig_h"]) * int(metadata["orig_w"]))
    print(f"device: {device}")
    print(f"model_variant: {model_variant}")
    print(f"output: {args.output}")
    print(f"metadata: {metadata_path}")
    print(f"format: {output_format}")
    print(f"y_shape: {tuple(int(v) for v in y.shape)}")
    print(f"z_shape: {tuple(int(v) for v in z.shape)}")
    print(f"package_bpp: {args.output.stat().st_size * 8.0 / pixels:.4f}")
    if args.timing:
        print(f"timing_load_checkpoint_ms={(t1 - t0) * 1000:.3f}")
        print(f"timing_preprocess_ms={(t2 - t1) * 1000:.3f}")
        print(f"timing_analysis_ms={(t3 - t2) * 1000:.3f}")
        print(f"timing_pack_prepare_ms={(t4 - t3) * 1000:.3f}")
        print(f"timing_write_package_ms={(t5 - t4) * 1000:.3f}")
        print(f"timing_total_ms={(t5 - t0) * 1000:.3f}")


if __name__ == "__main__":
    main()
