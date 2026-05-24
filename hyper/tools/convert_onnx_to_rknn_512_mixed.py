from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RK3588_QUANT_DTYPES = {"w8a8", "w16a16i", "w16a16i_dfp"}


def collect_images(path: Path, limit: int) -> list[Path]:
    if path.is_file():
        images = [path]
    elif path.is_dir():
        images = sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        )
    else:
        raise FileNotFoundError(path)
    if not images:
        raise RuntimeError(f"no calibration images found under: {path}")
    return images[: max(1, int(limit))]


def prepare_calibration_sample(image_path: Path, height: int, width: int, layout: str) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required when using --calib-images. Install it in the rknn env with: "
            "/home/zzw/miniconda3/envs/rknn/bin/python -m pip install pillow"
        ) from exc

    image = Image.open(image_path).convert("RGB")
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    array = np.asarray(image).astype(np.float32) / 255.0
    if layout == "nhwc":
        return np.expand_dims(array, axis=0).astype(np.float32, copy=False)
    return np.transpose(np.expand_dims(array, axis=0), (0, 3, 1, 2)).copy()


def make_calibration_dataset(args: argparse.Namespace) -> Path | None:
    if args.dataset is not None:
        return args.dataset
    if args.calib_images is None:
        return None

    images = collect_images(args.calib_images, args.calib_count)
    calib_dir = args.calib_cache_dir
    if calib_dir is None:
        calib_dir = args.output.parent / f"{args.output.stem}_calib_npy"
    calib_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = calib_dir / "dataset.txt"
    lines: list[str] = []
    for index, image_path in enumerate(images):
        sample = prepare_calibration_sample(
            image_path,
            height=int(args.height),
            width=int(args.width),
            layout=str(args.calib_layout),
        )
        sample_path = calib_dir / f"calib_{index:04d}.npy"
        np.save(sample_path, sample)
        lines.append(str(sample_path.resolve()))
    dataset_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"calibration_images: {len(images)}")
    print(f"calibration_dataset: {dataset_path}")
    return dataset_path


def default_input_name(part: str) -> str:
    if part == "decoder":
        return "y_hat"
    if part == "hyper-decoder":
        return "z_hat"
    return "input"


def default_input_shape(args: argparse.Namespace) -> list[int]:
    height = int(args.height)
    width = int(args.width)
    if args.part == "decoder":
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError("decoder input shape requires height/width divisible by 16")
        return [1, int(args.channels_y), height // 16, width // 16]
    if args.part == "hyper-decoder":
        if height % 64 != 0 or width % 64 != 0:
            raise ValueError("hyper-decoder input shape requires height/width divisible by 64")
        return [1, int(args.channels_z), height // 64, width // 64]
    return [1, 3, height, width]


def resolve_precision(args: argparse.Namespace, dataset: Path | None) -> str:
    if args.precision != "auto":
        precision = str(args.precision)
    elif args.part == "analysis" and dataset is not None:
        precision = "mixed"
    elif args.part == "generic" and dataset is not None:
        precision = "mixed"
    else:
        precision = "fp16"

    if precision in {"int8", "mixed"} and dataset is None:
        raise ValueError(f"--precision {precision} requires --dataset or --calib-images")
    return precision


def validate_quantized_dtype(args: argparse.Namespace, precision: str) -> None:
    if precision == "fp16":
        return
    target = str(args.target_platform).lower()
    dtype = str(args.quantized_dtype)
    if target == "rk3588" and dtype not in RK3588_QUANT_DTYPES:
        supported = ", ".join(sorted(RK3588_QUANT_DTYPES))
        raise ValueError(
            f"RK3588 does not support quantized_dtype={dtype!r}. "
            f"Supported quantized dtypes are: {supported}. "
            "For quality-first hyper_ms analysis export, use --precision fp16, "
            "or use --quantized-dtype w8a8 for RKNN auto-hybrid INT8+FP16."
        )


def build_custom_string(args: argparse.Namespace, precision: str, dataset: Path | None) -> str:
    payload = {
        "tool": "convert_onnx_to_rknn_512_mixed.py",
        "part": args.part,
        "precision": precision,
        "height": int(args.height),
        "width": int(args.width),
        "dataset": str(dataset) if dataset is not None else None,
        "policy": (
            "quality-first: analysis may use RKNN auto-hybrid INT8+FP16; "
            "hyper-decoder/decoder default to FP16"
        ),
    }
    return json.dumps(payload, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert 512x512 hyper_ms ONNX to RKNN with quality-first mixed precision policy."
    )
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--part",
        choices=("analysis", "decoder", "hyper-decoder", "generic"),
        default="analysis",
        help=(
            "analysis=image->y,z,scales,means; decoder=y_hat->image; "
            "hyper-decoder=z_hat->scales,means."
        ),
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "fp16", "int8", "mixed"),
        default="auto",
        help=(
            "auto keeps decoder/hyper-decoder FP16. analysis becomes mixed when "
            "a calibration dataset is provided, otherwise FP16."
        ),
    )
    parser.add_argument("--target-platform", default="rk3588")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--input-name", default=None)
    parser.add_argument("--force-input-size", action="store_true")
    parser.add_argument("--channels-y", type=int, default=192)
    parser.add_argument("--channels-z", type=int, default=128)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--calib-images", type=Path, default=None)
    parser.add_argument("--calib-cache-dir", type=Path, default=None)
    parser.add_argument("--calib-count", type=int, default=64)
    parser.add_argument(
        "--calib-layout",
        choices=("nchw", "nhwc"),
        default="nchw",
        help="Use nchw for ONNX exported by this repo. Use nhwc only for an explicitly NHWC ONNX.",
    )
    parser.add_argument("--quantized-dtype", default="w8a8")
    parser.add_argument("--quantized-algorithm", choices=("normal", "mmse", "kl_divergence"), default="mmse")
    parser.add_argument("--quantized-method", default="channel")
    parser.add_argument("--auto-hybrid-cos-thresh", type=float, default=0.995)
    parser.add_argument("--auto-hybrid-euc-thresh", type=float, default=None)
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--compress-weight", action="store_true")
    parser.add_argument("--single-core-mode", action="store_true")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.height <= 0 or args.width <= 0:
        raise ValueError("--height/--width must be positive")
    if args.part != "generic" and (args.height != 512 or args.width != 512):
        print(f"warning: this script is tuned for 512x512, got {args.width}x{args.height}")

    dataset = make_calibration_dataset(args)
    precision = resolve_precision(args, dataset)
    validate_quantized_dtype(args, precision)
    input_name = args.input_name or default_input_name(str(args.part))

    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise RuntimeError(
            "rknn-toolkit2 is not installed in this Python environment. "
            "Run this script in your PC rknn conversion environment."
        ) from exc

    print(f"onnx: {args.onnx}")
    print(f"output: {args.output}")
    print(f"part: {args.part}")
    print(f"precision: {precision}")
    if precision == "mixed":
        print(
            "mixed_policy: RKNN auto_hybrid with float_dtype=float16, "
            f"cos_thresh={args.auto_hybrid_cos_thresh}"
        )
    elif precision == "fp16":
        print("mixed_policy: disabled; exporting FP16 RKNN")
    else:
        print("mixed_policy: disabled; exporting full INT8 RKNN")

    rknn = RKNN(verbose=bool(args.verbose))
    try:
        config_kwargs: dict[str, Any] = {
            "target_platform": args.target_platform,
            "optimization_level": int(args.optimization_level),
            "float_dtype": "float16",
            "quantized_dtype": args.quantized_dtype,
            "quantized_algorithm": args.quantized_algorithm,
            "quantized_method": args.quantized_method,
            "compress_weight": bool(args.compress_weight),
            "single_core_mode": bool(args.single_core_mode),
            "custom_string": build_custom_string(args, precision, dataset),
            "auto_hybrid_cos_thresh": float(args.auto_hybrid_cos_thresh),
        }
        if args.auto_hybrid_euc_thresh is not None and not math.isnan(args.auto_hybrid_euc_thresh):
            config_kwargs["auto_hybrid_euc_thresh"] = float(args.auto_hybrid_euc_thresh)

        ret = rknn.config(**config_kwargs)
        if ret != 0:
            raise RuntimeError(f"rknn.config failed: {ret}")

        load_kwargs: dict[str, Any] = {"model": str(args.onnx)}
        if args.force_input_size:
            input_shape = default_input_shape(args)
            load_kwargs["inputs"] = [input_name]
            load_kwargs["input_size_list"] = [input_shape]
            print(f"forced_input: {input_name} {input_shape}")
        ret = rknn.load_onnx(**load_kwargs)
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx failed: {ret}")

        if precision == "fp16":
            ret = rknn.build(do_quantization=False)
        elif precision == "mixed":
            ret = rknn.build(
                do_quantization=True,
                dataset=str(dataset),
                auto_hybrid=True,
            )
        else:
            ret = rknn.build(do_quantization=True, dataset=str(dataset))
        if ret != 0:
            raise RuntimeError(f"rknn.build failed: {ret}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        ret = rknn.export_rknn(str(args.output))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn failed: {ret}")
        print(f"saved: {args.output}")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
