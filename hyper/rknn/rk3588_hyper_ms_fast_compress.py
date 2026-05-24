#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_rknn_lite_class() -> Any:
    try:
        from rknnlite.api import RKNNLite
    except ImportError as exc:
        raise RuntimeError(
            "rknnlite is not installed. Run this script on the RK3588 board "
            "inside your RKNNLite Python environment."
        ) from exc
    return RKNNLite


def resolve_core_mask(rknn_lite_class: Any, name: str) -> int:
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
        raise ValueError(f"unknown --core-mask value: {name}")
    if hasattr(rknn_lite_class, attr):
        return int(getattr(rknn_lite_class, attr))
    if normalized in {"auto", "any", "all", "0_1_2", "012"}:
        return int(rknn_lite_class.NPU_CORE_AUTO)
    raise ValueError(f"RKNNLite on this board does not expose {attr}")


class AnalysisRKNN:
    def __init__(self, rknn_path: Path, core_mask: str) -> None:
        self.rknn_path = Path(rknn_path)
        self.core_mask = str(core_mask)
        self.rknn: Any | None = None

    def __enter__(self) -> "AnalysisRKNN":
        rknn_lite_class = load_rknn_lite_class()
        self.rknn = rknn_lite_class()
        ret = self.rknn.load_rknn(str(self.rknn_path))
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")
        ret = self.rknn.init_runtime(core_mask=resolve_core_mask(rknn_lite_class, self.core_mask))
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        if self.rknn is not None:
            self.rknn.release()

    def infer(self, input_data: np.ndarray) -> list[np.ndarray]:
        if self.rknn is None:
            raise RuntimeError("RKNN runtime is not initialized")
        outputs = self.rknn.inference(inputs=[input_data])
        if outputs is None:
            raise RuntimeError("RKNN inference returned None")
        return [np.asarray(output) for output in outputs]


def load_params(path: Path) -> dict[str, Any]:
    params = json.loads(path.read_text(encoding="utf-8"))
    required = ("channels_y", "channels_z", "quant_step_y", "quant_step_z", "z_medians")
    for key in required:
        if key not in params:
            raise RuntimeError(f"params missing {key}: {path}")
    if not params.get("has_means_y", False):
        raise RuntimeError("expected mean-scale hyperprior params with has_means_y=true")
    return params


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def make_output_path(image_path: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_file():
        if output_root.suffix:
            return output_root
        return output_root / f"{image_path.stem}.npz"
    rel = image_path.relative_to(input_root)
    return output_root / rel.with_suffix(".npz")


def prepare_image(
    image_path: Path,
    width: int,
    height: int,
    resize: bool,
    input_layout: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    image = Image.open(image_path).convert("RGB")
    source_w, source_h = image.size
    if resize:
        image = image.resize((width, height), Image.Resampling.BICUBIC)
    elif image.size != (width, height):
        raise RuntimeError(
            f"{image_path} is {source_w}x{source_h}, but RKNN expects {width}x{height}; "
            "pass --resize or use a matching image."
        )

    array = np.asarray(image).astype(np.float32) / 255.0
    nhwc = np.expand_dims(array, axis=0).astype(np.float32, copy=False)
    if input_layout == "nchw":
        input_data = np.transpose(nhwc, (0, 3, 1, 2)).copy()
    else:
        input_data = nhwc

    metadata = {
        "format": "compressai-nano-hyper-ms-rknn-npz-v1",
        "image": str(image_path),
        "source_h": int(source_h),
        "source_w": int(source_w),
        "orig_h": int(height),
        "orig_w": int(width),
        "padded_h": int(height),
        "padded_w": int(width),
        "resize_input": bool(resize),
        "input_layout": str(input_layout),
        "input_dtype": "float32",
    }
    return input_data, metadata


def to_nchw(output: np.ndarray, channels: int, name: str) -> np.ndarray:
    array = np.asarray(output)
    if array.ndim != 4:
        raise RuntimeError(f"{name} must be 4D, got {array.shape}")
    if array.shape[1] == channels:
        return array.astype(np.float32, copy=False)
    if array.shape[-1] == channels:
        return np.transpose(array, (0, 3, 1, 2)).astype(np.float32, copy=False)
    raise RuntimeError(f"{name} has unexpected shape {array.shape}; expected {channels} channels")


def z_medians_array(params: dict[str, Any]) -> np.ndarray:
    channels_z = int(params["channels_z"])
    medians = np.asarray(params["z_medians"], dtype=np.float32)
    if medians.size != channels_z:
        raise RuntimeError(f"z_medians has {medians.size} values, expected {channels_z}")
    return medians.reshape(1, channels_z, 1, 1)


def quantize_centered(value: np.ndarray, center: np.ndarray, step: float) -> np.ndarray:
    return np.rint((value.astype(np.float32) - center.astype(np.float32)) / np.float32(step)).astype(np.int32)


def compact_symbols(symbols: np.ndarray) -> tuple[np.ndarray, str]:
    min_value = int(symbols.min())
    max_value = int(symbols.max())
    if -32768 <= min_value and max_value <= 32767:
        return symbols.astype("<i2", copy=False), "int16"
    return symbols.astype("<i4", copy=False), "int32"


def compress_one(
    analysis: AnalysisRKNN,
    image_path: Path,
    output_path: Path,
    params: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, float]:
    channels_y = int(params["channels_y"])
    channels_z = int(params["channels_z"])
    quant_step_y = float(params["quant_step_y"])
    quant_step_z = float(params["quant_step_z"])

    total_t0 = time.perf_counter()
    pre_t0 = time.perf_counter()
    input_data, metadata = prepare_image(
        image_path,
        width=int(args.width),
        height=int(args.height),
        resize=bool(args.resize),
        input_layout=str(args.input_layout),
    )
    pre_sec = time.perf_counter() - pre_t0

    infer_t0 = time.perf_counter()
    outputs = analysis.infer(input_data)
    infer_sec = time.perf_counter() - infer_t0
    if len(outputs) < 4:
        raise RuntimeError(f"analysis RKNN must output [y, z, scales_y, means_y], got {len(outputs)}")

    post_t0 = time.perf_counter()
    y = to_nchw(outputs[0], channels_y, "y")
    z = to_nchw(outputs[1], channels_z, "z")
    scales_y = to_nchw(outputs[2], channels_y, "scales_y")
    means_y = to_nchw(outputs[3], channels_y, "means_y")
    if y.shape != means_y.shape or y.shape != scales_y.shape:
        raise RuntimeError(f"shape mismatch: y={y.shape}, scales_y={scales_y.shape}, means_y={means_y.shape}")

    y_symbols_i32 = quantize_centered(y, means_y, quant_step_y)
    z_symbols_i32 = quantize_centered(z, z_medians_array(params), quant_step_z)
    y_symbols, y_dtype = compact_symbols(y_symbols_i32)
    z_symbols, z_dtype = compact_symbols(z_symbols_i32)

    metadata.update(
        {
            "model_variant": str(params.get("model_variant", "nano_hyper_ms_q_nano")),
            "model_type": str(params.get("model_type", "mean_scale_hyperprior")),
            "channels_y": channels_y,
            "channels_z": channels_z,
            "quant_step_y": quant_step_y,
            "quant_step_z": quant_step_z,
            "y_shape": [int(v) for v in y.shape],
            "z_shape": [int(v) for v in z.shape],
            "scales_shape": [int(v) for v in scales_y.shape],
            "means_shape": [int(v) for v in means_y.shape],
            "y_dtype": y_dtype,
            "z_dtype": z_dtype,
            "y_symbol_min": int(y_symbols_i32.min()),
            "y_symbol_max": int(y_symbols_i32.max()),
            "z_symbol_min": int(z_symbols_i32.min()),
            "z_symbol_max": int(z_symbols_i32.max()),
            "scale_min": float(np.min(scales_y)),
            "scale_mean": float(np.mean(scales_y)),
            "scale_max": float(np.max(scales_y)),
        }
    )
    post_sec = time.perf_counter() - post_t0

    save_t0 = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            y_symbols=y_symbols,
            z_symbols=z_symbols,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    save_sec = time.perf_counter() - save_t0
    total_sec = time.perf_counter() - total_t0

    pixels = float(args.width) * float(args.height)
    return {
        "pre_ms": pre_sec * 1000.0,
        "npu_ms": infer_sec * 1000.0,
        "post_ms": post_sec * 1000.0,
        "save_ms": save_sec * 1000.0,
        "total_ms": total_sec * 1000.0,
        "package_bpp": output_path.stat().st_size * 8.0 / pixels,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compress image(s) on RK3588 with a hyper_ms_nano analysis RKNN model."
    )
    parser.add_argument("--rknn", type=Path, required=True, help="analysis/compress RKNN, image -> y,z,scales,means")
    parser.add_argument("--params", type=Path, required=True, help="hyper_ms_nano_entropy_params.json")
    parser.add_argument("--input", type=Path, required=True, help="one image or an image directory")
    parser.add_argument("--output", type=Path, required=True, help="one .npz file or output directory")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--resize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--input-layout", choices=("nhwc", "nchw"), default="nhwc")
    parser.add_argument("--core-mask", default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = load_params(args.params)
    images = collect_images(args.input)
    if not images:
        raise RuntimeError(f"no images found: {args.input}")
    if args.input.is_dir():
        args.output.mkdir(parents=True, exist_ok=True)

    print(f"images: {len(images)}")
    print(f"rknn: {args.rknn}")
    print(f"params: {args.params}")
    print(f"shape: {args.width}x{args.height} resize={args.resize} layout={args.input_layout}")

    with AnalysisRKNN(args.rknn, args.core_mask) as analysis:
        for index, image_path in enumerate(images, start=1):
            output_path = make_output_path(image_path, args.input, args.output)
            stats = compress_one(analysis, image_path, output_path, params, args)
            print(
                f"[{index}/{len(images)}] {image_path} -> {output_path} "
                f"bpp={stats['package_bpp']:.4f} "
                f"npu={stats['npu_ms']:.2f}ms "
                f"post={stats['post_ms']:.2f}ms "
                f"save={stats['save_ms']:.2f}ms "
                f"total={stats['total_ms']:.2f}ms"
            )


if __name__ == "__main__":
    main()
