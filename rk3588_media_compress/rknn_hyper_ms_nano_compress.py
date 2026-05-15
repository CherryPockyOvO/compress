#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from rknnlite.api import RKNNLite


def load_image_rgb(
    image_path: Path,
    height: int | None,
    width: int | None,
    padding_multiple: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    image = Image.open(image_path).convert("RGB")
    source_w, source_h = image.size
    if height is not None or width is not None:
        if height is None or width is None:
            raise ValueError("--height and --width must be provided together")
        image = image.resize((width, height), Image.Resampling.BICUBIC)

    orig_w, orig_h = image.size
    array = np.asarray(image).astype(np.float32) / 255.0
    pad_h = (padding_multiple - orig_h % padding_multiple) % padding_multiple
    pad_w = (padding_multiple - orig_w % padding_multiple) % padding_multiple
    if pad_h > 0 or pad_w > 0:
        array = np.pad(array, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

    padded_h, padded_w = array.shape[:2]
    metadata: dict[str, Any] = {
        "format": "compressai-nano-hyper-ms-reference-v1",
        "image": str(image_path),
        "source_h": int(source_h),
        "source_w": int(source_w),
        "orig_h": int(orig_h),
        "orig_w": int(orig_w),
        "padded_h": int(padded_h),
        "padded_w": int(padded_w),
        "padding_multiple": int(padding_multiple),
        "input_dtype": "float32",
    }
    return np.expand_dims(array, axis=0).astype(np.float32), metadata


def resolve_core_mask(name: str) -> int:
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
    if hasattr(RKNNLite, attr):
        return int(getattr(RKNNLite, attr))
    if normalized in {"auto", "any", "all", "0_1_2", "012"}:
        return int(RKNNLite.NPU_CORE_AUTO)
    raise ValueError(f"RKNNLite on this board does not expose {attr}")


def to_nchw(output: np.ndarray, channels: int, name: str) -> np.ndarray:
    array = np.asarray(output)
    if array.ndim != 4:
        raise RuntimeError(f"{name} must be 4D, got {array.shape}")
    if array.shape[1] == channels:
        return array.astype(np.float32, copy=False)
    if array.shape[-1] == channels:
        return np.transpose(array, (0, 3, 1, 2)).astype(np.float32, copy=False)
    raise RuntimeError(f"{name} has unexpected shape {array.shape}; expected channel count {channels}")


def quantize_centered(value: np.ndarray, center: np.ndarray, step: float) -> np.ndarray:
    return np.rint((value - center) / np.float32(step)).astype(np.int32)


def quantize_z(value: np.ndarray, medians: np.ndarray, step: float) -> np.ndarray:
    center = medians.astype(np.float32).reshape(1, -1, 1, 1)
    return quantize_centered(value, center, step)


def compact_symbols(symbols: np.ndarray) -> tuple[np.ndarray, str]:
    min_value = int(symbols.min())
    max_value = int(symbols.max())
    if -32768 <= min_value and max_value <= 32767:
        return symbols.astype("<i2", copy=False), "int16"
    return symbols.astype("<i4", copy=False), "int32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the new hyper_ms_nano analysis RKNN model on RK3588 and write "
            "a reference compressed package. This is a CNZ5 placeholder, not "
            "the legacy CNZ4 production bitstream."
        )
    )
    parser.add_argument("--rknn", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--padding-multiple", type=int, default=64)
    parser.add_argument("--input-layout", choices=("nhwc", "nchw"), default="nhwc")
    parser.add_argument("--core-mask", default="all")
    parser.add_argument("--keep-float-outputs", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = json.loads(args.params.read_text(encoding="utf-8"))
    if not params.get("has_means_y", False):
        raise RuntimeError("This script expects mean-scale hyperprior params with has_means_y=true")

    channels_y = int(params["channels_y"])
    channels_z = int(params["channels_z"])
    quant_step_y = float(params["quant_step_y"])
    quant_step_z = float(params["quant_step_z"])
    z_medians = np.asarray(params["z_medians"], dtype=np.float32)
    if z_medians.size != channels_z:
        raise RuntimeError("z_medians size does not match channels_z")

    total_t0 = time.perf_counter()
    pre_t0 = time.perf_counter()
    input_data, metadata = load_image_rgb(
        args.image,
        height=args.height,
        width=args.width,
        padding_multiple=args.padding_multiple,
    )
    if args.input_layout == "nchw":
        input_data = np.transpose(input_data, (0, 3, 1, 2)).copy()
    pre_sec = time.perf_counter() - pre_t0

    runtime_t0 = time.perf_counter()
    rknn = RKNNLite()
    ret = rknn.load_rknn(str(args.rknn))
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")
    ret = rknn.init_runtime(core_mask=resolve_core_mask(args.core_mask))
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")
    runtime_sec = time.perf_counter() - runtime_t0

    try:
        infer_t0 = time.perf_counter()
        outputs = rknn.inference(inputs=[input_data])
        infer_sec = time.perf_counter() - infer_t0
    finally:
        rknn.release()

    if len(outputs) < 4:
        raise RuntimeError(
            f"Expected RKNN outputs [y, z, scales_y, means_y], got {len(outputs)} outputs"
        )

    post_t0 = time.perf_counter()
    y = to_nchw(outputs[0], channels_y, "y")
    z = to_nchw(outputs[1], channels_z, "z")
    scales_y = to_nchw(outputs[2], channels_y, "scales_y")
    means_y = to_nchw(outputs[3], channels_y, "means_y")
    if y.shape != means_y.shape or y.shape != scales_y.shape:
        raise RuntimeError(
            f"shape mismatch: y={y.shape}, scales_y={scales_y.shape}, means_y={means_y.shape}"
        )

    y_symbols_i32 = quantize_centered(y, means_y, quant_step_y)
    z_symbols_i32 = quantize_z(z, z_medians, quant_step_z)
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
    metadata_json = json.dumps(metadata, sort_keys=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        np.savez_compressed(
            handle,
            y_symbols=y_symbols,
            z_symbols=z_symbols,
            metadata=np.asarray(metadata_json),
        )

    meta_path = args.metadata_output or Path(str(args.output) + ".json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if args.keep_float_outputs is not None:
        args.keep_float_outputs.mkdir(parents=True, exist_ok=True)
        y.astype("<f4", copy=False).tofile(args.keep_float_outputs / "y.bin")
        z.astype("<f4", copy=False).tofile(args.keep_float_outputs / "z.bin")
        scales_y.astype("<f4", copy=False).tofile(args.keep_float_outputs / "scales_y.bin")
        means_y.astype("<f4", copy=False).tofile(args.keep_float_outputs / "means_y.bin")

    post_sec = time.perf_counter() - post_t0
    total_sec = time.perf_counter() - total_t0
    pixels = float(metadata["orig_h"]) * float(metadata["orig_w"])
    raw_symbol_bytes = int(y_symbols.nbytes + z_symbols.nbytes)
    package_bytes = args.output.stat().st_size

    print(f"input_shape: {input_data.shape}")
    print(f"y_shape: {y.shape}")
    print(f"z_shape: {z.shape}")
    print(f"saved_package: {args.output}")
    print(f"saved_metadata: {meta_path}")
    print(f"raw_symbol_bpp: {raw_symbol_bytes * 8.0 / pixels:.4f}")
    print(f"package_bpp: {package_bytes * 8.0 / pixels:.4f}")
    print(f"timing_preprocess_ms: {pre_sec * 1000.0:.3f}")
    print(f"timing_runtime_init_ms: {runtime_sec * 1000.0:.3f}")
    print(f"timing_npu_inference_ms: {infer_sec * 1000.0:.3f}")
    print(f"timing_postprocess_ms: {post_sec * 1000.0:.3f}")
    print(f"timing_total_ms: {total_sec * 1000.0:.3f}")


if __name__ == "__main__":
    main()
