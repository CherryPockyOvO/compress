#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
from rknnlite.api import RKNNLite


def load_image_rgb(
    image_path: Path,
    height: Optional[int],
    width: Optional[int],
    downsampling_factor: int = 16,
) -> Tuple[np.ndarray, Dict[str, object]]:
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

    metadata: Dict[str, object] = {
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
    return input_nhwc, metadata


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
        raise ValueError(
            "unknown --core-mask value: "
            f"{name}. Use auto, 0, 1, 2, 0_1, 1_2, 0_1_2, or all."
        )
    if hasattr(RKNNLite, attr):
        return int(getattr(RKNNLite, attr))
    if normalized in {"auto", "any", "all", "0_1_2", "012"}:
        return int(RKNNLite.NPU_CORE_AUTO)
    raise ValueError(f"RKNNLite on this board does not expose {attr}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode one RGB image into a float32 NCHW latent .bin with RKNNLite."
    )
    parser.add_argument("--rknn", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--meta-output", type=Path, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--downsampling-factor", type=int, default=16)
    parser.add_argument(
        "--input-layout",
        choices=["nhwc", "nchw"],
        default="nhwc",
        help="Use nchw only if your RKNN model input is NCHW.",
    )
    parser.add_argument(
        "--core-mask",
        default="auto",
        help="RK3588 NPU core mask: auto, 0, 1, 2, 0_1, 1_2, 0_1_2/all.",
    )
    args = parser.parse_args()

    total_t0 = time.perf_counter()
    preprocess_t0 = time.perf_counter()
    input_data, metadata = load_image_rgb(
        args.image,
        height=args.height,
        width=args.width,
        downsampling_factor=args.downsampling_factor,
    )
    if args.input_layout == "nchw":
        input_data = np.transpose(input_data, (0, 3, 1, 2)).copy()
    preprocess_sec = time.perf_counter() - preprocess_t0

    runtime_t0 = time.perf_counter()
    rknn = RKNNLite()
    ret = rknn.load_rknn(str(args.rknn))
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")

    core_mask = resolve_core_mask(args.core_mask)
    ret = rknn.init_runtime(core_mask=core_mask)
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")
    runtime_init_sec = time.perf_counter() - runtime_t0

    try:
        inference_t0 = time.perf_counter()
        outputs = rknn.inference(inputs=[input_data])
        inference_sec = time.perf_counter() - inference_t0
        if not outputs:
            raise RuntimeError("RKNN inference returned no outputs")

        save_t0 = time.perf_counter()
        latent = np.asarray(outputs[0])
        if latent.ndim != 4:
            raise RuntimeError(f"Unexpected latent shape: {latent.shape}")

        if latent.shape[1] != 128 and latent.shape[-1] == 128:
            latent = np.transpose(latent, (0, 3, 1, 2)).copy()

        latent = latent.astype("<f4", copy=False)
        metadata["latent_c"] = int(latent.shape[1])
        metadata["latent_h"] = int(latent.shape[2])
        metadata["latent_w"] = int(latent.shape[3])
        metadata["core_mask"] = args.core_mask

        args.output.parent.mkdir(parents=True, exist_ok=True)
        latent.tofile(args.output)

        meta_path = args.meta_output or Path(str(args.output) + ".json")
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        save_sec = time.perf_counter() - save_t0
        total_sec = time.perf_counter() - total_t0
        timing = {
            "preprocess_sec": round(preprocess_sec, 6),
            "runtime_init_sec": round(runtime_init_sec, 6),
            "npu_inference_sec": round(inference_sec, 6),
            "save_bin_sec": round(save_sec, 6),
            "encoder_total_sec": round(total_sec, 6),
        }

        print(f"input_shape: {input_data.shape}")
        print(f"latent_shape: {latent.shape}")
        print(f"core_mask: {args.core_mask}")
        print(f"saved latent: {args.output}")
        print(f"saved metadata: {meta_path}")
        print(f"timing_json: {json.dumps(timing, sort_keys=True)}")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
