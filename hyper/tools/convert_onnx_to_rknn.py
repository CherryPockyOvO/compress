from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an exported ONNX analysis/encoder model to RKNN. "
            "Run this on the PC environment that has rknn-toolkit2 installed."
        )
    )
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-platform", default="rk3588")
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument(
        "--force-input-size",
        action="store_true",
        help=(
            "Pass inputs/input_size_list to RKNN load_onnx. Normally unnecessary "
            "for the static-shape ONNX files exported by this project."
        ),
    )
    parser.add_argument(
        "--do-quantization",
        action="store_true",
        help="Enable RKNN post-training INT8 quantization. Omit for FP/FP16 RKNN.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Calibration dataset text file used only with --do-quantization.",
    )
    parser.add_argument("--optimization-level", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.do_quantization and args.dataset is None:
        raise ValueError("--dataset is required when --do-quantization is enabled")
    if (args.height is None) != (args.width is None):
        raise ValueError("--height and --width must be provided together")

    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise RuntimeError(
            "rknn-toolkit2 is not installed in this Python environment. "
            "Install the RKNN toolkit on the conversion PC first."
        ) from exc

    rknn = RKNN(verbose=True)
    try:
        ret = rknn.config(
            target_platform=args.target_platform,
            optimization_level=args.optimization_level,
        )
        if ret != 0:
            raise RuntimeError(f"rknn.config failed: {ret}")

        load_kwargs: dict[str, object] = {"model": str(args.onnx)}
        if args.force_input_size and args.height is not None and args.width is not None:
            load_kwargs["inputs"] = [args.input_name]
            load_kwargs["input_size_list"] = [[1, 3, int(args.height), int(args.width)]]

        ret = rknn.load_onnx(**load_kwargs)
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx failed: {ret}")

        build_kwargs: dict[str, object] = {"do_quantization": bool(args.do_quantization)}
        if args.dataset is not None:
            build_kwargs["dataset"] = str(args.dataset)
        ret = rknn.build(**build_kwargs)
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
