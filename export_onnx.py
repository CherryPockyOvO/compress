from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
from typing import Any

import torch

from compressai_nano import FactorizedPriorNano, get_quality_config


def preload_onnx_package() -> None:
    """Load the real onnx package before an output folder can shadow it."""

    existing = sys.modules.get("onnx")
    if existing is not None and hasattr(existing, "load_model_from_string"):
        return
    if existing is not None:
        del sys.modules["onnx"]

    project_root = Path(__file__).resolve().parent
    removed_entries = []
    for entry in list(sys.path):
        entry_path = Path(entry or ".").resolve()
        if entry_path == project_root:
            sys.path.remove(entry)
            removed_entries.append(entry)

    try:
        onnx = importlib.import_module("onnx")
    except ImportError as exc:
        raise RuntimeError(
            "ONNX export requires the real 'onnx' Python package. "
            "Install it with: pip install onnx"
        ) from exc
    finally:
        sys.path[:0] = removed_entries

    if not hasattr(onnx, "load_model_from_string"):
        raise RuntimeError(
            "Imported 'onnx' does not look like the official ONNX package. "
            "Avoid naming a local file or directory 'onnx'."
        )


def load_checkpoint(model: torch.nn.Module, checkpoint: str | None) -> None:
    if checkpoint is None:
        return

    raw: Any = torch.load(checkpoint, map_location="cpu")
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint must be a state_dict or contain a state_dict key")

    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[checkpoint] missing keys: {len(missing)}")
    if unexpected:
        print(f"[checkpoint] unexpected keys: {len(unexpected)}")


def export_encoder(
    model: FactorizedPriorNano,
    output_path: Path,
    dummy_x: torch.Tensor,
    opset: int,
) -> None:
    torch.onnx.export(
        model.encoder,
        dummy_x,
        output_path.as_posix(),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["y"],
        dynamic_axes=None,
    )


def export_decoder(
    model: FactorizedPriorNano,
    output_path: Path,
    dummy_y: torch.Tensor,
    opset: int,
) -> None:
    torch.onnx.export(
        model.decoder,
        dummy_y,
        output_path.as_posix(),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["y_hat"],
        output_names=["x_hat"],
        dynamic_axes=None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export compressai-nano ONNX models.")
    parser.add_argument("--quality-level", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=Path("onnx_models"))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--opset", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preload_onnx_package()
    config = get_quality_config(args.quality_level)
    model = FactorizedPriorNano(quality_level=args.quality_level)
    model.eval()
    load_checkpoint(model, args.checkpoint)

    factor = model.downsampling_factor
    if args.height % factor != 0 or args.width % factor != 0:
        raise ValueError(
            f"height and width must be divisible by {factor}; "
            f"got {args.height}x{args.width}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = args.output_dir / "encoder.onnx"
    decoder_path = args.output_dir / "decoder.onnx"

    dummy_x = torch.randn(1, 3, args.height, args.width)
    with torch.no_grad():
        dummy_y = model.encoder(dummy_x)

    export_encoder(model, encoder_path, dummy_x, args.opset)
    export_decoder(model, decoder_path, dummy_y, args.opset)

    print(f"quality={config.quality_level} ({config.name}), N={config.N}, M={config.M}")
    print(
        "decoder config: "
        f"channels={config.decoder_channels}, "
        f"res_blocks={config.decoder_res_blocks}, "
        f"refinement_blocks={config.refinement_blocks}"
    )
    print(f"encoder input : {tuple(dummy_x.shape)}")
    print(f"encoder output: {tuple(dummy_y.shape)}")
    print(f"decoder input : {tuple(dummy_y.shape)}")
    print(f"decoder output: {(1, 3, args.height, args.width)}")
    print(f"saved: {encoder_path}")
    print(f"saved: {decoder_path}")


if __name__ == "__main__":
    main()
