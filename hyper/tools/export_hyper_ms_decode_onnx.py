from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compressai_nano import get_model, infer_model_variant_from_checkpoint


def load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> None:
    raw = torch.load(checkpoint, map_location="cpu")
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"missing_keys: {len(missing)}")
    if missing:
        print("\n".join(f"  {key}" for key in missing[:20]))
    print(f"unexpected_keys: {len(unexpected)}")
    if unexpected:
        print("\n".join(f"  {key}" for key in unexpected[:20]))


class HyperDecoderExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.hyper_decoder = model.hyper_decoder

    def forward(self, z_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scales_y, means_y = self.hyper_decoder(z_hat)
        return scales_y, means_y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export hyper_ms decoder-side modules to ONNX for RKNN conversion."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("decoder", "hyper-decoder"), required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--opset", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.height % 64 != 0 or args.width % 64 != 0:
        raise ValueError("--height/--width must be divisible by 64 for hyper_ms export")

    raw = torch.load(args.checkpoint, map_location="cpu")
    model_variant = infer_model_variant_from_checkpoint(raw)
    model = get_model(model_variant=model_variant).eval()
    load_checkpoint(model, args.checkpoint)

    if not all(hasattr(model, name) for name in ("decoder", "hyper_decoder", "M", "Z")):
        raise RuntimeError(f"{model_variant} is not a hyper_ms model")

    if args.mode == "decoder":
        export_module: nn.Module = model.decoder
        dummy = torch.randn(1, int(model.M), args.height // 16, args.width // 16)
        input_names = ["y_hat"]
        output_names = ["x_hat"]
    else:
        export_module = HyperDecoderExportWrapper(model)
        dummy = torch.randn(1, int(model.Z), args.height // 64, args.width // 64)
        input_names = ["z_hat"]
        output_names = ["scales_y", "means_y"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_module,
        dummy,
        args.output.as_posix(),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
    )
    with torch.no_grad():
        exported = export_module(dummy)
    first = exported[0] if isinstance(exported, tuple) else exported
    print(f"mode         : {args.mode}")
    print(f"model_variant: {model_variant}")
    print(f"input_shape  : {tuple(dummy.shape)}")
    print(f"output_shape : {tuple(first.shape)}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
