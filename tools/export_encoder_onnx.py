from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compressai_nano import FactorizedPriorNano


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export only FactorizedPriorNano encoder to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--dynamic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = FactorizedPriorNano().eval()
    load_checkpoint(model, args.checkpoint)
    dummy = torch.randn(1, 3, args.height, args.width)
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "latent": {0: "batch", 2: "latent_height", 3: "latent_width"},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model.encoder,
        dummy,
        args.output.as_posix(),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["latent"],
        dynamic_axes=dynamic_axes,
    )
    with torch.no_grad():
        latent = model.encoder(dummy)
    print(f"encoder input : {tuple(dummy.shape)}")
    print(f"encoder output: {tuple(latent.shape)}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
