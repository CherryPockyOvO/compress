from __future__ import annotations

import argparse
from pathlib import Path

import torch

from compressai_nano import FactorizedPriorNano
from compressai_nano.cnz import MAGIC
from decode_cnz import crop_to_size, decode_cnz, decode_legacy, load_checkpoint, tensor_to_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode a nano bitstream on PC.")
    parser.add_argument("bitstream", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("recon.png"))
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = FactorizedPriorNano().to(device).eval()
    load_checkpoint(model, args.checkpoint)

    with torch.no_grad():
        prefix = args.bitstream.read_bytes()[:4]
        if prefix == MAGIC:
            x_hat, original_size = decode_cnz(args.bitstream, model, device, use_half=False)
        else:
            x_hat, original_size = decode_legacy(args.bitstream, model, device)
        x_hat = crop_to_size(x_hat, original_size)

    tensor_to_image(x_hat, args.output)
    print("model=single high-quality configuration")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
