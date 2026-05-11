from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
from PIL import Image

from compressai_nano import FactorizedPriorNano


def tensor_to_image(tensor: torch.Tensor, path: Path) -> None:
    tensor = tensor.squeeze(0).detach().cpu().clamp(0, 1)
    tensor = (tensor * 255.0).round().to(torch.uint8)
    tensor = tensor.permute(1, 2, 0).contiguous()
    height, width = tensor.shape[:2]
    image = Image.frombytes("RGB", (width, height), bytes(tensor.reshape(-1).tolist()))
    image.save(path)


def crop_to_size(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    height, width = size
    return x[..., :height, :width]


def load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> None:
    raw = torch.load(checkpoint, map_location="cpu")
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    model.load_state_dict(state_dict, strict=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode a nano bitstream on PC.")
    parser.add_argument("bitstream", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("recon.png"))
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package = pickle.loads(args.bitstream.read_bytes())
    if package.get("format") != "compressai-nano-v2":
        raise ValueError(f"Unsupported bitstream format: {package.get('format')}")

    quality_level = int(package["quality_level"])
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = FactorizedPriorNano(quality_level=quality_level).to(device).eval()
    load_checkpoint(model, args.checkpoint)

    with torch.no_grad():
        restored = model.decompress(package["strings"], package["shape"])
        x_hat = crop_to_size(restored["x_hat"], tuple(package["original_size"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_image(x_hat, args.output)
    print(f"quality_level={quality_level}")
    print(f"latent_shape={package['latent_shape']}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
