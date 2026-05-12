from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from compressai_nano import FactorizedPriorNano


def image_to_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    values = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    values = values.to(torch.float32).view(height, width, 3).permute(2, 0, 1)
    return values.unsqueeze(0) / 255.0


def tensor_to_image(tensor: torch.Tensor, path: Path) -> None:
    tensor = tensor.squeeze(0).detach().cpu().clamp(0, 1)
    tensor = (tensor * 255.0).round().to(torch.uint8)
    tensor = tensor.permute(1, 2, 0).contiguous()
    height, width = tensor.shape[:2]
    image = Image.frombytes("RGB", (width, height), tensor.numpy().tobytes())
    image.save(path)


def pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = x.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, (height, width)


def crop_to_size(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    height, width = size
    return x[..., :height, :width]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-image nano compression.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("recon.png"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = FactorizedPriorNano().eval()
    if args.checkpoint is not None:
        raw = torch.load(args.checkpoint, map_location="cpu")
        state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
        model.load_state_dict(state_dict, strict=False)

    x = image_to_tensor(args.image)
    x_padded, original_size = pad_to_multiple(x, model.downsampling_factor)

    with torch.no_grad():
        compressed = model.compress(x_padded)
        restored = model.decompress(compressed["strings"], compressed["shape"])
        x_hat = crop_to_size(restored["x_hat"], original_size)

    tensor_to_image(x_hat, args.output)
    bit_count = sum(len(item) * 8 for item in compressed["strings"])
    pixels = original_size[0] * original_size[1]
    print("model=single high-quality configuration")
    print(f"latent_shape={compressed['latent_shape']}")
    print(f"payload={bit_count / 8:.0f} bytes, bpp={bit_count / pixels:.4f}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
