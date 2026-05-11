from __future__ import annotations

import argparse
import pickle
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


def pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = x.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, (height, width)


def load_checkpoint(model: torch.nn.Module, checkpoint: Path | None) -> None:
    if checkpoint is None:
        return
    raw = torch.load(checkpoint, map_location="cpu")
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    model.load_state_dict(state_dict, strict=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode one image into a nano bitstream.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--quality-level", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("image.cnz"))
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = FactorizedPriorNano(quality_level=args.quality_level).to(device).eval()
    load_checkpoint(model, args.checkpoint)

    x = image_to_tensor(args.image).to(device)
    x_padded, original_size = pad_to_multiple(x, model.downsampling_factor)

    with torch.no_grad():
        compressed = model.compress(x_padded)

    package = {
        "format": "compressai-nano-v2",
        "quality_level": args.quality_level,
        "original_size": original_size,
        "padded_size": tuple(int(v) for v in x_padded.shape[-2:]),
        "shape": compressed["shape"],
        "latent_shape": compressed["latent_shape"],
        "quant_step": compressed["quant_step"],
        "strings": compressed["strings"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(pickle.dumps(package, protocol=pickle.HIGHEST_PROTOCOL))

    pixels = original_size[0] * original_size[1]
    latent_payload_bits = sum(len(item) * 8 for item in compressed["strings"])
    container_bits = args.output.stat().st_size * 8
    print(f"quality_level={args.quality_level}")
    print(f"original_size={original_size}")
    print(f"latent_shape={compressed['latent_shape']}")
    print(f"latent_payload_bpp={latent_payload_bits / pixels:.4f}")
    print(f"container_bpp={container_bits / pixels:.4f}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
