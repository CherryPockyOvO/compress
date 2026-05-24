from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from compressai_nano import get_model, infer_model_variant_from_checkpoint
from compressai_nano.cnz import build_cnz_bytes, quantize_latent, write_cnz_file


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
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"missing_keys: {len(missing)}")
    print(f"unexpected_keys: {len(unexpected)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode one image into a nano bitstream.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("image.cnz"))
    parser.add_argument("--codec", choices=("zlib", "none"), default="zlib")
    parser.add_argument("--zlib-level", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    raw = torch.load(args.checkpoint, map_location="cpu")
    model_variant = infer_model_variant_from_checkpoint(raw)
    model = get_model(model_variant=model_variant).to(device).eval()
    load_checkpoint(model, args.checkpoint)
    if not getattr(model, "supports_cnz_v4", False):
        raise RuntimeError(
            f"{model_variant} uses a scale hyperprior and cannot be encoded as CNZ4 yet. "
            "Use tools/export_encoder_onnx.py for analysis-side export, or add CNZ5 "
            "support with z/y streams before deployment."
        )

    x = image_to_tensor(args.image).to(device)
    x_padded, original_size = pad_to_multiple(x, model.downsampling_factor)

    with torch.no_grad():
        y = model.encoder(x_padded)
        symbols = quantize_latent(
            y,
            model.entropy_bottleneck.medians.detach(),
            float(model.entropy_bottleneck.quant_step.detach().cpu()),
        )
        blob, stats = build_cnz_bytes(
            symbols=symbols,
            medians=model.entropy_bottleneck.medians.detach(),
            quant_step=float(model.entropy_bottleneck.quant_step.detach().cpu()),
            orig_size=original_size,
            padded_size=tuple(int(v) for v in x_padded.shape[-2:]),
            down_factor=model.downsampling_factor,
            codec=args.codec,
            zlib_level=args.zlib_level,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_cnz_file(args.output, blob)

    pixels = original_size[0] * original_size[1]
    latent_payload_bits = int(stats["payload_size"]) * 8
    container_bits = args.output.stat().st_size * 8
    print(f"model_variant={model_variant}")
    print(f"original_size={original_size}")
    print(f"latent_shape={tuple(int(v) for v in symbols.shape)}")
    print(f"dtype={stats['dtype']}")
    print(f"codec={stats['codec']}")
    print(f"symbol_min={stats['min_symbol']}")
    print(f"symbol_max={stats['max_symbol']}")
    print(f"latent_payload_bpp={latent_payload_bits / pixels:.4f}")
    print(f"container_bpp={container_bits / pixels:.4f}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
