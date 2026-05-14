from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compressai_nano import get_model, infer_model_variant_from_checkpoint
from compressai_nano.cnz import build_cnz_bytes, quantize_latent


def load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> None:
    raw = torch.load(checkpoint, map_location="cpu")
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"missing_keys: {len(missing)}")
    print(f"unexpected_keys: {len(unexpected)}")


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


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Python encoder + CNZ reference compression path.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
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
            f"{model_variant} is not compatible with the CNZ4 benchmark path yet. "
            "Benchmark its analysis side with tools/export_encoder_onnx.py or add CNZ5."
        )
    x = image_to_tensor(args.image).to(device)
    x, original_size = pad_to_multiple(x, model.downsampling_factor)

    sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        y = model.encoder(x)
    sync(device)
    t1 = time.perf_counter()
    symbols = quantize_latent(
        y,
        model.entropy_bottleneck.medians.detach(),
        float(model.entropy_bottleneck.quant_step.detach().cpu()),
    )
    sync(device)
    t2 = time.perf_counter()
    blob, stats = build_cnz_bytes(
        symbols=symbols,
        medians=model.entropy_bottleneck.medians.detach(),
        quant_step=float(model.entropy_bottleneck.quant_step.detach().cpu()),
        orig_size=original_size,
        padded_size=tuple(int(v) for v in x.shape[-2:]),
        down_factor=model.downsampling_factor,
        codec=args.codec,
        zlib_level=args.zlib_level,
    )
    t3 = time.perf_counter()
    pixels = original_size[0] * original_size[1]
    print(f"encoder_ms={(t1 - t0) * 1000:.3f}")
    print(f"python_quantize_ms={(t2 - t1) * 1000:.3f}")
    print(f"python_byte_compress_ms={(t3 - t2) * 1000:.3f}")
    print(f"python_total_compress_ms={(t3 - t0) * 1000:.3f}")
    print(f"bpp={stats['payload_size'] * 8 / pixels:.6f}")
    print(f"symbol_min={stats['min_symbol']}")
    print(f"symbol_max={stats['max_symbol']}")
    print(f"dtype={stats['dtype']}")
    print(f"payload_size={stats['payload_size']}")
    print(f"container_size={len(blob)}")


if __name__ == "__main__":
    main()
