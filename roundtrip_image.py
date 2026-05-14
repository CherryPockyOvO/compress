from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from compressai_nano import get_model, infer_model_variant_from_checkpoint
from compressai_nano.cnz import build_cnz_bytes, quantize_latent, write_cnz_file
from decode_cnz import crop_to_size, tensor_to_image
from encode_image import image_to_tensor, load_checkpoint, pad_to_multiple


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def now_synced(device: torch.device) -> float:
    sync_device(device)
    return time.perf_counter()


def resolve_device(device_name: str, cpu: bool) -> torch.device:
    if cpu:
        return torch.device("cpu")
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return torch.device(device_name)


def default_output_dir(checkpoint: Path, image: Path) -> Path:
    checkpoint_parent = checkpoint.parent.name or "checkpoint"
    return Path("roundtrip") / checkpoint_parent / image.stem


def encode_decode_image(
    model: torch.nn.Module,
    image_path: Path,
    cnz_path: Path,
    recon_path: Path,
    device: torch.device,
    codec: str,
    zlib_level: int,
) -> dict[str, float | str | tuple[int, int] | tuple[int, ...]]:
    x = image_to_tensor(image_path).to(device)
    x_padded, original_size = pad_to_multiple(x, model.downsampling_factor)

    t0 = now_synced(device)
    with torch.inference_mode():
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
            codec=codec,
            zlib_level=zlib_level,
        )
    t1 = now_synced(device)

    cnz_path.parent.mkdir(parents=True, exist_ok=True)
    write_cnz_file(cnz_path, blob)
    t2 = now_synced(device)

    with torch.inference_mode():
        y_hat = (
            symbols.to(device=device, dtype=torch.float32)
            * float(model.entropy_bottleneck.quant_step.detach().cpu())
            + model.entropy_bottleneck.medians.detach().to(device=device).view(1, -1, 1, 1)
        )
        x_hat = model.decoder(y_hat)
        x_hat = crop_to_size(x_hat, original_size)
    t3 = now_synced(device)

    tensor_to_image(x_hat, recon_path)
    t4 = now_synced(device)

    pixels = original_size[0] * original_size[1]
    return {
        "original_size": original_size,
        "padded_size": tuple(int(v) for v in x_padded.shape[-2:]),
        "latent_shape": tuple(int(v) for v in symbols.shape),
        "dtype": str(stats["dtype"]),
        "codec": str(stats["codec"]),
        "payload_bpp": int(stats["payload_size"]) * 8 / pixels,
        "container_bpp": cnz_path.stat().st_size * 8 / pixels,
        "encode_ms": (t1 - t0) * 1000.0,
        "write_cnz_ms": (t2 - t1) * 1000.0,
        "decode_model_ms": (t3 - t2) * 1000.0,
        "save_recon_ms": (t4 - t3) * 1000.0,
        "total_ms": (t4 - t0) * 1000.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode one image to CNZ and immediately decode it with one checkpoint load."
    )
    parser.add_argument("image", type=Path, nargs="?", default=Path("samples/yn.png"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cnz-output", type=Path, default=None)
    parser.add_argument("--recon-output", type=Path, default=None)
    parser.add_argument("--codec", choices=("zlib", "none"), default="zlib")
    parser.add_argument("--zlib-level", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--timing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir(args.checkpoint, args.image)
    cnz_path = args.cnz_output or (output_dir / "image.cnz")
    recon_path = args.recon_output or (output_dir / "recon.png")

    device = resolve_device(args.device, args.cpu)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"device: cuda ({torch.cuda.get_device_name(device)})")
    else:
        print("device: cpu")

    raw = torch.load(args.checkpoint, map_location="cpu")
    model_variant = infer_model_variant_from_checkpoint(raw)
    model = get_model(model_variant=model_variant).to(device).eval()
    load_checkpoint(model, args.checkpoint)
    if not getattr(model, "supports_cnz_v4", False):
        raise RuntimeError(
            f"{model_variant} is a hyperprior model and needs CNZ5 support before "
            "roundtrip_image.py can encode/decode it."
        )

    stats = encode_decode_image(
        model=model,
        image_path=args.image,
        cnz_path=cnz_path,
        recon_path=recon_path,
        device=device,
        codec=args.codec,
        zlib_level=args.zlib_level,
    )

    print(f"image: {args.image}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"cnz: {cnz_path}")
    print(f"recon: {recon_path}")
    print(f"original_size: {stats['original_size']}")
    print(f"padded_size: {stats['padded_size']}")
    print(f"latent_shape: {stats['latent_shape']}")
    print(f"dtype: {stats['dtype']}")
    print(f"codec: {stats['codec']}")
    print(f"payload_bpp: {stats['payload_bpp']:.4f}")
    print(f"container_bpp: {stats['container_bpp']:.4f}")
    if args.timing:
        print(f"timing_encode_ms: {stats['encode_ms']:.3f}")
        print(f"timing_write_cnz_ms: {stats['write_cnz_ms']:.3f}")
        print(f"timing_decode_model_ms: {stats['decode_model_ms']:.3f}")
        print(f"timing_save_recon_ms: {stats['save_recon_ms']:.3f}")
        print(f"timing_total_ms: {stats['total_ms']:.3f}")


if __name__ == "__main__":
    main()
