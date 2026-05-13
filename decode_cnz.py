from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import zlib
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from compressai_nano import FactorizedPriorNano
from compressai_nano.cnz import MAGIC, cnz_to_y_hat, read_cnz_file


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def now_synced(device: torch.device) -> float:
    sync_device(device)
    return time.perf_counter()


def make_autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def tensor_to_image(tensor: torch.Tensor, path: Path) -> None:
    tensor = tensor.squeeze(0).detach().cpu().clamp(0, 1)
    tensor = (tensor * 255.0).round().to(torch.uint8)
    tensor = tensor.permute(1, 2, 0).contiguous()
    height, width = tensor.shape[:2]
    image = Image.frombytes("RGB", (width, height), tensor.numpy().tobytes())
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def crop_to_size(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    height, width = size
    return x[..., :height, :width]


def resize_tensor_to_size(
    x: torch.Tensor,
    size: tuple[int, int],
    mode: str = "bicubic",
) -> torch.Tensor:
    """Resize NCHW tensor to target (height, width)."""
    height, width = size
    if x.shape[-2:] == (height, width):
        return x

    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=(height, width),
            mode=mode,
            align_corners=False,
        )

    return F.interpolate(
        x,
        size=(height, width),
        mode=mode,
    )


def read_source_size_from_meta(meta_path: Path) -> tuple[int, int]:
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    if "source_h" not in metadata or "source_w" not in metadata:
        raise ValueError(f"metadata missing source_h/source_w: {meta_path}")

    source_h = int(metadata["source_h"])
    source_w = int(metadata["source_w"])

    if source_h <= 0 or source_w <= 0:
        raise ValueError(
            f"invalid source size in metadata: source_h={source_h}, source_w={source_w}"
        )

    return source_h, source_w


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


def legacy_pickle_payload_to_y_hat(
    package: dict[str, Any],
    model: FactorizedPriorNano,
    device: torch.device,
) -> torch.Tensor:
    strings = package["strings"]
    first = strings[0] if isinstance(strings, list) else strings
    if isinstance(first, bytes) and first[:4] == b"CNS1":
        restored = model.decompress(strings, package["shape"])
        return restored["x_hat"], restored["y_hat"]

    y_hats = []
    for string in strings:
        payload = pickle.loads(zlib.decompress(string))
        if payload.get("dtype") != "int32":
            raise ValueError(f"unsupported legacy dtype: {payload.get('dtype')}")
        sample_shape = tuple(int(v) for v in payload["shape"])
        symbols = torch.tensor(payload["symbols"], dtype=torch.int32).reshape(1, *sample_shape)
        y_hats.append(
            model.entropy_bottleneck.dequantize(
                symbols,
                quant_step=float(payload.get("quant_step", model.entropy_bottleneck.quant_step.item())),
                dtype=torch.float32,
                device=device,
            )
        )

    y_hat = torch.cat(y_hats, dim=0).to(device)
    x_hat = model.decoder(y_hat)
    return x_hat, y_hat


def decode_cnz(
    path: Path,
    model: FactorizedPriorNano,
    device: torch.device,
    use_half: bool,
) -> tuple[torch.Tensor, tuple[int, int]]:
    cnz_file = read_cnz_file(path)
    y_hat = cnz_to_y_hat(cnz_file, device=device)

    if use_half:
        y_hat = y_hat.to(torch.float16)

    with make_autocast(device, use_half):
        x_hat = model.decoder(y_hat)

    original_size = (cnz_file.header.orig_h, cnz_file.header.orig_w)

    print(f"format: CNZ4 v{cnz_file.header.version}")
    print(f"latent_shape: {(1, cnz_file.header.latent_c, cnz_file.header.latent_h, cnz_file.header.latent_w)}")
    print(f"payload_size: {cnz_file.header.payload_size}")
    print(f"encoded_original_size_from_cnz: {original_size}")

    return x_hat, original_size


def decode_legacy(
    path: Path,
    model: FactorizedPriorNano,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, int]]:
    package = pickle.loads(path.read_bytes())

    if package.get("format") not in {"compressai-nano-v2", "compressai-nano-v3"}:
        raise ValueError(f"unsupported legacy format: {package.get('format')}")

    x_hat, _y_hat = legacy_pickle_payload_to_y_hat(package, model, device)
    original_size = tuple(int(v) for v in package["original_size"])

    print(f"format: {package.get('format')}")
    print(f"latent_shape: {package.get('latent_shape')}")
    print(f"encoded_original_size_from_legacy: {original_size}")

    return x_hat, original_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode CNZ4 bitstream on PC using the PyTorch decoder.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("recon.png"))

    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Decode device. auto uses CUDA when available.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 autocast for the decoder on CUDA. Faster, with small numeric differences.",
    )
    parser.add_argument(
        "--channels-last",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use channels-last memory format on CUDA for faster convolutions.",
    )

    parser.add_argument(
        "--source-meta",
        type=Path,
        default=None,
        help=(
            "Optional latent metadata json, usually latent.bin.json. "
            "If provided with --resize-to-source, output is resized back to source_h/source_w."
        ),
    )
    parser.add_argument(
        "--resize-to-source",
        action="store_true",
        help=(
            "Resize decoded image back to source_h/source_w from --source-meta. "
            "Useful when RKNN input was resized to a fixed shape such as 512x512."
        ),
    )
    parser.add_argument(
        "--resize-mode",
        choices=("nearest", "bilinear", "bicubic"),
        default="bicubic",
        help="Resize mode used by --resize-to-source. Default: bicubic.",
    )

    parser.add_argument("--timing", action="store_true", help="Print decode stage timings.")
    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.cpu:
        args.device = "cpu"

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")

    if args.resize_to_source and args.source_meta is None:
        raise ValueError("--source-meta is required when --resize-to-source is used")

    if args.source_meta is not None and not args.source_meta.exists():
        raise FileNotFoundError(f"source metadata file not found: {args.source_meta}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"device: cuda ({torch.cuda.get_device_name(device)})")
    else:
        print("device: cpu")

    model = FactorizedPriorNano().to(device).eval()

    if device.type == "cuda" and args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.half:
        if device.type != "cuda":
            raise RuntimeError("--half is only supported with CUDA")
        model = model.half()

    t0 = now_synced(device)
    load_checkpoint(model, args.checkpoint)
    t1 = now_synced(device)

    with torch.inference_mode():
        t2 = now_synced(device)
        prefix = args.input.read_bytes()[:4]
        t3 = now_synced(device)

        if prefix == MAGIC:
            x_hat, original_size = decode_cnz(args.input, model, device, use_half=args.half)
        else:
            x_hat, original_size = decode_legacy(args.input, model, device)

        t4 = now_synced(device)

        # First crop to encoded size from CNZ/legacy header.
        # Example in fixed RKNN resize mode:
        # decoder output: 512x512
        # original_size from CNZ: 512x512
        x_hat = crop_to_size(x_hat, original_size)
        print(f"cropped_to_encoded_size: {original_size}")

        # Optional: resize back to the real source image size from latent metadata.
        # Example:
        # source image: 640x425
        # RKNN input: 512x512
        # decoded image: 512x512
        # final image: 640x425
        if args.resize_to_source:
            source_size = read_source_size_from_meta(args.source_meta)
            x_hat = resize_tensor_to_size(x_hat, source_size, mode=args.resize_mode)
            print(f"resized_to_source_size: {source_size} using {args.resize_mode}")

        t5 = now_synced(device)

    tensor_to_image(x_hat, args.output)
    t6 = now_synced(device)

    if args.timing:
        print(f"timing_load_checkpoint_ms={(t1 - t0) * 1000:.3f}")
        print(f"timing_read_magic_ms={(t3 - t2) * 1000:.3f}")
        print(f"timing_decode_model_ms={(t4 - t3) * 1000:.3f}")
        print(f"timing_crop_resize_ms={(t5 - t4) * 1000:.3f}")
        print(f"timing_save_image_ms={(t6 - t5) * 1000:.3f}")
        print(f"timing_total_ms={(t6 - t0) * 1000:.3f}")

    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()