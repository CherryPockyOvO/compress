from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compressai_nano import get_model, infer_model_variant_from_checkpoint


def load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> None:
    raw = torch.load(checkpoint, map_location="cpu")
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"missing_keys: {len(missing)}")
    print(f"unexpected_keys: {len(unexpected)}")


def image_to_tensor(
    path: Path,
    height: int | None,
    width: int | None,
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    source_w, source_h = image.size
    if height is not None or width is not None:
        if height is None or width is None:
            raise ValueError("--height and --width must be provided together")
        resample = getattr(Image, "Resampling", Image).BICUBIC
        image = image.resize((width, height), resample)
    values = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    width, height = image.size
    values = values.to(torch.float32).view(height, width, 3).permute(2, 0, 1)
    return values.unsqueeze(0) / 255.0, (source_h, source_w), (height, width)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump encoder latent y as float32 NCHW raw file.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Optional resize height before encoding. Defaults to the image's original height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional resize width before encoding. Defaults to the image's original width.",
    )
    parser.add_argument(
        "--meta-output",
        type=Path,
        default=None,
        help="Metadata JSON path. Defaults to <output>.json.",
    )
    parser.add_argument("--no-meta", action="store_true")
    parser.add_argument(
        "--dump-analysis-json",
        action="store_true",
        help="For hyperprior models, also write z/scales_y statistics to metadata.",
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    raw = torch.load(args.checkpoint, map_location="cpu")
    model_variant = infer_model_variant_from_checkpoint(raw)
    model = get_model(model_variant=model_variant).to(device).eval()
    load_checkpoint(model, args.checkpoint)
    x, source_size, encoded_size = image_to_tensor(args.image, args.height, args.width)
    x = x.to(device)
    factor = model.downsampling_factor
    encoded_h, encoded_w = encoded_size
    if encoded_h % factor != 0 or encoded_w % factor != 0:
        pad_h = (factor - encoded_h % factor) % factor
        pad_w = (factor - encoded_w % factor) % factor
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    with torch.no_grad():
        y = model.encoder(x).detach().cpu().contiguous()
        analysis = None
        if args.dump_analysis_json and hasattr(model, "analysis_transform"):
            analysis_outputs = model.analysis_transform(x)
            y_live, z_live, scales_live = analysis_outputs[:3]
            means_live = analysis_outputs[3] if len(analysis_outputs) > 3 else None
            analysis = {
                "z_shape": [int(v) for v in z_live.shape],
                "scales_y_shape": [int(v) for v in scales_live.shape],
                "means_y_shape": [int(v) for v in means_live.shape] if means_live is not None else [],
                "y_min": float(y_live.min().detach().cpu()),
                "y_max": float(y_live.max().detach().cpu()),
                "z_min": float(z_live.min().detach().cpu()),
                "z_max": float(z_live.max().detach().cpu()),
                "scale_min": float(scales_live.min().detach().cpu()),
                "scale_max": float(scales_live.max().detach().cpu()),
            }
            if means_live is not None:
                analysis["mean_min"] = float(means_live.min().detach().cpu())
                analysis["mean_max"] = float(means_live.max().detach().cpu())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    y.numpy().astype("<f4", copy=False).tofile(args.output)
    if not args.no_meta:
        meta_path = args.meta_output or Path(str(args.output) + ".json")
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format": "compressai-nano-latent-metadata-v1",
            "model_variant": model_variant,
            "image": str(args.image),
            "dtype": "float32",
            "layout": "NCHW",
            "source_h": int(source_size[0]),
            "source_w": int(source_size[1]),
            "orig_h": int(encoded_h),
            "orig_w": int(encoded_w),
            "padded_h": int(x.shape[-2]),
            "padded_w": int(x.shape[-1]),
            "latent_c": int(y.shape[1]),
            "latent_h": int(y.shape[2]),
            "latent_w": int(y.shape[3]),
            "downsampling_factor": int(factor),
        }
        if analysis is not None:
            metadata["analysis"] = analysis
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"source_size: {source_size}")
    print(f"encoded_size: {encoded_size}")
    print(f"padded_size: {tuple(int(v) for v in x.shape[-2:])}")
    print(f"latent_shape: {tuple(y.shape)}")
    print(f"model_variant: {model_variant}")
    print(f"saved: {args.output}")
    if not args.no_meta:
        print(f"metadata: {meta_path}")


if __name__ == "__main__":
    main()
