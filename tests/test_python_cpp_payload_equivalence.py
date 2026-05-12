from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compressai_nano import FactorizedPriorNano
from compressai_nano.cnz import cnz_to_y_hat, quantize_latent, read_cnz_file


def load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> None:
    raw = torch.load(checkpoint, map_location="cpu")
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"missing_keys: {len(missing)}")
    print(f"unexpected_keys: {len(unexpected)}")


def image_to_tensor(path: Path, height: int, width: int) -> torch.Tensor:
    resample = getattr(Image, "Resampling", Image).BICUBIC
    image = Image.open(path).convert("RGB").resize((width, height), resample)
    values = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    values = values.to(torch.float32).view(height, width, 3).permute(2, 0, 1)
    return values.unsqueeze(0) / 255.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Python quant/dequant with C++ CNZ encode output.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--cnz-encode-cli", type=Path, required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = FactorizedPriorNano().to(device).eval()
    load_checkpoint(model, args.checkpoint)
    x = image_to_tensor(args.image, args.height, args.width).to(device)
    with torch.no_grad():
        y = model.encoder(x)
        symbols = quantize_latent(
            y,
            model.entropy_bottleneck.medians.detach(),
            float(model.entropy_bottleneck.quant_step.detach().cpu()),
        )
        y_hat_ref = model.entropy_bottleneck.dequantize(symbols, device=device)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        latent_path = tmp_dir / "latent.bin"
        cnz_path = tmp_dir / "test.cnz"
        y.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tofile(latent_path)
        subprocess.run(
            [
                str(args.cnz_encode_cli),
                "--latent", str(latent_path),
                "--params", str(args.params),
                "--output", str(cnz_path),
                "--orig-h", str(args.height),
                "--orig-w", str(args.width),
                "--padded-h", str(args.height),
                "--padded-w", str(args.width),
                "--latent-c", str(y.shape[1]),
                "--latent-h", str(y.shape[2]),
                "--latent-w", str(y.shape[3]),
                "--codec", "zlib",
                "--zlib-level", "1",
            ],
            check=True,
        )
        y_hat_cpp = cnz_to_y_hat(read_cnz_file(cnz_path), device=device)

    diff = (y_hat_cpp - y_hat_ref).abs()
    print(f"max_abs_error={float(diff.max()):.10f}")
    print(f"mean_abs_error={float(diff.mean()):.10f}")
    if float(diff.max()) > 1e-6:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
