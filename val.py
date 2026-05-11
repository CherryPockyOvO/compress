from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from compressai_nano import FactorizedPriorNano
from train import IMAGE_EXTENSIONS, ToTensor, compute_bpp


class EvalImageFolder(Dataset):
    def __init__(self, root: Path, transform: Callable[[Image.Image], torch.Tensor]) -> None:
        self.root = Path(root)
        self.transform = transform
        self.paths = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found under: {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, int]:
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image), str(self.paths[index]), index


def load_checkpoint(path: Path, quality_level: int | None) -> tuple[FactorizedPriorNano, dict[str, Any]]:
    raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, dict):
        raise ValueError("checkpoint must be a dict saved by train.py")

    model_quality = int(raw.get("quality_level", quality_level or 2))
    model = FactorizedPriorNano(quality_level=model_quality)
    state_dict = raw.get("state_dict", raw)
    model.load_state_dict(state_dict, strict=False)
    return model, raw


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


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = F.mse_loss(x, y).item()
    if mse <= 0:
        return 99.0
    return -10.0 * math.log10(mse)


def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    channels = x.size(1)
    sigma = 1.5
    coords = torch.arange(window_size, device=x.device, dtype=x.dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    window = kernel_2d.expand(channels, 1, window_size, window_size)

    padding = window_size // 2
    mu_x = F.conv2d(x, window, padding=padding, groups=channels)
    mu_y = F.conv2d(y, window, padding=padding, groups=channels)
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=channels) - mu_xy

    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    )
    return score.mean()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.squeeze(0).detach().cpu().clamp(0, 1)
    tensor = (tensor * 255.0).round().to(torch.uint8)
    tensor = tensor.permute(1, 2, 0).contiguous()
    height, width = tensor.shape[:2]
    return Image.frombytes("RGB", (width, height), bytes(tensor.reshape(-1).tolist()))


def save_comparison(original: torch.Tensor, restored: torch.Tensor, path: Path) -> None:
    left = tensor_to_pil(original)
    right = tensor_to_pil(restored)
    canvas = Image.new("RGB", (left.width + right.width, left.height + 28), "white")
    canvas.paste(left, (0, 28))
    canvas.paste(right, (left.width, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "original", fill=(0, 0, 0))
    draw.text((left.width + 8, 8), "restored", fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    model, checkpoint = load_checkpoint(args.checkpoint, args.quality_level)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = model.to(device).eval()

    dataset = EvalImageFolder(args.data_dir, transform=ToTensor())
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    selected = set(random.Random(args.seed).sample(range(len(dataset)), min(args.visuals, len(dataset))))

    total_psnr = 0.0
    total_ssim = 0.0
    total_bpp = 0.0
    total_actual_bpp = 0.0

    for x, path_text, index in loader:
        del path_text
        x = x.to(device)
        x_padded, original_size = pad_to_multiple(x, model.downsampling_factor)
        output = model(x_padded)
        x_hat = crop_to_size(output["x_hat"], original_size)

        pixels = x.size(0) * x.size(2) * x.size(3)
        total_psnr += psnr(x, x_hat)
        total_ssim += float(ssim(x, x_hat))
        total_bpp += float(compute_bpp(output["likelihoods"], pixels))
        payload = model.entropy_bottleneck.compress(output["y"])
        total_actual_bpp += sum(len(item) * 8 for item in payload.strings) / pixels

        image_index = int(index.item())
        if image_index in selected:
            save_comparison(
                x,
                x_hat,
                args.results_dir / f"comparison_{image_index:04d}.png",
            )

    count = len(dataset)
    print(f"checkpoint: {args.checkpoint}")
    print(f"quality_level: {checkpoint.get('quality_level', args.quality_level)}")
    print(f"images: {count}")
    print(f"avg_psnr: {total_psnr / count:.4f} dB")
    print(f"avg_ssim: {total_ssim / count:.6f}")
    print(f"avg_estimated_bpp: {total_bpp / count:.6f}")
    print(f"avg_actual_zlib_bpp: {total_actual_bpp / count:.6f}")
    print(f"comparisons: {args.results_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate compressai-nano.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--quality-level", type=int, default=None, choices=(1, 2, 3))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--visuals", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
