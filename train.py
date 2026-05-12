from __future__ import annotations

import argparse
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from compressai_nano import FactorizedPriorNano


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_LAMBDA = 0.0067

TRAIN_PROFILES = {
    "balanced": {
        "lmbda": DEFAULT_LAMBDA,
        "ssim_weight": 0.2,
        "detail_weight": 0.0,
        "l1_weight": 0.0,
        "quant_step": None,
        "epochs": 100,
        "batch_size": 4,
        "crop_size": 256,
        "lr": 1e-4,
    },
    "detail": {
        "lmbda": 0.0130,
        "ssim_weight": 0.35,
        "detail_weight": 0.12,
        "l1_weight": 0.03,
        "quant_step": 0.50,
        "epochs": 80,
        "batch_size": 2,
        "crop_size": 384,
        "lr": 3e-5,
    },
}


class ImageFolderDataset(Dataset):
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

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image)


class Compose:
    def __init__(self, transforms: list[Callable[[Any], Any]]) -> None:
        self.transforms = transforms

    def __call__(self, value: Any) -> Any:
        for transform in self.transforms:
            value = transform(value)
        return value


class RandomCrop:
    def __init__(self, size: int) -> None:
        self.size = int(size)

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width < self.size or height < self.size:
            scale = max(self.size / width, self.size / height)
            width = max(self.size, math.ceil(width * scale))
            height = max(self.size, math.ceil(height * scale))
            resample = getattr(Image, "Resampling", Image).BICUBIC
            image = image.resize((width, height), resample=resample)

        left = random.randint(0, width - self.size)
        top = random.randint(0, height - self.size)
        return image.crop((left, top, left + self.size, top + self.size))


class CenterCrop:
    def __init__(self, size: int) -> None:
        self.size = int(size)

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width < self.size or height < self.size:
            scale = max(self.size / width, self.size / height)
            width = max(self.size, math.ceil(width * scale))
            height = max(self.size, math.ceil(height * scale))
            resample = getattr(Image, "Resampling", Image).BICUBIC
            image = image.resize((width, height), resample=resample)

        left = (width - self.size) // 2
        top = (height - self.size) // 2
        return image.crop((left, top, left + self.size, top + self.size))


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = float(p)

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return ImageOps.mirror(image)
        return image


class ToTensor:
    def __call__(self, image: Image.Image) -> torch.Tensor:
        width, height = image.size
        values = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        values = values.to(torch.float32).view(height, width, 3).permute(2, 0, 1)
        return values / 255.0


def make_train_transform(crop_size: int) -> Compose:
    return Compose(
        [
            RandomCrop(crop_size),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
        ]
    )


def make_eval_transform(crop_size: int) -> Compose:
    return Compose(
        [
            CenterCrop(crop_size),
            ToTensor(),
        ]
    )


def compute_bpp(likelihoods: dict[str, torch.Tensor], num_pixels: int) -> torch.Tensor:
    bits = torch.zeros((), device=next(iter(likelihoods.values())).device)
    for likelihood in likelihoods.values():
        bits = bits + torch.sum(-torch.log2(likelihood.clamp_min(1e-9)))
    return bits / float(num_pixels)


def ssim_index(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
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


def gradient_detail_loss(x_hat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    channels = x_hat.size(1)
    kernel_x = x_hat.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    kernel_y = x_hat.new_tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
    ).view(1, 1, 3, 3)
    kernel_x = kernel_x.expand(channels, 1, 3, 3) / 8.0
    kernel_y = kernel_y.expand(channels, 1, 3, 3) / 8.0

    pred_gx = F.conv2d(x_hat, kernel_x, padding=1, groups=channels)
    pred_gy = F.conv2d(x_hat, kernel_y, padding=1, groups=channels)
    target_gx = F.conv2d(target, kernel_x, padding=1, groups=channels)
    target_gy = F.conv2d(target, kernel_y, padding=1, groups=channels)
    eps = 1e-3
    return (
        torch.sqrt((pred_gx - target_gx).pow(2) + eps * eps).mean()
        + torch.sqrt((pred_gy - target_gy).pow(2) + eps * eps).mean()
    )


class RateDistortionLoss(nn.Module):
    def __init__(
        self,
        lmbda: float,
        ssim_weight: float = 0.2,
        detail_weight: float = 0.0,
        l1_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.lmbda = float(lmbda)
        self.ssim_weight = float(ssim_weight)
        self.detail_weight = float(detail_weight)
        self.l1_weight = float(l1_weight)

    def forward(self, output: dict[str, Any], target: torch.Tensor) -> dict[str, torch.Tensor]:
        x_hat = output["x_hat"]
        mse = F.mse_loss(x_hat, target)
        num_pixels = target.size(0) * target.size(2) * target.size(3)
        bpp = compute_bpp(output["likelihoods"], num_pixels)
        distortion = self.lmbda * (255.0**2) * mse
        ssim = ssim_index(x_hat, target)
        ssim_loss = 1.0 - ssim
        l1_loss = F.l1_loss(x_hat, target)
        if self.detail_weight > 0:
            detail_loss = gradient_detail_loss(x_hat, target)
        else:
            detail_loss = mse.new_zeros(())
        loss = (
            distortion
            + bpp
            + self.ssim_weight * ssim_loss
            + self.detail_weight * detail_loss
            + self.l1_weight * l1_loss
        )
        return {
            "loss": loss,
            "mse": mse,
            "bpp": bpp,
            "ssim": ssim,
            "ssim_loss": ssim_loss,
            "detail_loss": detail_loss,
            "l1_loss": l1_loss,
        }


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
) -> int:
    raw = torch.load(path, map_location="cpu")
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    model.load_state_dict(state_dict, strict=False)
    if optimizer is not None and isinstance(raw, dict) and "optimizer" in raw:
        optimizer.load_state_dict(raw["optimizer"])
    if scheduler is not None and isinstance(raw, dict) and "scheduler" in raw:
        scheduler.load_state_dict(raw["scheduler"])
    return int(raw.get("epoch", 0)) if isinstance(raw, dict) else 0


@torch.no_grad()
def set_quant_step(model: FactorizedPriorNano, quant_step: float | None) -> None:
    if quant_step is None:
        return
    if quant_step <= 0:
        raise ValueError(f"quant_step must be positive, got {quant_step}")
    model.entropy_bottleneck.quant_step.fill_(float(quant_step))


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
) -> None:
    payload = {
        "epoch": epoch,
        "quality_profile": args.quality_profile,
        "lambda": args.lmbda,
        "ssim_weight": args.ssim_weight,
        "detail_weight": args.detail_weight,
        "l1_weight": args.l1_weight,
        "quant_step": float(model.entropy_bottleneck.quant_step.detach().cpu()),
        "encoder_activation": args.encoder_activation,
        "decoder_activation": args.decoder_activation,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if args.lr_scheduler == "none":
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        threshold=args.lr_threshold,
        cooldown=args.lr_cooldown,
        min_lr=args.min_lr,
    )


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def make_autocast(enabled: bool):
    if enabled:
        return torch.amp.autocast("cuda")
    return nullcontext()


def train_one_epoch(
    model: FactorizedPriorNano,
    criterion: RateDistortionLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_clip: float,
    amp_enabled: bool,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "mse": 0.0,
        "bpp": 0.0,
        "ssim": 0.0,
        "detail_loss": 0.0,
        "l1_loss": 0.0,
    }

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with make_autocast(amp_enabled):
            output = model(batch)
            losses = criterion(output, batch)

        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        batch_size = batch.size(0)
        for key in totals:
            totals[key] += float(losses[key].detach()) * batch_size

    count = len(loader.dataset)
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def evaluate_loss(
    model: FactorizedPriorNano,
    criterion: RateDistortionLoss,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "mse": 0.0,
        "bpp": 0.0,
        "ssim": 0.0,
        "detail_loss": 0.0,
        "l1_loss": 0.0,
    }

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        output = model(batch)
        losses = criterion(output, batch)
        batch_size = batch.size(0)
        for key in totals:
            totals[key] += float(losses[key].detach()) * batch_size

    count = len(loader.dataset)
    return {f"val_{key}": value / count for key, value in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train compressai-nano.")
    parser.add_argument(
        "--quality-profile",
        choices=tuple(TRAIN_PROFILES.keys()),
        default="balanced",
        help="Training preset. detail is the high-precision fine-tuning preset for hair, lines, and texture.",
    )
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, default=None)
    parser.add_argument("--lambda", dest="lmbda", type=float, default=None)
    parser.add_argument("--ssim-weight", type=float, default=None)
    parser.add_argument(
        "--detail-weight",
        type=float,
        default=None,
        help="Weight for Sobel gradient detail loss. Use >0 to preserve lines and fine texture.",
    )
    parser.add_argument(
        "--l1-weight",
        type=float,
        default=None,
        help="Optional L1 reconstruction weight. Small values usually sharpen fine local contrast.",
    )
    parser.add_argument(
        "--quant-step",
        type=float,
        default=None,
        help="Override latent quantization step after model initialization/checkpoint load.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--lr-scheduler",
        choices=("reduce-on-plateau", "none"),
        default="reduce-on-plateau",
        help="Adapt learning rate from validation loss, or training loss when no validation set is given.",
    )
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--lr-threshold", type=float, default=1e-4)
    parser.add_argument("--lr-cooldown", type=int, default=0)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Load model weights only and start a new fine-tuning run from epoch 1.",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-activation", choices=("relu", "leaky_relu"), default="relu")
    parser.add_argument("--decoder-activation", choices=("relu", "leaky_relu"), default="leaky_relu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def apply_quality_profile(args: argparse.Namespace) -> None:
    profile = TRAIN_PROFILES[args.quality_profile]
    for key, value in profile.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    if args.init_checkpoint is not None and args.resume is not None:
        raise ValueError("--init-checkpoint and --resume are mutually exclusive")
    if args.detail_weight < 0:
        raise ValueError("--detail-weight must be non-negative")
    if args.l1_weight < 0:
        raise ValueError("--l1-weight must be non-negative")
    if args.quant_step is not None and args.quant_step <= 0:
        raise ValueError("--quant-step must be positive")


def main() -> None:
    args = parse_args()
    apply_quality_profile(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("device: cpu")

    transform = make_train_transform(args.crop_size)
    train_dataset = ImageFolderDataset(args.train_dir, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    val_loader = None
    if args.val_dir is not None:
        val_dataset = ImageFolderDataset(
            args.val_dir,
            transform=make_eval_transform(args.crop_size),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

    model = FactorizedPriorNano(
        activation=args.encoder_activation,
        decoder_activation=args.decoder_activation,
    ).to(device)
    set_quant_step(model, args.quant_step)
    criterion = RateDistortionLoss(
        args.lmbda,
        ssim_weight=args.ssim_weight,
        detail_weight=args.detail_weight,
        l1_weight=args.l1_weight,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = make_scheduler(optimizer, args)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 0
    if args.init_checkpoint is not None:
        source_epoch = load_checkpoint(args.init_checkpoint, model)
        set_quant_step(model, args.quant_step)
        print(f"initialized weights: {args.init_checkpoint} (source epoch {source_epoch})")
    elif args.resume is not None:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)
        set_quant_step(model, args.quant_step)
        print(f"resumed: {args.resume} at epoch {start_epoch}")

    quant_step = float(model.entropy_bottleneck.quant_step.detach().cpu())
    print(
        f"profile={args.quality_profile}, train images={len(train_dataset)}, "
        f"lambda={args.lmbda}, ssim_weight={args.ssim_weight}, "
        f"detail_weight={args.detail_weight}, l1_weight={args.l1_weight}, "
        f"quant_step={quant_step}, crop={args.crop_size}, "
        f"batch={args.batch_size}, lr={get_current_lr(optimizer):.2e}"
    )

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            scaler,
            device,
            args.grad_clip,
            amp_enabled,
        )
        metrics = dict(train_metrics)
        if val_loader is not None:
            metrics.update(evaluate_loss(model, criterion, val_loader, device))

        monitor_name = "val_loss" if val_loader is not None else "loss"
        monitor_value = metrics[monitor_name]
        old_lr = get_current_lr(optimizer)
        if scheduler is not None:
            scheduler.step(monitor_value)
        new_lr = get_current_lr(optimizer)
        metrics["lr"] = new_lr

        metric_text = ", ".join(f"{key}={value:.6f}" for key, value in metrics.items())
        lr_text = ""
        if new_lr < old_lr:
            lr_text = f" | lr reduced: {old_lr:.2e} -> {new_lr:.2e}"
        print(f"epoch {epoch:03d}: {metric_text} | monitor={monitor_name}{lr_text}")

        epoch_path = args.checkpoint_dir / f"epoch{epoch:03d}.pt"
        latest_path = args.checkpoint_dir / "latest.pt"
        save_checkpoint(epoch_path, model, optimizer, scheduler, epoch, args, metrics)
        save_checkpoint(latest_path, model, optimizer, scheduler, epoch, args, metrics)


if __name__ == "__main__":
    main()
