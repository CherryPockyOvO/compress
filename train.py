from __future__ import annotations

import argparse
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
Image.MAX_IMAGE_PIXELS = None
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from compressai_nano import FactorizedPriorNano


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_LAMBDA = 0.0067
OFFICIAL_MSE_LAMBDA_Q7 = 0.0932

TRAIN_PROFILES = {
    "balanced": {
        "lmbda": DEFAULT_LAMBDA,
        "rate_weight": 1.0,
        "target_bpp": None,
        "ssim_weight": 0.2,
        "detail_weight": 0.0,
        "l1_weight": 0.0,
        "lpips_weight": 0.0,
        "lpips_net": "alex",
        "quant_step": None,
        "epochs": 100,
        "batch_size": 128,
        "crop_size": 256,
        "lr": 1e-4,
    },
    "detail": {
        "lmbda": OFFICIAL_MSE_LAMBDA_Q7,
        "rate_weight": 1.0,
        "target_bpp": 1.0,
        "ssim_weight": 0.40,
        "detail_weight": 6.0,
        "l1_weight": 0.50,
        "lpips_weight": 0.03,
        "lpips_net": "alex",
        "quant_step": 0.45,
        "epochs": 80,
        "batch_size": 64,
        "crop_size": 384,
        "lr": 3e-5,
    },
}


@dataclass(frozen=True)
class CheckpointState:
    epoch: int
    global_step: int


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


def make_lpips_model(net: str) -> nn.Module:
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError(
            "LPIPS loss requires the 'lpips' package. "
            "Install it with: python -m pip install lpips"
        ) from exc

    model = lpips.LPIPS(net=net, verbose=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def to_lpips_range(value: torch.Tensor) -> torch.Tensor:
    return value.float().clamp(0.0, 1.0).mul(2.0).sub(1.0)


class RateDistortionLoss(nn.Module):
    def __init__(
        self,
        lmbda: float,
        rate_weight: float = 1.0,
        target_bpp: float | None = None,
        ssim_weight: float = 0.2,
        detail_weight: float = 0.0,
        l1_weight: float = 0.0,
        lpips_weight: float = 0.0,
        lpips_net: str = "alex",
    ) -> None:
        super().__init__()
        self.lmbda = float(lmbda)
        self.rate_weight = float(rate_weight)
        self.target_bpp = None if target_bpp is None else float(target_bpp)
        self.ssim_weight = float(ssim_weight)
        self.detail_weight = float(detail_weight)
        self.l1_weight = float(l1_weight)
        self.lpips_weight = float(lpips_weight)
        self.lpips_net = lpips_net
        self.lpips_model = make_lpips_model(lpips_net) if self.lpips_weight > 0 else None

    def forward(self, output: dict[str, Any], target: torch.Tensor) -> dict[str, torch.Tensor]:
        x_hat = output["x_hat"]
        mse = F.mse_loss(x_hat, target)
        num_pixels = target.size(0) * target.size(2) * target.size(3)
        bpp = compute_bpp(output["likelihoods"], num_pixels)
        if self.target_bpp is None:
            rate_loss = bpp
        else:
            rate_loss = torch.relu(bpp - bpp.new_tensor(self.target_bpp))
        distortion = self.lmbda * (255.0**2) * mse
        ssim = ssim_index(x_hat, target)
        ssim_loss = 1.0 - ssim
        l1_loss = F.l1_loss(x_hat, target)
        if self.detail_weight > 0:
            detail_loss = gradient_detail_loss(x_hat, target)
        else:
            detail_loss = mse.new_zeros(())
        if self.lpips_model is not None:
            if x_hat.device.type == "cuda":
                lpips_context = torch.amp.autocast("cuda", enabled=False)
            else:
                lpips_context = nullcontext()
            with lpips_context:
                lpips_loss = self.lpips_model(
                    to_lpips_range(x_hat),
                    to_lpips_range(target),
                ).mean()
        else:
            lpips_loss = mse.new_zeros(())
        loss = (
            distortion
            + self.rate_weight * rate_loss
            + self.ssim_weight * ssim_loss
            + self.detail_weight * detail_loss
            + self.l1_weight * l1_loss
            + self.lpips_weight * lpips_loss
        )
        return {
            "loss": loss,
            "mse": mse,
            "bpp": bpp,
            "rate_loss": rate_loss,
            "ssim": ssim,
            "ssim_loss": ssim_loss,
            "detail_loss": detail_loss,
            "l1_loss": l1_loss,
            "lpips_loss": lpips_loss,
        }


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
) -> CheckpointState:
    raw = torch.load(path, map_location="cpu")
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    model.load_state_dict(state_dict, strict=False)
    if optimizer is not None and isinstance(raw, dict) and "optimizer" in raw:
        optimizer.load_state_dict(raw["optimizer"])
    if scheduler is not None and isinstance(raw, dict) and "scheduler" in raw:
        scheduler.load_state_dict(raw["scheduler"])
    if isinstance(raw, dict):
        return CheckpointState(
            epoch=int(raw.get("epoch", 0)),
            global_step=int(raw.get("global_step", 0)),
        )
    return CheckpointState(epoch=0, global_step=0)


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
    global_step: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
) -> None:
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "quality_profile": args.quality_profile,
        "lambda": args.lmbda,
        "rate_weight": args.rate_weight,
        "target_bpp": args.target_bpp,
        "ssim_weight": args.ssim_weight,
        "detail_weight": args.detail_weight,
        "l1_weight": args.l1_weight,
        "lpips_weight": args.lpips_weight,
        "lpips_net": args.lpips_net,
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


def psnr_from_mse(mse: float) -> float:
    if mse <= 0.0:
        return math.inf
    return -10.0 * math.log10(mse)


def format_primary_metrics(metrics: dict[str, float], prefix: str = "") -> str:
    key = lambda name: f"{prefix}{name}" if prefix else name
    mse = metrics[key("mse")]
    psnr = psnr_from_mse(mse)
    return (
        f"loss={metrics[key('loss')]:.4f} "
        f"bpp={metrics[key('bpp')]:.3f} "
        f"mse={mse:.6f} "
        f"psnr={psnr:.2f} "
        f"ssim={metrics[key('ssim')]:.4f}"
    )


def format_detail_metrics(
    metrics: dict[str, float],
    prefix: str = "",
    include_lpips: bool = False,
) -> str:
    key = lambda name: f"{prefix}{name}" if prefix else name
    parts = [
        f"rate={metrics[key('rate_loss')]:.4f}",
        f"grad={metrics[key('detail_loss')]:.5f}",
        f"l1={metrics[key('l1_loss')]:.5f}",
    ]
    if include_lpips or metrics.get(key("lpips_loss"), 0.0) > 0.0:
        parts.append(f"lpips={metrics[key('lpips_loss')]:.5f}")
    skipped_key = key("skipped_batches")
    if metrics.get(skipped_key, 0.0) > 0.0:
        parts.append(f"skipped={metrics[skipped_key]:.0f}")
    return " ".join(parts)


def format_checkpoint_summary(
    checkpoint_name: str,
    epoch: int,
    global_step: int,
    metrics: dict[str, float],
    monitor_name: str,
    old_lr: float,
    new_lr: float,
    include_lpips: bool,
) -> str:
    primary_parts = [
        f"{checkpoint_name} epoch {epoch:03d} step {global_step}",
        f"train {format_primary_metrics(metrics)}",
    ]
    if "val_loss" in metrics:
        primary_parts.append(f"val {format_primary_metrics(metrics, 'val_')}")

    detail_parts = [
        f"train {format_detail_metrics(metrics, include_lpips=include_lpips)}",
    ]
    if "val_loss" in metrics:
        detail_parts.append(
            f"val {format_detail_metrics(metrics, 'val_', include_lpips=include_lpips)}"
        )

    lr_text = f"lr={new_lr:.2e} monitor={monitor_name}={metrics[monitor_name]:.4f}"
    if new_lr < old_lr:
        lr_text += f" reduced {old_lr:.2e}->{new_lr:.2e}"

    return "\n".join(
        [
            " | ".join(primary_parts),
            "  detail: " + " | ".join(detail_parts),
            "  " + lr_text,
        ]
    )


def print_run_config(
    args: argparse.Namespace,
    train_dataset: ImageFolderDataset,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    model: FactorizedPriorNano,
    optimizer: torch.optim.Optimizer,
    amp_enabled: bool,
    global_step: int,
) -> None:
    quant_step = float(model.entropy_bottleneck.quant_step.detach().cpu())
    val_text = "none" if val_loader is None else str(len(val_loader.dataset))
    lpips_text = "off"
    if args.lpips_weight > 0:
        lpips_text = f"{args.lpips_weight:g}/{args.lpips_net}"
    print("run config:")
    print(
        f"  data: train={len(train_dataset)} val={val_text} "
        f"batch={args.batch_size} crop={args.crop_size} workers={args.num_workers}"
    )
    print(
        f"  objective: profile={args.quality_profile} lambda={args.lmbda:g} "
        f"target_bpp={args.target_bpp} rate={args.rate_weight:g} "
        f"ssim={args.ssim_weight:g} grad={args.detail_weight:g} "
        f"l1={args.l1_weight:g} lpips={lpips_text} quant_step={quant_step:g}"
    )
    print(
        f"  schedule: epochs={args.epochs} max_steps={args.max_steps} "
        f"steps/epoch={len(train_loader)} start_step={global_step} "
        f"lr={get_current_lr(optimizer):.2e} amp={amp_enabled}"
    )
    print(
        f"  checkpoints: every {args.checkpoint_interval_steps} steps -> eN.pt/latest.pt "
        f"eval_interval={args.eval_interval_steps} progress={args.progress}"
    )


def make_autocast(enabled: bool):
    if enabled:
        return torch.amp.autocast("cuda")
    return nullcontext()


def tensor_is_finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.detach()).all().item())


def metrics_are_finite(metrics: dict[str, float]) -> bool:
    return all(math.isfinite(value) for value in metrics.values())


def model_parameters_are_finite(model: nn.Module) -> bool:
    with torch.no_grad():
        return all(torch.isfinite(param).all().item() for param in model.parameters())


def train_one_epoch(
    model: FactorizedPriorNano,
    criterion: RateDistortionLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_clip: float,
    amp_enabled: bool,
    epoch: int,
    global_step: int,
    max_steps: int | None,
    checkpoint_interval_steps: int,
    progress_enabled: bool,
    on_checkpoint: Callable[[int, int, dict[str, float]], bool] | None,
) -> tuple[dict[str, float], int, bool]:
    model.train()
    totals = {
        "loss": 0.0,
        "mse": 0.0,
        "bpp": 0.0,
        "rate_loss": 0.0,
        "ssim": 0.0,
        "detail_loss": 0.0,
        "l1_loss": 0.0,
        "lpips_loss": 0.0,
    }
    interval_totals = {key: 0.0 for key in totals}
    processed_samples = 0
    interval_samples = 0
    skipped_batches = 0
    interval_skipped_batches = 0
    stop_training = False

    remaining_steps = None if max_steps is None else max(0, max_steps - global_step)
    total_batches = len(loader)
    if remaining_steps is not None:
        total_batches = min(total_batches, remaining_steps)
    progress = tqdm(
        loader,
        total=total_batches,
        desc=f"train e{epoch:03d}",
        unit="step",
        dynamic_ncols=True,
        leave=False,
        disable=not progress_enabled,
    )

    for batch in progress:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with make_autocast(amp_enabled):
            output = model(batch)
            losses = criterion(output, batch)

        if not tensor_is_finite(losses["loss"]):
            skipped_batches += 1
            interval_skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(step=global_step, skipped=skipped_batches)
            continue

        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not tensor_is_finite(grad_norm):
            skipped_batches += 1
            interval_skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            progress.set_postfix(step=global_step, skipped=skipped_batches)
            continue

        scaler.step(optimizer)
        scaler.update()

        batch_size = batch.size(0)
        processed_samples += batch_size
        interval_samples += batch_size
        global_step += 1
        for key in totals:
            value = float(losses[key].detach()) * batch_size
            totals[key] += value
            interval_totals[key] += value

        progress.set_postfix(
            step=global_step,
            loss=f"{float(losses['loss'].detach()):.4f}",
            bpp=f"{float(losses['bpp'].detach()):.3f}",
        )

        reached_max_steps = max_steps is not None and global_step >= max_steps
        should_checkpoint = (
            checkpoint_interval_steps > 0
            and interval_samples > 0
            and (global_step % checkpoint_interval_steps == 0 or reached_max_steps)
        )
        if should_checkpoint and on_checkpoint is not None:
            interval_count = max(1, interval_samples)
            interval_metrics = {
                key: value / interval_count for key, value in interval_totals.items()
            }
            interval_metrics["skipped_batches"] = float(interval_skipped_batches)
            if not on_checkpoint(epoch, global_step, interval_metrics):
                stop_training = True
                break
            model.train()
            interval_totals = {key: 0.0 for key in totals}
            interval_samples = 0
            interval_skipped_batches = 0

        if reached_max_steps:
            stop_training = True
            break

    count = max(1, processed_samples)
    metrics = {key: value / count for key, value in totals.items()}
    metrics["skipped_batches"] = float(skipped_batches)
    return metrics, global_step, stop_training


@torch.no_grad()
def evaluate_loss(
    model: FactorizedPriorNano,
    criterion: RateDistortionLoss,
    loader: DataLoader,
    device: torch.device,
    progress_enabled: bool = True,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "mse": 0.0,
        "bpp": 0.0,
        "rate_loss": 0.0,
        "ssim": 0.0,
        "detail_loss": 0.0,
        "l1_loss": 0.0,
        "lpips_loss": 0.0,
    }
    processed_samples = 0
    skipped_batches = 0

    progress = tqdm(
        loader,
        desc="val",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        disable=not progress_enabled,
    )
    for batch in progress:
        batch = batch.to(device, non_blocking=True)
        output = model(batch)
        losses = criterion(output, batch)
        if not tensor_is_finite(losses["loss"]):
            skipped_batches += 1
            progress.set_postfix(skipped=skipped_batches)
            continue
        batch_size = batch.size(0)
        processed_samples += batch_size
        for key in totals:
            totals[key] += float(losses[key].detach()) * batch_size
        progress.set_postfix(
            loss=f"{float(losses['loss'].detach()):.4f}",
            bpp=f"{float(losses['bpp'].detach()):.3f}",
        )

    count = max(1, processed_samples)
    metrics = {f"val_{key}": value / count for key, value in totals.items()}
    metrics["val_skipped_batches"] = float(skipped_batches)
    return metrics


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
    parser.add_argument(
        "--rate-weight",
        type=float,
        default=None,
        help="Weight of the rate term. Lower values trade more bits for quality.",
    )
    parser.add_argument(
        "--target-bpp",
        type=float,
        default=None,
        help="If set, only penalize estimated bpp above this budget.",
    )
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
        "--lpips-weight",
        type=float,
        default=None,
        help="Optional LPIPS perceptual loss weight. Start around 0.03 for detail fine-tuning.",
    )
    parser.add_argument(
        "--lpips-net",
        choices=("alex", "vgg", "squeeze"),
        default=None,
        help="Backbone used by LPIPS when --lpips-weight is greater than zero.",
    )
    parser.add_argument(
        "--quant-step",
        type=float,
        default=None,
        help="Override latent quantization step after model initialization/checkpoint load.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after this many optimizer updates. This is useful for large datasets where epochs are too coarse.",
    )
    parser.add_argument(
        "--checkpoint-interval-steps",
        type=int,
        default=100,
        help="Save step checkpoints every N optimizer updates. Default saves e1.pt at step 100.",
    )
    parser.add_argument(
        "--eval-interval-steps",
        type=int,
        default=100,
        help=(
            "Run validation every N optimizer updates. Default matches the "
            "100-step checkpoint cadence. Use 0 to disable step validation."
        ),
    )
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
    parser.add_argument(
        "--log-style",
        choices=("compact", "full"),
        default="compact",
        help="compact prints grouped epoch summaries; full prints every metric key.",
    )
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def apply_quality_profile(args: argparse.Namespace) -> None:
    profile = TRAIN_PROFILES[args.quality_profile]
    for key, value in profile.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    if args.init_checkpoint is not None and args.resume is not None:
        raise ValueError("--init-checkpoint and --resume are mutually exclusive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.checkpoint_interval_steps <= 0:
        raise ValueError("--checkpoint-interval-steps must be positive")
    if args.eval_interval_steps < 0:
        raise ValueError("--eval-interval-steps must be non-negative")
    if args.rate_weight < 0:
        raise ValueError("--rate-weight must be non-negative")
    if args.target_bpp is not None and args.target_bpp < 0:
        raise ValueError("--target-bpp must be non-negative")
    if args.detail_weight < 0:
        raise ValueError("--detail-weight must be non-negative")
    if args.l1_weight < 0:
        raise ValueError("--l1-weight must be non-negative")
    if args.lpips_weight < 0:
        raise ValueError("--lpips-weight must be non-negative")
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
        rate_weight=args.rate_weight,
        target_bpp=args.target_bpp,
        ssim_weight=args.ssim_weight,
        detail_weight=args.detail_weight,
        l1_weight=args.l1_weight,
        lpips_weight=args.lpips_weight,
        lpips_net=args.lpips_net,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = make_scheduler(optimizer, args)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    global_step = 0
    start_epoch = 0
    if args.init_checkpoint is not None:
        state = load_checkpoint(args.init_checkpoint, model)
        set_quant_step(model, args.quant_step)
        print(f"initialized weights: {args.init_checkpoint} (source epoch {state.epoch})")
    elif args.resume is not None:
        state = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = state.epoch
        global_step = state.global_step or start_epoch * len(train_loader)
        set_quant_step(model, args.quant_step)
        print(f"resumed: {args.resume} at epoch {start_epoch}, step {global_step}")

    print_run_config(
        args,
        train_dataset,
        train_loader,
        val_loader,
        model,
        optimizer,
        amp_enabled,
        global_step,
    )

    if args.max_steps is not None and global_step >= args.max_steps:
        print(f"already reached max_steps={args.max_steps}; nothing to train")
        return

    end_epoch = args.epochs
    if args.max_steps is not None:
        remaining_steps = args.max_steps - global_step
        needed_epochs = start_epoch + math.ceil(remaining_steps / len(train_loader))
        end_epoch = max(end_epoch, needed_epochs)

    last_checkpoint_step = global_step
    best_val_loss = math.inf

    def save_step_checkpoint(
        epoch: int,
        step: int,
        train_metrics: dict[str, float],
    ) -> bool:
        nonlocal best_val_loss, last_checkpoint_step

        metrics = dict(train_metrics)
        is_final_step = args.max_steps is not None and step >= args.max_steps
        should_eval = (
            val_loader is not None
            and args.eval_interval_steps > 0
            and (step % args.eval_interval_steps == 0 or is_final_step)
        )
        if should_eval:
            metrics.update(evaluate_loss(model, criterion, val_loader, device, args.progress))

        if not metrics_are_finite(metrics) or not model_parameters_are_finite(model):
            tqdm.write(
                f"non-finite training state at epoch {epoch:03d}, step {step}; "
                "checkpoint was not saved. Resume from the previous good checkpoint."
            )
            return False

        monitor_name = "val_loss" if "val_loss" in metrics else "loss"
        monitor_value = metrics[monitor_name]
        old_lr = get_current_lr(optimizer)
        if scheduler is not None and (val_loader is None or should_eval):
            scheduler.step(monitor_value)
        new_lr = get_current_lr(optimizer)
        metrics["lr"] = new_lr

        checkpoint_index = max(1, math.ceil(step / args.checkpoint_interval_steps))
        checkpoint_name = f"e{checkpoint_index}.pt"
        checkpoint_path = args.checkpoint_dir / checkpoint_name
        latest_path = args.checkpoint_dir / "latest.pt"
        best_path = args.checkpoint_dir / "best.pt"
        improved = "val_loss" in metrics and metrics["val_loss"] < best_val_loss

        if args.log_style == "full":
            metric_text = ", ".join(f"{key}={value:.6f}" for key, value in metrics.items())
            lr_text = ""
            if new_lr < old_lr:
                lr_text = f" | lr reduced: {old_lr:.2e} -> {new_lr:.2e}"
            best_text = " | best" if improved else ""
            tqdm.write(
                f"{checkpoint_name} epoch {epoch:03d} step {step}: "
                f"{metric_text} | monitor={monitor_name}{lr_text}{best_text}"
            )
        else:
            summary = format_checkpoint_summary(
                checkpoint_name,
                epoch,
                step,
                metrics,
                monitor_name,
                old_lr,
                new_lr,
                include_lpips=(args.lpips_weight > 0),
            )
            if improved:
                summary += "\n  best: updated best.pt"
            tqdm.write(summary)

        save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, step, args, metrics)
        save_checkpoint(latest_path, model, optimizer, scheduler, epoch, step, args, metrics)
        if improved:
            best_val_loss = metrics["val_loss"]
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, step, args, metrics)
        last_checkpoint_step = step
        return True

    for epoch in range(start_epoch + 1, end_epoch + 1):
        train_metrics, global_step, stop_training = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            scaler,
            device,
            args.grad_clip,
            amp_enabled,
            epoch,
            global_step,
            args.max_steps,
            args.checkpoint_interval_steps,
            args.progress,
            save_step_checkpoint,
        )

        if epoch == end_epoch and global_step != last_checkpoint_step:
            if not save_step_checkpoint(epoch, global_step, train_metrics):
                break

        if stop_training:
            print(f"stopped at max_steps={args.max_steps}")
            break


if __name__ == "__main__":
    main()
