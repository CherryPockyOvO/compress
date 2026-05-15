from __future__ import annotations

import argparse
import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
Image.MAX_IMAGE_PIXELS = None
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from compressai_nano import (
    MODEL_CONFIGS,
    MODEL_VARIANT_HYPER_MS_Q,
    MODEL_VARIANT_HYPER_MS_Q_NANO,
    MODEL_VARIANT_HYPER_RESIDUAL_Q,
    MODEL_VARIANT_NANO,
    QATSettings,
    get_model,
    infer_model_variant_from_checkpoint,
    model_config_to_dict,
    normalize_model_variant,
)


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
        "highlight_weight": 0.0,
        "highlight_under_weight": 1.0,
        "highlight_lap_weight": 0.8,
        "texture_lap_weight": 1.0,
        "texture_contrast_weight": 0.4,
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
        "lmbda": 0.05,
        "rate_weight": 0.25,
        "target_bpp": None,
        "ssim_weight": 0.05,
        "detail_weight": 1.8,
        "highlight_weight": 1.5,
        "highlight_under_weight": 1.0,
        "highlight_lap_weight": 0.8,
        "texture_lap_weight": 1.0,
        "texture_contrast_weight": 0.4,
        "l1_weight": 0.10,
        "lpips_weight": 0.003,
        "lpips_net": "alex",
        "quant_step": 0.50,
        "epochs": 80,
        "batch_size": 32,
        "crop_size": 384,
        "lr": 5e-6,
    },
    "hyper_quality_fp": {
        "model_variant": MODEL_VARIANT_HYPER_RESIDUAL_Q,
        "lmbda": 0.06,
        "rate_weight": 0.35,
        "target_bpp": None,
        "ssim_weight": 0.05,
        "detail_weight": 1.2,
        "highlight_weight": 0.8,
        "highlight_under_weight": 1.0,
        "highlight_lap_weight": 0.8,
        "texture_lap_weight": 1.0,
        "texture_contrast_weight": 0.4,
        "l1_weight": 0.08,
        "lpips_weight": 0.002,
        "lpips_net": "alex",
        "quant_step": 0.45,
        "epochs": 100,
        "batch_size": 24,
        "crop_size": 384,
        "lr": 1e-4,
        "enable_latent_fake_quant": False,
        "enable_z_fake_quant": False,
        "enable_scale_fake_quant": False,
        "latent_range_weight": 0.0,
        "z_range_weight": 0.0,
        "symbol_range_weight": 0.0,
        "scale_range_weight": 0.0,
    },
    "hyper_quality_qat8": {
        "model_variant": MODEL_VARIANT_HYPER_RESIDUAL_Q,
        "lmbda": 0.06,
        "rate_weight": 0.35,
        "target_bpp": None,
        "ssim_weight": 0.05,
        "detail_weight": 1.2,
        "highlight_weight": 0.8,
        "highlight_under_weight": 1.0,
        "highlight_lap_weight": 0.8,
        "texture_lap_weight": 1.0,
        "texture_contrast_weight": 0.4,
        "l1_weight": 0.08,
        "lpips_weight": 0.002,
        "lpips_net": "alex",
        "quant_step": 0.45,
        "epochs": 30,
        "batch_size": 24,
        "crop_size": 384,
        "lr": 5e-6,
        "enable_latent_fake_quant": True,
        "latent_fake_quant_bits": 8,
        "latent_fake_quant_clip": 6.0,
        "enable_z_fake_quant": True,
        "z_fake_quant_bits": 8,
        "z_fake_quant_clip": 6.0,
        "enable_scale_fake_quant": True,
        "scale_fake_quant_bits": 8,
        "scale_fake_quant_clip": 8.0,
        "latent_range_weight": 0.01,
        "z_range_weight": 0.01,
        "symbol_range_weight": 0.001,
        "scale_range_weight": 0.001,
    },
    "hyper_ms_mini_fp": {
        "model_variant": MODEL_VARIANT_HYPER_MS_Q,
        "lmbda": OFFICIAL_MSE_LAMBDA_Q7,
        "rate_weight": 1.0,
        "target_bpp": None,
        "ssim_weight": 0.03,
        "detail_weight": 0.8,
        "highlight_weight": 0.6,
        "highlight_under_weight": 1.0,
        "highlight_lap_weight": 0.8,
        "texture_lap_weight": 1.0,
        "texture_contrast_weight": 0.4,
        "l1_weight": 0.04,
        "lpips_weight": 0.0,
        "lpips_net": "alex",
        "quant_step": 0.35,
        "epochs": 120,
        "batch_size": 12,
        "crop_size": 384,
        "lr": 1e-4,
        "enable_latent_fake_quant": False,
        "enable_z_fake_quant": False,
        "enable_scale_fake_quant": False,
        "latent_range_weight": 0.0,
        "z_range_weight": 0.0,
        "symbol_range_weight": 0.0,
        "scale_range_weight": 0.0,
    },
    "hyper_ms_mini_hq": {
        "model_variant": MODEL_VARIANT_HYPER_MS_Q,
        "lmbda": 0.16,
        "rate_weight": 0.08,
        "target_bpp": 2.80,
        "ssim_weight": 0.03,
        "detail_weight": 1.3,
        "highlight_weight": 1.4,
        "highlight_under_weight": 1.3,
        "highlight_lap_weight": 1.2,
        "texture_lap_weight": 1.5,
        "texture_contrast_weight": 0.7,
        "l1_weight": 0.08,
        "lpips_weight": 0.0003,
        "lpips_net": "alex",
        "quant_step": 0.30,
        "epochs": 30,
        "batch_size": 12,
        "crop_size": 384,
        "lr": 5e-6,
        "enable_latent_fake_quant": False,
        "enable_z_fake_quant": False,
        "enable_scale_fake_quant": False,
        "latent_range_weight": 0.002,
        "z_range_weight": 0.0,
        "symbol_range_weight": 0.0002,
        "scale_range_weight": 0.0,
    },
    "hyper_ms_mini_qat8": {
        "model_variant": MODEL_VARIANT_HYPER_MS_Q,
        "lmbda": 0.16,
        "rate_weight": 0.05,
        "target_bpp": 2.80,
        "ssim_weight": 0.03,
        "detail_weight": 1.3,
        "highlight_weight": 1.4,
        "highlight_under_weight": 1.3,
        "highlight_lap_weight": 1.2,
        "texture_lap_weight": 1.5,
        "texture_contrast_weight": 0.7,
        "l1_weight": 0.08,
        "lpips_weight": 0.0003,
        "lpips_net": "alex",
        "quant_step": 0.30,
        "epochs": 30,
        "batch_size": 12,
        "crop_size": 384,
        "lr": 3e-6,
        "enable_latent_fake_quant": True,
        "latent_fake_quant_bits": 8,
        "latent_fake_quant_clip": 6.0,
        "enable_z_fake_quant": False,
        "z_fake_quant_bits": 8,
        "z_fake_quant_clip": 6.0,
        "enable_scale_fake_quant": False,
        "scale_fake_quant_bits": 8,
        "scale_fake_quant_clip": 8.0,
        "latent_range_weight": 0.005,
        "z_range_weight": 0.0,
        "symbol_range_weight": 0.0005,
        "scale_range_weight": 0.0,
    },
    "hyper_ms_nano_fp": {
        "model_variant": MODEL_VARIANT_HYPER_MS_Q_NANO,
        "lmbda": OFFICIAL_MSE_LAMBDA_Q7,
        "rate_weight": 1.0,
        "target_bpp": None,
        "ssim_weight": 0.03,
        "detail_weight": 0.8,
        "highlight_weight": 0.6,
        "highlight_under_weight": 1.0,
        "highlight_lap_weight": 0.8,
        "texture_lap_weight": 1.0,
        "texture_contrast_weight": 0.4,
        "l1_weight": 0.04,
        "lpips_weight": 0.0,
        "lpips_net": "alex",
        "quant_step": 0.38,
        "epochs": 120,
        "batch_size": 18,
        "crop_size": 384,
        "lr": 1e-4,
        "enable_latent_fake_quant": False,
        "enable_z_fake_quant": False,
        "enable_scale_fake_quant": False,
        "latent_range_weight": 0.0,
        "z_range_weight": 0.0,
        "symbol_range_weight": 0.0,
        "scale_range_weight": 0.0,
    },
    "hyper_ms_nano_qat8": {
        "model_variant": MODEL_VARIANT_HYPER_MS_Q_NANO,
        "lmbda": 0.14,
        "rate_weight": 0.06,
        "target_bpp": 2.40,
        "ssim_weight": 0.03,
        "detail_weight": 1.2,
        "highlight_weight": 1.2,
        "highlight_under_weight": 1.2,
        "highlight_lap_weight": 1.0,
        "texture_lap_weight": 1.3,
        "texture_contrast_weight": 0.6,
        "l1_weight": 0.07,
        "lpips_weight": 0.0003,
        "lpips_net": "alex",
        "quant_step": 0.33,
        "epochs": 30,
        "batch_size": 18,
        "crop_size": 384,
        "lr": 3e-6,
        "enable_latent_fake_quant": True,
        "latent_fake_quant_bits": 8,
        "latent_fake_quant_clip": 6.0,
        "enable_z_fake_quant": False,
        "z_fake_quant_bits": 8,
        "z_fake_quant_clip": 6.0,
        "enable_scale_fake_quant": False,
        "scale_fake_quant_bits": 8,
        "scale_fake_quant_clip": 8.0,
        "latent_range_weight": 0.005,
        "z_range_weight": 0.0,
        "symbol_range_weight": 0.0005,
        "scale_range_weight": 0.0,
    },
}


@dataclass(frozen=True)
class CheckpointState:
    epoch: int
    global_step: int
    model_variant: str = MODEL_VARIANT_NANO


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass
class RunningBppStats:
    minimum: float = math.inf
    maximum: float = -math.inf
    total: float = 0.0
    total_sq: float = 0.0
    count: int = 0

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().float().reshape(-1)
        if values.numel() == 0:
            return
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        self.total += float(values.sum())
        self.total_sq += float((values * values).sum())
        self.count += int(values.numel())

    def add_to(self, metrics: dict[str, float], prefix: str = "") -> None:
        key = lambda name: f"{prefix}{name}" if prefix else name
        if self.count == 0:
            metrics[key("bpp_min")] = 0.0
            metrics[key("bpp_max")] = 0.0
            metrics[key("bpp_std")] = 0.0
            return

        mean = self.total / self.count
        variance = max(0.0, self.total_sq / self.count - mean * mean)
        metrics[key("bpp_min")] = self.minimum
        metrics[key("bpp_max")] = self.maximum
        metrics[key("bpp_std")] = math.sqrt(variance)


BASE_METRIC_KEYS = [
    "loss",
    "mse",
    "bpp",
    "bpp_y",
    "bpp_z",
    "bpp_total",
    "rate_loss",
    "ssim",
    "detail_loss",
    "highlight_loss",
    "highlight_under_loss",
    "peak_under",
    "highlight_lap",
    "highlight_contrast",
    "l1_loss",
    "lpips_loss",
    "latent_range_loss",
    "z_range_loss",
    "symbol_range_loss",
    "scale_range_loss",
    "latent_y_min",
    "latent_y_max",
    "latent_y_mean",
    "latent_y_std",
    "latent_y_p01",
    "latent_y_p99",
    "latent_y_clip_ratio",
    "latent_z_min",
    "latent_z_max",
    "latent_z_mean",
    "latent_z_std",
    "latent_z_p01",
    "latent_z_p99",
    "latent_z_clip_ratio",
    "scale_min",
    "scale_max",
    "scale_mean",
    "scale_std",
    "scale_p01",
    "scale_p99",
    "mean_y_min",
    "mean_y_max",
    "mean_y_mean",
    "mean_y_std",
    "mean_y_p01",
    "mean_y_p99",
    "mean_y_clip_ratio",
    "symbol_y_min",
    "symbol_y_max",
    "symbol_y_std",
    "symbol_y_p99_abs",
    "symbol_z_min",
    "symbol_z_max",
    "symbol_z_std",
    "symbol_z_p99_abs",
    "fake_quant_y_error",
    "fake_quant_z_error",
    "fake_quant_scale_error",
]


def make_metric_totals() -> dict[str, float]:
    return {key: 0.0 for key in BASE_METRIC_KEYS}


def read_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def init_distributed() -> DistributedContext:
    world_size = read_env_int("WORLD_SIZE", 1)
    rank = read_env_int("RANK", 0)
    local_rank = read_env_int("LOCAL_RANK", 0)
    if world_size <= 1:
        return DistributedContext(enabled=False)

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA. Launch with one process per GPU.")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA devices are visible"
        )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return DistributedContext(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def cleanup_distributed(distributed: DistributedContext) -> None:
    if distributed.enabled and dist.is_initialized():
        dist.destroy_process_group()


def distributed_device(distributed: DistributedContext) -> torch.device:
    if distributed.enabled:
        return torch.device("cuda", distributed.local_rank)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def all_reduce_bool(value: bool, device: torch.device, distributed: DistributedContext) -> bool:
    if not distributed.enabled:
        return value
    flag = torch.tensor(1 if value else 0, dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def add_reduced_bpp_stats(
    metrics: dict[str, float],
    stats: RunningBppStats,
    device: torch.device,
    distributed: DistributedContext,
    prefix: str = "",
) -> None:
    if not distributed.enabled:
        stats.add_to(metrics, prefix=prefix)
        return

    key = lambda name: f"{prefix}{name}" if prefix else name
    count = int(stats.count)

    minimum = torch.tensor(
        stats.minimum if count > 0 else math.inf,
        dtype=torch.float64,
        device=device,
    )
    maximum = torch.tensor(
        stats.maximum if count > 0 else -math.inf,
        dtype=torch.float64,
        device=device,
    )
    sums = torch.tensor(
        [stats.total, stats.total_sq, float(count)],
        dtype=torch.float64,
        device=device,
    )

    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    dist.all_reduce(sums, op=dist.ReduceOp.SUM)

    global_count = int(sums[2].item())
    if global_count == 0:
        metrics[key("bpp_min")] = 0.0
        metrics[key("bpp_max")] = 0.0
        metrics[key("bpp_std")] = 0.0
        return

    mean = sums[0].item() / global_count
    variance = max(0.0, sums[1].item() / global_count - mean * mean)
    metrics[key("bpp_min")] = float(minimum.item())
    metrics[key("bpp_max")] = float(maximum.item())
    metrics[key("bpp_std")] = math.sqrt(variance)


def average_reduced_metrics(
    totals: dict[str, float],
    processed_samples: int,
    skipped_batches: int,
    bpp_stats: RunningBppStats,
    device: torch.device,
    distributed: DistributedContext,
    prefix: str = "",
) -> dict[str, float]:
    keys = list(totals.keys())
    values = [totals[key] for key in keys]
    values.extend([float(processed_samples), float(skipped_batches)])
    reduced = torch.tensor(values, dtype=torch.float64, device=device)
    if distributed.enabled:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)

    sample_count = max(1.0, float(reduced[len(keys)].item()))
    key_name = lambda name: f"{prefix}{name}" if prefix else name
    metrics = {
        key_name(key): float(reduced[index].item() / sample_count)
        for index, key in enumerate(keys)
    }
    metrics[key_name("skipped_batches")] = float(reduced[len(keys) + 1].item())
    add_reduced_bpp_stats(metrics, bpp_stats, device, distributed, prefix=prefix)
    return metrics


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


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


def resize_to_min_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    if width >= size and height >= size:
        return image

    scale = max(size / width, size / height)
    width = max(size, math.ceil(width * scale))
    height = max(size, math.ceil(height * scale))
    resample = getattr(Image, "Resampling", Image).BICUBIC
    return image.resize((width, height), resample=resample)


class RandomCrop:
    def __init__(self, size: int) -> None:
        self.size = int(size)

    def __call__(self, image: Image.Image) -> Image.Image:
        image = resize_to_min_crop(image, self.size)
        width, height = image.size
        left = random.randint(0, width - self.size)
        top = random.randint(0, height - self.size)
        return image.crop((left, top, left + self.size, top + self.size))


class DetailAwareRandomCrop:
    def __init__(self, size: int, p_detail: float = 0.3) -> None:
        self.size = int(size)
        self.p_detail = float(p_detail)
        self.fallback = RandomCrop(size)

    def __call__(self, image: Image.Image) -> Image.Image:
        image = resize_to_min_crop(image, self.size)
        if random.random() >= self.p_detail:
            return self.fallback(image)

        try:
            crop = self.detail_crop(image)
        except (RuntimeError, ValueError):
            crop = None
        if crop is None:
            return self.fallback(image)
        return crop

    def detail_crop(self, image: Image.Image) -> Image.Image | None:
        width, height = image.size
        score = self.score_map(image)
        if score is None or score.numel() == 0:
            return None

        flat_score = score.flatten()
        if float(flat_score.max()) <= 1e-4:
            return None

        index = int(torch.multinomial(flat_score.clamp_min(0.0), num_samples=1).item())
        score_height, score_width = score.shape
        center_y = (index // score_width + 0.5) * height / score_height
        center_x = (index % score_width + 0.5) * width / score_width

        left = round(center_x - self.size / 2)
        top = round(center_y - self.size / 2)
        left = min(max(0, left), width - self.size)
        top = min(max(0, top), height - self.size)
        return image.crop((left, top, left + self.size, top + self.size))

    @staticmethod
    def score_map(image: Image.Image) -> torch.Tensor | None:
        width, height = image.size
        max_side = max(width, height)
        if max_side > 256:
            scale = 256 / max_side
            score_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            resample = getattr(Image, "Resampling", Image).BILINEAR
            image = image.resize(score_size, resample=resample)
            width, height = image.size

        gray_image = ImageOps.grayscale(image)
        gray = torch.frombuffer(bytearray(gray_image.tobytes()), dtype=torch.uint8)
        gray = gray.to(torch.float32).view(height, width) / 255.0
        if height < 3 or width < 3:
            return None

        kernel = gray.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        laplacian = F.conv2d(
            gray.view(1, 1, height, width),
            kernel.view(1, 1, 3, 3),
            padding=1,
        ).abs()
        score = laplacian.squeeze(0).squeeze(0) + 2.0 * torch.relu(gray - 0.7)
        return score


class CenterCrop:
    def __init__(self, size: int) -> None:
        self.size = int(size)

    def __call__(self, image: Image.Image) -> Image.Image:
        image = resize_to_min_crop(image, self.size)
        width, height = image.size
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


def make_train_transform(crop_size: int, quality_profile: str = "balanced") -> Compose:
    crop: Callable[[Image.Image], Image.Image]
    if quality_profile == "detail" or quality_profile.startswith(("hyper_quality", "hyper_ms")):
        crop = DetailAwareRandomCrop(crop_size, p_detail=0.3)
    else:
        crop = RandomCrop(crop_size)

    return Compose(
        [
            crop,
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


def compute_bpp_per_image(
    likelihoods: dict[str, torch.Tensor],
    num_pixels_per_image: int,
) -> torch.Tensor:
    first_likelihood = next(iter(likelihoods.values()))
    bits = torch.zeros(first_likelihood.size(0), device=first_likelihood.device)
    for likelihood in likelihoods.values():
        bits = bits + torch.sum(-torch.log2(likelihood.clamp_min(1e-9)).flatten(1), dim=1)
    return bits / float(num_pixels_per_image)


def compute_likelihood_bpp(likelihood: torch.Tensor | None, num_pixels_per_image: int) -> torch.Tensor:
    if likelihood is None:
        return torch.zeros(())
    bits = torch.sum(-torch.log2(likelihood.clamp_min(1e-9)).flatten(1), dim=1)
    return (bits / float(num_pixels_per_image)).mean()


def tensor_stats(
    tensor: torch.Tensor | None,
    prefix: str,
    clip: float | None = None,
) -> dict[str, torch.Tensor]:
    if tensor is None:
        zero = torch.zeros(())
        return {
            f"{prefix}_min": zero,
            f"{prefix}_max": zero,
            f"{prefix}_mean": zero,
            f"{prefix}_std": zero,
            f"{prefix}_p01": zero,
            f"{prefix}_p99": zero,
            f"{prefix}_clip_ratio": zero,
        }

    values = tensor.detach().float().reshape(-1)
    zero = values.new_zeros(())
    if values.numel() == 0:
        return {
            f"{prefix}_min": zero,
            f"{prefix}_max": zero,
            f"{prefix}_mean": zero,
            f"{prefix}_std": zero,
            f"{prefix}_p01": zero,
            f"{prefix}_p99": zero,
            f"{prefix}_clip_ratio": zero,
        }
    clip_ratio = zero
    if clip is not None and clip > 0:
        clip_ratio = (values.abs() >= float(clip) * 0.999).to(values.dtype).mean()
    return {
        f"{prefix}_min": values.min(),
        f"{prefix}_max": values.max(),
        f"{prefix}_mean": values.mean(),
        f"{prefix}_std": values.std(unbiased=False),
        f"{prefix}_p01": torch.quantile(values, 0.01),
        f"{prefix}_p99": torch.quantile(values, 0.99),
        f"{prefix}_clip_ratio": clip_ratio,
    }


def scale_stats(tensor: torch.Tensor | None) -> dict[str, torch.Tensor]:
    if tensor is None:
        zero = torch.zeros(())
        return {
            "scale_min": zero,
            "scale_max": zero,
            "scale_mean": zero,
            "scale_std": zero,
            "scale_p01": zero,
            "scale_p99": zero,
        }
    values = tensor.detach().float().reshape(-1)
    zero = values.new_zeros(())
    if values.numel() == 0:
        return {
            "scale_min": zero,
            "scale_max": zero,
            "scale_mean": zero,
            "scale_std": zero,
            "scale_p01": zero,
            "scale_p99": zero,
        }
    return {
        "scale_min": values.min(),
        "scale_max": values.max(),
        "scale_mean": values.mean(),
        "scale_std": values.std(unbiased=False),
        "scale_p01": torch.quantile(values, 0.01),
        "scale_p99": torch.quantile(values, 0.99),
    }


def symbol_stats(symbols: torch.Tensor | None, prefix: str) -> dict[str, torch.Tensor]:
    if symbols is None:
        zero = torch.zeros(())
        return {
            f"{prefix}_min": zero,
            f"{prefix}_max": zero,
            f"{prefix}_std": zero,
            f"{prefix}_p99_abs": zero,
        }
    values = symbols.detach().float().reshape(-1)
    zero = values.new_zeros(())
    if values.numel() == 0:
        return {
            f"{prefix}_min": zero,
            f"{prefix}_max": zero,
            f"{prefix}_std": zero,
            f"{prefix}_p99_abs": zero,
        }
    return {
        f"{prefix}_min": values.min(),
        f"{prefix}_max": values.max(),
        f"{prefix}_std": values.std(unbiased=False),
        f"{prefix}_p99_abs": torch.quantile(values.abs(), 0.99),
    }


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


def rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def rgb_to_y(x: torch.Tensor) -> torch.Tensor:
    return rgb_to_luma(x)


def laplacian_map(x: torch.Tensor) -> torch.Tensor:
    channels = x.size(1)
    kernel = x.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    kernel = kernel.view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    return F.conv2d(x, kernel, padding=1, groups=channels)


def make_highlight_texture_weight(target: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        target = target.detach()
        y = rgb_to_luma(target)
        bright = torch.sigmoid((y - 0.72) / 0.08)
        lap = torch.abs(laplacian_map(y))
        lap = lap / (lap.mean(dim=(2, 3), keepdim=True) + 1e-6)
        weight = 1.0 + 2.5 * bright + 1.5 * lap.clamp(0.0, 3.0)
        return weight.clamp(1.0, 6.0)


def make_highlight_focus_weight(
    target: torch.Tensor,
    threshold: float = 0.72,
    softness: float = 0.08,
    peak_threshold: float = 0.88,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        luma = rgb_to_luma(target.detach())
        bright = torch.sigmoid((luma - threshold) / softness)
        peak = torch.sigmoid((luma - peak_threshold) / max(softness * 0.5, 1e-6))
        lap = torch.abs(laplacian_map(luma))
        lap = lap / (lap.mean(dim=(2, 3), keepdim=True) + 1e-6)
        edge = lap.clamp(0.0, 4.0)

        focus = bright * (1.0 + 1.5 * edge) + 1.5 * peak
        return focus.clamp(0.0, 8.0), peak.clamp(0.0, 1.0)


def highlight_aware_terms(
    x_hat: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    focus_weight, peak_weight = make_highlight_focus_weight(target)

    rgb_l1 = (focus_weight * torch.abs(x_hat - target)).mean()

    y_hat = rgb_to_luma(x_hat)
    y_tgt = rgb_to_luma(target)
    luma_l1 = (focus_weight * torch.abs(y_hat - y_tgt)).mean()

    lap_hat = laplacian_map(y_hat)
    lap_tgt = laplacian_map(y_tgt)
    lap_loss = (focus_weight * torch.abs(lap_hat - lap_tgt)).mean()

    under_loss = (peak_weight * torch.relu(y_tgt - y_hat)).mean()
    return rgb_l1, luma_l1, lap_loss, under_loss


def combine_highlight_aware_terms(
    terms: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    under_weight: float = 1.0,
    lap_weight: float = 0.8,
    peak_under_weight: float | None = None,
) -> torch.Tensor:
    if peak_under_weight is not None:
        under_weight = peak_under_weight
    rgb_l1, luma_l1, lap_loss, under_loss = terms
    return rgb_l1 + 0.75 * luma_l1 + float(lap_weight) * lap_loss + float(under_weight) * under_loss


def highlight_aware_loss(
    x_hat: torch.Tensor,
    target: torch.Tensor,
    under_weight: float = 1.0,
    lap_weight: float = 0.8,
    peak_under_weight: float | None = None,
) -> torch.Tensor:
    if peak_under_weight is not None:
        under_weight = peak_under_weight
    return combine_highlight_aware_terms(
        highlight_aware_terms(x_hat, target),
        under_weight=under_weight,
        lap_weight=lap_weight,
    )


def highlight_texture_loss(
    x_hat: torch.Tensor,
    target: torch.Tensor,
    texture_lap_weight: float = 1.0,
    texture_contrast_weight: float = 0.4,
) -> torch.Tensor:
    weight = make_highlight_texture_weight(target)
    l1 = (weight * torch.abs(x_hat - target)).mean()

    lap_x = laplacian_map(x_hat)
    lap_t = laplacian_map(target)
    lap_loss = (weight * torch.abs(lap_x - lap_t)).mean()

    y_hat = rgb_to_luma(x_hat)
    y_tgt = rgb_to_luma(target)
    mu_hat = F.avg_pool2d(y_hat, kernel_size=7, stride=1, padding=3)
    mu_tgt = F.avg_pool2d(y_tgt, kernel_size=7, stride=1, padding=3)
    std_hat = torch.sqrt(F.avg_pool2d((y_hat - mu_hat) ** 2, 7, 1, 3) + 1e-6)
    std_tgt = torch.sqrt(F.avg_pool2d((y_tgt - mu_tgt) ** 2, 7, 1, 3) + 1e-6)
    contrast_loss = (weight * torch.abs(std_hat - std_tgt)).mean()

    return l1 + float(texture_lap_weight) * lap_loss + float(texture_contrast_weight) * contrast_loss


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


def detail_reconstruction_loss(
    x_hat: torch.Tensor,
    target: torch.Tensor,
    texture_lap_weight: float = 1.0,
    texture_contrast_weight: float = 0.4,
) -> torch.Tensor:
    return (
        highlight_texture_loss(
            x_hat,
            target,
            texture_lap_weight=texture_lap_weight,
            texture_contrast_weight=texture_contrast_weight,
        )
        + 0.2 * gradient_detail_loss(x_hat, target)
    )


def local_luma_std(y: torch.Tensor) -> torch.Tensor:
    mu = F.avg_pool2d(y, kernel_size=7, stride=1, padding=3)
    return torch.sqrt(F.avg_pool2d((y - mu) ** 2, 7, 1, 3) + 1e-6)


def highlight_quality_metrics(
    x_hat: torch.Tensor,
    target: torch.Tensor,
    peak_threshold: float = 0.88,
    highlight_threshold: float = 0.72,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        y_hat = rgb_to_luma(x_hat.detach().clamp(0.0, 1.0))
        y_tgt = rgb_to_luma(target.detach().clamp(0.0, 1.0))

        peak_mask = (y_tgt > peak_threshold).to(dtype=y_tgt.dtype)
        peak_denom = peak_mask.sum() + 1e-6
        peak_under = (peak_mask * torch.relu(y_tgt - y_hat)).sum() / peak_denom

        highlight_mask = (y_tgt > highlight_threshold).to(dtype=y_tgt.dtype)
        highlight_denom = highlight_mask.sum() + 1e-6
        lap_metric = (
            highlight_mask * torch.abs(laplacian_map(y_hat) - laplacian_map(y_tgt))
        ).sum() / highlight_denom

        contrast_metric = (
            highlight_mask * torch.abs(local_luma_std(y_hat) - local_luma_std(y_tgt))
        ).sum() / highlight_denom

    return peak_under, lap_metric, contrast_metric


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
        highlight_weight: float = 0.0,
        highlight_under_weight: float = 1.0,
        highlight_lap_weight: float = 0.8,
        texture_lap_weight: float = 1.0,
        texture_contrast_weight: float = 0.4,
        l1_weight: float = 0.0,
        lpips_weight: float = 0.0,
        lpips_net: str = "alex",
        latent_range_weight: float = 0.0,
        z_range_weight: float = 0.0,
        symbol_range_weight: float = 0.0,
        scale_range_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.lmbda = float(lmbda)
        self.rate_weight = float(rate_weight)
        self.target_bpp = None if target_bpp is None else float(target_bpp)
        self.ssim_weight = float(ssim_weight)
        self.detail_weight = float(detail_weight)
        self.highlight_weight = float(highlight_weight)
        self.highlight_under_weight = float(highlight_under_weight)
        self.highlight_lap_weight = float(highlight_lap_weight)
        self.texture_lap_weight = float(texture_lap_weight)
        self.texture_contrast_weight = float(texture_contrast_weight)
        self.l1_weight = float(l1_weight)
        self.lpips_weight = float(lpips_weight)
        self.lpips_net = lpips_net
        self.latent_range_weight = float(latent_range_weight)
        self.z_range_weight = float(z_range_weight)
        self.symbol_range_weight = float(symbol_range_weight)
        self.scale_range_weight = float(scale_range_weight)
        self.lpips_model = make_lpips_model(lpips_net) if self.lpips_weight > 0 else None

    def forward(self, output: dict[str, Any], target: torch.Tensor) -> dict[str, torch.Tensor]:
        x_hat = output["x_hat"]
        mse = F.mse_loss(x_hat, target)
        num_pixels_per_image = target.size(2) * target.size(3)
        bpp_per_image = compute_bpp_per_image(output["likelihoods"], num_pixels_per_image)
        bpp = bpp_per_image.mean()
        bpp_y = compute_likelihood_bpp(output["likelihoods"].get("y"), num_pixels_per_image).to(
            device=mse.device,
            dtype=mse.dtype,
        )
        bpp_z = compute_likelihood_bpp(output["likelihoods"].get("z"), num_pixels_per_image).to(
            device=mse.device,
            dtype=mse.dtype,
        )
        bpp_total = bpp_y + bpp_z
        if self.target_bpp is None:
            rate_loss = bpp
        else:
            rate_loss = torch.relu(bpp - bpp.new_tensor(self.target_bpp))
        distortion = self.lmbda * (255.0**2) * mse
        ssim = ssim_index(x_hat, target)
        ssim_loss = 1.0 - ssim
        l1_loss = F.l1_loss(x_hat, target)
        if self.detail_weight > 0:
            detail_loss = detail_reconstruction_loss(
                x_hat,
                target,
                texture_lap_weight=self.texture_lap_weight,
                texture_contrast_weight=self.texture_contrast_weight,
            )
        else:
            detail_loss = mse.new_zeros(())
        if self.highlight_weight > 0:
            highlight_terms = highlight_aware_terms(x_hat, target)
            highlight_loss = combine_highlight_aware_terms(
                highlight_terms,
                under_weight=self.highlight_under_weight,
                lap_weight=self.highlight_lap_weight,
            )
            highlight_under_loss = highlight_terms[3]
        else:
            highlight_loss = mse.new_zeros(())
            highlight_under_loss = mse.new_zeros(())
        peak_under, highlight_lap, highlight_contrast = highlight_quality_metrics(x_hat, target)
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

        y = output.get("y")
        z = output.get("z")
        scales_y = output.get("scales_y")
        means_y = output.get("means_y")
        symbols = output.get("symbols", {})
        symbol_y = symbols.get("y") if isinstance(symbols, dict) else None
        symbol_z = symbols.get("z") if isinstance(symbols, dict) else None

        y_clip = output.get("latent_clip", None)
        z_clip = output.get("z_clip", None)
        if y_clip is None and y is not None:
            y_clip = 6.0 if output.get("model_variant") == MODEL_VARIANT_HYPER_RESIDUAL_Q else None
        if z_clip is None and z is not None:
            z_clip = 6.0 if output.get("model_variant") == MODEL_VARIANT_HYPER_RESIDUAL_Q else None

        if self.latent_range_weight > 0 and y is not None and y_clip is not None and y_clip > 0:
            latent_range_loss = torch.relu(y.abs() - 0.9 * float(y_clip)).mean()
        else:
            latent_range_loss = mse.new_zeros(())
        if self.z_range_weight > 0 and z is not None and z_clip is not None and z_clip > 0:
            z_range_loss = torch.relu(z.abs() - 0.9 * float(z_clip)).mean()
        else:
            z_range_loss = mse.new_zeros(())
        quant_step = output.get("quant_step")
        y_for_symbol = output.get("y_for_hyper", y)
        if self.symbol_range_weight > 0 and y_for_symbol is not None and quant_step is not None:
            step = torch.as_tensor(quant_step, device=y_for_symbol.device, dtype=y_for_symbol.dtype)
            symbol_proxy = y_for_symbol / step.clamp_min(1e-9)
            symbol_range_loss = torch.relu(symbol_proxy.abs() - 127.0).mean()
        else:
            symbol_range_loss = mse.new_zeros(())
        if self.scale_range_weight > 0 and scales_y is not None:
            scale_min_value = float(output.get("scale_min_value", 1e-3))
            scale_max_value = float(output.get("scale_max_value", 20.0))
            scale_range_loss = (
                torch.relu(scales_y - scale_max_value).mean()
                + torch.relu(scale_min_value - scales_y).mean()
            )
        else:
            scale_range_loss = mse.new_zeros(())

        loss = (
            distortion
            + self.rate_weight * rate_loss
            + self.ssim_weight * ssim_loss
            + self.detail_weight * detail_loss
            + self.highlight_weight * highlight_loss
            + self.l1_weight * l1_loss
            + self.lpips_weight * lpips_loss
            + self.latent_range_weight * latent_range_loss
            + self.z_range_weight * z_range_loss
            + self.symbol_range_weight * symbol_range_loss
            + self.scale_range_weight * scale_range_loss
        )
        result = {
            "loss": loss,
            "mse": mse,
            "bpp": bpp,
            "bpp_y": bpp_y,
            "bpp_z": bpp_z,
            "bpp_total": bpp_total,
            "bpp_per_image": bpp_per_image,
            "rate_loss": rate_loss,
            "ssim": ssim,
            "ssim_loss": ssim_loss,
            "detail_loss": detail_loss,
            "highlight_loss": highlight_loss,
            "highlight_under_loss": highlight_under_loss,
            "peak_under": peak_under,
            "highlight_lap": highlight_lap,
            "highlight_contrast": highlight_contrast,
            "l1_loss": l1_loss,
            "lpips_loss": lpips_loss,
            "latent_range_loss": latent_range_loss,
            "z_range_loss": z_range_loss,
            "symbol_range_loss": symbol_range_loss,
            "scale_range_loss": scale_range_loss,
        }
        result.update(tensor_stats(y, "latent_y", clip=y_clip))
        result.update(tensor_stats(z, "latent_z", clip=z_clip))
        result.update(scale_stats(scales_y))
        result.update(tensor_stats(means_y, "mean_y"))
        result.update(symbol_stats(symbol_y, "symbol_y"))
        result.update(symbol_stats(symbol_z, "symbol_z"))
        fake_quant_errors = output.get("fake_quant_errors", {})
        if not isinstance(fake_quant_errors, dict):
            fake_quant_errors = {}
        result["fake_quant_y_error"] = fake_quant_errors.get("y", mse.new_zeros(()))
        result["fake_quant_z_error"] = fake_quant_errors.get("z", mse.new_zeros(()))
        result["fake_quant_scale_error"] = fake_quant_errors.get("scale", mse.new_zeros(()))
        return result


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
) -> CheckpointState:
    raw = torch.load(path, map_location="cpu")
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    model_variant = infer_model_variant_from_checkpoint(raw)
    model.load_state_dict(state_dict, strict=False)
    if optimizer is not None and isinstance(raw, dict) and "optimizer" in raw:
        optimizer.load_state_dict(raw["optimizer"])
    if scheduler is not None and isinstance(raw, dict) and "scheduler" in raw:
        scheduler.load_state_dict(raw["scheduler"])
    if isinstance(raw, dict):
        return CheckpointState(
            epoch=int(raw.get("epoch", 0)),
            global_step=int(raw.get("global_step", 0)),
            model_variant=model_variant,
        )
    return CheckpointState(epoch=0, global_step=0, model_variant=model_variant)


@torch.no_grad()
def set_quant_step(model: nn.Module, quant_step: float | None) -> None:
    if quant_step is None:
        return
    if quant_step <= 0:
        raise ValueError(f"quant_step must be positive, got {quant_step}")
    if hasattr(model, "set_quant_step"):
        model.set_quant_step(float(quant_step))
    elif hasattr(model, "entropy_bottleneck"):
        model.entropy_bottleneck.quant_step.fill_(float(quant_step))
    else:
        raise AttributeError("model does not expose set_quant_step or entropy_bottleneck")


def get_model_quant_step(model: nn.Module) -> float:
    model = unwrap_model(model)
    if hasattr(model, "get_quant_step"):
        return float(model.get_quant_step())
    return float(model.entropy_bottleneck.quant_step.detach().cpu())


def checkpoint_model_variant(path: Path | None) -> str | None:
    if path is None:
        return None
    raw = torch.load(path, map_location="cpu")
    return infer_model_variant_from_checkpoint(raw)


def make_qat_settings(args: argparse.Namespace) -> QATSettings:
    return QATSettings(
        enable_latent_fake_quant=bool(args.enable_latent_fake_quant),
        latent_fake_quant_bits=int(args.latent_fake_quant_bits),
        latent_fake_quant_clip=float(args.latent_fake_quant_clip),
        enable_z_fake_quant=bool(args.enable_z_fake_quant),
        z_fake_quant_bits=int(args.z_fake_quant_bits),
        z_fake_quant_clip=float(args.z_fake_quant_clip),
        enable_scale_fake_quant=bool(args.enable_scale_fake_quant),
        scale_fake_quant_bits=int(args.scale_fake_quant_bits),
        scale_fake_quant_clip=float(args.scale_fake_quant_clip),
    )


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
    model = unwrap_model(model)
    model_variant = getattr(model, "model_variant", MODEL_VARIANT_NANO)
    model_config = (
        model.model_config_dict()
        if hasattr(model, "model_config_dict")
        else model_config_to_dict(MODEL_CONFIGS[MODEL_VARIANT_NANO])
    )
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "model_variant": model_variant,
        "model_config": model_config,
        "quality_profile": args.quality_profile,
        "lambda": args.lmbda,
        "rate_weight": args.rate_weight,
        "target_bpp": args.target_bpp,
        "ssim_weight": args.ssim_weight,
        "detail_weight": args.detail_weight,
        "highlight_weight": args.highlight_weight,
        "highlight_under_weight": args.highlight_under_weight,
        "highlight_peak_under_weight": args.highlight_under_weight,
        "highlight_lap_weight": args.highlight_lap_weight,
        "texture_lap_weight": args.texture_lap_weight,
        "texture_contrast_weight": args.texture_contrast_weight,
        "l1_weight": args.l1_weight,
        "lpips_weight": args.lpips_weight,
        "lpips_net": args.lpips_net,
        "quant_step": get_model_quant_step(model),
        "enable_latent_fake_quant": args.enable_latent_fake_quant,
        "latent_fake_quant_bits": args.latent_fake_quant_bits,
        "latent_fake_quant_clip": args.latent_fake_quant_clip,
        "enable_z_fake_quant": args.enable_z_fake_quant,
        "z_fake_quant_bits": args.z_fake_quant_bits,
        "z_fake_quant_clip": args.z_fake_quant_clip,
        "enable_scale_fake_quant": args.enable_scale_fake_quant,
        "scale_fake_quant_bits": args.scale_fake_quant_bits,
        "scale_fake_quant_clip": args.scale_fake_quant_clip,
        "latent_range_weight": args.latent_range_weight,
        "z_range_weight": args.z_range_weight,
        "symbol_range_weight": args.symbol_range_weight,
        "scale_range_weight": args.scale_range_weight,
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
        f"hi={metrics[key('highlight_loss')]:.5f}",
        f"peak_under={metrics[key('peak_under')]:.5f}",
        f"hi_lap={metrics[key('highlight_lap')]:.5f}",
        f"hi_contrast={metrics[key('highlight_contrast')]:.5f}",
        f"l1={metrics[key('l1_loss')]:.5f}",
    ]
    if all(key(name) in metrics for name in ("bpp_min", "bpp_max", "bpp_std")):
        parts.extend(
            [
                f"bpp_min={metrics[key('bpp_min')]:.3f}",
                f"bpp_max={metrics[key('bpp_max')]:.3f}",
                f"bpp_std={metrics[key('bpp_std')]:.3f}",
            ]
        )
    if metrics.get(key("bpp_z"), 0.0) > 0.0:
        parts.extend(
            [
                f"bpp_y={metrics[key('bpp_y')]:.3f}",
                f"bpp_z={metrics[key('bpp_z')]:.3f}",
            ]
        )
    if key("latent_y_p99") in metrics and metrics.get(key("latent_y_p99"), 0.0) != 0.0:
        parts.extend(
            [
                f"y_p99={metrics[key('latent_y_p99')]:.2f}",
                f"sym_y_p99={metrics[key('symbol_y_p99_abs')]:.1f}",
                f"y_clip={100.0 * metrics.get(key('latent_y_clip_ratio'), 0.0):.2f}%",
            ]
        )
    if key("latent_z_p99") in metrics and metrics.get(key("latent_z_p99"), 0.0) != 0.0:
        parts.extend(
            [
                f"z_p99={metrics[key('latent_z_p99')]:.2f}",
                f"sym_z_p99={metrics[key('symbol_z_p99_abs')]:.1f}",
                f"z_clip={100.0 * metrics.get(key('latent_z_clip_ratio'), 0.0):.2f}%",
            ]
        )
    if metrics.get(key("scale_mean"), 0.0) > 0.0:
        parts.append(f"scale_mean={metrics[key('scale_mean')]:.3f}")
    if key("mean_y_std") in metrics and metrics.get(key("mean_y_std"), 0.0) > 0.0:
        parts.append(f"mean_y_std={metrics[key('mean_y_std')]:.3f}")
    if metrics.get(key("fake_quant_y_error"), 0.0) > 0.0:
        parts.append(f"fq_y={metrics[key('fake_quant_y_error')]:.5f}")
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
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    amp_enabled: bool,
    global_step: int,
    distributed: DistributedContext,
) -> None:
    model = unwrap_model(model)
    quant_step = get_model_quant_step(model)
    val_text = "none" if val_loader is None else str(len(val_loader.dataset))
    global_batch = args.batch_size * distributed.world_size
    lpips_text = "off"
    if args.lpips_weight > 0:
        lpips_text = f"{args.lpips_weight:g}/{args.lpips_net}"
    print("run config:")
    print(
        f"  data: train={len(train_dataset)} val={val_text} "
        f"batch_per_gpu={args.batch_size} global_batch={global_batch} "
        f"crop={args.crop_size} workers_per_rank={args.num_workers}"
    )
    if distributed.enabled:
        print(
            f"  distributed: ddp world_size={distributed.world_size} "
            f"rank={distributed.rank} local_rank={distributed.local_rank}"
        )
    print(
        f"  objective: profile={args.quality_profile} model_variant={args.model_variant} "
        f"lambda={args.lmbda:g} "
        f"target_bpp={args.target_bpp} rate={args.rate_weight:g} "
        f"ssim={args.ssim_weight:g} grad={args.detail_weight:g} "
        f"highlight={args.highlight_weight:g} under={args.highlight_under_weight:g} "
        f"hi_lap={args.highlight_lap_weight:g} texture_lap={args.texture_lap_weight:g} "
        f"texture_contrast={args.texture_contrast_weight:g} l1={args.l1_weight:g} "
        f"lpips={lpips_text} quant_step={quant_step:g}"
    )
    if args.model_variant != MODEL_VARIANT_NANO:
        print(
            "  qat: "
            f"latent={args.enable_latent_fake_quant}/{args.latent_fake_quant_bits}b "
            f"z={args.enable_z_fake_quant}/{args.z_fake_quant_bits}b "
            f"scale={args.enable_scale_fake_quant}/{args.scale_fake_quant_bits}b "
            f"range_w=({args.latent_range_weight:g}, {args.z_range_weight:g}, "
            f"{args.symbol_range_weight:g}, {args.scale_range_weight:g})"
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
    model: nn.Module,
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
    distributed: DistributedContext,
) -> tuple[dict[str, float], int, bool]:
    model.train()
    totals = make_metric_totals()
    interval_totals = {key: 0.0 for key in totals}
    bpp_stats = RunningBppStats()
    interval_bpp_stats = RunningBppStats()
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
        disable=not (progress_enabled and distributed.is_main),
    )

    for batch in progress:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with make_autocast(amp_enabled):
            output = model(batch)
            losses = criterion(output, batch)

        loss_is_finite = all_reduce_bool(
            tensor_is_finite(losses["loss"]),
            device,
            distributed,
        )
        if not loss_is_finite:
            skipped_batches += 1
            interval_skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(step=global_step, skipped=skipped_batches)
            continue

        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        grad_is_finite = all_reduce_bool(
            tensor_is_finite(grad_norm),
            device,
            distributed,
        )
        if not grad_is_finite:
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
        bpp_values = losses.get("bpp_per_image", losses["bpp"]).detach()
        bpp_stats.update(bpp_values)
        interval_bpp_stats.update(bpp_values)

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
            interval_metrics = average_reduced_metrics(
                interval_totals,
                interval_samples,
                interval_skipped_batches,
                interval_bpp_stats,
                device,
                distributed,
            )
            if not on_checkpoint(epoch, global_step, interval_metrics):
                stop_training = True
                break
            model.train()
            interval_totals = {key: 0.0 for key in totals}
            interval_bpp_stats = RunningBppStats()
            interval_samples = 0
            interval_skipped_batches = 0

        if reached_max_steps:
            stop_training = True
            break

    metrics = average_reduced_metrics(
        totals,
        processed_samples,
        skipped_batches,
        bpp_stats,
        device,
        distributed,
    )
    return metrics, global_step, stop_training


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    criterion: RateDistortionLoss,
    loader: DataLoader,
    device: torch.device,
    progress_enabled: bool = True,
    distributed: DistributedContext = DistributedContext(enabled=False),
) -> dict[str, float]:
    model.eval()
    totals = make_metric_totals()
    bpp_stats = RunningBppStats()
    processed_samples = 0
    skipped_batches = 0

    progress = tqdm(
        loader,
        desc="val",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        disable=not (progress_enabled and distributed.is_main),
    )
    for batch in progress:
        batch = batch.to(device, non_blocking=True)
        output = model(batch)
        losses = criterion(output, batch)
        loss_is_finite = all_reduce_bool(
            tensor_is_finite(losses["loss"]),
            device,
            distributed,
        )
        if not loss_is_finite:
            skipped_batches += 1
            progress.set_postfix(skipped=skipped_batches)
            continue
        batch_size = batch.size(0)
        processed_samples += batch_size
        for key in totals:
            totals[key] += float(losses[key].detach()) * batch_size
        bpp_stats.update(losses.get("bpp_per_image", losses["bpp"]).detach())
        progress.set_postfix(
            loss=f"{float(losses['loss'].detach()):.4f}",
            bpp=f"{float(losses['bpp'].detach()):.3f}",
        )

    return average_reduced_metrics(
        totals,
        processed_samples,
        skipped_batches,
        bpp_stats,
        device,
        distributed,
        prefix="val_",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train compressai-nano.")
    parser.add_argument(
        "--quality-profile",
        choices=tuple(TRAIN_PROFILES.keys()),
        default="balanced",
        help=(
            "Training preset. detail is the legacy nano profile; hyper_quality_* "
            "is the scale-only hyperprior baseline; hyper_ms_* is the mean-scale route."
        ),
    )
    parser.add_argument(
        "--model-variant",
        choices=tuple(MODEL_CONFIGS.keys()),
        default=None,
        help="Model family. Defaults to nano unless the profile or checkpoint selects another variant.",
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
        help="Weight for high-frequency detail reconstruction loss.",
    )
    parser.add_argument(
        "--highlight-weight",
        type=float,
        default=None,
        help=(
            "Weight for the explicit highlight-aware loss. Use >0 to focus "
            "fine-tuning on bright edges, reflections, and specular texture."
        ),
    )
    parser.add_argument(
        "--highlight-peak-under-weight",
        type=float,
        default=None,
        help="Deprecated alias for --highlight-under-weight.",
    )
    parser.add_argument(
        "--highlight-under-weight",
        type=float,
        default=None,
        help=(
            "Internal weight for penalizing under-reconstructed strong highlight "
            "peaks inside --highlight-weight."
        ),
    )
    parser.add_argument(
        "--highlight-lap-weight",
        type=float,
        default=None,
        help=(
            "Internal Laplacian weight inside --highlight-weight for bright "
            "edge and water-ripple sharpness."
        ),
    )
    parser.add_argument(
        "--texture-lap-weight",
        type=float,
        default=None,
        help=(
            "Laplacian weight inside --detail-weight for highlight texture "
            "and thin reflective edges."
        ),
    )
    parser.add_argument(
        "--texture-contrast-weight",
        type=float,
        default=None,
        help=(
            "Local luminance contrast weight inside --detail-weight for "
            "bright texture retention."
        ),
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
        help="Optional LPIPS perceptual loss weight. Start around 0.003 for detail fine-tuning.",
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
    parser.add_argument("--encoder-activation", choices=("relu", "relu6", "leaky_relu"), default=None)
    parser.add_argument("--decoder-activation", choices=("relu", "leaky_relu"), default="leaky_relu")
    parser.add_argument(
        "--enable-latent-fake-quant",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--latent-fake-quant-bits", type=int, choices=(8,), default=None)
    parser.add_argument("--latent-fake-quant-clip", type=float, default=None)
    parser.add_argument(
        "--enable-z-fake-quant",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--z-fake-quant-bits", type=int, choices=(8,), default=None)
    parser.add_argument("--z-fake-quant-clip", type=float, default=None)
    parser.add_argument(
        "--enable-scale-fake-quant",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--scale-fake-quant-bits", type=int, choices=(8,), default=None)
    parser.add_argument("--scale-fake-quant-clip", type=float, default=None)
    parser.add_argument("--latent-range-weight", type=float, default=None)
    parser.add_argument("--z-range-weight", type=float, default=None)
    parser.add_argument("--symbol-range-weight", type=float, default=None)
    parser.add_argument("--scale-range-weight", type=float, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--local-rank",
        type=int,
        default=None,
        help="Ignored compatibility flag for torchrun/launch; LOCAL_RANK env is used.",
    )
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
    legacy_highlight_under_weight = args.highlight_peak_under_weight
    for key, value in profile.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if legacy_highlight_under_weight is not None:
        args.highlight_under_weight = legacy_highlight_under_weight
    args.highlight_peak_under_weight = args.highlight_under_weight

    checkpoint_path = args.resume or args.init_checkpoint
    checkpoint_variant = checkpoint_model_variant(checkpoint_path)
    if args.model_variant is None and checkpoint_variant is not None:
        args.model_variant = checkpoint_variant
    if args.model_variant is None:
        args.model_variant = MODEL_VARIANT_NANO
    args.model_variant = normalize_model_variant(args.model_variant)

    config = MODEL_CONFIGS[args.model_variant]
    if args.encoder_activation is None:
        args.encoder_activation = config.activation
    default_qat = {
        "enable_latent_fake_quant": False,
        "latent_fake_quant_bits": 8,
        "latent_fake_quant_clip": 6.0,
        "enable_z_fake_quant": False,
        "z_fake_quant_bits": 8,
        "z_fake_quant_clip": 6.0,
        "enable_scale_fake_quant": False,
        "scale_fake_quant_bits": 8,
        "scale_fake_quant_clip": 8.0,
        "latent_range_weight": 0.0,
        "z_range_weight": 0.0,
        "symbol_range_weight": 0.0,
        "scale_range_weight": 0.0,
    }
    for key, value in default_qat.items():
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
    if args.highlight_weight < 0:
        raise ValueError("--highlight-weight must be non-negative")
    if args.highlight_under_weight < 0:
        raise ValueError("--highlight-under-weight must be non-negative")
    if args.highlight_lap_weight < 0:
        raise ValueError("--highlight-lap-weight must be non-negative")
    if args.texture_lap_weight < 0:
        raise ValueError("--texture-lap-weight must be non-negative")
    if args.texture_contrast_weight < 0:
        raise ValueError("--texture-contrast-weight must be non-negative")
    if args.l1_weight < 0:
        raise ValueError("--l1-weight must be non-negative")
    if args.lpips_weight < 0:
        raise ValueError("--lpips-weight must be non-negative")
    if args.quant_step is not None and args.quant_step <= 0:
        raise ValueError("--quant-step must be positive")
    for key in (
        "latent_fake_quant_clip",
        "z_fake_quant_clip",
        "scale_fake_quant_clip",
    ):
        if getattr(args, key) <= 0:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    for key in (
        "latent_range_weight",
        "z_range_weight",
        "symbol_range_weight",
        "scale_range_weight",
    ):
        if getattr(args, key) < 0:
            raise ValueError(f"--{key.replace('_', '-')} must be non-negative")


def main() -> None:
    args = parse_args()
    apply_quality_profile(args)
    distributed = init_distributed()

    try:
        random.seed(args.seed + distributed.rank)
        torch.manual_seed(args.seed + distributed.rank)

        device = distributed_device(distributed)
        amp_enabled = bool(args.amp and device.type == "cuda")
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            if distributed.is_main:
                print(f"device: cuda ({torch.cuda.get_device_name(device)})")
        elif distributed.is_main:
            print("device: cpu")

        transform = make_train_transform(args.crop_size, args.quality_profile)
        train_dataset = ImageFolderDataset(args.train_dir, transform=transform)
        train_sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=distributed.world_size,
                rank=distributed.rank,
                shuffle=True,
                seed=args.seed,
                drop_last=False,
            )
            if distributed.enabled
            else None
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
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
            val_sampler = (
                DistributedSampler(
                    val_dataset,
                    num_replicas=distributed.world_size,
                    rank=distributed.rank,
                    shuffle=False,
                    drop_last=False,
                )
                if distributed.enabled
                else None
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
                drop_last=False,
            )

        base_model = get_model(
            model_variant=args.model_variant,
            activation=args.encoder_activation,
            decoder_activation=args.decoder_activation,
            qat=make_qat_settings(args),
        ).to(device)
        set_quant_step(base_model, args.quant_step)
        criterion = RateDistortionLoss(
            args.lmbda,
            rate_weight=args.rate_weight,
            target_bpp=args.target_bpp,
            ssim_weight=args.ssim_weight,
            detail_weight=args.detail_weight,
            highlight_weight=args.highlight_weight,
            highlight_under_weight=args.highlight_under_weight,
            highlight_lap_weight=args.highlight_lap_weight,
            texture_lap_weight=args.texture_lap_weight,
            texture_contrast_weight=args.texture_contrast_weight,
            l1_weight=args.l1_weight,
            lpips_weight=args.lpips_weight,
            lpips_net=args.lpips_net,
            latent_range_weight=args.latent_range_weight,
            z_range_weight=args.z_range_weight,
            symbol_range_weight=args.symbol_range_weight,
            scale_range_weight=args.scale_range_weight,
        ).to(device)

        model: nn.Module = base_model
        if distributed.enabled:
            model = DistributedDataParallel(
                base_model,
                device_ids=[distributed.local_rank],
                output_device=distributed.local_rank,
            )

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = make_scheduler(optimizer, args)
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        global_step = 0
        start_epoch = 0
        if args.init_checkpoint is not None:
            state = load_checkpoint(args.init_checkpoint, base_model)
            set_quant_step(base_model, args.quant_step)
            if distributed.is_main:
                print(f"initialized weights: {args.init_checkpoint} (source epoch {state.epoch})")
        elif args.resume is not None:
            state = load_checkpoint(args.resume, base_model, optimizer, scheduler)
            start_epoch = state.epoch
            global_step = state.global_step or start_epoch * len(train_loader)
            set_quant_step(base_model, args.quant_step)
            if distributed.is_main:
                print(f"resumed: {args.resume} at epoch {start_epoch}, step {global_step}")

        if distributed.is_main:
            print_run_config(
                args,
                train_dataset,
                train_loader,
                val_loader,
                base_model,
                optimizer,
                amp_enabled,
                global_step,
                distributed,
            )

        if args.max_steps is not None and global_step >= args.max_steps:
            if distributed.is_main:
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
                metrics.update(
                    evaluate_loss(
                        model,
                        criterion,
                        val_loader,
                        device,
                        args.progress,
                        distributed,
                    )
                )

            local_state_is_finite = metrics_are_finite(metrics) and model_parameters_are_finite(
                base_model
            )
            state_is_finite = all_reduce_bool(local_state_is_finite, device, distributed)
            if not state_is_finite:
                if distributed.is_main:
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

            if distributed.is_main:
                if args.log_style == "full":
                    metric_text = ", ".join(
                        f"{key}={value:.6f}" for key, value in metrics.items()
                    )
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

                save_checkpoint(
                    checkpoint_path,
                    base_model,
                    optimizer,
                    scheduler,
                    epoch,
                    step,
                    args,
                    metrics,
                )
                save_checkpoint(
                    latest_path,
                    base_model,
                    optimizer,
                    scheduler,
                    epoch,
                    step,
                    args,
                    metrics,
                )
                if improved:
                    save_checkpoint(
                        best_path,
                        base_model,
                        optimizer,
                        scheduler,
                        epoch,
                        step,
                        args,
                        metrics,
                    )

            if improved:
                best_val_loss = metrics["val_loss"]
            last_checkpoint_step = step
            if distributed.enabled:
                dist.barrier()
            return True

        for epoch in range(start_epoch + 1, end_epoch + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
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
                distributed,
            )

            if epoch == end_epoch and global_step != last_checkpoint_step:
                if not save_step_checkpoint(epoch, global_step, train_metrics):
                    break

            if stop_training:
                if distributed.is_main:
                    print(f"stopped at max_steps={args.max_steps}")
                break
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
