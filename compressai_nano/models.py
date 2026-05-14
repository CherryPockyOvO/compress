from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .entropy import EntropyPayload, NanoEntropyBottleneck
from .layers import ConvNormAct, conv, deconv, init_module, make_activation


MODEL_VARIANT_NANO = "nano"
MODEL_VARIANT_HYPER_RESIDUAL_Q = "nano_hyper_residual_q"


@dataclass(frozen=True)
class ModelConfig:
    N: int
    M: int
    quant_step: float
    decoder_channels: int
    decoder_res_blocks: int
    refinement_blocks: int
    name: str
    model_variant: str = MODEL_VARIANT_NANO
    model_type: str = "factorized"
    encoder_type: str = "default"
    activation: str = "relu"
    encoder_norm: str = "bn"
    Z: int | None = None
    latent_clip: float | None = None
    z_clip: float | None = None
    scale_min: float = 1e-3
    scale_max: float = 20.0


MODEL_CONFIGS: dict[str, ModelConfig] = {
    MODEL_VARIANT_NANO: ModelConfig(
        N=128,
        M=128,
        quant_step=0.67,
        decoder_channels=256,
        decoder_res_blocks=3,
        refinement_blocks=5,
        name="high-quality",
        model_variant=MODEL_VARIANT_NANO,
        model_type="factorized",
        encoder_type="default",
        activation="relu",
        encoder_norm="bn",
        latent_clip=None,
    ),
    MODEL_VARIANT_HYPER_RESIDUAL_Q: ModelConfig(
        N=128,
        M=160,
        Z=96,
        quant_step=0.45,
        decoder_channels=256,
        decoder_res_blocks=4,
        refinement_blocks=6,
        name=MODEL_VARIANT_HYPER_RESIDUAL_Q,
        model_variant=MODEL_VARIANT_HYPER_RESIDUAL_Q,
        model_type="scale_hyperprior",
        encoder_type="residual_quant_friendly",
        activation="relu6",
        encoder_norm="none",
        latent_clip=6.0,
        z_clip=6.0,
        scale_min=1e-3,
        scale_max=20.0,
    ),
}

# Backward-compatible alias used by older scripts/imports.
MODEL_CONFIG = MODEL_CONFIGS[MODEL_VARIANT_NANO]


def normalize_model_variant(model_variant: str | None = None) -> str:
    if model_variant is None:
        return MODEL_VARIANT_NANO
    normalized = str(model_variant).strip()
    if not normalized:
        return MODEL_VARIANT_NANO
    if normalized not in MODEL_CONFIGS:
        choices = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(f"unknown model_variant={normalized!r}; choices: {choices}")
    return normalized


def get_model_config(model_variant: str | None = None) -> ModelConfig:
    return MODEL_CONFIGS[normalize_model_variant(model_variant)]


def model_config_to_dict(config: ModelConfig) -> dict[str, Any]:
    return asdict(config)


def infer_model_variant_from_checkpoint(raw: object) -> str:
    if not isinstance(raw, dict):
        return MODEL_VARIANT_NANO

    variant = raw.get("model_variant")
    if isinstance(variant, str) and variant:
        return normalize_model_variant(variant)

    config = raw.get("model_config")
    if isinstance(config, dict):
        variant = config.get("model_variant") or config.get("name")
        if isinstance(variant, str) and variant in MODEL_CONFIGS:
            return normalize_model_variant(variant)

    return MODEL_VARIANT_NANO


def clip_latent(value: Tensor, clip: float | None) -> Tensor:
    if clip is None or clip <= 0:
        return value
    clip_tensor = value.new_tensor(float(clip))
    return clip_tensor * torch.tanh(value / clip_tensor)


def fake_quant_symmetric_ste(x: Tensor, bits: int = 8, clip: float = 6.0) -> Tensor:
    if bits <= 0:
        raise ValueError(f"fake quant bits must be positive, got {bits}")
    if clip <= 0:
        return x
    qmax = 2 ** (bits - 1) - 1
    scale = float(clip) / float(qmax)
    x_clip = x.clamp(-float(clip), float(clip))
    q = torch.round(x_clip / scale).clamp(-qmax, qmax)
    x_q = q * scale
    return x + (x_q - x).detach()


def fake_quant_positive_ste(x: Tensor, bits: int = 8, clip: float = 8.0) -> Tensor:
    if bits <= 0:
        raise ValueError(f"fake quant bits must be positive, got {bits}")
    if clip <= 0:
        return x
    qmax = 2**bits - 1
    scale = float(clip) / float(qmax)
    x_clip = x.clamp(0.0, float(clip))
    q = torch.round(x_clip / scale).clamp(0, qmax)
    x_q = q * scale
    return x + (x_q - x).detach()


@dataclass
class QATSettings:
    enable_latent_fake_quant: bool = False
    latent_fake_quant_bits: int = 8
    latent_fake_quant_clip: float = 6.0
    enable_z_fake_quant: bool = False
    z_fake_quant_bits: int = 8
    z_fake_quant_clip: float = 6.0
    enable_scale_fake_quant: bool = False
    scale_fake_quant_bits: int = 8
    scale_fake_quant_clip: float = 8.0


class Encoder(nn.Module):
    """RK3588-friendly analysis transform g_a: image x -> latent y."""

    def __init__(self, N: int = 128, M: int = 128, activation: str = "relu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvNormAct(3, N, activation=activation),
            ConvNormAct(N, N, activation=activation),
            ConvNormAct(N, N, activation=activation),
            conv(N, M),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ConvAct(nn.Sequential):
    """Conv2d + ReLU/ReLU6 without normalization for RKNN quantization experiments."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "relu6",
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        bias: bool = True,
    ) -> None:
        if kernel_size not in {1, 3, 5}:
            raise ValueError("ConvAct kernel_size must be 1, 3, or 5")
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            make_activation(activation),
        )


class QuantResidualBlock(nn.Module):
    """Quantization-friendly residual block: Conv3x3 -> ReLU6 -> Conv3x3 -> Add -> ReLU6."""

    def __init__(self, channels: int, activation: str = "relu6") -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act1 = make_activation(activation)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act2 = make_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.conv2(self.act1(self.conv1(x)))
        return self.act2(x + residual)


class DownsampleResidualBlock(nn.Module):
    """Stride-2 residual downsampler with an explicit 1x1 skip branch."""

    def __init__(self, in_channels: int, out_channels: int, activation: str = "relu6") -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            make_activation(activation),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        self.out_act = make_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_act(self.main(x) + self.skip(x))


class QuantFriendlyResidualEncoder(nn.Module):
    """Residual analysis transform prepared for RKNN INT8/FP16 mixed precision."""

    def __init__(
        self,
        N: int = 128,
        M: int = 160,
        activation: str = "relu6",
        latent_clip: float | None = 6.0,
    ) -> None:
        super().__init__()
        self.latent_clip = latent_clip
        self.down1 = DownsampleResidualBlock(3, N, activation=activation)
        self.res1 = QuantResidualBlock(N, activation=activation)
        self.down2 = DownsampleResidualBlock(N, N, activation=activation)
        self.res2 = QuantResidualBlock(N, activation=activation)
        self.down3 = DownsampleResidualBlock(N, N, activation=activation)
        self.res3 = QuantResidualBlock(N, activation=activation)
        self.down4 = DownsampleResidualBlock(N, M, activation=activation)

    def forward(self, x: Tensor) -> Tensor:
        x = self.res1(self.down1(x))
        x = self.res2(self.down2(x))
        x = self.res3(self.down3(x))
        y = self.down4(x)
        return clip_latent(y, self.latent_clip)


class HyperEncoder(nn.Module):
    """Lightweight scale-only hyper encoder h_a: y -> z."""

    def __init__(
        self,
        M: int,
        N: int,
        Z: int,
        activation: str = "relu6",
        z_clip: float | None = 6.0,
    ) -> None:
        super().__init__()
        self.z_clip = z_clip
        self.net = nn.Sequential(
            ConvAct(M, N, activation=activation, kernel_size=3, stride=1),
            ConvAct(N, N, activation=activation, kernel_size=3, stride=2),
            nn.Conv2d(N, Z, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, y: Tensor) -> Tensor:
        return clip_latent(self.net(y), self.z_clip)


class HyperDecoder(nn.Module):
    """Lightweight scale-only hyper decoder h_s: z_hat -> scales_y."""

    def __init__(
        self,
        Z: int,
        N: int,
        M: int,
        activation: str = "relu6",
        scale_min: float = 1e-3,
        scale_max: float = 20.0,
    ) -> None:
        super().__init__()
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.net = nn.Sequential(
            deconv(Z, N, kernel_size=3, stride=2),
            make_activation(activation),
            deconv(N, N, kernel_size=3, stride=2),
            make_activation(activation),
            nn.Conv2d(N, M, kernel_size=3, padding=1),
        )

    def make_positive_scale(self, raw: Tensor) -> Tensor:
        scale = F.softplus(raw) + self.scale_min
        return scale.clamp(self.scale_min, self.scale_max)

    def forward(self, z_hat: Tensor) -> Tensor:
        return self.make_positive_scale(self.net(z_hat))


class GaussianConditionalEntropy(nn.Module):
    """Scale-only Gaussian conditional entropy model for y."""

    def __init__(
        self,
        quant_step: float = 1.0,
        scale_min: float = 1e-3,
        scale_max: float = 20.0,
        likelihood_bound: float = 1e-9,
    ) -> None:
        super().__init__()
        if quant_step <= 0:
            raise ValueError("quant_step must be positive")
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.likelihood_bound = float(likelihood_bound)
        self.register_buffer("quant_step", torch.tensor(float(quant_step)))

    def forward(
        self,
        y: Tensor,
        scales_y: Tensor,
        training: bool | None = None,
    ) -> tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        step = self._step_like(y)
        if training:
            y_hat = y + (torch.rand_like(y) - 0.5) * step
        else:
            y_hat = self.quantize(y).to(dtype=y.dtype) * step
        likelihoods = self._likelihood(y_hat, scales_y)
        return y_hat, likelihoods

    def quantize(self, y: Tensor) -> Tensor:
        step = self._step_like(y)
        return torch.round(y / step).to(torch.int32)

    def dequantize(
        self,
        symbols: Tensor,
        quant_step: float | Tensor | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if device is None:
            device = symbols.device
        symbols = symbols.to(device=device, dtype=dtype)
        if quant_step is None:
            step = self.quant_step.to(device=device, dtype=dtype)
        else:
            step = torch.as_tensor(quant_step, dtype=dtype, device=device)
        return symbols * step

    def _likelihood(self, y_hat: Tensor, scales_y: Tensor) -> Tensor:
        step = self._step_like(y_hat)
        scales = scales_y.to(device=y_hat.device, dtype=y_hat.dtype).clamp(
            self.scale_min,
            self.scale_max,
        )
        upper = self._standardized_cumulative((y_hat + 0.5 * step) / scales)
        lower = self._standardized_cumulative((y_hat - 0.5 * step) / scales)
        return (upper - lower).clamp_min(self.likelihood_bound)

    @staticmethod
    def _standardized_cumulative(inputs: Tensor) -> Tensor:
        return 0.5 * torch.erfc(-inputs / (2.0**0.5))

    def _step_like(self, tensor: Tensor) -> Tensor:
        return self.quant_step.to(device=tensor.device, dtype=tensor.dtype)


class ResidualBlock(nn.Module):
    """PC-side residual block used only by the decoder."""

    def __init__(self, channels: int, activation: str = "leaky_relu") -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            make_activation(activation),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + 0.1 * self.body(x)


class UpsampleResidualBlock(nn.Module):
    """2x upsampling followed by several residual refinement blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = [
            deconv(in_channels, out_channels),
            make_activation(activation),
        ]
        blocks.extend(ResidualBlock(out_channels, activation) for _ in range(num_blocks))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """PC-quality synthesis transform g_s: quantized latent y_hat -> x_hat."""

    def __init__(
        self,
        N: int = 128,
        M: int = 128,
        decoder_channels: int = 256,
        decoder_res_blocks: int = 3,
        refinement_blocks: int = 5,
        activation: str = "leaky_relu",
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        del N
        c0 = int(decoder_channels)
        c1 = max(128, c0)
        c2 = max(96, c0 // 2)
        c3 = max(64, c0 // 4)

        self.stem = nn.Sequential(
            nn.Conv2d(M, c0, kernel_size=3, padding=1),
            make_activation(activation),
            *[ResidualBlock(c0, activation) for _ in range(decoder_res_blocks)],
        )
        self.up1 = UpsampleResidualBlock(c0, c1, decoder_res_blocks, activation)
        self.up2 = UpsampleResidualBlock(c1, c2, decoder_res_blocks, activation)
        self.up3 = UpsampleResidualBlock(c2, c3, decoder_res_blocks, activation)
        self.up4 = UpsampleResidualBlock(c3, 64, refinement_blocks, activation)
        self.to_rgb = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            make_activation(activation),
            *[ResidualBlock(64, activation) for _ in range(refinement_blocks)],
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
        )
        self.clamp_output = bool(clamp_output)

    def forward(self, y_hat: Tensor) -> Tensor:
        x = self.stem(y_hat)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x_hat = self.to_rgb(x)
        if self.clamp_output:
            x_hat = torch.sigmoid(x_hat)
        return x_hat


class FactorizedPriorNano(nn.Module):
    """Backward-compatible nano factorized-prior image codec."""

    model_variant = MODEL_VARIANT_NANO
    supports_cnz_v4 = True

    def __init__(
        self,
        activation: str | None = None,
        decoder_activation: str = "leaky_relu",
        clamp_decoder_output: bool = True,
    ) -> None:
        super().__init__()
        config = get_model_config(MODEL_VARIANT_NANO)
        if activation is None:
            activation = config.activation

        self.model_variant = MODEL_VARIANT_NANO
        self.model_name = config.name
        self.config = config
        self.N = config.N
        self.M = config.M
        self.decoder_channels = config.decoder_channels
        self.downsampling_factor = 2**4

        self.encoder = Encoder(N=self.N, M=self.M, activation=activation)
        self.decoder = Decoder(
            N=self.N,
            M=self.M,
            decoder_channels=config.decoder_channels,
            decoder_res_blocks=config.decoder_res_blocks,
            refinement_blocks=config.refinement_blocks,
            activation=decoder_activation,
            clamp_output=clamp_decoder_output,
        )
        self.entropy_bottleneck = NanoEntropyBottleneck(
            channels=self.M,
            quant_step=config.quant_step,
        )

        init_module(self.encoder)
        init_module(self.decoder)

    @property
    def g_a(self) -> Encoder:
        return self.encoder

    @property
    def g_s(self) -> Decoder:
        return self.decoder

    def model_config_dict(self) -> dict[str, Any]:
        return model_config_to_dict(self.config)

    def set_quant_step(self, quant_step: float) -> None:
        self.entropy_bottleneck.quant_step.fill_(float(quant_step))

    def get_quant_step(self) -> float:
        return float(self.entropy_bottleneck.quant_step.detach().cpu())

    def forward(self, x: Tensor) -> dict[str, Any]:
        y = self.encoder(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.decoder(y_hat)
        return {
            "x_hat": x_hat,
            "y": y,
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods},
            "symbols": {"y": self.entropy_bottleneck.quantize(y).detach()},
            "quant_step": self.entropy_bottleneck.quant_step,
            "latent_clip": None,
            "z_clip": None,
            "model_variant": self.model_variant,
        }

    @torch.no_grad()
    def compress(self, x: Tensor) -> dict[str, object]:
        y = self.encoder(x)
        payload: EntropyPayload = self.entropy_bottleneck.compress(y)
        return {
            "strings": payload.strings,
            "shape": payload.shape,
            "latent_shape": payload.latent_shape,
            "quant_step": payload.quant_step,
            "dtype": payload.dtype,
            "codec": payload.codec,
        }

    @torch.no_grad()
    def decompress(
        self,
        strings: bytes | list[bytes],
        shape: tuple[int, int] | None = None,
    ) -> dict[str, Tensor]:
        device = next(self.parameters()).device
        y_hat = self.entropy_bottleneck.decompress(strings, shape=shape, device=device)
        x_hat = self.decoder(y_hat)
        return {"x_hat": x_hat, "y_hat": y_hat}


class NanoHyperResidualQ(nn.Module):
    """High-precision residual scale-hyperprior codec prepared for QAT and RKNN mixed precision."""

    model_variant = MODEL_VARIANT_HYPER_RESIDUAL_Q
    supports_cnz_v4 = False

    def __init__(
        self,
        activation: str | None = None,
        decoder_activation: str = "leaky_relu",
        clamp_decoder_output: bool = True,
        qat: QATSettings | None = None,
    ) -> None:
        super().__init__()
        config = get_model_config(MODEL_VARIANT_HYPER_RESIDUAL_Q)
        if activation is None:
            activation = config.activation
        if config.Z is None:
            raise ValueError("nano_hyper_residual_q requires config.Z")

        self.model_variant = MODEL_VARIANT_HYPER_RESIDUAL_Q
        self.model_name = config.name
        self.config = config
        self.N = config.N
        self.M = config.M
        self.Z = int(config.Z)
        self.decoder_channels = config.decoder_channels
        self.downsampling_factor = 2**4
        self.scale_min = float(config.scale_min)
        self.scale_max = float(config.scale_max)
        self.latent_clip = config.latent_clip
        self.z_clip = config.z_clip
        self.qat = qat if qat is not None else QATSettings()

        self.encoder = QuantFriendlyResidualEncoder(
            N=self.N,
            M=self.M,
            activation=activation,
            latent_clip=config.latent_clip,
        )
        self.hyper_encoder = HyperEncoder(
            M=self.M,
            N=self.N,
            Z=self.Z,
            activation=activation,
            z_clip=config.z_clip,
        )
        self.entropy_bottleneck_z = NanoEntropyBottleneck(
            channels=self.Z,
            quant_step=1.0,
        )
        self.hyper_decoder = HyperDecoder(
            Z=self.Z,
            N=self.N,
            M=self.M,
            activation=activation,
            scale_min=config.scale_min,
            scale_max=config.scale_max,
        )
        self.conditional_entropy_y = GaussianConditionalEntropy(
            quant_step=config.quant_step,
            scale_min=config.scale_min,
            scale_max=config.scale_max,
        )
        self.decoder = Decoder(
            N=self.N,
            M=self.M,
            decoder_channels=config.decoder_channels,
            decoder_res_blocks=config.decoder_res_blocks,
            refinement_blocks=config.refinement_blocks,
            activation=decoder_activation,
            clamp_output=clamp_decoder_output,
        )

        init_module(self.encoder)
        init_module(self.hyper_encoder)
        init_module(self.hyper_decoder)
        init_module(self.decoder)

    @property
    def g_a(self) -> QuantFriendlyResidualEncoder:
        return self.encoder

    @property
    def h_a(self) -> HyperEncoder:
        return self.hyper_encoder

    @property
    def h_s(self) -> HyperDecoder:
        return self.hyper_decoder

    @property
    def g_s(self) -> Decoder:
        return self.decoder

    def model_config_dict(self) -> dict[str, Any]:
        return model_config_to_dict(self.config)

    def set_qat_settings(self, qat: QATSettings) -> None:
        self.qat = qat

    def set_quant_step(self, quant_step: float) -> None:
        self.conditional_entropy_y.quant_step.fill_(float(quant_step))

    def get_quant_step(self) -> float:
        return float(self.conditional_entropy_y.quant_step.detach().cpu())

    def analysis_transform(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        y = self.encoder(x)
        z = self.hyper_encoder(y)
        z_hat, _ = self.entropy_bottleneck_z(z, training=False)
        scales_y = self.hyper_decoder(z_hat)
        return y, z, scales_y

    def _maybe_fake_quant_latent(self, y: Tensor) -> tuple[Tensor, Tensor]:
        if not self.qat.enable_latent_fake_quant:
            return y, y.new_zeros(())
        y_q = fake_quant_symmetric_ste(
            y,
            bits=self.qat.latent_fake_quant_bits,
            clip=self.qat.latent_fake_quant_clip,
        )
        return y_q, torch.mean(torch.abs(y_q.detach() - y.detach()))

    def _maybe_fake_quant_z(self, z: Tensor) -> tuple[Tensor, Tensor]:
        if not self.qat.enable_z_fake_quant:
            return z, z.new_zeros(())
        z_q = fake_quant_symmetric_ste(
            z,
            bits=self.qat.z_fake_quant_bits,
            clip=self.qat.z_fake_quant_clip,
        )
        return z_q, torch.mean(torch.abs(z_q.detach() - z.detach()))

    def _maybe_fake_quant_scale(self, scales: Tensor) -> tuple[Tensor, Tensor]:
        if not self.qat.enable_scale_fake_quant:
            return scales, scales.new_zeros(())
        scales_q = fake_quant_positive_ste(
            scales,
            bits=self.qat.scale_fake_quant_bits,
            clip=self.qat.scale_fake_quant_clip,
        ).clamp(self.scale_min, self.scale_max)
        return scales_q, torch.mean(torch.abs(scales_q.detach() - scales.detach()))

    def forward(self, x: Tensor) -> dict[str, Any]:
        y = self.encoder(x)
        y_for_hyper, fq_y_error = self._maybe_fake_quant_latent(y)
        z = self.hyper_encoder(y_for_hyper)
        z_for_entropy, fq_z_error = self._maybe_fake_quant_z(z)
        z_hat, z_likelihoods = self.entropy_bottleneck_z(z_for_entropy)
        scales_y = self.hyper_decoder(z_hat)
        scales_y, fq_scale_error = self._maybe_fake_quant_scale(scales_y)
        y_hat, y_likelihoods = self.conditional_entropy_y(y_for_hyper, scales_y)
        x_hat = self.decoder(y_hat)
        return {
            "x_hat": x_hat,
            "y": y,
            "y_for_hyper": y_for_hyper,
            "y_hat": y_hat,
            "z": z,
            "z_for_entropy": z_for_entropy,
            "z_hat": z_hat,
            "scales_y": scales_y,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "symbols": {
                "y": self.conditional_entropy_y.quantize(y_for_hyper).detach(),
                "z": self.entropy_bottleneck_z.quantize(z_for_entropy).detach(),
            },
            "fake_quant_errors": {
                "y": fq_y_error,
                "z": fq_z_error,
                "scale": fq_scale_error,
            },
            "quant_step": self.conditional_entropy_y.quant_step,
            "latent_clip": self.latent_clip,
            "z_clip": self.z_clip,
            "scale_min_value": self.scale_min,
            "scale_max_value": self.scale_max,
            "model_variant": self.model_variant,
        }

    @torch.no_grad()
    def compress(self, x: Tensor) -> dict[str, object]:
        raise NotImplementedError(
            "nano_hyper_residual_q training/export is implemented, but CNZ hyperprior "
            "bitstream support requires a future CNZ5 format with z and y streams."
        )

    @torch.no_grad()
    def decompress(
        self,
        strings: bytes | list[bytes],
        shape: tuple[int, int] | None = None,
    ) -> dict[str, Tensor]:
        del strings, shape
        raise NotImplementedError(
            "nano_hyper_residual_q cannot decode CNZ4 streams. Extend the bitstream "
            "to carry z, y, hyperprior shape, and model_variant before deployment."
        )


def get_model(
    model_variant: str | None = None,
    activation: str | None = None,
    decoder_activation: str = "leaky_relu",
    clamp_decoder_output: bool = True,
    qat: QATSettings | None = None,
) -> nn.Module:
    variant = normalize_model_variant(model_variant)
    if variant == MODEL_VARIANT_NANO:
        return FactorizedPriorNano(
            activation=activation,
            decoder_activation=decoder_activation,
            clamp_decoder_output=clamp_decoder_output,
        )
    if variant == MODEL_VARIANT_HYPER_RESIDUAL_Q:
        return NanoHyperResidualQ(
            activation=activation,
            decoder_activation=decoder_activation,
            clamp_decoder_output=clamp_decoder_output,
            qat=qat,
        )
    raise AssertionError(f"unhandled model_variant={variant}")
