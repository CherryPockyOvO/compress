from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .entropy import EntropyPayload, NanoEntropyBottleneck
from .layers import ConvNormAct, conv, deconv, init_module, make_activation


@dataclass(frozen=True)
class ModelConfig:
    N: int
    M: int
    quant_step: float
    decoder_channels: int
    decoder_res_blocks: int
    refinement_blocks: int
    name: str


MODEL_CONFIG = ModelConfig(
    N=128,
    M=128,
    quant_step=0.67,
    decoder_channels=256,
    decoder_res_blocks=3,
    refinement_blocks=5,
    name="high-quality",
)


def get_model_config() -> ModelConfig:
    return MODEL_CONFIG


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
    """PC-quality synthesis transform g_s: quantized latent y_hat -> x_hat.

    The encoder remains small and RK3588 friendly. The decoder is intentionally
    heavier because it runs on the PC side after the latent bitstream is
    transmitted back from the RK3588 device.
    """

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
    """Asymmetric image codec for RK3588 encode and PC decode.

    Encoder: RK3588-friendly Conv+BatchNorm+ReLU only.
    Decoder: PC-side residual decoder with much higher reconstruction capacity.

    Note: checkpoints trained with the old lightweight decoder or the old
    N=64, M=64 layout are shape incompatible with this model and must be
    retrained.
    """

    def __init__(
        self,
        activation: str = "relu",
        decoder_activation: str = "leaky_relu",
        clamp_decoder_output: bool = True,
    ) -> None:
        super().__init__()
        config = get_model_config()

        self.model_name = config.name
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

    def forward(self, x: Tensor) -> dict[str, Tensor | dict[str, Tensor]]:
        y = self.encoder(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.decoder(y_hat)
        return {
            "x_hat": x_hat,
            "y": y,
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods},
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


def get_model(**kwargs: object) -> FactorizedPriorNano:
    return FactorizedPriorNano(**kwargs)
