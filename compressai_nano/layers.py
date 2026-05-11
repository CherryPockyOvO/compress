from __future__ import annotations

import torch.nn as nn


def conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 5,
    stride: int = 2,
    bias: bool = True,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        bias=bias,
    )


def deconv(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 5,
    stride: int = 2,
    bias: bool = True,
) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        output_padding=stride - 1,
        bias=bias,
    )


def make_activation(name: str = "relu", inplace: bool = True) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name in {"leaky_relu", "leaky-relu", "lrelu"}:
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    raise ValueError(f"Unsupported activation: {name}")


class ConvNormAct(nn.Sequential):
    """Conv2d + BatchNorm2d + activation replacement for CompressAI GDN blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "relu",
        kernel_size: int = 5,
        stride: int = 2,
    ) -> None:
        super().__init__(
            conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels),
            make_activation(activation),
        )


class DeconvNormAct(nn.Sequential):
    """ConvTranspose2d + BatchNorm2d + activation replacement for IGDN blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "relu",
        kernel_size: int = 5,
        stride: int = 2,
    ) -> None:
        super().__init__(
            deconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels),
            make_activation(activation),
        )


def init_module(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
