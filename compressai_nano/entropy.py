from __future__ import annotations

import math
import pickle
import zlib
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class EntropyPayload:
    strings: list[bytes]
    shape: tuple[int, int]
    latent_shape: tuple[int, int, int, int]
    quant_step: float


class NanoEntropyBottleneck(nn.Module):
    """Small CPU-side entropy bottleneck interface.

    The forward path keeps the FactorizedPrior training/inference contract:
    latent y is quantized to y_hat and assigned factorized likelihoods. The
    compress/decompress path stores integer symbols with zlib so the nano
    project is runnable without CompressAI C++ rANS extensions. Replace this
    class with a real CDF/rANS coder after training if production bitstreams
    are required.
    """

    def __init__(
        self,
        channels: int,
        quant_step: float = 1.0,
        likelihood_bound: float = 1e-9,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if quant_step <= 0:
            raise ValueError("quant_step must be positive")

        self.channels = int(channels)
        self.likelihood_bound = float(likelihood_bound)
        self.medians = nn.Parameter(torch.zeros(channels))
        self.log_scales = nn.Parameter(torch.zeros(channels))
        self.register_buffer("quant_step", torch.tensor(float(quant_step)))

    def forward(self, y: Tensor, training: bool | None = None) -> tuple[Tensor, Tensor]:
        if training is None:
            training = self.training

        step = self._step_like(y)
        medians = self._channel_param(self.medians, y)
        if training:
            y_hat = y + (torch.rand_like(y) - 0.5) * step
        else:
            y_hat = torch.round((y - medians) / step) * step + medians

        likelihoods = self._likelihood(y_hat)
        return y_hat, likelihoods

    @torch.no_grad()
    def compress(self, y: Tensor) -> EntropyPayload:
        symbols = self.quantize(y).cpu().contiguous()
        if symbols.dim() != 4:
            raise ValueError(f"expected NCHW latent tensor, got shape {tuple(symbols.shape)}")

        strings = []
        for sample in symbols:
            payload = {
                "shape": tuple(int(v) for v in sample.shape),
                "dtype": "int32",
                "quant_step": float(self.quant_step.detach().cpu()),
                "symbols": sample.reshape(-1).tolist(),
            }
            strings.append(zlib.compress(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)))

        latent_shape = tuple(int(v) for v in symbols.shape)
        return EntropyPayload(
            strings=strings,
            shape=latent_shape[-2:],
            latent_shape=latent_shape,
            quant_step=float(self.quant_step.detach().cpu()),
        )

    @torch.no_grad()
    def decompress(
        self,
        strings: bytes | Iterable[bytes],
        shape: tuple[int, int] | None = None,
        device: torch.device | str | None = None,
    ) -> Tensor:
        del shape
        if isinstance(strings, bytes):
            strings = [strings]

        y_hats = []
        for string in strings:
            payload = pickle.loads(zlib.decompress(string))
            if payload.get("dtype") != "int32":
                raise ValueError(f"unsupported payload dtype: {payload.get('dtype')}")

            sample_shape = tuple(int(v) for v in payload["shape"])
            flat = torch.tensor(payload["symbols"], dtype=torch.int32, device=device)
            symbols = flat.reshape(1, *sample_shape)
            y_hats.append(
                self.dequantize(
                    symbols,
                    quant_step=float(payload.get("quant_step", self.quant_step.item())),
                    dtype=torch.float32,
                    device=device,
                )
            )
        return torch.cat(y_hats, dim=0)

    def quantize(self, y: Tensor) -> Tensor:
        step = self._step_like(y)
        medians = self._channel_param(self.medians, y)
        return torch.round((y - medians) / step).to(torch.int32)

    def dequantize(
        self,
        symbols: Tensor,
        quant_step: float | Tensor | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if device is None:
            device = self.medians.device

        symbols = symbols.to(device=device, dtype=dtype)
        if quant_step is None:
            step = self.quant_step.to(device=device, dtype=dtype)
        else:
            step = torch.as_tensor(quant_step, dtype=dtype, device=device)

        medians = self._channel_param(self.medians.to(device=device, dtype=dtype), symbols)
        return symbols * step + medians

    def estimate_bits(self, likelihoods: Tensor) -> Tensor:
        return -torch.log2(likelihoods.clamp_min(self.likelihood_bound)).sum()

    def _likelihood(self, y_hat: Tensor) -> Tensor:
        step = self._step_like(y_hat)
        medians = self._channel_param(self.medians, y_hat)
        scales = torch.exp(self._channel_param(self.log_scales, y_hat)).clamp_min(1e-3)

        centered = y_hat - medians
        upper = self._standardized_cumulative((centered + 0.5 * step) / scales)
        lower = self._standardized_cumulative((centered - 0.5 * step) / scales)
        return (upper - lower).clamp_min(self.likelihood_bound)

    @staticmethod
    def _standardized_cumulative(inputs: Tensor) -> Tensor:
        return 0.5 * torch.erfc(-inputs / math.sqrt(2.0))

    def _step_like(self, tensor: Tensor) -> Tensor:
        return self.quant_step.to(device=tensor.device, dtype=tensor.dtype)

    @staticmethod
    def _channel_param(param: Tensor, reference: Tensor) -> Tensor:
        return param.to(device=reference.device, dtype=reference.dtype).view(
            1,
            -1,
            *([1] * (reference.dim() - 2)),
        )
