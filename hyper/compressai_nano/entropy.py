from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
from torch import Tensor

from .cnz import (
    CODEC_ZLIB,
    compress_raw_bytes,
    dequantize_symbols,
    decompress_raw_bytes,
    dtype_itemsize,
    pack_symbols,
    unpack_symbols,
)


SAMPLE_PAYLOAD_FORMAT = "<4sIIIQ"
SAMPLE_PAYLOAD_MAGIC = b"CNS1"
SAMPLE_PAYLOAD_HEADER_SIZE = struct.calcsize(SAMPLE_PAYLOAD_FORMAT)


@dataclass(frozen=True)
class EntropyPayload:
    strings: list[bytes]
    shape: tuple[int, int]
    latent_shape: tuple[int, int, int, int]
    quant_step: float
    dtype: str
    codec: str


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
        dtype_name = "int16"
        for sample in symbols:
            sample_symbols = sample.unsqueeze(0)
            packed = pack_symbols(sample_symbols)
            dtype_name = "int16" if packed.dtype_code == 1 else "int32"
            compressed = compress_raw_bytes(packed.raw_bytes, codec=CODEC_ZLIB, level=1)
            payload_header = struct.pack(
                SAMPLE_PAYLOAD_FORMAT,
                SAMPLE_PAYLOAD_MAGIC,
                int(packed.dtype_code),
                int(CODEC_ZLIB),
                int(len(packed.raw_bytes)),
                int(len(compressed)),
            )
            strings.append(payload_header + compressed)

        latent_shape = tuple(int(v) for v in symbols.shape)
        return EntropyPayload(
            strings=strings,
            shape=latent_shape[-2:],
            latent_shape=latent_shape,
            quant_step=float(self.quant_step.detach().cpu()),
            dtype=dtype_name,
            codec="zlib",
        )

    @torch.no_grad()
    def decompress(
        self,
        strings: bytes | Iterable[bytes],
        shape: tuple[int, int] | None = None,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if isinstance(strings, bytes):
            strings = [strings]

        y_hats = []
        for string in strings:
            if len(string) < SAMPLE_PAYLOAD_HEADER_SIZE:
                raise ValueError("entropy sample payload is too small")
            magic, dtype_code, codec_code, raw_size, payload_size = struct.unpack(
                SAMPLE_PAYLOAD_FORMAT,
                string[:SAMPLE_PAYLOAD_HEADER_SIZE],
            )
            if magic != SAMPLE_PAYLOAD_MAGIC:
                raise ValueError(f"invalid entropy sample payload magic: {magic!r}")
            payload = string[SAMPLE_PAYLOAD_HEADER_SIZE:]
            if len(payload) != payload_size:
                raise ValueError(f"payload size mismatch: got {len(payload)}, expected {payload_size}")
            raw = decompress_raw_bytes(payload, codec_code, int(raw_size))

            if shape is None:
                raise ValueError("shape is required to decompress raw symbol payloads")
            sample_shape = (1, self.channels, int(shape[0]), int(shape[1]))
            expected_size = self.channels * int(shape[0]) * int(shape[1]) * dtype_itemsize(dtype_code)
            if len(raw) != expected_size:
                raise ValueError(f"raw size mismatch: got {len(raw)}, expected {expected_size}")
            symbols = unpack_symbols(raw, dtype_code, sample_shape)
            y_hats.append(dequantize_symbols(
                symbols,
                self.medians.detach(),
                float(self.quant_step.item()),
                dtype=torch.float32,
                device=device,
            ))
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
