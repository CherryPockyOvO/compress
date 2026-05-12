from __future__ import annotations

import dataclasses
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch


MAGIC = b"CNZ4"
VERSION = 1

DTYPE_INT16 = 1
DTYPE_INT32 = 2

CODEC_NONE = 1
CODEC_ZLIB = 2
CODEC_LZ4 = 3
CODEC_ZSTD = 4

CODEC_NAMES = {
    CODEC_NONE: "none",
    CODEC_ZLIB: "zlib",
    CODEC_LZ4: "lz4",
    CODEC_ZSTD: "zstd",
}
CODEC_BY_NAME = {value: key for key, value in CODEC_NAMES.items()}

FIXED_HEADER_FORMAT = "<4s" + ("I" * 12) + "fIQ"
FIXED_HEADER_SIZE = struct.calcsize(FIXED_HEADER_FORMAT)


@dataclasses.dataclass(frozen=True)
class CnzHeader:
    version: int
    header_size: int
    orig_h: int
    orig_w: int
    padded_h: int
    padded_w: int
    latent_c: int
    latent_h: int
    latent_w: int
    down_factor: int
    dtype: int
    codec: int
    quant_step: float
    num_medians: int
    payload_size: int


@dataclasses.dataclass(frozen=True)
class CnzFile:
    header: CnzHeader
    medians: np.ndarray
    payload: bytes


@dataclasses.dataclass(frozen=True)
class PackedSymbols:
    symbols: torch.Tensor
    dtype_code: int
    raw_bytes: bytes
    min_symbol: int
    max_symbol: int


def round_to_even_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x)


def quantize_latent(
    y: torch.Tensor,
    medians: torch.Tensor,
    quant_step: float | torch.Tensor,
) -> torch.Tensor:
    if y.dim() != 4:
        raise ValueError(f"expected NCHW latent tensor, got {tuple(y.shape)}")
    if y.size(0) != 1:
        raise ValueError("CNZ4 deployment path currently supports batch=1")
    step = torch.as_tensor(quant_step, dtype=y.dtype, device=y.device)
    medians = medians.to(device=y.device, dtype=y.dtype).view(1, -1, 1, 1)
    return round_to_even_tensor((y - medians) / step).to(torch.int32)


def dequantize_symbols(
    symbols: torch.Tensor,
    medians: torch.Tensor,
    quant_step: float | torch.Tensor,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if device is None:
        device = symbols.device
    symbols = symbols.to(device=device, dtype=dtype)
    step = torch.as_tensor(quant_step, dtype=dtype, device=device)
    medians = medians.to(device=device, dtype=dtype).view(1, -1, 1, 1)
    return symbols * step + medians


def choose_symbol_dtype(symbols: torch.Tensor) -> tuple[int, int, int]:
    min_symbol = int(symbols.min().item())
    max_symbol = int(symbols.max().item())
    if -32768 <= min_symbol and max_symbol <= 32767:
        return DTYPE_INT16, min_symbol, max_symbol
    return DTYPE_INT32, min_symbol, max_symbol


def pack_symbols(symbols: torch.Tensor) -> PackedSymbols:
    symbols = symbols.detach().cpu().contiguous().to(torch.int32)
    dtype_code, min_symbol, max_symbol = choose_symbol_dtype(symbols)
    array = symbols.numpy()
    if dtype_code == DTYPE_INT16:
        raw = array.astype("<i2", copy=False).tobytes(order="C")
    else:
        raw = array.astype("<i4", copy=False).tobytes(order="C")
    return PackedSymbols(
        symbols=symbols,
        dtype_code=dtype_code,
        raw_bytes=raw,
        min_symbol=min_symbol,
        max_symbol=max_symbol,
    )


def unpack_symbols(raw: bytes, dtype_code: int, latent_shape: tuple[int, int, int, int]) -> torch.Tensor:
    expected_items = int(np.prod(latent_shape))
    if dtype_code == DTYPE_INT16:
        dtype = np.dtype("<i2")
    elif dtype_code == DTYPE_INT32:
        dtype = np.dtype("<i4")
    else:
        raise ValueError(f"unsupported symbol dtype code: {dtype_code}")
    expected_bytes = expected_items * dtype.itemsize
    if len(raw) != expected_bytes:
        raise ValueError(f"raw symbol size mismatch: got {len(raw)}, expected {expected_bytes}")
    array = np.frombuffer(raw, dtype=dtype).astype(np.int32, copy=True)
    return torch.from_numpy(array.reshape(latent_shape))


def compress_raw_bytes(raw: bytes, codec: int | str = CODEC_ZLIB, level: int = 1) -> bytes:
    codec_code = resolve_codec(codec)
    if codec_code == CODEC_NONE:
        return raw
    if codec_code == CODEC_ZLIB:
        return zlib.compress(raw, level)
    raise ValueError(f"codec not available in Python reference path: {CODEC_NAMES.get(codec_code, codec_code)}")


def decompress_raw_bytes(payload: bytes, codec: int | str, expected_size: int) -> bytes:
    codec_code = resolve_codec(codec)
    if codec_code == CODEC_NONE:
        raw = payload
    elif codec_code == CODEC_ZLIB:
        raw = zlib.decompress(payload)
    else:
        raise ValueError(f"codec not available in Python reference path: {CODEC_NAMES.get(codec_code, codec_code)}")
    if len(raw) != expected_size:
        raise ValueError(f"decompressed size mismatch: got {len(raw)}, expected {expected_size}")
    return raw


def resolve_codec(codec: int | str) -> int:
    if isinstance(codec, int):
        return codec
    normalized = codec.strip().lower()
    if normalized not in CODEC_BY_NAME:
        raise ValueError(f"unsupported codec: {codec}")
    return CODEC_BY_NAME[normalized]


def dtype_itemsize(dtype_code: int) -> int:
    if dtype_code == DTYPE_INT16:
        return 2
    if dtype_code == DTYPE_INT32:
        return 4
    raise ValueError(f"unsupported dtype code: {dtype_code}")


def build_cnz_bytes(
    *,
    symbols: torch.Tensor,
    medians: torch.Tensor,
    quant_step: float,
    orig_size: tuple[int, int],
    padded_size: tuple[int, int],
    down_factor: int,
    codec: int | str = CODEC_ZLIB,
    zlib_level: int = 1,
) -> tuple[bytes, dict[str, Any]]:
    if symbols.dim() != 4 or symbols.size(0) != 1:
        raise ValueError(f"expected latent symbols shape [1,C,H,W], got {tuple(symbols.shape)}")
    packed = pack_symbols(symbols)
    codec_code = resolve_codec(codec)
    payload = compress_raw_bytes(packed.raw_bytes, codec_code, level=zlib_level)
    latent_c = int(symbols.size(1))
    latent_h = int(symbols.size(2))
    latent_w = int(symbols.size(3))
    medians_np = medians.detach().cpu().to(torch.float32).numpy().astype("<f4", copy=False)
    if medians_np.size != latent_c:
        raise ValueError(f"median count {medians_np.size} does not match latent channels {latent_c}")

    header_size = FIXED_HEADER_SIZE + int(medians_np.size) * 4
    header = struct.pack(
        FIXED_HEADER_FORMAT,
        MAGIC,
        VERSION,
        header_size,
        int(orig_size[0]),
        int(orig_size[1]),
        int(padded_size[0]),
        int(padded_size[1]),
        latent_c,
        latent_h,
        latent_w,
        int(down_factor),
        int(packed.dtype_code),
        int(codec_code),
        float(quant_step),
        int(medians_np.size),
        int(len(payload)),
    )
    blob = header + medians_np.tobytes(order="C") + payload
    stats = {
        "dtype": "int16" if packed.dtype_code == DTYPE_INT16 else "int32",
        "dtype_code": packed.dtype_code,
        "codec": CODEC_NAMES[codec_code],
        "payload_size": len(payload),
        "container_size": len(blob),
        "raw_size": len(packed.raw_bytes),
        "min_symbol": packed.min_symbol,
        "max_symbol": packed.max_symbol,
    }
    return blob, stats


def parse_cnz_bytes(data: bytes) -> CnzFile:
    if len(data) < FIXED_HEADER_SIZE:
        raise ValueError("CNZ4 file is too small for the fixed header")
    fixed = struct.unpack(FIXED_HEADER_FORMAT, data[:FIXED_HEADER_SIZE])
    magic = fixed[0]
    if magic != MAGIC:
        raise ValueError(f"invalid CNZ magic: {magic!r}")
    (
        _magic,
        version,
        header_size,
        orig_h,
        orig_w,
        padded_h,
        padded_w,
        latent_c,
        latent_h,
        latent_w,
        down_factor,
        dtype_code,
        codec_code,
        quant_step,
        num_medians,
        payload_size,
    ) = fixed
    if version != VERSION:
        raise ValueError(f"unsupported CNZ version: {version}")
    if latent_c == 0 or latent_h == 0 or latent_w == 0:
        raise ValueError("latent dimensions must be positive")
    if num_medians != latent_c:
        raise ValueError(f"num_medians {num_medians} does not match latent_c {latent_c}")
    if quant_step <= 0:
        raise ValueError(f"quant_step must be positive, got {quant_step}")
    expected_header_size = FIXED_HEADER_SIZE + num_medians * 4
    if header_size != expected_header_size:
        raise ValueError(f"header size mismatch: got {header_size}, expected {expected_header_size}")
    if len(data) < header_size + payload_size:
        raise ValueError("CNZ4 file is truncated")

    medians_start = FIXED_HEADER_SIZE
    medians_end = medians_start + num_medians * 4
    medians = np.frombuffer(data[medians_start:medians_end], dtype="<f4").copy()
    payload = data[header_size:header_size + payload_size]
    header = CnzHeader(
        version=version,
        header_size=header_size,
        orig_h=orig_h,
        orig_w=orig_w,
        padded_h=padded_h,
        padded_w=padded_w,
        latent_c=latent_c,
        latent_h=latent_h,
        latent_w=latent_w,
        down_factor=down_factor,
        dtype=dtype_code,
        codec=codec_code,
        quant_step=quant_step,
        num_medians=num_medians,
        payload_size=payload_size,
    )
    return CnzFile(header=header, medians=medians, payload=payload)


def read_cnz_file(path: str | Path) -> CnzFile:
    return parse_cnz_bytes(Path(path).read_bytes())


def write_cnz_file(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def cnz_to_y_hat(cnz: CnzFile, device: torch.device | str | None = None) -> torch.Tensor:
    header = cnz.header
    itemsize = dtype_itemsize(header.dtype)
    expected_raw_size = header.latent_c * header.latent_h * header.latent_w * itemsize
    raw = decompress_raw_bytes(cnz.payload, header.codec, expected_raw_size)
    symbols = unpack_symbols(raw, header.dtype, (1, header.latent_c, header.latent_h, header.latent_w))
    medians = torch.from_numpy(cnz.medians)
    return dequantize_symbols(symbols, medians, header.quant_step, device=device)
