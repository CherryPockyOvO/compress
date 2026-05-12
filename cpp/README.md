# CNZ4 C++ Post-Processing

This directory contains the RK3588-side CPU post-processing path for
`compressai-nano`.

The deployment split is:

```text
RKNN NPU: image -> Encoder -> latent y
C++ CPU : y -> quantize -> int16/int32 symbols -> zlib -> CNZ4 bitstream
PC      : CNZ4 -> y_hat -> PyTorch Decoder -> reconstructed image
```

The model architecture and Python training path are unchanged.

## Build

```bash
mkdir -p build
cd build
cmake ..
make -j
```

Default dependency:

- C++17
- zlib

Optional CMake flags exist for future codecs, but they are off by default:

```bash
cmake .. -DENABLE_LZ4=OFF -DENABLE_ZSTD=OFF
```

## One-Command C++ Test

From the project root:

```bash
cpp/scripts/cpp_roundtrip_test.sh \
  --latent latent.bin \
  --params entropy_params.json \
  --cnz test.cnz \
  --yhat y_hat.bin
```

This runs the full C++ post-processing path:

```text
latent.bin -> CNZ4 package -> y_hat.bin
```

`y_hat.bin` is the PC-side decoder input tensor. The final RGB image is still
produced by the PyTorch decoder.

## Deployment Packages

From the project root:

```bash
cpp/scripts/package_cnz_tools.sh
```

This writes:

```text
dist/cnz_cpp_tools/rk3588_encode_source/
dist/cnz_cpp_tools/pc_decode_bin/
```

The RK3588 folder contains C++ source plus `run_encode.sh`; build it on the
RK3588 board or cross-compile it. The PC folder contains the current-machine
`cnz_decode_cli` binary plus `run_decode_yhat.sh`.

## Export Parameters

From the Python project root:

```bash
python tools/export_entropy_params.py \
  --checkpoint checkpoints/latest.pt \
  --output entropy_params.json
```

## Export Encoder ONNX

```bash
python tools/export_encoder_onnx.py \
  --checkpoint checkpoints/latest.pt \
  --output encoder.onnx \
  --height 512 \
  --width 512
```

Only `model.encoder` is exported. The decoder and entropy path are not exported.

## Simulate RKNN Latent

```bash
python tools/dump_latent.py \
  --image test.png \
  --checkpoint checkpoints/latest.pt \
  --output latent.bin
```

`latent.bin` is float32 NCHW, batch=1. This matches the C++ CLI input. The
script also writes `latent.bin.json` metadata by default. By default, the image
is encoded at its original pixel size and padded to the model downsampling
factor. Pass `--height` and `--width` only when you intentionally need a
fixed-size RKNN input simulation.

## Encode CNZ4

```bash
./cnz_encode_cli \
  --latent latent.bin \
  --params entropy_params.json \
  --output test.cnz \
  --codec zlib \
  --zlib-level 1
```

If `latent.bin.json` is next to `latent.bin`, the encoder reads the image and
latent dimensions from that metadata. Without metadata, square latents are
inferred from the raw file size and entropy parameter channel count. For
non-square raw latents, pass `--latent-h` and `--latent-w`, or provide
`--metadata path/to/latent.bin.json`.

The encoder scans the quantized symbols and stores int16 when safe; otherwise it
falls back to int32. The dtype is recorded in the CNZ4 header.

## Decode Latent For Testing

```bash
./cnz_decode_cli \
  --input test.cnz \
  --output-yhat y_hat.bin
```

`y_hat.bin` is float32 NCHW.

## Benchmark

```bash
./cnz_benchmark_cli \
  --latent latent.bin \
  --params entropy_params.json \
  --latent-c 128 --latent-h 32 --latent-w 32 \
  --orig-h 512 --orig-w 512 \
  --padded-h 512 --padded-w 512 \
  --output bench.cnz \
  --codec zlib \
  --zlib-level 1
```

## CNZ4 Format

All fields are little-endian:

```text
magic:          4 bytes  "CNZ4"
version:        uint32
header_size:    uint32
orig_h:         uint32
orig_w:         uint32
padded_h:       uint32
padded_w:       uint32
latent_c:       uint32
latent_h:       uint32
latent_w:       uint32
down_factor:    uint32
dtype:          uint32   1=int16, 2=int32
codec:          uint32   1=none, 2=zlib, 3=lz4, 4=zstd
quant_step:     float32
num_medians:    uint32
payload_size:   uint64
medians:        float32[num_medians]
payload:        bytes
```

The C++ reader checks magic, version, header size, latent dimensions, median
count, dtype, codec, payload size, and decompressed raw size.
