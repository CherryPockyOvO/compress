# RK3588 Deployment For `hyper_ms_nano`

This note is for the current deployable high-precision nano route:

```text
profile:       hyper_ms_nano_fp / hyper_ms_nano_qat8
model_variant: nano_hyper_ms_q_nano
model type:    mean-scale hyperprior
```

Do not confuse it with the legacy `nano` model used by `balanced/detail`.
Legacy `nano` already has a CNZ4 bitstream path. The new `hyper_ms_nano`
model has the better official-style mean-scale hyperprior structure, but it
needs a new CNZ5 bitstream before production entropy coding is complete.

## Current Flow

Training and PyTorch evaluation currently do this:

```text
x
 -> g_a encoder
 -> y
 -> h_a hyper encoder
 -> z
 -> quantize z -> z_hat
 -> h_s hyper decoder
 -> scales_y, means_y
 -> centered quantize y with means_y
 -> y_hat
 -> g_s decoder
 -> x_hat
```

For RK3588 deployment the split should be:

```text
RK3588 NPU:
  x -> g_a/h_a/h_s analysis RKNN -> y, z, scales_y, means_y

RK3588 CPU reference path:
  z -> z_symbols
  y + means_y -> y_symbols
  package z_symbols + y_symbols

Future production CPU path:
  entropy-code z_symbols
  entropy-code y_symbols with the Gaussian mean/scale contract
  write CNZ5

Decode side:
  read z_symbols -> z_hat
  h_s(z_hat) -> scales_y, means_y
  read y_symbols + means_y -> y_hat
  g_s(y_hat) -> x_hat
```

The old CNZ4 C++ tools only support the legacy factorized `nano` stream:

```text
image -> encoder -> y -> C++ CNZ4
```

They do not support mean-scale hyperprior because CNZ4 has no `z` stream and no
`means_y/scales_y` contract. For `hyper_ms_nano`, use the reference RK script
below until CNZ5 is implemented.

## 1. Export ONNX On PC

Use a trained `hyper_ms_nano_fp` checkpoint. Use `best.pt` for quality tests or
`latest.pt` for quick iteration.

```bash
/home/zzw/miniconda3/envs/net/bin/python tools/export_encoder_onnx.py \
  --checkpoint checkpoints_hyper_ms_nano_fp/best.pt \
  --output dist/rk3588_hyper_ms_nano/analysis_hyper_ms_nano_1280x768.onnx \
  --export-mode analysis \
  --height 768 \
  --width 1280 \
  --opset 12
```

`--export-mode analysis` exports:

```text
input -> y, z, scales_y, means_y
```

The export wrapper avoids likelihood-only operators and uses:

```text
z -> round/dequantize -> h_s
```

so RKNN does not need to carry the training likelihood branch.

For a fixed RKNN input, prefer static shapes. For 720p images, pad to
`1280x768`; the hyperprior path needs dimensions aligned so `y` and
`means/scales` match.

## 2. Export Hyper Entropy Parameters

```bash
/home/zzw/miniconda3/envs/net/bin/python tools/export_entropy_params.py \
  --checkpoint checkpoints_hyper_ms_nano_fp/best.pt \
  --output dist/rk3588_hyper_ms_nano/hyper_ms_nano_entropy_params.json
```

For `hyper_ms_nano`, this JSON contains:

```text
model_variant
channels_y
channels_z
quant_step_y
quant_step_z
z_medians
scale_min / scale_max
has_means_y=true
```

The old C++ CNZ4 encoder cannot consume this JSON for production compression.
The RK reference script uses it to quantize `z` and `y` correctly.

## 3. Convert ONNX To RKNN On PC

Install RKNN-Toolkit2 in your RK conversion environment, then run:

```bash
python tools/convert_onnx_to_rknn.py \
  --onnx dist/rk3588_hyper_ms_nano/analysis_hyper_ms_nano_1280x768.onnx \
  --output dist/rk3588_hyper_ms_nano/analysis_hyper_ms_nano_1280x768_fp16.rknn \
  --target-platform rk3588 \
  --height 768 \
  --width 1280
```

This builds a non-INT8 RKNN model. In RKNN workflows this is the FP/FP16 path.
Use it first to validate quality and operator support.

INT8 conversion is a later step:

```bash
python tools/convert_onnx_to_rknn.py \
  --onnx dist/rk3588_hyper_ms_nano/analysis_hyper_ms_nano_1280x768.onnx \
  --output dist/rk3588_hyper_ms_nano/analysis_hyper_ms_nano_1280x768_int8.rknn \
  --target-platform rk3588 \
  --height 768 \
  --width 1280 \
  --do-quantization \
  --dataset calibration_images.txt
```

Only do this after the FP/FP16 RKNN output is close to the PyTorch output.

## 4. Copy Files To RK3588

Copy these files to the board:

```text
dist/rk3588_hyper_ms_nano/analysis_hyper_ms_nano_1280x768_fp16.rknn
dist/rk3588_hyper_ms_nano/hyper_ms_nano_entropy_params.json
rk3588_media_compress/rknn_hyper_ms_nano_compress.py
```

Install runtime dependencies on RK3588:

```bash
pip install pillow numpy
```

Install the RKNN Lite runtime package that matches your board image and RKNN
Toolkit2 version. The script imports:

```python
from rknnlite.api import RKNNLite
```

## 5. Run Reference Compression On RK3588

```bash
python3 rk3588_media_compress/rknn_hyper_ms_nano_compress.py \
  --rknn analysis_hyper_ms_nano_1280x768_fp16.rknn \
  --params hyper_ms_nano_entropy_params.json \
  --image test.jpg \
  --output test.hyper_ms_nano_ref.npz \
  --height 768 \
  --width 1280 \
  --core-mask all
```

The script prints:

```text
input_shape
y_shape
z_shape
raw_symbol_bpp
package_bpp
timing_npu_inference_ms
timing_total_ms
```

The `.npz` package stores:

```text
y_symbols
z_symbols
metadata
```

This is a reference package, not the final production bitstream. It is useful
for validating RKNN output shape, quantization ranges, symbol dtype, estimated
bpp, and NPU speed before writing CNZ5.

For debugging, also save float outputs:

```bash
python3 rk3588_media_compress/rknn_hyper_ms_nano_compress.py \
  --rknn analysis_hyper_ms_nano_1280x768_fp16.rknn \
  --params hyper_ms_nano_entropy_params.json \
  --image test.jpg \
  --output test.hyper_ms_nano_ref.npz \
  --height 768 \
  --width 1280 \
  --core-mask all \
  --keep-float-outputs debug_hyper_ms_outputs
```

This writes:

```text
debug_hyper_ms_outputs/y.bin
debug_hyper_ms_outputs/z.bin
debug_hyper_ms_outputs/scales_y.bin
debug_hyper_ms_outputs/means_y.bin
```

## What Is Python And What Is C++ Today?

### Legacy `detail/nano`

This path is already split:

```text
Python/RKNNLite:
  preprocess image
  run RKNN encoder
  save latent.bin

C++:
  quantize y with medians/quant_step
  pack int16/int32 symbols
  zlib payload
  write CNZ4

PC/Python:
  CNZ4 -> y_hat
  PyTorch decoder -> image
```

So yes: for the old `nano`, Python orchestrates preprocessing and RKNN
inference; C++ does the CNZ4 entropy packaging.

### New `hyper_ms_nano`

Current state:

```text
Python/RKNNLite:
  preprocess image
  run RKNN analysis
  get y, z, scales_y, means_y

Python reference postprocess:
  z -> z_symbols
  y + means_y -> y_symbols
  write .npz reference package
```

C++ entropy coding for this model is not finished yet. The missing production
piece is CNZ5:

```text
CNZ5 header:
  model_variant
  original/padded sizes
  y shape and z shape
  quant_step_y, quant_step_z or ids
  stream descriptors

CNZ5 payload:
  z_symbols stream
  y_symbols stream
```

The decoder must reconstruct `means_y/scales_y` from `z_hat`, then dequantize
`y_symbols` around `means_y`.

## Practical Recommendation

Use this order:

1. Train `hyper_ms_nano_fp`.
2. Test PyTorch with `roundtrip_image.py --mode forward`.
3. Export analysis ONNX.
4. Convert FP/FP16 RKNN.
5. Run `rknn_hyper_ms_nano_compress.py` on RK3588.
6. Compare RKNN shapes, symbol ranges, package bpp, and speed.
7. Only after quality and speed are acceptable, implement CNZ5 C++ entropy
   coding.

