# nano_hyper_ms_q

`nano_hyper_ms_q` is the current high-precision hyperprior route. It keeps the
project's lightweight deployment goals, but follows the official CompressAI
mean-scale hyperprior shape instead of the earlier scale-only experiment. The
old `nano_hyper_residual_q` model is retained as a baseline through the
`hyper_quality_*` profiles; new high-precision work should use `hyper_ms_*`.

For the current high-precision training route, start with
[`high_precision_training.md`](high_precision_training.md). That runbook trains
`nano_hyper_ms_q` through `hyper_ms_mini_fp`, `hyper_ms_mini_hq`, and
`hyper_ms_mini_qat8`.

## Goals

- Improve reconstruction of highlights, water texture, line art, and fine local
  contrast over the lightweight factorized nano model.
- Keep the RK3588 analysis side friendly to NPU quantization and mixed
  precision.
- Support quality-first mixed QAT: fake-quantize the main `y` latent while
  keeping the hyperprior `z`, `means_y`, and `scales_y` in FP/FP16.
- Keep ReLU6 in the hidden INT8-friendly encoder blocks while using a signed
  clipped final `y` latent output for better PSNR headroom.
- Prepare clean ONNX exports for RKNN hybrid quantization experiments.

## Differences From `nano`

`nano` keeps the original factorized-prior path:

```text
x -> g_a -> y -> factorized entropy -> y_hat -> g_s -> x_hat
```

`nano_hyper_ms_q` adds a residual encoder and a mean-scale hyperprior:

```text
x -> g_a -> y
       y -> h_a -> z -> entropy bottleneck -> z_hat -> h_s -> scales_y, means_y
       y + means_y + scales_y -> centered Gaussian conditional -> y_hat -> g_s -> x_hat
```

The older `nano_hyper_residual_q` route predicts scales only and is kept as a
comparison baseline. The new main route predicts means and scales so high bpp
can be converted into useful reconstruction quality instead of only wider
scale estimates.

## Residual Encoder

The analysis transform uses RKNN-friendly blocks:

- `DownsampleResidualBlock`: stride-2 3x3 main branch plus 1x1 stride-2 skip.
- `QuantResidualBlock`: 3x3, ReLU6, 3x3, add, ReLU6.
- No GDN, attention, LayerNorm, GroupNorm, or dynamic-shape operators.
- No BatchNorm in the new encoder.
- `latent_clip * tanh(y / latent_clip)` with default `latent_clip=6.0`.

Recommended mini config:

```text
N=160, M=256, Z=160
quant_step=0.35 FP / 0.30 HQ-QAT
decoder_channels=320
decoder_res_blocks=4
refinement_blocks=6
activation=relu6
encoder_norm=none
latent_clip=6.0
z_clip=6.0
scale_min=1e-3
scale_max=20.0
```

The narrower speed config is `nano_hyper_ms_q_nano` with
`N=128, M=192, Z=128`.

## Mean-Scale Hyperprior

`h_a` maps `y -> z`; `z` is clipped with `z_clip * tanh(z / z_clip)`.

`h_s` maps `z_hat -> (scales_y, means_y)`. Means are raw Conv outputs. Scales
use:

```python
scale = softplus(raw) + scale_min
scale = scale.clamp(scale_min, scale_max)
```

This is intentionally simple for the first version. It improves spatial rate
allocation without adding autoregressive context, ELIC-style context models, or
heavy attention.

## QAT

QAT is disabled for FP profiles. The mean-scale mixed-QAT profile fake-quantizes
only:

- `y` latent.

It keeps `z`, `means_y`, and `scales_y` in FP/FP16. This keeps the hyperprior
stable while the main transform learns the INT8 deployment error.

Available fake-quant flags:

```text
--enable-latent-fake-quant
--latent-fake-quant-bits 8
--latent-fake-quant-clip 6.0

--enable-z-fake-quant
--z-fake-quant-bits 8
--z-fake-quant-clip 6.0

--enable-scale-fake-quant
--scale-fake-quant-bits 8
--scale-fake-quant-clip 8.0
```

Optional range regularizers are available:

```text
--latent-range-weight
--z-range-weight
--symbol-range-weight
--scale-range-weight
```

## Why Mixed Precision

Full INT8 is a useful failure baseline, but it is not the expected deployment
target. The recommended RKNN hybrid quantization split is:

- ordinary `g_a` Conv/residual layers: INT8 candidates,
- final `y` output layer: FP16,
- final `z` output from `h_a`: FP16,
- `h_s` all layers or at least final scale layers: FP16,
- `scales_y` output: FP16.

This keeps quantization noise away from the tensors that directly determine
entropy coding and reconstruction.

## Training Flow

Stage 1: FP mean-scale training. Export this checkpoint for FP16/RKNN-FP16
experiments.

```bash
python train.py \
  --quality-profile hyper_ms_mini_fp \
  --train-dir data/train \
  --val-dir data/val \
  --checkpoint-dir checkpoints_hyper_ms_mini_fp
```

Stage 2: quality fine-tuning

```bash
python train.py \
  --quality-profile hyper_ms_mini_hq \
  --train-dir data/train \
  --val-dir data/val \
  --init-checkpoint checkpoints_hyper_ms_mini_fp/best.pt \
  --checkpoint-dir checkpoints_hyper_ms_mini_hq
```

Stage 3: mixed INT8/FP16 QAT fine-tuning

```bash
python train.py \
  --quality-profile hyper_ms_mini_qat8 \
  --train-dir data/train \
  --val-dir data/val \
  --init-checkpoint checkpoints_hyper_ms_mini_hq/best.pt \
  --checkpoint-dir checkpoints_hyper_ms_mini_qat8
```

Stage 3: RKNN mixed precision exploration

1. Export FP ONNX.
2. Convert FP RKNN and confirm quality.
3. Convert the mixed-QAT path as the quantized target.
4. Use RKNN hybrid quantization where needed:
   - ordinary Conv/residual layers INT8,
   - signed `y` latent fake-quantized during training,
   - `z` output FP16,
   - `h_s`, `means_y`, and `scales_y` FP16.
5. Compare PyTorch, RKNN FP, and RKNN mixed latent statistics and reconstructions.

## ONNX Export

Export only `image -> y`:

```bash
python tools/export_encoder_onnx.py \
  --checkpoint checkpoints_hyper_ms_mini_qat8/best.pt \
  --output encoder_hyper_ms_y.onnx \
  --height 768 \
  --width 1280
```

Export analysis side `image -> y, z, scales_y, means_y`:

```bash
python tools/export_encoder_onnx.py \
  --checkpoint checkpoints_hyper_ms_mini_qat8/best.pt \
  --output analysis_hyper_ms.onnx \
  --export-mode analysis \
  --height 768 \
  --width 1280
```

## Deployment Status

Current CNZ4 support remains unchanged and is for the old `nano` model:

- one `y` latent stream,
- factorized entropy parameters,
- zlib/CNZ container.

`nano_hyper_ms_q` does not yet support full CNZ deployment. It needs a
new bitstream version, suggested as CNZ5, with:

- `model_variant` in the header,
- `z` stream,
- `y` stream,
- `z` entropy parameters,
- hyperprior shape,
- decoder-side `h_s(z_hat)` to reconstruct `means_y` and `scales_y`,
- conditional y decoding with `means_y` and `scales_y`.

Until CNZ5 exists, use this variant for PyTorch training, ONNX export, RKNN FP
validation, and RKNN mixed precision analysis-side experiments.
