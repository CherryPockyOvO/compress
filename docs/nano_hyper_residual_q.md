# nano_hyper_residual_q

`nano_hyper_residual_q` is an independent high-precision model variant for the
existing nano codec project. The default model is still `nano`; this variant is
enabled only with `--model-variant nano_hyper_residual_q` or with one of the
`hyper_quality_*` training profiles.

For the current three-stage high-precision training route, start with
[`high_precision_training.md`](high_precision_training.md). That runbook trains
this `nano_hyper_residual_q` model through `hyper_quality_fp`,
`hyper_quality_qat16`, and `hyper_quality_qat8`.

## Goals

- Improve reconstruction of highlights, water texture, line art, and fine local
  contrast over the lightweight factorized nano model.
- Keep the RK3588 analysis side friendly to NPU quantization and mixed
  precision.
- Support staged QAT for latent, hyper-latent, and scale tensors without forcing
  every intermediate activation through fake quant.
- Prepare clean ONNX exports for RKNN hybrid quantization experiments.

## Differences From `nano`

`nano` keeps the original factorized-prior path:

```text
x -> g_a -> y -> factorized entropy -> y_hat -> g_s -> x_hat
```

`nano_hyper_residual_q` adds a residual encoder and a scale-only hyperprior:

```text
x -> g_a -> y
       y -> h_a -> z -> entropy bottleneck -> z_hat -> h_s -> scales_y
       y + scales_y -> Gaussian conditional entropy -> y_hat -> g_s -> x_hat
```

The first hyperprior version predicts scales only. Mean prediction is reserved
for a later version because mean errors are more sensitive during INT8/mixed
precision deployment.

## Residual Encoder

The analysis transform uses RKNN-friendly blocks:

- `DownsampleResidualBlock`: stride-2 3x3 main branch plus 1x1 stride-2 skip.
- `QuantResidualBlock`: 3x3, ReLU6, 3x3, add, ReLU6.
- No GDN, attention, LayerNorm, GroupNorm, or dynamic-shape operators.
- No BatchNorm in the new encoder.
- `latent_clip * tanh(y / latent_clip)` with default `latent_clip=6.0`.

Recommended config:

```text
N=128, M=160, Z=96
quant_step=0.45
decoder_channels=256
decoder_res_blocks=4
refinement_blocks=6
activation=relu6
encoder_norm=none
latent_clip=6.0
z_clip=6.0
scale_min=1e-3
scale_max=20.0
```

## Scale-Only Hyperprior

`h_a` maps `y -> z`; `z` is clipped with `z_clip * tanh(z / z_clip)`.

`h_s` maps `z_hat -> scales_y`; scales use:

```python
scale = softplus(raw) + scale_min
scale = scale.clamp(scale_min, scale_max)
```

This is intentionally simple for the first version. It improves spatial rate
allocation without adding autoregressive context, ELIC-style context models, or
heavy attention.

## QAT

QAT is disabled by default. The first implementation fake-quantizes only:

- `y` latent,
- `z` hyper-latent,
- `scales_y`.

It does not fake-quantize every activation. This keeps FP training stable and
lets RKNN mixed precision decide which Conv/residual layers should become INT8.

Available flags:

```text
--enable-latent-fake-quant
--latent-fake-quant-bits 8|16
--latent-fake-quant-clip 6.0

--enable-z-fake-quant
--z-fake-quant-bits 8|16
--z-fake-quant-clip 6.0

--enable-scale-fake-quant
--scale-fake-quant-bits 8|16
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

Stage 1: FP training

```bash
python train.py \
  --quality-profile hyper_quality_fp \
  --train-dir data/train \
  --val-dir data/val \
  --checkpoint-dir checkpoints_hyper_quality_fp
```

Stage 2: QAT16 fine-tuning

```bash
python train.py \
  --quality-profile hyper_quality_qat16 \
  --train-dir data/train \
  --val-dir data/val \
  --init-checkpoint checkpoints_hyper_quality_fp/best.pt \
  --checkpoint-dir checkpoints_hyper_quality_qat16
```

Stage 3: QAT8 fine-tuning

```bash
python train.py \
  --quality-profile hyper_quality_qat8 \
  --train-dir data/train \
  --val-dir data/val \
  --init-checkpoint checkpoints_hyper_quality_qat16/best.pt \
  --checkpoint-dir checkpoints_hyper_quality_qat8
```

Stage 4: RKNN mixed precision exploration

1. Export FP ONNX.
2. Convert FP RKNN and confirm quality.
3. Convert full INT8 as a failure/control baseline.
4. Use RKNN hybrid quantization:
   - ordinary Conv/residual layers INT8,
   - `y` output FP16,
   - `z` output FP16,
   - `h_s` and `scales_y` FP16.
5. Compare PyTorch, RKNN FP, and RKNN mixed latent statistics and reconstructions.

## ONNX Export

Export only `image -> y`:

```bash
python tools/export_encoder_onnx.py \
  --checkpoint checkpoints_hyper_quality_qat8/best.pt \
  --output encoder_hyper_y.onnx \
  --height 720 \
  --width 1280
```

Export analysis side `image -> y, z, scales_y`:

```bash
python tools/export_encoder_onnx.py \
  --checkpoint checkpoints_hyper_quality_qat8/best.pt \
  --output analysis_hyper.onnx \
  --export-mode analysis \
  --height 720 \
  --width 1280
```

## Deployment Status

Current CNZ4 support remains unchanged and is for the old `nano` model:

- one `y` latent stream,
- factorized entropy parameters,
- zlib/CNZ container.

`nano_hyper_residual_q` does not yet support full CNZ deployment. It needs a
new bitstream version, suggested as CNZ5, with:

- `model_variant` in the header,
- `z` stream,
- `y` stream,
- `z` entropy parameters,
- hyperprior shape,
- decoder-side `h_s(z_hat)` to reconstruct `scales_y`,
- conditional y decoding with `scales_y`.

Until CNZ5 exists, use this variant for PyTorch training, ONNX export, RKNN FP
validation, and RKNN mixed precision analysis-side experiments.
