# High-Precision Training Runbook

This document is the operating guide for the current high-precision route:
`nano_hyper_ms_q`.

This route uses:

- a quantization-friendly residual encoder built from
  `DownsampleResidualBlock` and `QuantResidualBlock`;
- a mean-scale hyperprior where `h_a` predicts `z`, `h_s` reconstructs
  `scales_y` and `means_y`, and `y` is modeled with centered Gaussian
  conditional entropy.

The training path is:

```text
x -> residual g_a -> y
       y -> h_a -> z -> entropy bottleneck -> z_hat -> h_s -> scales_y, means_y
       y + means_y + scales_y -> Gaussian conditional entropy -> y_hat -> g_s -> x_hat
```

The `detail` profile is the legacy `nano` CNZ4 fine-tuning profile. It does not
use the hyperprior or the residual encoder.

## Precision Stages

The current route keeps three precision stages:

```text
hyper_ms_mini_fp:
  full-precision mean-scale baseline; fake quant disabled

hyper_ms_mini_hq:
  quality-first fine-tune from FP; higher MSE/highlight priority

hyper_ms_mini_qat8:
  mixed-QAT from high-quality FP; fake quant y to 8 bits, keep z/means/scales FP
```

The mini profiles select:

```text
model_variant=nano_hyper_ms_q
N=160
M=256
Z=160
quant_step=0.35 for FP, 0.30 for HQ/QAT
decoder_res_blocks=4
refinement_blocks=6
activation=relu6
latent_clip=6.0
z_clip=6.0
```

Do not initialize this model from `checkpoints_detail/*.pt` or
`checkpoints_balanced/*.pt`. Those checkpoints are for the old `nano` model
with `M=128` and no hyperprior. Start `hyper_ms_mini_fp` from scratch unless
you already have a matching `nano_hyper_ms_q` checkpoint.

## Local Environment

The training environment on this workstation is `net`:

```bash
source /home/zzw/miniconda3/bin/activate net
```

You can also call the tools directly without activating:

```bash
/home/zzw/miniconda3/envs/net/bin/python
/home/zzw/miniconda3/envs/net/bin/torchrun
```

## Stage 1: FP Mean-Scale Training

```bash
cd /home/zzw/workspace/compressai-nano

CUDA_VISIBLE_DEVICES=0,1,2 /home/zzw/miniconda3/envs/net/bin/torchrun \
  --standalone \
  --nproc_per_node=3 \
  train.py \
  --quality-profile hyper_ms_mini_fp \
  --train-dir data/train \
  --val-dir data/val \
  --checkpoint-dir checkpoints_hyper_ms_mini_fp \
  --checkpoint-interval-steps 100 \
  --eval-interval-steps 100 \
  --max-steps 8000 \
  --num-workers 4
```

Resume the same FP stage by increasing `--max-steps` and using `--resume`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 /home/zzw/miniconda3/envs/net/bin/torchrun \
  --standalone \
  --nproc_per_node=3 \
  train.py \
  --quality-profile hyper_ms_mini_fp \
  --resume checkpoints_hyper_ms_mini_fp/latest.pt \
  --train-dir data/train \
  --val-dir data/val \
  --checkpoint-dir checkpoints_hyper_ms_mini_fp \
  --checkpoint-interval-steps 100 \
  --eval-interval-steps 100 \
  --max-steps 12000 \
  --num-workers 4
```

## Stage 2: Quality Fine-Tuning

Run this only after Stage 1 has produced a useful `best.pt`:

```bash
cd /home/zzw/workspace/compressai-nano

CUDA_VISIBLE_DEVICES=0,1,2 /home/zzw/miniconda3/envs/net/bin/torchrun \
  --standalone \
  --nproc_per_node=3 \
  train.py \
  --quality-profile hyper_ms_mini_hq \
  --init-checkpoint checkpoints_hyper_ms_mini_fp/best.pt \
  --train-dir data/train \
  --val-dir data/val \
  --checkpoint-dir checkpoints_hyper_ms_mini_hq \
  --checkpoint-interval-steps 100 \
  --eval-interval-steps 100 \
  --max-steps 3000 \
  --num-workers 4
```

## Stage 3: Mixed INT8/FP16 QAT Fine-Tuning

Run this after Stage 2 has produced `checkpoints_hyper_ms_mini_hq/best.pt`:

```bash
cd /home/zzw/workspace/compressai-nano

CUDA_VISIBLE_DEVICES=0,1,2 /home/zzw/miniconda3/envs/net/bin/torchrun \
  --standalone \
  --nproc_per_node=3 \
  train.py \
  --quality-profile hyper_ms_mini_qat8 \
  --init-checkpoint checkpoints_hyper_ms_mini_hq/best.pt \
  --train-dir data/train \
  --val-dir data/val \
  --checkpoint-dir checkpoints_hyper_ms_mini_qat8 \
  --checkpoint-interval-steps 100 \
  --eval-interval-steps 100 \
  --max-steps 2000 \
  --num-workers 4
```

This profile is quality-first and matches the planned mixed precision split:

```text
y: 8-bit symmetric fake quant, clip=6.0
z: FP/FP16, fake quant disabled
means_y/scales_y: FP/FP16, fake quant disabled
```

The encoder keeps ReLU6 in the hidden residual blocks for INT8 friendliness,
but the final `y` latent is signed and clipped. This model must be trained from
scratch or from matching `nano_hyper_ms_q` checkpoints; do not warm-start it
from the old scale-only `nano_hyper_residual_q` checkpoints.

## Batch Size And Steps

In DDP mode, `--batch-size` is per GPU. The `hyper_ms_mini_*` profiles use
`batch_size=12`, so three GPUs train with a global batch of `36`.

With the current local split of about 40001 train images, one epoch is about
556 optimizer steps:

```text
hyper_ms_mini_fp:   8000 steps, about 7.2 epochs
hyper_ms_mini_hq:   3000 steps, about 2.7 epochs
hyper_ms_mini_qat8: 2000 steps, about 1.8 epochs
```

## Encoder Complexity

The complexity tool counts Conv/Deconv parameters and FLOPs. FLOPs are reported
as `2 * MACs`. FP16 and INT8 have the same parameter count and FLOPs; their
weight storage differs.

For the legacy `nano` model used by `balanced` and `detail`, the encoder is the
same network for both profiles. The checkpoint changes quality/rate behavior,
but not encoder parameter count or FLOPs.

```bash
/home/zzw/miniconda3/envs/net/bin/python tools/encoder_complexity.py \
  --model-variant nano \
  --height 720 \
  --width 1280 \
  --mode both
```

Current legacy `nano` result at `720x1280`:

```text
encoder_y / analysis_y_only:
  params: 1,238,912
  FP16 param size: 2.363 MiB
  INT8 param size: 1.182 MiB
  MACs: 33.178 GMACs
  FLOPs: 66.355 GFLOPs
```

For 720p content the hyperprior analysis path should be padded to `768x1280`
so `y`, `scales_y`, and `means_y` have matching spatial shapes.

```bash
/home/zzw/miniconda3/envs/net/bin/python tools/encoder_complexity.py \
  --height 768 \
  --width 1280 \
  --mode both
```

Recalculate after choosing mini or nano width:

```text
encoder_y: image -> y
analysis_y_z_scales_means: image -> (y, z, scales_y, means_y)
```

`encoder_y` is `image -> y`. `analysis_y_z_scales_means` is
`image -> (y, z, scales_y, means_y)` for the mean-scale route.

## Metrics To Watch

Primary metrics:

```text
val_loss
val_bpp
val_bpp_y
val_bpp_z
val_mse
val_ssim
```

Hyper/QAT metrics:

```text
val_latent_y_p99
val_latent_z_p99
val_symbol_y_p99_abs
val_symbol_z_p99_abs
val_scale_mean
val_mean_y_std
val_fake_quant_y_error
val_fake_quant_z_error
val_fake_quant_scale_error
```

Detail metrics still matter because the loss keeps highlight and texture terms:

```text
val_peak_under
val_highlight_lap
val_highlight_contrast
val_lpips_loss
```

## Export

Export the FP checkpoint for FP16/RKNN-FP16 conversion:

```bash
/home/zzw/miniconda3/envs/net/bin/python tools/export_encoder_onnx.py \
  --checkpoint checkpoints_hyper_ms_mini_fp/best.pt \
  --output encoder_hyper_ms_fp16_y.onnx \
  --height 768 \
  --width 1280
```

Export the INT8/QAT8 checkpoint:

```bash
/home/zzw/miniconda3/envs/net/bin/python tools/export_encoder_onnx.py \
  --checkpoint checkpoints_hyper_ms_mini_qat8/best.pt \
  --output encoder_hyper_ms_int8_y.onnx \
  --height 768 \
  --width 1280

/home/zzw/miniconda3/envs/net/bin/python tools/export_encoder_onnx.py \
  --checkpoint checkpoints_hyper_ms_mini_qat8/best.pt \
  --output analysis_hyper_ms_int8.onnx \
  --export-mode analysis \
  --height 768 \
  --width 1280
```

`nano_hyper_ms_q` currently supports training, ONNX export, RKNN FP
validation, and RKNN mixed-precision analysis. It does not yet support CNZ4
encode/decode. Full bitstream deployment needs a future CNZ5 format carrying
`z`, `y`, `means_y/scales_y`, hyperprior shape, entropy parameters, and
`model_variant`.

## PyTorch Roundtrip

Use `roundtrip_image.py` in `forward` mode to inspect the new hyperprior model
before CNZ5 exists:

```bash
/home/zzw/miniconda3/envs/net/bin/python roundtrip_image.py \
  samples/test.jpg \
  --checkpoint checkpoints_hyper_ms_mini_fp/latest.pt \
  --mode forward \
  --output-dir roundtrip_hyper_ms_test \
  --timing
```

This is not a real bitstream roundtrip; `--mode cnz4` is only for old `nano`
checkpoints that support CNZ4.
