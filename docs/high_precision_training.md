# High-Precision Training Runbook

This document is the operating guide for the current high-precision route:
`nano_hyper_residual_q`.

This route uses both pieces of the new architecture:

- a quantization-friendly residual encoder built from
  `DownsampleResidualBlock` and `QuantResidualBlock`;
- a scale-only hyperprior: `h_a` predicts `z`, `h_s` reconstructs `scales_y`,
  and `y` is modeled with Gaussian conditional entropy.

The training path is:

```text
x -> residual g_a -> y
       y -> h_a -> z -> entropy bottleneck -> z_hat -> h_s -> scales_y
       y + scales_y -> Gaussian conditional entropy -> y_hat -> g_s -> x_hat
```

The old `detail` and `detail_peak` profiles are legacy `nano` fine-tuning
profiles. They do not use hyperprior or the new residual encoder.

## Three Precision Stages

The current high-precision route has three stages:

```text
hyper_quality_fp:
  full-precision training; fake quant disabled

hyper_quality_qat16:
  fine-tune from FP; fake quant y/z/scales_y to 16 bits

hyper_quality_qat8:
  fine-tune from QAT16; fake quant y/z/scales_y to 8 bits
```

All three profiles select:

```text
model_variant=nano_hyper_residual_q
N=128
M=160
Z=96
quant_step=0.45
decoder_res_blocks=4
refinement_blocks=6
activation=relu6
latent_clip=6.0
z_clip=6.0
```

Do not initialize this model from `checkpoints_detail/*.pt` or
`checkpoints_balanced/*.pt`. Those checkpoints are for the old `nano` model
with `M=128` and no hyperprior. Start `hyper_quality_fp` from scratch unless
you already have a `nano_hyper_residual_q` checkpoint.

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

The expected local GPU layout is three RTX 4090 cards:

```bash
nvidia-smi
```

The `hyper_quality_*` profiles use LPIPS by default (`lpips_weight=0.002`), so
the `lpips` package must import in this environment.

## Stage 1: FP Training

Start here:

```bash
cd /home/zzw/workspace/compressai-nano

CUDA_VISIBLE_DEVICES=0,1,2 /home/zzw/miniconda3/envs/net/bin/torchrun \
  --standalone \
  --nproc_per_node=3 \
  train.py \
  --quality-profile hyper_quality_fp \
  --train-dir data/train \
  --val-dir data/val \
  --checkpoint-dir checkpoints_hyper_quality_fp \
  --checkpoint-interval-steps 100 \
  --eval-interval-steps 100 \
  --max-steps 8000 \
  --num-workers 8
```

This stage uses full precision for `y`, `z`, and `scales_y`.

## Stage 2: QAT16 Fine-Tuning

Run this after stage 1 has produced `checkpoints_hyper_quality_fp/best.pt`:

```bash
cd /home/zzw/workspace/compressai-nano

CUDA_VISIBLE_DEVICES=0,1,2 /home/zzw/miniconda3/envs/net/bin/torchrun \
  --standalone \
  --nproc_per_node=3 \
  train.py \
  --quality-profile hyper_quality_qat16 \
  --init-checkpoint checkpoints_hyper_quality_fp/best.pt \
  --train-dir data/train \
  --val-dir data/val \
  --checkpoint-dir checkpoints_hyper_quality_qat16 \
  --checkpoint-interval-steps 100 \
  --eval-interval-steps 100 \
  --max-steps 3000 \
  --num-workers 8
```

This enables fake quant for:

```text
y: 16-bit symmetric fake quant, clip=6.0
z: 16-bit symmetric fake quant, clip=6.0
scales_y: 16-bit positive fake quant, clip=8.0
```

## Stage 3: QAT8 Fine-Tuning

Run this after stage 2 has produced `checkpoints_hyper_quality_qat16/best.pt`:

```bash
cd /home/zzw/workspace/compressai-nano

CUDA_VISIBLE_DEVICES=0,1,2 /home/zzw/miniconda3/envs/net/bin/torchrun \
  --standalone \
  --nproc_per_node=3 \
  train.py \
  --quality-profile hyper_quality_qat8 \
  --init-checkpoint checkpoints_hyper_quality_qat16/best.pt \
  --train-dir data/train \
  --val-dir data/val \
  --checkpoint-dir checkpoints_hyper_quality_qat8 \
  --checkpoint-interval-steps 100 \
  --eval-interval-steps 100 \
  --max-steps 2000 \
  --num-workers 8
```

This enables fake quant for:

```text
y: 8-bit symmetric fake quant, clip=6.0
z: 8-bit symmetric fake quant, clip=6.0
scales_y: 8-bit positive fake quant, clip=8.0
```

## Resume

Use `--resume` only when continuing the same stage, because it restores the
optimizer, scheduler, epoch, and global step. Example for interrupted stage 1:

```bash
cd /home/zzw/workspace/compressai-nano

CUDA_VISIBLE_DEVICES=0,1,2 /home/zzw/miniconda3/envs/net/bin/torchrun \
  --standalone \
  --nproc_per_node=3 \
  train.py \
  --quality-profile hyper_quality_fp \
  --resume checkpoints_hyper_quality_fp/latest.pt \
  --train-dir data/train \
  --val-dir data/val \
  --checkpoint-dir checkpoints_hyper_quality_fp \
  --checkpoint-interval-steps 100 \
  --eval-interval-steps 100 \
  --max-steps 9000 \
  --num-workers 8
```

Increase `--max-steps` beyond the saved `global_step`. If the checkpoint has
already reached the requested max step, the script exits without training.

## Batch Size And Steps

In DDP mode, `--batch-size` is per GPU. The `hyper_quality_*` profiles use
`batch_size=24`, so three GPUs train with a global batch of `72`.

With the current local split of about 40001 train images, one epoch is about
556 optimizer steps. The suggested starts are:

```text
hyper_quality_fp:    8000 steps, about 14.4 epochs
hyper_quality_qat16: 3000 steps, about 5.4 epochs
hyper_quality_qat8:  2000 steps, about 3.6 epochs
```

If memory is tight:

```bash
--batch-size 16
```

If data loading is the bottleneck, try `--num-workers 10` or `--num-workers 12`.
The value is per rank, so `--num-workers 8` starts 24 loader workers across
three ranks.

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

## Export After Training

Export only `image -> y`:

```bash
/home/zzw/miniconda3/envs/net/bin/python tools/export_encoder_onnx.py \
  --checkpoint checkpoints_hyper_quality_qat8/best.pt \
  --output encoder_hyper_y.onnx \
  --height 720 \
  --width 1280
```

Export analysis side `image -> y, z, scales_y`:

```bash
/home/zzw/miniconda3/envs/net/bin/python tools/export_encoder_onnx.py \
  --checkpoint checkpoints_hyper_quality_qat8/best.pt \
  --output analysis_hyper.onnx \
  --export-mode analysis \
  --height 720 \
  --width 1280
```

`nano_hyper_residual_q` currently supports training, ONNX export, RKNN FP
validation, and RKNN mixed-precision analysis. It does not yet support CNZ4
encode/decode. Full bitstream deployment needs a future CNZ5 format carrying
`z`, `y`, hyperprior shape, entropy parameters, and `model_variant`.

## PyTorch Roundtrip

Use `roundtrip_image.py` in `forward` mode to inspect the new hyperprior model
before CNZ5 exists:

```bash
/home/zzw/miniconda3/envs/net/bin/python roundtrip_image.py \
  samples/test.jpg \
  --checkpoint checkpoints_hyper_quality_fp/latest.pt \
  --mode forward \
  --output-dir roundtrip_hyper_test \
  --timing
```

This runs:

```text
image -> residual encoder -> y/z/scales_y -> y_hat -> decoder -> recon.png
```

It also prints `bpp_y`, `bpp_z`, `estimated_bpp`, and the shapes of `y`, `z`,
and `scales_y`. This is not a real bitstream roundtrip; `--mode cnz4` is only
for old `nano` checkpoints that support CNZ4.
