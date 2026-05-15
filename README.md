# compressai-nano

`compress` is a stripped-down FactorizedPrior-style image compression
project derived from the CompressAI model layout. It keeps only the pieces
needed for a single-image codec and fixed-shape RK3588-friendly ONNX export.

The current architecture is asymmetric: RK3588 only runs the encoder, while the
PC runs the heavier residual decoder for better reconstruction quality.

## Directory Tree

```text
compress/
|-- README.md
|-- requirements.txt
|-- encode_image.py
|-- decode_cnz.py
|-- tools/
|-- cpp/
|-- tests/
|-- legacy/
`-- compressai_nano/
    |-- __init__.py
    |-- cnz.py
    |-- entropy.py
    |-- layers.py
    `-- models.py
```

## What Changed From CompressAI

- The model is based on `bmshj2018-factorized`.
- Encoder channels are fixed to `N=128, M=128`.
- GDN/IGDN blocks are replaced by `BatchNorm2d + ReLU`.
- The decoder is a PC-side residual reconstruction network and is intentionally
  heavier than the encoder.
- The old three-level interface has been removed. This project now keeps only
  the former level-3 high-quality configuration.
- Encoder and decoder are separate modules for ONNX export.
- The CPU entropy path is a small pure Python/PyTorch symbol codec. It is meant
  for runnable prototyping; use a trained CDF/rANS implementation for production
  bitstreams.

## Model Configuration

```python
from compressai_nano import FactorizedPriorNano

model = FactorizedPriorNano()
```

The single retained configuration uses `N=128, M=128`, `quant_step=0.67`,
`decoder_channels=256`, `decoder_res_blocks=3`, and `refinement_blocks=5`.
Old lower-quality checkpoints are not useful for this code path. Old level-3
checkpoints can still be loaded manually with `--resume` or `--checkpoint`.

The current high-precision training route is `nano_hyper_ms_q`. It follows the
CompressAI mean-scale hyperprior pattern in a RK3588-friendly mini/nano shape:
quant-friendly residual transforms, ReLU6 hidden blocks, signed clipped `y`
latents, and `h_s(z_hat) -> (scales_y, means_y)`. The deployment target is
mixed precision: main transform INT8/QAT candidates, with the hyperprior
`z/means/scales` kept FP/FP16. The old `nano_hyper_residual_q` scale-only model
is retained only as a baseline. This route does not decode the old `nano` CNZ4
bitstream; CNZ5 support is still needed for full hyperprior deployment. See
[`docs/nano_hyper_residual_q.md`](docs/nano_hyper_residual_q.md).

## Export Encoder ONNX

From this directory:

```powershell
python tools/export_encoder_onnx.py --checkpoint checkpoints/latest.pt --output encoder.onnx --height 720 --width 1280
```

This writes the RK3588-side encoder model:

```text
encoder.onnx
```

The PC-side decoder stays in PyTorch. The old full encoder+decoder ONNX export
script is archived under `legacy/`.

## Single Image PC Run

```powershell
python encode_image.py path\to\image.png --checkpoint checkpoints\latest.pt --output stream.cnz
python decode_cnz.py stream.cnz --checkpoint checkpoints\latest.pt --output recon.png
```

## C++ CNZ Roundtrip

```bash
cpp/scripts/cpp_roundtrip_test.sh \
  --latent latent.bin \
  --params entropy_params.json \
  --cnz test.cnz \
  --yhat y_hat.bin
```

This tests the C++ deployment path: `latent.bin -> CNZ4 -> y_hat.bin`.

## Train And Validate

Prepare a 1000-image split:

```powershell
python prepare_data.py --count 1000 --source unsplash-api --unsplash-access-key $env:UNSPLASH_ACCESS_KEY --threads 8 --overwrite
```

Prepare a larger mixed 5000-image compression dataset:

```powershell
python expand_dataset.py --count 5000 --threads 12
```

Add high-resolution detail patches from DIV2K/Flickr2K, with COCO as a fallback
source when more images are needed:

```powershell
python expand_hq_dataset.py --output-dir data --target-count 30000
```

On mainland China servers, avoid full dataset snapshots when possible. Download
only the HR files from ModelScope, then import the local folder:

```bash
modelscope download --dataset OmniData/DIV2K --local_dir ./data/_raw_hq/modelscope_div2k_hr --include "*DIV2K_train_HR*" "*DIV2K_valid_HR*"
python expand_hq_dataset.py --output-dir data --target-count 30000 --local-inputs ./data/_raw_hq/modelscope_div2k_hr --patches-per-local 24 --skip-remote-downloads
```

If you want to use COCO from a mainland China mirror instead of the official
COCO host:

```bash
pip install -U modelscope
python expand_hq_dataset.py --output-dir data --target-count 30000 --download-coco-cn --patch-size 384 --min-size 384 --skip-remote-downloads
```

If the ModelScope CLI reports `0 files`, the script falls back to
`MsDataset.load("COCO2017_Instance_Segmentation", split="subtrain"/"validation")`.

```powershell
python train.py --train-dir data\train --val-dir data\val --epochs 150 --batch-size 4 --num-workers 4 --lr 1e-4 --ssim-weight 0.2
python val.py --data-dir D:\data\images\test --checkpoint checkpoints\latest.pt --results-dir results
```

### High-Precision Hyperprior Training

Use this as the current high-precision route. It trains `nano_hyper_ms_q`, a
mean-scale hyperprior model distilled from the official CompressAI high-quality
route into RK3588-friendly mini/nano widths:

```text
x -> residual g_a -> y
       y -> h_a -> z -> entropy bottleneck -> z_hat -> h_s -> scales_y, means_y
       y + means_y + scales_y -> Gaussian conditional entropy -> y_hat -> g_s -> x_hat
```

This is different from the legacy `detail` route. `detail` keeps the old
`nano` factorized-prior shape and CNZ4 path; it does not use the new residual
encoder or hyperprior.

The full runbook is in
[`docs/high_precision_training.md`](docs/high_precision_training.md). On this
workstation the intended Python environment is:

```bash
source /home/zzw/miniconda3/bin/activate net
```

or call the environment tools directly:

```bash
/home/zzw/miniconda3/envs/net/bin/python
/home/zzw/miniconda3/envs/net/bin/torchrun
```

The current precision route has three stages:

1. `hyper_ms_mini_fp`: FP mean-scale baseline from scratch.
2. `hyper_ms_mini_hq`: quality-first fine-tune from the FP baseline.
3. `hyper_ms_mini_qat8`: mixed-QAT fine-tune. It fake-quantizes the main `y`
   latent to 8 bits and leaves the hyperprior `z`, `means_y`, and `scales_y`
   in FP/FP16.

Start stage 1 from scratch. Do not initialize from `checkpoints_detail/*.pt`,
because the old `nano` model uses `M=128` factorized latents while
`nano_hyper_ms_q` uses `M=256`, `Z=160`, and a mean-scale hyperprior.

```bash
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

After stage 1, run the quality-first fine-tune:

```bash
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

Then continue with the mixed INT8/FP16 QAT route:

```bash
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

In DDP mode `--batch-size` is per GPU. The `hyper_ms_mini_*` presets use
`batch_size=24`, so three GPUs train with an effective global batch of 72.
With the current local split of about 40001 train images, one epoch is about
556 optimizer steps. Only rank 0 prints progress and writes `eN.pt`,
`latest.pt`, and `best.pt`.

Encoder-side complexity for a 720p frame padded to `768x1280`:

```text
Run tools/encoder_complexity.py after choosing mini or nano width.
```

Recalculate with:

```bash
/home/zzw/miniconda3/envs/net/bin/python tools/encoder_complexity.py --height 768 --width 1280 --mode both
```

Export the high-precision analysis model after training:

```bash
/home/zzw/miniconda3/envs/net/bin/python tools/export_encoder_onnx.py --checkpoint checkpoints_hyper_ms_mini_fp/best.pt --output encoder_hyper_ms_fp16_y.onnx --height 768 --width 1280
/home/zzw/miniconda3/envs/net/bin/python tools/export_encoder_onnx.py --checkpoint checkpoints_hyper_ms_mini_qat8/best.pt --output encoder_hyper_ms_int8_y.onnx --height 768 --width 1280
/home/zzw/miniconda3/envs/net/bin/python tools/export_encoder_onnx.py --checkpoint checkpoints_hyper_ms_mini_qat8/best.pt --output analysis_hyper_ms_int8.onnx --export-mode analysis --height 768 --width 1280
```

The hyperprior model currently supports training and ONNX/RKNN analysis export.
It does not yet support CNZ4 encode/decode; full bitstream deployment needs a
future CNZ5 format carrying `z`, `y`, and the mean/scale contract.

For a PyTorch reconstruction roundtrip before CNZ5 exists:

```bash
/home/zzw/miniconda3/envs/net/bin/python roundtrip_image.py samples/test.jpg --checkpoint checkpoints_hyper_ms_mini_fp/latest.pt --mode forward --output-dir roundtrip_hyper_ms_test --timing
```

## RK3588 Deployment Path

The production deployment is split across RK3588 and PC:

```text
Python training:
  image -> Encoder -> quantize/dequantize -> Decoder -> x_hat

RK3588:
  RKNN NPU -> Encoder only -> latent y
  C++ CPU  -> quantize -> int16/int32 symbols -> zlib -> CNZ4 bitstream

PC:
  CNZ4 -> symbols -> dequantize -> PyTorch Decoder -> reconstructed image
```

Do not export the full `FactorizedPriorNano` to RKNN. Do not put Python
`pickle`, Python dictionaries, or zlib calls inside RKNN. RKNN only runs the
encoder.

### Export Encoder ONNX

```bash
python tools/export_encoder_onnx.py \
  --checkpoint checkpoints/latest.pt \
  --output encoder.onnx \
  --height 512 \
  --width 512
```

Convert `encoder.onnx` to RKNN with RKNN-Toolkit2 on your RK3588 deployment
workflow.

### Export Entropy Parameters

```bash
python tools/export_entropy_params.py \
  --checkpoint checkpoints/latest.pt \
  --output entropy_params.json
```

The JSON contains `quant_step`, `medians`, channel count, config name, and
downsampling factor. It is consumed by the C++ post-processing CLI.

### Simulate RKNN Output

```bash
python tools/dump_latent.py \
  --image test.png \
  --checkpoint checkpoints/latest.pt \
  --output latent.bin
```

`latent.bin` is float32 NCHW, batch=1, matching the C++ encoder input format.
The tool also writes `latent.bin.json` metadata by default, so the C++ encoder
can fill image and latent dimensions automatically. By default, the image is
encoded at its original pixel size and padded to the model downsampling factor.
Pass `--height` and `--width` only when you intentionally need a fixed-size
RKNN input simulation.

### Build C++ Post-Processor

```bash
cd cpp
mkdir -p build
cd build
cmake ..
make -j
```

Default C++ dependency is zlib. Optional LZ4/Zstd switches exist but are off by
default:

```bash
cmake .. -DENABLE_LZ4=OFF -DENABLE_ZSTD=OFF
```

### C++ Encode

```bash
./cnz_encode_cli \
  --latent latent.bin \
  --params ../../entropy_params.json \
  --output test.cnz \
  --codec zlib \
  --zlib-level 1
```

If `latent.bin.json` is present, image and latent dimensions are read from it.
Without metadata, square latents are inferred from the raw file size and entropy
parameter channel count. For non-square raw latents, pass `--latent-h` and
`--latent-w`, or provide `--metadata path/to/latent.bin.json`.

The C++ encoder uses PyTorch-compatible round-to-even quantization:

```text
symbols = round_to_even((y - median[c]) / quant_step)
y_hat   = symbols * quant_step + median[c]
```

It stores int16 when all symbols fit `[-32768, 32767]`; otherwise it falls back
to int32 and records the dtype in the CNZ4 header.

### PC Decode

```bash
python decode_cnz.py test.cnz \
  --checkpoint checkpoints/latest.pt \
  --output recon.png
```

`decode_cnz.py` supports new CNZ4 files and falls back to old pickle v2/v3
bitstreams when possible. C++ does not support old pickle files.

### Consistency Tests

```bash
python tests/test_rounding_consistency.py
python tests/test_python_cpp_payload_equivalence.py \
  --image test.png \
  --checkpoint checkpoints/latest.pt \
  --params entropy_params.json \
  --cnz-encode-cli cpp/build/cnz_encode_cli
```

The Python/C++ equivalence test compares C++ CNZ dequantized `y_hat` with the
Python reference quantize/dequantize path and reports max/mean absolute error.

### Benchmarks

Python:

```bash
python tools/benchmark_pipeline.py \
  --image test.png \
  --checkpoint checkpoints/latest.pt \
  --codec zlib \
  --zlib-level 1
```

C++:

```bash
cpp/build/cnz_benchmark_cli \
  --latent latent.bin \
  --params entropy_params.json \
  --latent-c 128 --latent-h 32 --latent-w 32 \
  --orig-h 512 --orig-w 512 \
  --padded-h 512 --padded-w 512 \
  --output bench.cnz \
  --codec zlib \
  --zlib-level 1
```

## CNZ4 Bitstream

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
