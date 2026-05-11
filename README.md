# compressai-nano

`compressai-nano` is a stripped-down FactorizedPrior-style image compression
project derived from the CompressAI model layout. It keeps only the pieces
needed for a single-image three-level codec and fixed-shape RK3588-friendly
ONNX export.

The current architecture is asymmetric: RK3588 only runs the encoder, while the
PC runs the heavier residual decoder for better reconstruction quality.

## Directory Tree

```text
compressai-nano/
|-- README.md
|-- requirements.txt
|-- export_onnx.py
|-- demo_codec.py
|-- encode_image.py
|-- decode_image.py
`-- compressai_nano/
    |-- __init__.py
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
- Three quality levels are exposed through `quality_level=1,2,3`.
- Encoder and decoder are separate modules for ONNX export.
- The CPU entropy path is a small pure Python/PyTorch symbol codec. It is meant
  for runnable prototyping; use a trained CDF/rANS implementation for production
  bitstreams.

## Quality Levels

```python
from compressai_nano import FactorizedPriorNano

model_q1 = FactorizedPriorNano(quality_level=1)  # lowest bitrate, coarser symbols
model_q2 = FactorizedPriorNano(quality_level=2)  # balanced default
model_q3 = FactorizedPriorNano(quality_level=3)  # finer symbols, higher bitrate
```

All three levels keep the RK3588 encoder at `N=128, M=128`. Higher quality
levels use a finer latent quantization step and a stronger PC-side decoder.
Train each level separately before using the codec for real image quality.

## Export ONNX

From this directory:

```powershell
python export_onnx.py --quality-level 2 --height 256 --width 256 --output-dir onnx_models
```

This writes:

```text
onnx_models/encoder.onnx
onnx_models/decoder.onnx
```

The encoder input is fixed to `(1, 3, 256, 256)`. The decoder input is fixed to
the latent tensor shape produced by the encoder, `(1, 128, 16, 16)` for 256x256.

Use a checkpoint when you have trained weights:

```powershell
python export_onnx.py --quality-level 2 --checkpoint checkpoints/q2.pth --output-dir onnx_models
```

## Single Image Smoke Run

```powershell
python demo_codec.py path\to\image.png --quality-level 2 --output recon.png
```

Without a trained checkpoint, the script validates the codec path but the
reconstruction will not be meaningful.

## Train And Validate

Prepare a 1000-image split:

```powershell
python prepare_data.py --count 1000 --source unsplash-api --unsplash-access-key $env:UNSPLASH_ACCESS_KEY --threads 8 --overwrite
```

Prepare a larger mixed 5000-image compression dataset:

```powershell
python expand_dataset.py --count 5000 --threads 12
```

```powershell
python train.py --train-dir data\train --val-dir data\val --quality-level 2 --epochs 150 --batch-size 4 --num-workers 4 --lr 1e-4 --ssim-weight 0.2
python val.py --data-dir D:\data\images\test --checkpoint checkpoints\q2_latest.pt --results-dir results
```

Simulate RK3588 encode and PC decode:

```powershell
python encode_image.py test.jpg --checkpoint checkpoints\q2_latest.pt --quality-level 2 --output stream.cnz
python decode_image.py stream.cnz --checkpoint checkpoints\q2_latest.pt --output recon.png
```

Check exported encoder ONNX after running `export_onnx.py`:

```powershell
python check_npu_compatibility.py --encoder onnx_models\encoder.onnx --checkpoint checkpoints\q2_latest.pt --quality-level 2
```
