# Detail Subproject

`detail` is the legacy/factorized-prior demo-quality route. It uses the CNZ4 bitstream:

```text
image -> encoder/RKNN or PyTorch -> y symbols -> C++ CNZ entropy pack -> .cnz
.cnz -> Python decoder -> reconstructed image
```

The shared dataset remains outside this folder:

```text
../data/train
../data/val
```

## Train

```bash
cd detail
CUDA_VISIBLE_DEVICES=0,1,2 /home/zzw/miniconda3/envs/net/bin/torchrun \
  --standalone \
  --nproc_per_node=3 \
  train.py \
  --quality-profile detail \
  --train-dir ../data/train \
  --val-dir ../data/val \
  --checkpoint-dir checkpoints_detail \
  --checkpoint-interval-steps 100 \
  --eval-interval-steps 100 \
  --num-workers 4
```

## Local Python Roundtrip

```bash
cd detail
/home/zzw/miniconda3/envs/net/bin/python roundtrip_image.py \
  ../samples/test.jpg \
  --checkpoint artifacts/checkpoints/best_detail.pt \
  --output-dir roundtrip_detail \
  --mode cnz4 \
  --timing
```

## Local Python Encode / Decode

Encode:

```bash
cd detail
/home/zzw/miniconda3/envs/net/bin/python encode_image.py \
  ../samples/test.jpg \
  --checkpoint artifacts/checkpoints/best_detail.pt \
  --output out/image.cnz
```

Decode:

```bash
cd detail
/home/zzw/miniconda3/envs/net/bin/python decode_cnz.py \
  out/image.cnz \
  --checkpoint artifacts/checkpoints/best_detail.pt \
  --output out/recon.png \
  --half \
  --timing
```

## Build C++ Entropy Tools

```bash
cd detail/cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The main tools are:

```text
detail/cpp/build/cnz_encode_cli
detail/cpp/build/cnz_decode_cli
```

## RKNN Export

Export ONNX:

```bash
cd detail
/home/zzw/miniconda3/envs/net/bin/python tools/export_encoder_onnx.py \
  --checkpoint artifacts/checkpoints/best_detail.pt \
  --export-mode encoder \
  --height 512 \
  --width 512 \
  --output artifacts/export/detail_encoder_512x512.onnx
```

Convert to RKNN FP16:

```bash
cd detail
/home/zzw/miniconda3/envs/rknn/bin/python tools/convert_onnx_to_rknn.py \
  --onnx artifacts/export/detail_encoder_512x512.onnx \
  --output artifacts/export/detail_encoder_512x512_fp16.rknn \
  --target-platform rk3588
```

## RKNN Board Compression

Use RKNN for image-to-latent and C++ for entropy packing:

```bash
cd detail
python3 rknn/rk3588_fast_compress.py \
  --input image_or_dir \
  --output out_cnz \
  --rknn artifacts/export/detail_encoder_512x512_fp16.rknn \
  --params artifacts/export/detail_entropy_params.json \
  --cnz-encode-cli cpp/build/cnz_encode_cli \
  --height 512 \
  --width 512 \
  --core-masks 0,1,2 \
  --entropy-workers 3
```

If `detail_entropy_params.json` is missing, export it:

```bash
cd detail
/home/zzw/miniconda3/envs/net/bin/python tools/export_entropy_params.py \
  --checkpoint artifacts/checkpoints/best_detail.pt \
  --output artifacts/export/detail_entropy_params.json
```
