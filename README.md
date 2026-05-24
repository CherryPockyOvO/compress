# compressai-nano

This workspace now keeps two independent compression routes:

- `detail/`: legacy demo route. It keeps the old detail model and CNZ4-style
  C++ entropy toolchain.
- `hyper/`: current high-quality route. It keeps the mean-scale hyperprior nano
  model, local Python decoding, RK3588 RKNN compression scripts, and the C++
  entropy encoder path.

Shared data and samples stay at the workspace root:

```text
data/
samples/
```

Use the subproject README for the actual commands:

```bash
cd detail
cat README.md

cd ../hyper
cat README.md
```

## Quick Start

Detail single-image roundtrip:

```bash
cd detail
/home/zzw/miniconda3/envs/net/bin/python roundtrip_image.py \
  ../samples/test.jpg \
  --checkpoint artifacts/checkpoints/best_detail.pt \
  --output-dir roundtrip_detail \
  --mode cnz4 \
  --timing
```

Hyper local quality check:

```bash
cd hyper
/home/zzw/miniconda3/envs/net/bin/python roundtrip_image.py \
  ../samples/test.jpg \
  --checkpoint artifacts/checkpoints/best_hyper.pt \
  --output-dir roundtrip_hyper \
  --mode forward \
  --timing
```

Hyper RK3588 compression package decode on PC:

```bash
cd hyper
/home/zzw/miniconda3/envs/net/bin/python decode_hyper_ms_npz.py \
  out/test.npz \
  --checkpoint artifacts/checkpoints/best_hyper.pt \
  --output out/recon.png \
  --device cuda \
  --half \
  --resize-to-source \
  --timing
```

## Repository Layout

```text
compressai-nano/
|-- README.md
|-- requirements.txt
|-- data/                 # shared, ignored by git
|-- samples/              # shared test media, ignored by git if image/video
|-- _legacy_root_backup/  # reversible backup of old root-level scripts/artifacts
|-- detail/               # old detail/demo codec route
`-- hyper/                # mean-scale hyperprior codec route
```

Large outputs, checkpoints, ONNX, RKNN, and temporary compression products are
kept inside each subproject but ignored by git.
