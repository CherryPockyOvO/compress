# RK3588 media CNZ batch compressor

This folder is a deployable RK3588-side batch pipeline:

```text
image/video frame -> RKNN encoder -> latent .bin -> C++ CNZ encoder -> .cnz
```

After each frame is converted to `.cnz`, the intermediate `.bin` is deleted by
default. For video input, frames are extracted to a temporary directory with
`ffmpeg` and removed automatically when the run finishes.

## Files

```text
rk3588_media_compress/
|-- batch_compress.py       # batch entry point
|-- rk3588_fast_compress.py # optimized persistent-worker RK3588 compressor
|-- rknn_encode_frame.py    # single-frame RKNN encoder, compatible with your run.py
|-- compress_media.sh       # shell wrapper
|-- pc_decode_cnz_video.py  # PC-side CNZ folder decoder and video builder
|-- decode_cnz_video.sh     # PC-side shell wrapper
|-- requirements.txt
`-- README.md
```

## RK3588 prerequisites

Install Python dependencies and make sure the C++ CNZ tool is built on the
board:

```bash
pip3 install -r rk3588_media_compress/requirements.txt

# From the original repo, on RK3588:
cpp/scripts/build_cpp_tools.sh
```

For video input, install `ffmpeg`:

```bash
sudo apt-get install ffmpeg
```

## Compress A Video

Example for the same 720x1280 RKNN input shape shown in your command:

Fast path, recommended on RK3588:

```bash
python3 rk3588_media_compress/rk3588_fast_compress.py \
  --input input.mp4 \
  --output out_cnz \
  --rknn encoder_fp.rknn \
  --params entropy_params2.json \
  --cnz-encode-cli rk3588_media_compress/cnz_encode_cli \
  --height 720 \
  --width 1280 \
  --workers-per-core 2 \
  --entropy-workers 3
```

This keeps RKNN runtimes alive. With `--core-masks 0,1,2` and
`--workers-per-core 2`, it starts six workers: two feeding each NPU core.
Use `--workers-per-core 3` only after the board runs stably.

Frames are assigned to per-core queues, and workers on the same NPU core share
that core's queue. This keeps task counts balanced across core0/core1/core2
while still allowing multiple workers to overlap preprocessing, NPU inference,
and entropy coding.

`--entropy-workers` limits concurrent `cnz_encode_cli` processes. If the board
stalls or entropy time jumps, keep it at `3` or reduce it to `2`.

By default, RKNN inference and entropy coding run as separate pipeline stages:
RKNN workers write `.bin` files and immediately continue feeding the NPU, while
CPU entropy workers turn `.bin` into `.cnz` in parallel. If you want the older
single-worker behavior for debugging, pass `--no-separate-entropy`.

If one NPU core stops returning frames while other cores keep running, the
script aborts with `core-stall` instead of waiting forever. If core0 repeatedly
stalls on your board, run only the stable cores:

```bash
python3 rk3588_media_compress/rk3588_fast_compress.py --input input.mp4 --output out_cnz --rknn encoder_fp.rknn --params entropy_params2.json --cnz-encode-cli rk3588_media_compress/cnz_encode_cli --height 720 --width 1280 --core-masks 1,2 --workers-per-core 2 --entropy-workers 2
```

Compatibility path:

```bash
bash rk3588_media_compress/compress_media.sh \
  --input input.mp4 \
  --output out_cnz \
  --rknn encoder_fp.rknn \
  --params entropy_params2.json \
  --height 720 \
  --width 1280 \
  --workers 3 \
  --core-masks 0,1,2
```

Output:

```text
out_cnz/
|-- manifest.json
`-- frames/
    |-- frame_00000000.cnz
    |-- frame_00000001.cnz
    `-- ...
```

Terminal progress includes separate RKNN/bin and entropy/CNZ timing:

```text
[1/300] frame 00000000 -> frames/frame_00000000.cnz total=0.184s bin_total=0.122s npu_infer=0.087s entropy=0.061s frame_fps=5.43 avg_fps=5.12 (63819 bytes)
avg_stage_time: bin_total=0.120s, npu_infer=0.086s, entropy=0.060s
done: 300 frames, total_time=55.238s, avg_fps=5.43
```

`bin_total` is the full single-frame RKNN encoder command time, including image
load/preprocess, RKNN model/runtime setup, NPU inference, and writing `.bin`.
`npu_infer` is the measured `rknn.inference()` time inside
`rknn_encode_frame.py`. If you pass your old `run.py` with
`--encoder-script /path/to/run.py --no-core-mask-arg`, `npu_infer` will be
`n/a`, but `bin_total` and `entropy` are still measured.

## Output Location

With `--output out_cnz`, the persistent output is:

```text
out_cnz/
|-- manifest.json              # per-frame timing, output paths, sizes
`-- frames/
    |-- frame_00000000.cnz
    |-- frame_00000001.cnz
    `-- ...
```

Intermediate `.bin` files are deleted by default. If you pass `--keep-bin`, they
are kept here:

```text
out_cnz/_work/latent_bins/
```

## PC Decode CNZ Folder To Video

Run this on your local PC from the repository root. It loads the PyTorch decoder
checkpoint once, decodes all RK-generated `.cnz` frames to images, then combines
them with `ffmpeg`.

Install the PC-side Python dependencies from the repository root if needed:

```bash
pip install -r requirements.txt
```

```bash
python3 rk3588_media_compress/pc_decode_cnz_video.py \
  --input out_cnz \
  --checkpoint best2.pt \
  --output-video recon.mp4 \
  --fps 30 \
  --overwrite
```

If `--fps` is omitted, the script tries to read the original video FPS from
`out_cnz/manifest.json`; if unavailable it uses `30`.

Local output:

```text
recon.mp4
recon.mp4.decode_manifest.json
recon_frames/
|-- frame_00000000.png
|-- frame_00000001.png
`-- ...
```

Terminal progress reports the decode stages separately:

```text
[1/102] frame_00000000.cnz -> frame_00000000.png decode=0.704s unpack=0.012s torch=0.582s save=0.104s fps=1.42
avg_decode_stage_time: unpack=0.012s, torch=0.580s, save_pipe=0.105s
```

Use CUDA half precision when available:

```bash
python3 rk3588_media_compress/pc_decode_cnz_video.py \
  --input out_cnz \
  --checkpoint best2.pt \
  --output-video recon.mp4 \
  --fps 30 \
  --device cuda \
  --half \
  --batch-size 4 \
  --pipe-video \
  --overwrite
```

`--batch-size 4` sends four latent frames through the PyTorch decoder at once.
`--pipe-video` skips intermediate PNG/JPG files and streams RGB frames directly
to `ffmpeg`, which is much faster when you only need the final video.

For short clips, if the first CUDA batch is much slower than later batches, try:

```bash
--no-cudnn-benchmark
```

Use multiple local GPUs:

```bash
python3 rk3588_media_compress/pc_decode_cnz_video.py --input out_cnz --checkpoint best2.pt --output-video recon.mp4 --fps 30 --devices cuda:0,cuda:1,cuda:2 --half --batch-size 4 --pipe-video --preset ultrafast --overwrite
```

In multi-GPU mode, each GPU loads one decoder copy and processes batches in
parallel. The main process writes decoded raw RGB frames to `ffmpeg` in the
original frame order.

Delete decoded PNG frames after the video is written:

```bash
--cleanup-frames
```

For faster local testing, writing JPG frames is usually faster than PNG:

```bash
--image-format jpg
```

## Compress One Image

```bash
bash rk3588_media_compress/compress_media.sh \
  --input test5.jpg \
  --output out_one_image \
  --rknn encoder_fp.rknn \
  --params entropy_params2.json \
  --height 720 \
  --width 1280
```

## Compress An Image Folder

```bash
bash rk3588_media_compress/compress_media.sh \
  --input frames_dir \
  --output out_cnz \
  --rknn encoder_fp.rknn \
  --params entropy_params2.json \
  --height 720 \
  --width 1280 \
  --recursive
```

## CNZ Tool Path

The script auto-detects these locations:

```text
rk3588_media_compress/rk3588_encode_cnz.sh
rk3588_media_compress/cnz_encode_cli
rk3588_media_compress/bin/cnz_encode_cli
../cpp/scripts/rk3588_encode_cnz.sh
cpp/scripts/rk3588_encode_cnz.sh
./cnz_encode_cli
```

If your tool is elsewhere, pass it explicitly:

```bash
--cnz-tool /path/to/rk3588_encode_cnz.sh
```

or:

```bash
--cnz-tool /path/to/cnz_encode_cli
```

## Useful Options

```text
--workers 3              Run three parallel frame pipelines for RK3588.
--core-masks 0,1,2       Assign NPU cores round-robin.
--codec zlib             CNZ payload codec. Use "none" for no zlib.
--zlib-level 1           Fast zlib level.
--start-time 00:00:05    Start video extraction from this point.
--duration 10            Only process a 10 second segment.
--fps 10                 Optional downsample. Omit this to keep every frame.
--frame-limit 100        Debug only the first 100 frames.
--keep-bin               Keep intermediate latent .bin files.
--keep-latent-metadata   Keep intermediate .bin.json files.
```

If you want to call your original `run.py` instead of `rknn_encode_frame.py`,
use:

```bash
--encoder-script /path/to/run.py --no-core-mask-arg
```
