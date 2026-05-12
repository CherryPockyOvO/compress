# Legacy and Debug Entrypoints

This folder keeps scripts that are not part of the current RK3588/CNZ demo
path. They are archived here instead of deleted.

- `decode_image.py`: older compatibility wrapper around CNZ decoding.
- `demo_codec.py`: single-image smoke test using the model's Python
  compress/decompress helpers.
- `export_onnx.py`: exports both encoder and decoder ONNX. RK3588 deployment
  uses `tools/export_encoder_onnx.py` instead.
- `anime.py`: dataset collection helper. It is not needed for the current
  compression/decompression demo.

Current recommended entrypoints live in the project root, `tools/`, and
`cpp/scripts/`.
