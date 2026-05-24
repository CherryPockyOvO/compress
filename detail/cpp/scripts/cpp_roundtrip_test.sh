#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  cpp_roundtrip_test.sh --latent latent.bin --params entropy_params.json [options]

Runs the full C++ post-processing roundtrip:
  latent.bin -> test.cnz -> y_hat.bin

Options:
  --latent PATH       Default: latent.bin
  --params PATH       Default: entropy_params.json
  --cnz PATH          Default: test.cnz
  --yhat PATH         Default: y_hat.bin
  --codec NAME        Default: zlib
  --zlib-level N      Default: 1
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LATENT="latent.bin"
PARAMS="entropy_params.json"
CNZ="test.cnz"
YHAT="y_hat.bin"
CODEC="zlib"
ZLIB_LEVEL="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --latent) LATENT="$2"; shift 2 ;;
    --params) PARAMS="$2"; shift 2 ;;
    --cnz|--output-cnz) CNZ="$2"; shift 2 ;;
    --yhat|--output-yhat) YHAT="$2"; shift 2 ;;
    --codec) CODEC="$2"; shift 2 ;;
    --zlib-level) ZLIB_LEVEL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

"$SCRIPT_DIR/build_cpp_tools.sh"

echo
echo "[1/2] RK3588-side C++ encode: latent -> CNZ"
"$SCRIPT_DIR/rk3588_encode_cnz.sh" \
  --latent "$LATENT" \
  --params "$PARAMS" \
  --output "$CNZ" \
  --codec "$CODEC" \
  --zlib-level "$ZLIB_LEVEL"

echo
echo "[2/2] PC-side C++ decode: CNZ -> y_hat"
"$SCRIPT_DIR/pc_decode_yhat.sh" \
  --input "$CNZ" \
  --output-yhat "$YHAT"

echo
echo "roundtrip artifacts:"
ls -lh "$CNZ" "$YHAT"
