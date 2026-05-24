#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  pc_decode_yhat.sh --input test.cnz --output-yhat y_hat.bin

This is the C++ PC-side CNZ4 unpack/dequantize step. It outputs decoder-ready
float32 NCHW y_hat, not the final RGB image. Final RGB reconstruction still uses
the PyTorch decoder.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/CMakeLists.txt" && -d "$SCRIPT_DIR/src" ]]; then
  CPP_DIR="$SCRIPT_DIR"
else
  CPP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

find_decode_cli() {
  local candidates=(
    "$CPP_DIR/build/cnz_decode_cli"
    "$CPP_DIR/bin/cnz_decode_cli"
    "$SCRIPT_DIR/build/cnz_decode_cli"
    "$SCRIPT_DIR/bin/cnz_decode_cli"
    "$SCRIPT_DIR/cnz_decode_cli"
  )
  for path in "${candidates[@]}"; do
    if [[ -x "$path" ]]; then
      echo "$path"
      return 0
    fi
  done
  if [[ -f "$CPP_DIR/CMakeLists.txt" ]]; then
    if [[ -x "$SCRIPT_DIR/build_cpp_tools.sh" ]]; then
      "$SCRIPT_DIR/build_cpp_tools.sh" >/dev/null
    elif [[ -x "$SCRIPT_DIR/scripts/build_cpp_tools.sh" ]]; then
      "$SCRIPT_DIR/scripts/build_cpp_tools.sh" >/dev/null
    fi
    if [[ -x "$CPP_DIR/build/cnz_decode_cli" ]]; then
      echo "$CPP_DIR/build/cnz_decode_cli"
      return 0
    fi
  fi
  return 1
}

INPUT="test.cnz"
OUTPUT_YHAT="y_hat.bin"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT="$2"; shift 2 ;;
    --output-yhat|--output) OUTPUT_YHAT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

DECODE_CLI="$(find_decode_cli)" || {
  echo "cnz_decode_cli not found. Run cpp/scripts/build_cpp_tools.sh first." >&2
  exit 1
}

"$DECODE_CLI" --input "$INPUT" --output-yhat "$OUTPUT_YHAT"
