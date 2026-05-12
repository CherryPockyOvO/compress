#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  rk3588_encode_cnz.sh --latent latent.bin --params entropy_params.json --output test.cnz [options]

Options:
  --latent PATH       float32 NCHW latent file from RKNN encoder. Default: latent.bin
  --params PATH       entropy params JSON. Default: entropy_params.json
  --output PATH       output CNZ4 file. Default: test.cnz
  --metadata PATH     latent metadata JSON. Default: auto-detect <latent>.json
  --codec NAME        none or zlib. Default: zlib
  --zlib-level N      zlib level. Default: 1
  --orig-h N          override original height
  --orig-w N          override original width
  --padded-h N        override padded height
  --padded-w N        override padded width
  --latent-c N        override latent channels
  --latent-h N        override latent height
  --latent-w N        override latent width
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/CMakeLists.txt" && -d "$SCRIPT_DIR/src" ]]; then
  CPP_DIR="$SCRIPT_DIR"
else
  CPP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

find_encode_cli() {
  local candidates=(
    "$CPP_DIR/build/cnz_encode_cli"
    "$CPP_DIR/bin/cnz_encode_cli"
    "$SCRIPT_DIR/build/cnz_encode_cli"
    "$SCRIPT_DIR/bin/cnz_encode_cli"
    "$SCRIPT_DIR/cnz_encode_cli"
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
    if [[ -x "$CPP_DIR/build/cnz_encode_cli" ]]; then
      echo "$CPP_DIR/build/cnz_encode_cli"
      return 0
    fi
  fi
  return 1
}

LATENT="latent.bin"
PARAMS="entropy_params.json"
OUTPUT="test.cnz"
CODEC="zlib"
ZLIB_LEVEL="1"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --latent) LATENT="$2"; shift 2 ;;
    --params) PARAMS="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --metadata|--meta) EXTRA_ARGS+=("--metadata" "$2"); shift 2 ;;
    --codec) CODEC="$2"; shift 2 ;;
    --zlib-level) ZLIB_LEVEL="$2"; shift 2 ;;
    --orig-h|--orig-w|--padded-h|--padded-w|--latent-c|--latent-h|--latent-w)
      EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

ENCODE_CLI="$(find_encode_cli)" || {
  echo "cnz_encode_cli not found. Run cpp/scripts/build_cpp_tools.sh first." >&2
  exit 1
}

"$ENCODE_CLI" \
  --latent "$LATENT" \
  --params "$PARAMS" \
  --output "$OUTPUT" \
  --codec "$CODEC" \
  --zlib-level "$ZLIB_LEVEL" \
  "${EXTRA_ARGS[@]}"
