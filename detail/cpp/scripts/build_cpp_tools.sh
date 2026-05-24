#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$CPP_DIR/build}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

cmake -S "$CPP_DIR" -B "$BUILD_DIR" \
  -DENABLE_LZ4="${ENABLE_LZ4:-OFF}" \
  -DENABLE_ZSTD="${ENABLE_ZSTD:-OFF}"

cmake --build "$BUILD_DIR" --parallel "$JOBS"

echo "built:"
echo "  $BUILD_DIR/cnz_encode_cli"
echo "  $BUILD_DIR/cnz_decode_cli"
echo "  $BUILD_DIR/cnz_benchmark_cli"
