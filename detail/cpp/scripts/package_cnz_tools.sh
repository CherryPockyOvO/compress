#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  package_cnz_tools.sh [--output-dir dist/cnz_cpp_tools] [--params entropy_params.json]

Creates two deployment folders:
  rk3588_encode_source/  source + scripts to build/run cnz_encode_cli on RK3588
  pc_decode_bin/         current-machine cnz_decode_cli binary + run script

Note: binaries built on this PC are not RK3588 binaries. Build the source pack
on the RK3588 board, or cross-compile it with your RK3588 toolchain.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$CPP_DIR/.." && pwd)"
OUT_DIR="$ROOT_DIR/dist/cnz_cpp_tools"
PARAMS="$ROOT_DIR/entropy_params.json"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) OUT_DIR="$2"; shift 2 ;;
    --params) PARAMS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

"$SCRIPT_DIR/build_cpp_tools.sh"

RK_DIR="$OUT_DIR/rk3588_encode_source"
PC_DIR="$OUT_DIR/pc_decode_bin"

mkdir -p "$RK_DIR/include" "$RK_DIR/src" "$RK_DIR/scripts"
mkdir -p "$PC_DIR/bin" "$PC_DIR/scripts"

cp -f "$CPP_DIR/CMakeLists.txt" "$RK_DIR/CMakeLists.txt"
cp -f "$CPP_DIR/include/cnz_codec.h" "$RK_DIR/include/cnz_codec.h"
cp -f "$CPP_DIR/src/cnz_codec.cpp" "$RK_DIR/src/cnz_codec.cpp"
cp -f "$CPP_DIR/src/cnz_encode_cli.cpp" "$RK_DIR/src/cnz_encode_cli.cpp"
cp -f "$CPP_DIR/src/cnz_decode_cli.cpp" "$RK_DIR/src/cnz_decode_cli.cpp"
cp -f "$CPP_DIR/src/cnz_benchmark_cli.cpp" "$RK_DIR/src/cnz_benchmark_cli.cpp"
cp -f "$SCRIPT_DIR/build_cpp_tools.sh" "$RK_DIR/scripts/build_cpp_tools.sh"
cp -f "$SCRIPT_DIR/rk3588_encode_cnz.sh" "$RK_DIR/run_encode.sh"

if [[ -f "$PARAMS" ]]; then
  cp -f "$PARAMS" "$RK_DIR/entropy_params.json"
fi

cp -f "$CPP_DIR/build/cnz_decode_cli" "$PC_DIR/bin/cnz_decode_cli"
cp -f "$CPP_DIR/build/cnz_benchmark_cli" "$PC_DIR/bin/cnz_benchmark_cli"
cp -f "$SCRIPT_DIR/pc_decode_yhat.sh" "$PC_DIR/run_decode_yhat.sh"

cat > "$RK_DIR/README.md" <<'EOF'
# RK3588 CNZ Encode Pack

Build on the RK3588 board:

```bash
scripts/build_cpp_tools.sh
```

Encode RKNN encoder output:

```bash
./run_encode.sh \
  --latent latent.bin \
  --params entropy_params.json \
  --output frame.cnz
```

`latent.bin` must be float32 NCHW, batch=1. If `latent.bin.json` is present,
dimensions are read automatically.
EOF

cat > "$PC_DIR/README.md" <<'EOF'
# PC CNZ Decode Pack

Decode a CNZ4 package to decoder-ready y_hat:

```bash
./run_decode_yhat.sh \
  --input frame.cnz \
  --output-yhat y_hat.bin
```

`y_hat.bin` is float32 NCHW. Final RGB reconstruction still uses the PyTorch
decoder from the main project.
EOF

chmod +x "$RK_DIR/scripts/build_cpp_tools.sh" "$RK_DIR/run_encode.sh" "$PC_DIR/run_decode_yhat.sh"

echo "packages written:"
echo "  $RK_DIR"
echo "  $PC_DIR"
