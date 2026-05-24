# CNZ C++ helper scripts

Common commands from the repository root:

```bash
cpp/scripts/build_cpp_tools.sh
```

Run the full C++ post-processing roundtrip:

```bash
cpp/scripts/cpp_roundtrip_test.sh \
  --latent latent.bin \
  --params entropy_params.json \
  --cnz test.cnz \
  --yhat y_hat.bin
```

Only run RK3588-side post-processing:

```bash
cpp/scripts/rk3588_encode_cnz.sh \
  --latent latent.bin \
  --params entropy_params.json \
  --output test.cnz
```

Only run PC-side C++ unpack/dequantize:

```bash
cpp/scripts/pc_decode_yhat.sh \
  --input test.cnz \
  --output-yhat y_hat.bin
```

Create deployment folders:

```bash
cpp/scripts/package_cnz_tools.sh
```

The RK3588 package contains source because PC-built binaries do not run on
RK3588. Build that package on the board or cross-compile it.
