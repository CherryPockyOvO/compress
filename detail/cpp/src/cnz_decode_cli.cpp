#include "cnz_codec.h"

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace {

std::map<std::string, std::string> parse_args(int argc, char** argv) {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("--", 0) != 0 || i + 1 >= argc) {
            throw std::runtime_error("expected --key value argument pair, got: " + key);
        }
        args[key] = argv[++i];
    }
    return args;
}

std::string require_arg(const std::map<std::string, std::string>& args, const std::string& key) {
    auto it = args.find(key);
    if (it == args.end()) {
        throw std::runtime_error("missing required argument: " + key);
    }
    return it->second;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_args(argc, argv);
        const std::string input_path = require_arg(args, "--input");
        const std::string output_path = require_arg(args, "--output-yhat");

        const auto file = cnz::read_cnz_file(input_path);
        const auto& h = file.header;
        const size_t expected_raw = static_cast<size_t>(h.latent_c) * h.latent_h * h.latent_w * cnz::dtype_size(h.dtype);
        const auto raw = cnz::decompress_raw_bytes(file.payload, h.codec, expected_raw);
        const auto symbols = cnz::unpack_symbols_bytes(raw, h.dtype);
        const auto y_hat = cnz::dequantize_to_float(
            symbols.data(),
            file.medians.data(),
            static_cast<int>(h.latent_c),
            static_cast<int>(h.latent_h),
            static_cast<int>(h.latent_w),
            h.quant_step
        );
        cnz::write_float32_file(output_path, y_hat);

        const double pixels = static_cast<double>(h.orig_h) * static_cast<double>(h.orig_w);
        const double bpp = pixels > 0.0 ? static_cast<double>(h.payload_size * 8) / pixels : 0.0;
        std::cout << "input: " << input_path << "\n";
        std::cout << "output_yhat: " << output_path << "\n";
        std::cout << "orig_size: " << h.orig_h << "x" << h.orig_w << "\n";
        std::cout << "padded_size: " << h.padded_h << "x" << h.padded_w << "\n";
        std::cout << "latent_shape: [1," << h.latent_c << "," << h.latent_h << "," << h.latent_w << "]\n";
        std::cout << "dtype: " << cnz::dtype_name(h.dtype) << "\n";
        std::cout << "codec: " << cnz::codec_name(h.codec) << "\n";
        std::cout << "payload_size: " << h.payload_size << "\n";
        std::cout << "payload_bpp: " << bpp << "\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "cnz_decode_cli error: " << exc.what() << "\n";
        return 1;
    }
}
