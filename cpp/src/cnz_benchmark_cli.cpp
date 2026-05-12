#include "cnz_codec.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace {

using Clock = std::chrono::high_resolution_clock;

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

double ms_since(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_args(argc, argv);
        const std::string latent_path = require_arg(args, "--latent");
        const std::string params_path = require_arg(args, "--params");
        const int C = std::stoi(require_arg(args, "--latent-c"));
        const int H = std::stoi(require_arg(args, "--latent-h"));
        const int W = std::stoi(require_arg(args, "--latent-w"));
        const int orig_h = args.count("--orig-h") ? std::stoi(args.at("--orig-h")) : H * 16;
        const int orig_w = args.count("--orig-w") ? std::stoi(args.at("--orig-w")) : W * 16;
        const int padded_h = args.count("--padded-h") ? std::stoi(args.at("--padded-h")) : orig_h;
        const int padded_w = args.count("--padded-w") ? std::stoi(args.at("--padded-w")) : orig_w;
        const std::string codec_name = args.count("--codec") ? args.at("--codec") : "zlib";
        const int zlib_level = args.count("--zlib-level") ? std::stoi(args.at("--zlib-level")) : 1;
        const std::string output_path = args.count("--output") ? args.at("--output") : "";

        const auto t0 = Clock::now();
        const auto params = cnz::load_entropy_params_json(params_path);
        const auto latent = cnz::read_float32_file(latent_path);
        const auto t1 = Clock::now();
        const auto symbols = cnz::quantize_to_int32(latent.data(), params.medians.data(), C, H, W, params.quant_step);
        const auto t2 = Clock::now();
        auto minmax = std::minmax_element(symbols.begin(), symbols.end());
        const uint32_t dtype = cnz::can_store_int16(symbols) ? cnz::kDtypeInt16 : cnz::kDtypeInt32;
        const auto t3 = Clock::now();
        const auto raw = cnz::pack_symbols_bytes(symbols, dtype);
        const uint32_t codec = cnz::codec_from_name(codec_name);
        const auto payload = cnz::compress_raw_bytes(raw, codec, zlib_level);
        const auto t4 = Clock::now();
        if (!output_path.empty()) {
            cnz::CnzHeader header;
            header.orig_h = static_cast<uint32_t>(orig_h);
            header.orig_w = static_cast<uint32_t>(orig_w);
            header.padded_h = static_cast<uint32_t>(padded_h);
            header.padded_w = static_cast<uint32_t>(padded_w);
            header.latent_c = static_cast<uint32_t>(C);
            header.latent_h = static_cast<uint32_t>(H);
            header.latent_w = static_cast<uint32_t>(W);
            header.down_factor = params.downsampling_factor;
            header.dtype = dtype;
            header.codec = codec;
            header.quant_step = params.quant_step;
            cnz::write_cnz_file(output_path, header, params.medians, payload);
        }
        const auto t5 = Clock::now();

        const double pixels = static_cast<double>(orig_h) * static_cast<double>(orig_w);
        const double bpp = pixels > 0.0 ? static_cast<double>(payload.size() * 8) / pixels : 0.0;
        std::cout << "load_ms=" << ms_since(t0, t1) << "\n";
        std::cout << "quantize_ms=" << ms_since(t1, t2) << "\n";
        std::cout << "dtype_scan_ms=" << ms_since(t2, t3) << "\n";
        std::cout << "compress_ms=" << ms_since(t3, t4) << "\n";
        std::cout << "write_file_ms=" << ms_since(t4, t5) << "\n";
        std::cout << "total_ms=" << ms_since(t0, t5) << "\n";
        std::cout << "payload_size=" << payload.size() << "\n";
        std::cout << "raw_size=" << raw.size() << "\n";
        std::cout << "bpp=" << bpp << "\n";
        std::cout << "symbol_min=" << *minmax.first << "\n";
        std::cout << "symbol_max=" << *minmax.second << "\n";
        std::cout << "dtype=" << cnz::dtype_name(dtype) << "\n";
        std::cout << "codec=" << cnz::codec_name(codec) << "\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "cnz_benchmark_cli error: " << exc.what() << "\n";
        return 1;
    }
}
