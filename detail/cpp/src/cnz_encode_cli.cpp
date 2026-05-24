#include "cnz_codec.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>

namespace {

struct LatentShape {
    int orig_h = 0;
    int orig_w = 0;
    int padded_h = 0;
    int padded_w = 0;
    int latent_c = 0;
    int latent_h = 0;
    int latent_w = 0;
};

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

int optional_int(
    const std::map<std::string, std::string>& args,
    const std::string& key,
    int fallback
) {
    auto it = args.find(key);
    if (it == args.end()) {
        return fallback;
    }
    return std::stoi(it->second);
}

bool file_exists(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    return input.good();
}

std::string read_text_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open metadata JSON: " + path);
    }
    return std::string(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()
    );
}

int extract_int_optional(const std::string& text, const std::string& key, int fallback) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*([-+0-9]+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        return fallback;
    }
    return std::stoi(match[1].str());
}

LatentShape load_metadata(const std::string& path) {
    const std::string text = read_text_file(path);
    LatentShape shape;
    shape.orig_h = extract_int_optional(text, "orig_h", 0);
    shape.orig_w = extract_int_optional(text, "orig_w", 0);
    shape.padded_h = extract_int_optional(text, "padded_h", 0);
    shape.padded_w = extract_int_optional(text, "padded_w", 0);
    shape.latent_c = extract_int_optional(text, "latent_c", 0);
    shape.latent_h = extract_int_optional(text, "latent_h", 0);
    shape.latent_w = extract_int_optional(text, "latent_w", 0);
    return shape;
}

std::string resolve_metadata_path(
    const std::map<std::string, std::string>& args,
    const std::string& latent_path
) {
    auto it = args.find("--metadata");
    if (it != args.end()) {
        return it->second;
    }
    it = args.find("--meta");
    if (it != args.end()) {
        return it->second;
    }
    const std::string default_path = latent_path + ".json";
    if (file_exists(default_path)) {
        return default_path;
    }
    return "";
}

bool is_perfect_square(size_t value, int& root) {
    const double root_f = std::sqrt(static_cast<double>(value));
    const auto rounded = static_cast<size_t>(std::llround(root_f));
    if (rounded * rounded != value) {
        return false;
    }
    root = static_cast<int>(rounded);
    return true;
}

void fill_inferred_shape(
    LatentShape& shape,
    size_t latent_float_count,
    const cnz::EntropyParams& params
) {
    if (shape.latent_c <= 0) {
        shape.latent_c = static_cast<int>(params.channels);
    }
    if (shape.latent_c <= 0) {
        throw std::runtime_error("latent_c is unknown");
    }
    if (latent_float_count % static_cast<size_t>(shape.latent_c) != 0) {
        throw std::runtime_error("latent file size is not divisible by latent_c");
    }

    const size_t plane = latent_float_count / static_cast<size_t>(shape.latent_c);
    if (shape.latent_h <= 0 && shape.latent_w <= 0) {
        int side = 0;
        if (!is_perfect_square(plane, side)) {
            throw std::runtime_error(
                "cannot infer non-square latent_h/latent_w from raw latent.bin; "
                "provide --latent-h/--latent-w or a metadata JSON"
            );
        }
        shape.latent_h = side;
        shape.latent_w = side;
    } else if (shape.latent_h <= 0) {
        if (plane % static_cast<size_t>(shape.latent_w) != 0) {
            throw std::runtime_error("cannot infer latent_h from latent_w and file size");
        }
        shape.latent_h = static_cast<int>(plane / static_cast<size_t>(shape.latent_w));
    } else if (shape.latent_w <= 0) {
        if (plane % static_cast<size_t>(shape.latent_h) != 0) {
            throw std::runtime_error("cannot infer latent_w from latent_h and file size");
        }
        shape.latent_w = static_cast<int>(plane / static_cast<size_t>(shape.latent_h));
    }

    const size_t expected = static_cast<size_t>(shape.latent_c) *
                            static_cast<size_t>(shape.latent_h) *
                            static_cast<size_t>(shape.latent_w);
    if (expected != latent_float_count) {
        throw std::runtime_error("latent shape does not match latent file size");
    }

    if (shape.padded_h <= 0) {
        shape.padded_h = shape.latent_h * static_cast<int>(params.downsampling_factor);
    }
    if (shape.padded_w <= 0) {
        shape.padded_w = shape.latent_w * static_cast<int>(params.downsampling_factor);
    }
    if (shape.orig_h <= 0) {
        shape.orig_h = shape.padded_h;
    }
    if (shape.orig_w <= 0) {
        shape.orig_w = shape.padded_w;
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_args(argc, argv);
        const std::string latent_path = require_arg(args, "--latent");
        const std::string params_path = require_arg(args, "--params");
        const std::string output_path = require_arg(args, "--output");
        const std::string codec_name = args.count("--codec") ? args.at("--codec") : "zlib";
        const int zlib_level = args.count("--zlib-level") ? std::stoi(args.at("--zlib-level")) : 1;

        const auto params = cnz::load_entropy_params_json(params_path);
        const auto latent = cnz::read_float32_file(latent_path);

        LatentShape shape;
        const std::string metadata_path = resolve_metadata_path(args, latent_path);
        if (!metadata_path.empty()) {
            shape = load_metadata(metadata_path);
        }
        shape.orig_h = optional_int(args, "--orig-h", shape.orig_h);
        shape.orig_w = optional_int(args, "--orig-w", shape.orig_w);
        shape.padded_h = optional_int(args, "--padded-h", shape.padded_h);
        shape.padded_w = optional_int(args, "--padded-w", shape.padded_w);
        shape.latent_c = optional_int(args, "--latent-c", shape.latent_c);
        shape.latent_h = optional_int(args, "--latent-h", shape.latent_h);
        shape.latent_w = optional_int(args, "--latent-w", shape.latent_w);
        fill_inferred_shape(shape, latent.size(), params);

        const uint32_t codec = cnz::codec_from_name(codec_name);
        auto result = cnz::encode_latent_to_payload(
            latent.data(),
            shape.latent_c,
            shape.latent_h,
            shape.latent_w,
            params,
            codec,
            zlib_level
        );
        result.header.orig_h = static_cast<uint32_t>(shape.orig_h);
        result.header.orig_w = static_cast<uint32_t>(shape.orig_w);
        result.header.padded_h = static_cast<uint32_t>(shape.padded_h);
        result.header.padded_w = static_cast<uint32_t>(shape.padded_w);
        cnz::write_cnz_file(output_path, result.header, params.medians, result.compressed_payload);

        const double pixels = static_cast<double>(shape.orig_h) * static_cast<double>(shape.orig_w);
        const double bpp = pixels > 0.0 ? static_cast<double>(result.compressed_payload.size() * 8) / pixels : 0.0;
        std::cout << "saved: " << output_path << "\n";
        if (!metadata_path.empty()) {
            std::cout << "metadata: " << metadata_path << "\n";
        }
        std::cout << "original_size: " << shape.orig_h << "x" << shape.orig_w << "\n";
        std::cout << "padded_size: " << shape.padded_h << "x" << shape.padded_w << "\n";
        std::cout << "latent_shape: [1," << shape.latent_c << "," << shape.latent_h << "," << shape.latent_w << "]\n";
        std::cout << "dtype: " << cnz::dtype_name(result.header.dtype) << "\n";
        std::cout << "codec: " << cnz::codec_name(result.header.codec) << "\n";
        std::cout << "symbol_min: " << result.min_symbol << "\n";
        std::cout << "symbol_max: " << result.max_symbol << "\n";
        std::cout << "raw_size: " << result.raw_bytes.size() << "\n";
        std::cout << "payload_size: " << result.compressed_payload.size() << "\n";
        std::cout << "payload_bpp: " << bpp << "\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "cnz_encode_cli error: " << exc.what() << "\n";
        return 1;
    }
}
