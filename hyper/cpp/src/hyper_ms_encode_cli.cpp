#include "cnz_codec.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

constexpr uint32_t kVersion = 1;
constexpr uint8_t kMagic[4] = {'H', 'M', 'S', '1'};

struct Shape4 {
    int n = 1;
    int c = 0;
    int h = 0;
    int w = 0;

    size_t count() const {
        return static_cast<size_t>(n) * static_cast<size_t>(c) *
               static_cast<size_t>(h) * static_cast<size_t>(w);
    }
};

struct HyperParams {
    int channels_y = 0;
    int channels_z = 0;
    float quant_step_y = 1.0f;
    float quant_step_z = 1.0f;
    std::string model_variant;
    std::vector<float> z_medians;
};

struct Metadata {
    int source_h = 0;
    int source_w = 0;
    int orig_h = 0;
    int orig_w = 0;
    int padded_h = 0;
    int padded_w = 0;
    Shape4 y_shape;
    Shape4 z_shape;
};

template <typename T>
void append_le(std::vector<uint8_t>& out, T value) {
    static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value,
                  "append_le only supports scalar values");
    uint8_t bytes[sizeof(T)];
    std::memcpy(bytes, &value, sizeof(T));
    out.insert(out.end(), bytes, bytes + sizeof(T));
}

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
    const auto it = args.find(key);
    if (it == args.end()) {
        throw std::runtime_error("missing required argument: " + key);
    }
    return it->second;
}

std::string optional_arg(
    const std::map<std::string, std::string>& args,
    const std::string& key,
    const std::string& fallback
) {
    const auto it = args.find(key);
    return it == args.end() ? fallback : it->second;
}

std::string read_text_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open text file: " + path);
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void write_binary_file(const std::string& path, const std::vector<uint8_t>& data) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open output file: " + path);
    }
    if (!data.empty()) {
        output.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    }
}

float extract_float(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*([-+0-9.eE]+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("missing numeric JSON key: " + key);
    }
    return std::stof(match[1].str());
}

int extract_int(const std::string& text, const std::string& key) {
    return static_cast<int>(extract_float(text, key));
}

std::string extract_string_optional(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        return "";
    }
    return match[1].str();
}

std::vector<float> extract_float_array(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("missing float array JSON key: " + key);
    }
    std::vector<float> values;
    const std::string array_text = match[1].str();
    const std::regex number_pattern("[-+0-9.eE]+");
    auto begin = std::sregex_iterator(array_text.begin(), array_text.end(), number_pattern);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        values.push_back(std::stof(it->str()));
    }
    return values;
}

std::vector<int> extract_int_array(const std::string& text, const std::string& key) {
    const auto floats = extract_float_array(text, key);
    std::vector<int> values;
    values.reserve(floats.size());
    for (float value : floats) {
        values.push_back(static_cast<int>(value));
    }
    return values;
}

HyperParams load_params(const std::string& path) {
    const std::string text = read_text_file(path);
    HyperParams params;
    params.channels_y = extract_int(text, "channels_y");
    params.channels_z = extract_int(text, "channels_z");
    params.quant_step_y = extract_float(text, "quant_step_y");
    params.quant_step_z = extract_float(text, "quant_step_z");
    params.model_variant = extract_string_optional(text, "model_variant");
    params.z_medians = extract_float_array(text, "z_medians");
    if (params.channels_y <= 0 || params.channels_z <= 0) {
        throw std::runtime_error("channels_y/channels_z must be positive");
    }
    if (params.quant_step_y <= 0.0f || params.quant_step_z <= 0.0f) {
        throw std::runtime_error("quant steps must be positive");
    }
    if (params.z_medians.size() != static_cast<size_t>(params.channels_z)) {
        throw std::runtime_error("z_medians count does not match channels_z");
    }
    return params;
}

Shape4 shape_from_array(const std::vector<int>& values, const std::string& key) {
    if (values.size() != 4) {
        throw std::runtime_error(key + " must have 4 values");
    }
    Shape4 shape;
    shape.n = values[0];
    shape.c = values[1];
    shape.h = values[2];
    shape.w = values[3];
    if (shape.n != 1 || shape.c <= 0 || shape.h <= 0 || shape.w <= 0) {
        throw std::runtime_error("invalid shape in " + key);
    }
    return shape;
}

Metadata load_metadata(const std::string& path) {
    const std::string text = read_text_file(path);
    Metadata metadata;
    metadata.source_h = extract_int(text, "source_h");
    metadata.source_w = extract_int(text, "source_w");
    metadata.orig_h = extract_int(text, "orig_h");
    metadata.orig_w = extract_int(text, "orig_w");
    metadata.padded_h = extract_int(text, "padded_h");
    metadata.padded_w = extract_int(text, "padded_w");
    metadata.y_shape = shape_from_array(extract_int_array(text, "y_shape"), "y_shape");
    metadata.z_shape = shape_from_array(extract_int_array(text, "z_shape"), "z_shape");
    return metadata;
}

std::vector<int32_t> quantize_y(
    const std::vector<float>& y,
    const std::vector<float>& means,
    float quant_step
) {
    if (y.size() != means.size()) {
        throw std::runtime_error("y and means_y sizes do not match");
    }
    std::vector<int32_t> symbols(y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        symbols[i] = cnz::round_to_even((y[i] - means[i]) / quant_step);
    }
    return symbols;
}

std::vector<int32_t> quantize_z(
    const std::vector<float>& z,
    const Shape4& shape,
    const std::vector<float>& medians,
    float quant_step
) {
    if (shape.count() != z.size()) {
        throw std::runtime_error("z shape does not match z file size");
    }
    if (medians.size() != static_cast<size_t>(shape.c)) {
        throw std::runtime_error("z_medians count does not match z channels");
    }
    const size_t plane = static_cast<size_t>(shape.h) * static_cast<size_t>(shape.w);
    std::vector<int32_t> symbols(z.size());
    for (int c = 0; c < shape.c; ++c) {
        const float median = medians[static_cast<size_t>(c)];
        for (size_t i = 0; i < plane; ++i) {
            const size_t offset = static_cast<size_t>(c) * plane + i;
            symbols[offset] = cnz::round_to_even((z[offset] - median) / quant_step);
        }
    }
    return symbols;
}

struct StreamPayload {
    uint32_t dtype = cnz::kDtypeInt16;
    std::vector<int32_t> symbols;
    std::vector<uint8_t> raw;
    std::vector<uint8_t> payload;
    int32_t min_symbol = 0;
    int32_t max_symbol = 0;
};

StreamPayload encode_stream(
    const std::vector<int32_t>& symbols,
    uint32_t codec,
    int zlib_level
) {
    if (symbols.empty()) {
        throw std::runtime_error("cannot encode empty symbol stream");
    }
    StreamPayload stream;
    stream.symbols = symbols;
    auto minmax = std::minmax_element(stream.symbols.begin(), stream.symbols.end());
    stream.min_symbol = *minmax.first;
    stream.max_symbol = *minmax.second;
    stream.dtype = cnz::can_store_int16(stream.symbols) ? cnz::kDtypeInt16 : cnz::kDtypeInt32;
    stream.raw = cnz::pack_symbols_bytes(stream.symbols, stream.dtype);
    stream.payload = cnz::compress_raw_bytes(stream.raw, codec, zlib_level);
    return stream;
}

std::string json_escape(const std::string& value) {
    std::string out;
    for (char ch : value) {
        if (ch == '\\' || ch == '"') {
            out.push_back('\\');
        }
        out.push_back(ch);
    }
    return out;
}

std::string build_header_json(
    const Metadata& metadata,
    const HyperParams& params,
    const StreamPayload& y,
    const StreamPayload& z,
    uint32_t codec
) {
    std::ostringstream out;
    out << "{";
    out << "\"format\":\"compressai-nano-hyper-ms-cpp-v1\",";
    out << "\"model_variant\":\"" << json_escape(params.model_variant) << "\",";
    out << "\"model_type\":\"mean_scale_hyperprior\",";
    out << "\"source_h\":" << metadata.source_h << ",";
    out << "\"source_w\":" << metadata.source_w << ",";
    out << "\"orig_h\":" << metadata.orig_h << ",";
    out << "\"orig_w\":" << metadata.orig_w << ",";
    out << "\"padded_h\":" << metadata.padded_h << ",";
    out << "\"padded_w\":" << metadata.padded_w << ",";
    out << "\"y_shape\":[1," << metadata.y_shape.c << "," << metadata.y_shape.h << "," << metadata.y_shape.w << "],";
    out << "\"z_shape\":[1," << metadata.z_shape.c << "," << metadata.z_shape.h << "," << metadata.z_shape.w << "],";
    out << "\"channels_y\":" << params.channels_y << ",";
    out << "\"channels_z\":" << params.channels_z << ",";
    out << "\"quant_step_y\":" << params.quant_step_y << ",";
    out << "\"quant_step_z\":" << params.quant_step_z << ",";
    out << "\"codec\":\"" << cnz::codec_name(codec) << "\",";
    out << "\"y_dtype\":\"" << cnz::dtype_name(y.dtype) << "\",";
    out << "\"z_dtype\":\"" << cnz::dtype_name(z.dtype) << "\",";
    out << "\"y_symbol_min\":" << y.min_symbol << ",";
    out << "\"y_symbol_max\":" << y.max_symbol << ",";
    out << "\"z_symbol_min\":" << z.min_symbol << ",";
    out << "\"z_symbol_max\":" << z.max_symbol << ",";
    out << "\"y_raw_size\":" << y.raw.size() << ",";
    out << "\"z_raw_size\":" << z.raw.size() << ",";
    out << "\"y_payload_size\":" << y.payload.size() << ",";
    out << "\"z_payload_size\":" << z.payload.size();
    out << "}";
    return out.str();
}

std::vector<uint8_t> build_hms_file(
    const std::string& header_json,
    const StreamPayload& y,
    const StreamPayload& z,
    uint32_t codec
) {
    std::vector<uint8_t> output;
    output.insert(output.end(), kMagic, kMagic + 4);
    append_le<uint32_t>(output, kVersion);
    append_le<uint32_t>(output, static_cast<uint32_t>(header_json.size()));
    append_le<uint32_t>(output, codec);
    append_le<uint32_t>(output, y.dtype);
    append_le<uint32_t>(output, z.dtype);
    append_le<uint64_t>(output, static_cast<uint64_t>(y.raw.size()));
    append_le<uint64_t>(output, static_cast<uint64_t>(z.raw.size()));
    append_le<uint64_t>(output, static_cast<uint64_t>(y.payload.size()));
    append_le<uint64_t>(output, static_cast<uint64_t>(z.payload.size()));
    output.insert(output.end(), header_json.begin(), header_json.end());
    output.insert(output.end(), y.payload.begin(), y.payload.end());
    output.insert(output.end(), z.payload.begin(), z.payload.end());
    return output;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_args(argc, argv);
        const std::string y_path = require_arg(args, "--y");
        const std::string z_path = require_arg(args, "--z");
        const std::string means_path = require_arg(args, "--means");
        const std::string params_path = require_arg(args, "--params");
        const std::string metadata_path = require_arg(args, "--metadata");
        const std::string output_path = require_arg(args, "--output");
        const std::string codec_name = optional_arg(args, "--codec", "zlib");
        const int zlib_level = std::stoi(optional_arg(args, "--zlib-level", "1"));

        const HyperParams params = load_params(params_path);
        const Metadata metadata = load_metadata(metadata_path);
        if (metadata.y_shape.c != params.channels_y || metadata.z_shape.c != params.channels_z) {
            throw std::runtime_error("metadata shapes do not match params channels");
        }

        const auto y_float = cnz::read_float32_file(y_path);
        const auto z_float = cnz::read_float32_file(z_path);
        const auto means_float = cnz::read_float32_file(means_path);
        if (y_float.size() != metadata.y_shape.count()) {
            throw std::runtime_error("y file size does not match y_shape");
        }
        if (z_float.size() != metadata.z_shape.count()) {
            throw std::runtime_error("z file size does not match z_shape");
        }
        if (means_float.size() != metadata.y_shape.count()) {
            throw std::runtime_error("means file size does not match y_shape");
        }

        const uint32_t codec = cnz::codec_from_name(codec_name);
        const auto y_symbols = quantize_y(y_float, means_float, params.quant_step_y);
        const auto z_symbols = quantize_z(z_float, metadata.z_shape, params.z_medians, params.quant_step_z);
        const StreamPayload y_stream = encode_stream(y_symbols, codec, zlib_level);
        const StreamPayload z_stream = encode_stream(z_symbols, codec, zlib_level);
        const std::string header_json = build_header_json(metadata, params, y_stream, z_stream, codec);
        const auto blob = build_hms_file(header_json, y_stream, z_stream, codec);
        write_binary_file(output_path, blob);

        const double pixels = static_cast<double>(metadata.orig_h) * static_cast<double>(metadata.orig_w);
        const double bpp = pixels > 0.0 ? static_cast<double>(blob.size() * 8) / pixels : 0.0;
        std::cout << "saved: " << output_path << "\n";
        std::cout << "metadata: " << metadata_path << "\n";
        std::cout << "y_shape: [1," << metadata.y_shape.c << "," << metadata.y_shape.h << "," << metadata.y_shape.w << "]\n";
        std::cout << "z_shape: [1," << metadata.z_shape.c << "," << metadata.z_shape.h << "," << metadata.z_shape.w << "]\n";
        std::cout << "y_dtype: " << cnz::dtype_name(y_stream.dtype) << "\n";
        std::cout << "z_dtype: " << cnz::dtype_name(z_stream.dtype) << "\n";
        std::cout << "y_symbol_min: " << y_stream.min_symbol << "\n";
        std::cout << "y_symbol_max: " << y_stream.max_symbol << "\n";
        std::cout << "z_symbol_min: " << z_stream.min_symbol << "\n";
        std::cout << "z_symbol_max: " << z_stream.max_symbol << "\n";
        std::cout << "payload_size: " << blob.size() << "\n";
        std::cout << "payload_bpp: " << bpp << "\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "hyper_ms_encode_cli error: " << exc.what() << "\n";
        return 1;
    }
}
