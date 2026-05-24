#include "cnz_codec.h"

#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace cnz {
namespace {

constexpr uint8_t kMagic[4] = {'C', 'N', 'Z', '4'};
constexpr size_t kFixedHeaderSize = 68;

template <typename T>
void append_le(std::vector<uint8_t>& out, T value) {
    static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value,
                  "append_le only supports scalar values");
    uint8_t bytes[sizeof(T)];
    std::memcpy(bytes, &value, sizeof(T));
    out.insert(out.end(), bytes, bytes + sizeof(T));
}

template <typename T>
T read_le(const std::vector<uint8_t>& data, size_t& offset) {
    if (offset + sizeof(T) > data.size()) {
        throw std::runtime_error("unexpected end of file while reading CNZ header");
    }
    T value{};
    std::memcpy(&value, data.data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}

std::vector<uint8_t> read_binary_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open file for reading: " + path);
    }
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    if (size < 0) {
        throw std::runtime_error("failed to determine file size: " + path);
    }
    input.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    if (!data.empty()) {
        input.read(reinterpret_cast<char*>(data.data()), size);
    }
    return data;
}

void write_binary_file(const std::string& path, const std::vector<uint8_t>& data) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open file for writing: " + path);
    }
    if (!data.empty()) {
        output.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    }
}

std::string read_text_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open JSON file: " + path);
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

float extract_float(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*([-+0-9.eE]+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("missing numeric JSON key: " + key);
    }
    return std::stof(match[1].str());
}

uint32_t extract_uint(const std::string& text, const std::string& key) {
    const float value = extract_float(text, key);
    if (value < 0.0f) {
        throw std::runtime_error("negative JSON integer key: " + key);
    }
    return static_cast<uint32_t>(value);
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

void validate_header(const CnzHeader& header) {
    if (header.version != kVersion) {
        throw std::runtime_error("unsupported CNZ version: " + std::to_string(header.version));
    }
    if (header.latent_c == 0 || header.latent_h == 0 || header.latent_w == 0) {
        throw std::runtime_error("latent dimensions must be positive");
    }
    if (header.quant_step <= 0.0f) {
        throw std::runtime_error("quant_step must be positive");
    }
    if (header.num_medians != header.latent_c) {
        throw std::runtime_error("num_medians does not match latent_c");
    }
    const uint32_t expected_header_size = static_cast<uint32_t>(kFixedHeaderSize + header.num_medians * sizeof(float));
    if (header.header_size != expected_header_size) {
        throw std::runtime_error("CNZ header_size mismatch");
    }
    if (header.dtype != kDtypeInt16 && header.dtype != kDtypeInt32) {
        throw std::runtime_error("unsupported CNZ dtype: " + std::to_string(header.dtype));
    }
    if (header.codec != kCodecNone && header.codec != kCodecZlib &&
        header.codec != kCodecLz4 && header.codec != kCodecZstd) {
        throw std::runtime_error("unsupported CNZ codec: " + std::to_string(header.codec));
    }
}

}  // namespace

EntropyParams load_entropy_params_json(const std::string& path) {
    const std::string text = read_text_file(path);
    EntropyParams params;
    params.channels = extract_uint(text, "channels");
    params.quant_step = extract_float(text, "quant_step");
    params.downsampling_factor = extract_uint(text, "downsampling_factor");
    params.model_config_name = extract_string_optional(text, "model_config_name");
    params.medians = extract_float_array(text, "medians");
    if (params.channels == 0) {
        throw std::runtime_error("entropy params channels must be positive");
    }
    if (params.quant_step <= 0.0f) {
        throw std::runtime_error("entropy params quant_step must be positive");
    }
    if (params.medians.size() != params.channels) {
        throw std::runtime_error("entropy params medians count does not match channels");
    }
    return params;
}

int32_t round_to_even(float x) {
    // Match PyTorch torch.round: nearest integer, ties go to nearest even.
    // Non-.5 values follow normal nearest rounding. std::round is not used
    // because it rounds .5 away from zero on many platforms.
    const float lower_f = std::floor(x);
    const float frac = x - lower_f;
    int64_t lower = static_cast<int64_t>(lower_f);
    int64_t result = lower;
    if (frac > 0.5f) {
        result = lower + 1;
    } else if (frac == 0.5f) {
        result = (lower % 2 == 0) ? lower : lower + 1;
    }
    if (result < std::numeric_limits<int32_t>::min() ||
        result > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("rounded symbol is outside int32 range");
    }
    return static_cast<int32_t>(result);
}

void validate_shape(int C, int H, int W, const EntropyParams& params) {
    if (C <= 0 || H <= 0 || W <= 0) {
        throw std::runtime_error("latent dimensions must be positive");
    }
    if (static_cast<uint32_t>(C) != params.channels) {
        throw std::runtime_error("latent C does not match entropy params channels");
    }
    if (params.medians.size() != static_cast<size_t>(C)) {
        throw std::runtime_error("median count does not match latent C");
    }
    if (params.quant_step <= 0.0f) {
        throw std::runtime_error("quant_step must be positive");
    }
}

std::vector<int32_t> quantize_to_int32(
    const float* y,
    const float* medians,
    int C,
    int H,
    int W,
    float quant_step
) {
    if (!y || !medians) {
        throw std::runtime_error("quantize_to_int32 received null pointer");
    }
    if (C <= 0 || H <= 0 || W <= 0 || quant_step <= 0.0f) {
        throw std::runtime_error("invalid quantize_to_int32 shape or quant_step");
    }
    const size_t plane = static_cast<size_t>(H) * static_cast<size_t>(W);
    std::vector<int32_t> symbols(static_cast<size_t>(C) * plane);
    for (int c = 0; c < C; ++c) {
        const float median = medians[c];
        for (size_t i = 0; i < plane; ++i) {
            const size_t offset = static_cast<size_t>(c) * plane + i;
            symbols[offset] = round_to_even((y[offset] - median) / quant_step);
        }
    }
    return symbols;
}

std::vector<int32_t> quantize_latent(
    const float* y,
    const float* medians,
    int C,
    int H,
    int W,
    float quant_step
) {
    return quantize_to_int32(y, medians, C, H, W, quant_step);
}

std::vector<float> dequantize_to_float(
    const int32_t* symbols,
    const float* medians,
    int C,
    int H,
    int W,
    float quant_step
) {
    if (!symbols || !medians) {
        throw std::runtime_error("dequantize_to_float received null pointer");
    }
    if (C <= 0 || H <= 0 || W <= 0 || quant_step <= 0.0f) {
        throw std::runtime_error("invalid dequantize_to_float shape or quant_step");
    }
    const size_t plane = static_cast<size_t>(H) * static_cast<size_t>(W);
    std::vector<float> output(static_cast<size_t>(C) * plane);
    for (int c = 0; c < C; ++c) {
        const float median = medians[c];
        for (size_t i = 0; i < plane; ++i) {
            const size_t offset = static_cast<size_t>(c) * plane + i;
            output[offset] = static_cast<float>(symbols[offset]) * quant_step + median;
        }
    }
    return output;
}

std::vector<float> dequantize_latent(
    const int32_t* symbols,
    const float* medians,
    int C,
    int H,
    int W,
    float quant_step
) {
    return dequantize_to_float(symbols, medians, C, H, W, quant_step);
}

bool can_store_int16(const std::vector<int32_t>& symbols) {
    for (int32_t value : symbols) {
        if (value < std::numeric_limits<int16_t>::min() ||
            value > std::numeric_limits<int16_t>::max()) {
            return false;
        }
    }
    return true;
}

uint32_t codec_from_name(const std::string& codec) {
    if (codec == "none") return kCodecNone;
    if (codec == "zlib") return kCodecZlib;
    if (codec == "lz4") return kCodecLz4;
    if (codec == "zstd") return kCodecZstd;
    throw std::runtime_error("unsupported codec name: " + codec);
}

std::string codec_name(uint32_t codec) {
    if (codec == kCodecNone) return "none";
    if (codec == kCodecZlib) return "zlib";
    if (codec == kCodecLz4) return "lz4";
    if (codec == kCodecZstd) return "zstd";
    return "unknown";
}

std::string dtype_name(uint32_t dtype) {
    if (dtype == kDtypeInt16) return "int16";
    if (dtype == kDtypeInt32) return "int32";
    return "unknown";
}

size_t dtype_size(uint32_t dtype) {
    if (dtype == kDtypeInt16) return sizeof(int16_t);
    if (dtype == kDtypeInt32) return sizeof(int32_t);
    throw std::runtime_error("unsupported dtype code: " + std::to_string(dtype));
}

std::vector<uint8_t> pack_symbols_bytes(const std::vector<int32_t>& symbols, uint32_t dtype) {
    std::vector<uint8_t> raw;
    if (dtype == kDtypeInt16) {
        raw.resize(symbols.size() * sizeof(int16_t));
        for (size_t i = 0; i < symbols.size(); ++i) {
            if (symbols[i] < std::numeric_limits<int16_t>::min() ||
                symbols[i] > std::numeric_limits<int16_t>::max()) {
                throw std::runtime_error("symbol overflows int16");
            }
            const int16_t value = static_cast<int16_t>(symbols[i]);
            std::memcpy(raw.data() + i * sizeof(int16_t), &value, sizeof(int16_t));
        }
    } else if (dtype == kDtypeInt32) {
        raw.resize(symbols.size() * sizeof(int32_t));
        std::memcpy(raw.data(), symbols.data(), raw.size());
    } else {
        throw std::runtime_error("unsupported dtype in pack_symbols_bytes");
    }
    return raw;
}

std::vector<int32_t> unpack_symbols_bytes(const std::vector<uint8_t>& raw, uint32_t dtype) {
    if (dtype == kDtypeInt16) {
        if (raw.size() % sizeof(int16_t) != 0) {
            throw std::runtime_error("int16 raw payload size is not aligned");
        }
        std::vector<int32_t> symbols(raw.size() / sizeof(int16_t));
        for (size_t i = 0; i < symbols.size(); ++i) {
            int16_t value = 0;
            std::memcpy(&value, raw.data() + i * sizeof(int16_t), sizeof(int16_t));
            symbols[i] = static_cast<int32_t>(value);
        }
        return symbols;
    }
    if (dtype == kDtypeInt32) {
        if (raw.size() % sizeof(int32_t) != 0) {
            throw std::runtime_error("int32 raw payload size is not aligned");
        }
        std::vector<int32_t> symbols(raw.size() / sizeof(int32_t));
        std::memcpy(symbols.data(), raw.data(), raw.size());
        return symbols;
    }
    throw std::runtime_error("unsupported dtype in unpack_symbols_bytes");
}

std::vector<uint8_t> encode_symbols_to_payload(
    const std::vector<int32_t>& symbols,
    uint32_t dtype,
    uint32_t codec,
    int zlib_level
) {
    const auto raw = pack_symbols_bytes(symbols, dtype);
    return compress_raw_bytes(raw, codec, zlib_level);
}

std::vector<int32_t> decode_symbols_from_payload(
    const std::vector<uint8_t>& payload,
    uint32_t dtype,
    uint32_t codec,
    size_t expected_symbol_count
) {
    const size_t expected_raw_size = expected_symbol_count * dtype_size(dtype);
    const auto raw = decompress_raw_bytes(payload, codec, expected_raw_size);
    auto symbols = unpack_symbols_bytes(raw, dtype);
    if (symbols.size() != expected_symbol_count) {
        throw std::runtime_error("decoded symbol count mismatch");
    }
    return symbols;
}

std::vector<uint8_t> compress_raw_bytes(
    const std::vector<uint8_t>& raw,
    uint32_t codec,
    int zlib_level
) {
    if (codec == kCodecNone) {
        return raw;
    }
    if (codec == kCodecZlib) {
        uLongf bound = compressBound(static_cast<uLong>(raw.size()));
        std::vector<uint8_t> output(bound);
        int rc = compress2(output.data(), &bound, raw.data(), static_cast<uLong>(raw.size()), zlib_level);
        if (rc != Z_OK) {
            throw std::runtime_error("zlib compress2 failed with code " + std::to_string(rc));
        }
        output.resize(bound);
        return output;
    }
    if (codec == kCodecLz4) {
        throw std::runtime_error("LZ4 support was not compiled in");
    }
    if (codec == kCodecZstd) {
        throw std::runtime_error("Zstd support was not compiled in");
    }
    throw std::runtime_error("unsupported codec");
}

std::vector<uint8_t> decompress_raw_bytes(
    const std::vector<uint8_t>& payload,
    uint32_t codec,
    size_t expected_raw_size
) {
    if (codec == kCodecNone) {
        if (payload.size() != expected_raw_size) {
            throw std::runtime_error("uncompressed payload size mismatch");
        }
        return payload;
    }
    if (codec == kCodecZlib) {
        std::vector<uint8_t> raw(expected_raw_size);
        uLongf dest_len = static_cast<uLongf>(raw.size());
        int rc = uncompress(raw.data(), &dest_len, payload.data(), static_cast<uLong>(payload.size()));
        if (rc != Z_OK) {
            throw std::runtime_error("zlib uncompress failed with code " + std::to_string(rc));
        }
        if (dest_len != expected_raw_size) {
            throw std::runtime_error("zlib decompressed size mismatch");
        }
        return raw;
    }
    if (codec == kCodecLz4) {
        throw std::runtime_error("LZ4 support was not compiled in");
    }
    if (codec == kCodecZstd) {
        throw std::runtime_error("Zstd support was not compiled in");
    }
    throw std::runtime_error("unsupported codec");
}

void write_cnz_file(
    const std::string& path,
    const CnzHeader& input_header,
    const std::vector<float>& medians,
    const std::vector<uint8_t>& payload
) {
    CnzHeader header = input_header;
    header.version = kVersion;
    header.num_medians = static_cast<uint32_t>(medians.size());
    header.header_size = static_cast<uint32_t>(kFixedHeaderSize + medians.size() * sizeof(float));
    header.payload_size = static_cast<uint64_t>(payload.size());
    validate_header(header);

    std::vector<uint8_t> output;
    output.reserve(header.header_size + payload.size());
    output.insert(output.end(), kMagic, kMagic + 4);
    append_le<uint32_t>(output, header.version);
    append_le<uint32_t>(output, header.header_size);
    append_le<uint32_t>(output, header.orig_h);
    append_le<uint32_t>(output, header.orig_w);
    append_le<uint32_t>(output, header.padded_h);
    append_le<uint32_t>(output, header.padded_w);
    append_le<uint32_t>(output, header.latent_c);
    append_le<uint32_t>(output, header.latent_h);
    append_le<uint32_t>(output, header.latent_w);
    append_le<uint32_t>(output, header.down_factor);
    append_le<uint32_t>(output, header.dtype);
    append_le<uint32_t>(output, header.codec);
    append_le<float>(output, header.quant_step);
    append_le<uint32_t>(output, header.num_medians);
    append_le<uint64_t>(output, header.payload_size);
    for (float median : medians) {
        append_le<float>(output, median);
    }
    output.insert(output.end(), payload.begin(), payload.end());
    write_binary_file(path, output);
}

CnzFile read_cnz_file(const std::string& path) {
    const std::vector<uint8_t> data = read_binary_file(path);
    if (data.size() < kFixedHeaderSize) {
        throw std::runtime_error("CNZ file is too small");
    }
    if (!std::equal(kMagic, kMagic + 4, data.begin())) {
        throw std::runtime_error("invalid CNZ magic");
    }
    size_t offset = 4;
    CnzHeader header;
    header.version = read_le<uint32_t>(data, offset);
    header.header_size = read_le<uint32_t>(data, offset);
    header.orig_h = read_le<uint32_t>(data, offset);
    header.orig_w = read_le<uint32_t>(data, offset);
    header.padded_h = read_le<uint32_t>(data, offset);
    header.padded_w = read_le<uint32_t>(data, offset);
    header.latent_c = read_le<uint32_t>(data, offset);
    header.latent_h = read_le<uint32_t>(data, offset);
    header.latent_w = read_le<uint32_t>(data, offset);
    header.down_factor = read_le<uint32_t>(data, offset);
    header.dtype = read_le<uint32_t>(data, offset);
    header.codec = read_le<uint32_t>(data, offset);
    header.quant_step = read_le<float>(data, offset);
    header.num_medians = read_le<uint32_t>(data, offset);
    header.payload_size = read_le<uint64_t>(data, offset);
    validate_header(header);
    if (data.size() < header.header_size + header.payload_size) {
        throw std::runtime_error("CNZ file is truncated");
    }
    std::vector<float> medians(header.num_medians);
    for (uint32_t i = 0; i < header.num_medians; ++i) {
        medians[i] = read_le<float>(data, offset);
    }
    if (offset != header.header_size) {
        throw std::runtime_error("CNZ median section size mismatch");
    }
    std::vector<uint8_t> payload(
        data.begin() + static_cast<std::ptrdiff_t>(header.header_size),
        data.begin() + static_cast<std::ptrdiff_t>(header.header_size + header.payload_size)
    );
    return CnzFile{header, medians, payload};
}

std::vector<float> read_float32_file(const std::string& path) {
    const std::vector<uint8_t> bytes = read_binary_file(path);
    if (bytes.size() % sizeof(float) != 0) {
        throw std::runtime_error("float32 file size is not divisible by 4");
    }
    std::vector<float> values(bytes.size() / sizeof(float));
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
}

void write_float32_file(const std::string& path, const std::vector<float>& data) {
    std::vector<uint8_t> bytes(data.size() * sizeof(float));
    std::memcpy(bytes.data(), data.data(), bytes.size());
    write_binary_file(path, bytes);
}

EncodeResult encode_latent_to_payload(
    const float* latent,
    int C,
    int H,
    int W,
    const EntropyParams& params,
    uint32_t codec,
    int zlib_level
) {
    validate_shape(C, H, W, params);
    EncodeResult result;
    result.symbols = quantize_to_int32(latent, params.medians.data(), C, H, W, params.quant_step);
    auto minmax = std::minmax_element(result.symbols.begin(), result.symbols.end());
    result.min_symbol = *minmax.first;
    result.max_symbol = *minmax.second;
    result.header.dtype = can_store_int16(result.symbols) ? kDtypeInt16 : kDtypeInt32;
    result.header.codec = codec;
    result.raw_bytes = pack_symbols_bytes(result.symbols, result.header.dtype);
    result.compressed_payload = compress_raw_bytes(result.raw_bytes, codec, zlib_level);
    result.header.latent_c = static_cast<uint32_t>(C);
    result.header.latent_h = static_cast<uint32_t>(H);
    result.header.latent_w = static_cast<uint32_t>(W);
    result.header.quant_step = params.quant_step;
    result.header.down_factor = params.downsampling_factor;
    result.header.num_medians = params.channels;
    result.header.payload_size = result.compressed_payload.size();
    result.header.header_size = static_cast<uint32_t>(kFixedHeaderSize + params.medians.size() * sizeof(float));
    return result;
}

}  // namespace cnz
