#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cnz {

constexpr uint32_t kVersion = 1;
constexpr uint32_t kDtypeInt16 = 1;
constexpr uint32_t kDtypeInt32 = 2;
constexpr uint32_t kCodecNone = 1;
constexpr uint32_t kCodecZlib = 2;
constexpr uint32_t kCodecLz4 = 3;
constexpr uint32_t kCodecZstd = 4;

struct CnzHeader {
    uint32_t version = kVersion;
    uint32_t header_size = 0;
    uint32_t orig_h = 0;
    uint32_t orig_w = 0;
    uint32_t padded_h = 0;
    uint32_t padded_w = 0;
    uint32_t latent_c = 0;
    uint32_t latent_h = 0;
    uint32_t latent_w = 0;
    uint32_t down_factor = 16;
    uint32_t dtype = kDtypeInt16;
    uint32_t codec = kCodecZlib;
    float quant_step = 1.0f;
    uint32_t num_medians = 0;
    uint64_t payload_size = 0;
};

struct EntropyParams {
    uint32_t channels = 0;
    float quant_step = 1.0f;
    uint32_t downsampling_factor = 16;
    std::string model_config_name;
    std::vector<float> medians;
};

struct CnzFile {
    CnzHeader header;
    std::vector<float> medians;
    std::vector<uint8_t> payload;
};

struct EncodeResult {
    CnzHeader header;
    std::vector<int32_t> symbols;
    std::vector<uint8_t> raw_bytes;
    std::vector<uint8_t> compressed_payload;
    int32_t min_symbol = 0;
    int32_t max_symbol = 0;
};

EntropyParams load_entropy_params_json(const std::string& path);

int32_t round_to_even(float x);

std::vector<int32_t> quantize_to_int32(
    const float* y,
    const float* medians,
    int C,
    int H,
    int W,
    float quant_step
);

std::vector<int32_t> quantize_latent(
    const float* y,
    const float* medians,
    int C,
    int H,
    int W,
    float quant_step
);

std::vector<float> dequantize_to_float(
    const int32_t* symbols,
    const float* medians,
    int C,
    int H,
    int W,
    float quant_step
);

std::vector<float> dequantize_latent(
    const int32_t* symbols,
    const float* medians,
    int C,
    int H,
    int W,
    float quant_step
);

bool can_store_int16(const std::vector<int32_t>& symbols);
uint32_t codec_from_name(const std::string& codec);
std::string codec_name(uint32_t codec);
std::string dtype_name(uint32_t dtype);
size_t dtype_size(uint32_t dtype);

std::vector<uint8_t> pack_symbols_bytes(const std::vector<int32_t>& symbols, uint32_t dtype);
std::vector<int32_t> unpack_symbols_bytes(const std::vector<uint8_t>& raw, uint32_t dtype);
std::vector<uint8_t> encode_symbols_to_payload(
    const std::vector<int32_t>& symbols,
    uint32_t dtype,
    uint32_t codec,
    int zlib_level
);
std::vector<int32_t> decode_symbols_from_payload(
    const std::vector<uint8_t>& payload,
    uint32_t dtype,
    uint32_t codec,
    size_t expected_symbol_count
);

std::vector<uint8_t> compress_raw_bytes(
    const std::vector<uint8_t>& raw,
    uint32_t codec,
    int zlib_level
);

std::vector<uint8_t> decompress_raw_bytes(
    const std::vector<uint8_t>& payload,
    uint32_t codec,
    size_t expected_raw_size
);

void write_cnz_file(
    const std::string& path,
    const CnzHeader& header,
    const std::vector<float>& medians,
    const std::vector<uint8_t>& payload
);

CnzFile read_cnz_file(const std::string& path);

std::vector<float> read_float32_file(const std::string& path);
void write_float32_file(const std::string& path, const std::vector<float>& data);

EncodeResult encode_latent_to_payload(
    const float* latent,
    int C,
    int H,
    int W,
    const EntropyParams& params,
    uint32_t codec,
    int zlib_level
);

void validate_shape(int C, int H, int W, const EntropyParams& params);

}  // namespace cnz
