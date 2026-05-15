cmake_minimum_required(VERSION 3.16)
project(compress_cnz_codec LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_LZ4 "Enable optional LZ4 codec support" OFF)
option(ENABLE_ZSTD "Enable optional Zstandard codec support" OFF)

find_package(ZLIB REQUIRED)

add_library(cnz_codec
    src/cnz_codec.cpp
)
target_include_directories(cnz_codec PUBLIC include)
target_link_libraries(cnz_codec PUBLIC ZLIB::ZLIB)

if(ENABLE_LZ4)
    find_library(LZ4_LIBRARY lz4)
    if(LZ4_LIBRARY)
        target_compile_definitions(cnz_codec PUBLIC CNZ_ENABLE_LZ4=1)
        target_link_libraries(cnz_codec PUBLIC ${LZ4_LIBRARY})
    else()
        message(WARNING "ENABLE_LZ4=ON but lz4 was not found; building without LZ4")
    endif()
endif()

if(ENABLE_ZSTD)
    find_library(ZSTD_LIBRARY zstd)
    if(ZSTD_LIBRARY)
        target_compile_definitions(cnz_codec PUBLIC CNZ_ENABLE_ZSTD=1)
        target_link_libraries(cnz_codec PUBLIC ${ZSTD_LIBRARY})
    else()
        message(WARNING "ENABLE_ZSTD=ON but zstd was not found; building without ZSTD")
    endif()
endif()

add_executable(cnz_encode_cli src/cnz_encode_cli.cpp)
target_link_libraries(cnz_encode_cli PRIVATE cnz_codec)

add_executable(cnz_decode_cli src/cnz_decode_cli.cpp)
target_link_libraries(cnz_decode_cli PRIVATE cnz_codec)

add_executable(cnz_benchmark_cli src/cnz_benchmark_cli.cpp)
target_link_libraries(cnz_benchmark_cli PRIVATE cnz_codec)

add_executable(hyper_ms_encode_cli src/hyper_ms_encode_cli.cpp)
target_link_libraries(hyper_ms_encode_cli PRIVATE cnz_codec)
