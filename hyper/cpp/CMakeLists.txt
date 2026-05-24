cmake_minimum_required(VERSION 3.16)
project(hyper_ms_entropy_codec LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ZLIB REQUIRED)

add_library(cnz_codec src/cnz_codec.cpp)
target_include_directories(cnz_codec PUBLIC include)
target_link_libraries(cnz_codec PUBLIC ZLIB::ZLIB)

add_executable(hyper_ms_encode_cli src/hyper_ms_encode_cli.cpp)
target_link_libraries(hyper_ms_encode_cli PRIVATE cnz_codec)
