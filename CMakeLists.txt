cmake_minimum_required(VERSION 3.4...3.18)
project(py-qwen3-asr-cpp)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -fPIC")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
endif()

add_subdirectory(pybind11)
add_subdirectory(qwen3-asr.cpp)

pybind11_add_module(_py_qwen3_asr_cpp
	src/main.cpp
)

target_link_libraries (_py_qwen3_asr_cpp PRIVATE qwen3_asr forced_aligner)
