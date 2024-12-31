cmake_minimum_required(VERSION 3.14..3.21)

project(simplellama CXX)

include(FetchContent)
set(GGML_CCACHE OFF)
option(GGML_VULKAN "Run with vulkan backend" OFF)
option(GGML_CUDA "Run with CUDA backend" OFF)
option(GGML_PERF ON)
option(SIMPLELLAMA_BUILD_EXAMPLE "Build example" ON)

find_package(SDL2 REQUIRED)


if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(CMAKE_CXX_STANDARD 17)
else()
    set(SIMPLELLAMA_BUILD_EXAMPLE OFF)
endif()

if(TARGET ggml)
else()
FetchContent_Declare(
    ggml
    GIT_REPOSITORY https://github.com/ggerganov/ggml.git
    GIT_TAG 6a7a034e117f189df4d13665b9b604638ddca468
)
FetchContent_MakeAvailable(ggml)
endif()

add_library(llama ${CMAKE_CURRENT_LIST_DIR}/llama/llama.cpp
                  ${CMAKE_CURRENT_LIST_DIR}/llama/llama-grammar.cpp
                  ${CMAKE_CURRENT_LIST_DIR}/llama/llama-sampling.cpp
                  ${CMAKE_CURRENT_LIST_DIR}/llama/llama-vocab.cpp
                  ${CMAKE_CURRENT_LIST_DIR}/llama/unicode.cpp
                  ${CMAKE_CURRENT_LIST_DIR}/llama/unicode-data.cpp)
target_include_directories(llama PUBLIC ${CMAKE_CURRENT_LIST_DIR}/llama)

target_link_libraries(llama ggml)

add_library(simplellama ${CMAKE_CURRENT_LIST_DIR}/src/simplellama.cpp)
target_include_directories(simplellama PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src)
target_link_libraries(simplellama llama ggml ${SDL2_LIBRARIES})

if(SIMPLELLAMA_BUILD_EXAMPLE)

add_executable(simple_text_demo ${CMAKE_CURRENT_LIST_DIR}/example/simple_text_demo/simple_text_demo.cpp)
target_link_libraries(simple_text_demo simplellama)

if(POLICY CMP0135)
	cmake_policy(SET CMP0135 NEW)
	set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
endif()

FetchContent_Declare(
    phi-2
    URL https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q5_0.gguf
    SOURCE_DIR ${CMAKE_BINARY_DIR}
    URL_HASH SHA256=93143aff6d8566aac56fa88720dd8099ff076efac065dcd3538ae61a61ddd335
    DOWNLOAD_NO_EXTRACT YES
)
FetchContent_MakeAvailable(phi-2)

endif()