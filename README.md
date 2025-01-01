# SimpleLLama.cpp
Simple c++ api for LLama.Cpp. Which has way less breaking changes to the API. 

- Plain C/C++ implementation with minimal dependencies (Only LLama.cpp (but it is included))
- Runs on ARM as well (Tested on RPI 3,4 and 5)
- Written with performance in mind, still supports many backends

## API Features
This library offers support for:
- Running GGUF formatted large language models such as Phi-2, LLama, e.g.

### Basic usage

This is some example code for running basic inference:

```cpp
#include <simplellama.hpp>
...
/* Create new model params struct with for the model settings */
simplellama_model_params model_params;
model_params.model_llama = "phi-2.Q5_0.gguf"; /* Model name, downloaded automatically by cmake */

/* Make a new instance of SimpleLLama */
SimpleLLama sl(model_params);
    
/* Initialize the model runtime */
sl.init();

/* Some example questions that we want to get answered :) */
std::string question = "What came first, the egg or the chicken?";
    
/* Let the LLM answer */
std::cout << question << '\n';
std::string response = "response:" + sl.do_inference(question);
std::cout << response << '\n'; 
```
output:
```text
What came first, the egg or the chicken?
response: The egg
```

### Example code
For a full example, see the example code in [example/simple_text_demo/simple_text_demo.cpp](example/simple_text_demo/simple_text_demo.cpp).

## Building with CMake
Before using this library you will need the following packages installed:

- Working C++ compiler (GCC, Clang, MSVC (2017 or Higher))
- CMake
- Ninja (**Optional**, but preferred)

### Running the examples (CPU)
1. Clone this repo
2. Run:
```bash
cmake . -B build -G Ninja
```
3. Let CMake generate and run:
```bash
cd build && ninja
```
4. After building you can run (linux & mac):
```bash
./simple_text_demo
```
or (if using windows)
```bat
simple_text_demo.exe
```
## Selecting backends
Depending on your hardware configuration different software backends might leverage your hardware compute power better.
So check this out! As it might make the difference between 2 tok/s and 40 tok/s.

These backends are available:

| Backend | Best suited for: | CMake? |
|--|--|--|
| Vulkan | Generic Intel, AMD & NVidia gpu's | Add -DGGML_GGUF to configure command or add set(GGML_GGUF ON) to your CMakeLists before importing project|
| Cuda | NVidia GPU |  Add -DGGML_CUDA to configure command or add set(GGML_CUDA ON) to your CMakeLists before importing project|
| BLIS | All |  Add -DGGML_BLIS to configure command or add set(GGML_BLIS ON) to your CMakeLists before importing project|
| BLAS | All |  Add -DGGML_BLAS to configure command or add set(GGML_BLAS ON) to your CMakeLists before importing project|
| SYCL | Intel(>12th gen core) & NVidia GPU |  Add -DGGML_SYCL to configure command or add set(GGML_SYCL ON) to your CMakeLists before importing project|
| HIP |  AMD GPU | Add -DGGML_HIP to configure command or add set(GGML_HIP ON) to your CMakeLists before importing project|

### Using it in your project as library
Add this to your top-level CMakeLists file:
```cmake
include(FetchContent)
FetchContent_Declare(
    SimpleLLama
    GIT_REPOSITORY https://github.com/HCL_Hbot/SimpleLLama
    GIT_TAG main
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/lib/SimpleLLama
)
FetchContent_MakeAvailable(SimpleLLama)
...
target_link_libraries(YOUR_EXECUTABLE simplellama)
```
Or manually clone this repo and add the library to your project using:
```cmake
add_subdirectory(lowwi)
...
target_link_libraries(YOUR_EXECUTABLE simplellama)
```


## Aditional documentation
See our [wiki](https://HCL-Hbot.github.io/SimpleLLama/)...

## Todo
- Make it ovos compatible
- Add compatibility for ARM SOC

## License
This work is licensed under the MIT License.

