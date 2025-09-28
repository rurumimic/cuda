cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

option(ENABLE_CUDA_DEBUG "Enable CUDA debug (use -G)" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures (e.g. -DCMAKE_CUDA_ARCHITECTURES=86;80;75)")

project(sync VERSION 1.0 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(${PROJECT_NAME}
  src/main.cu
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

if (DEFINED CMAKE_CUDA_COMPILER_ID AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  if(ENABLE_CUDA_DEBUG)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
  else()
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  endif()
  target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    # $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
  )
endif()

target_compile_options(${PROJECT_NAME} PRIVATE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>>:-Wall -Wextra -Wpedantic>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
)

target_include_directories(${PROJECT_NAME}
  PRIVATE ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)

include(cmake/format.cmake)
