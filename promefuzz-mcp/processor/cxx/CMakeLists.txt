cmake_minimum_required(VERSION 3.15)
project(PromeFuzzProcessor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use clang as the compiler
set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/lib/llvm-18/bin/clang++")

# Direct include and library paths
set(LLVM_ROOT "/usr/lib/llvm-18")
include_directories(
    ${LLVM_ROOT}/include
    ${LLVM_ROOT}/lib/clang/18/include
)

link_directories(${LLVM_ROOT}/lib)

add_executable(preprocessor
    preprocessor.cc
)

add_executable(cgprocessor
    cgprocessor.cc
)

# Link with clang shared library which includes everything
target_link_libraries(preprocessor
    PRIVATE
    clang
    ${LLVM_ROOT}/lib/libclang-cpp.so.18.1
    ${LLVM_ROOT}/lib/libLLVM.so.18.1
    z
    m
    tinfo
    dl
    pthread
)

target_link_libraries(cgprocessor
    PRIVATE
    clang
    ${LLVM_ROOT}/lib/libclang-cpp.so.18.1
    ${LLVM_ROOT}/lib/libLLVM.so.18.1
    z
    m
    tinfo
    dl
    pthread
)
