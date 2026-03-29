cmake_minimum_required(VERSION 3.15)
project(PromeFuzzProcessor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Use clang when available, but don't hardcode distro-specific paths.
find_program(CLANG_BIN NAMES clang)
find_program(CLANGXX_BIN NAMES clang++)
if(CLANG_BIN)
    set(CMAKE_C_COMPILER "${CLANG_BIN}" CACHE FILEPATH "" FORCE)
endif()
if(CLANGXX_BIN)
    set(CMAKE_CXX_COMPILER "${CLANGXX_BIN}" CACHE FILEPATH "" FORCE)
endif()

# Resolve LLVM installation dynamically.
find_program(LLVM_CONFIG_BIN NAMES llvm-config llvm-config-18 llvm-config-17 llvm-config-16 llvm-config-15 llvm-config-14)
if(NOT LLVM_CONFIG_BIN)
    message(FATAL_ERROR "llvm-config not found; install llvm dev packages")
endif()

execute_process(
    COMMAND ${LLVM_CONFIG_BIN} --prefix
    OUTPUT_VARIABLE LLVM_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND ${LLVM_CONFIG_BIN} --includedir
    OUTPUT_VARIABLE LLVM_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND ${LLVM_CONFIG_BIN} --libdir
    OUTPUT_VARIABLE LLVM_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND ${LLVM_CONFIG_BIN} --version
    OUTPUT_VARIABLE LLVM_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX MATCH "^[0-9]+" LLVM_VERSION_MAJOR "${LLVM_VERSION}")

include_directories(${LLVM_INCLUDE_DIR})
if(LLVM_VERSION_MAJOR AND EXISTS "${LLVM_LIB_DIR}/clang/${LLVM_VERSION_MAJOR}/include")
    include_directories("${LLVM_LIB_DIR}/clang/${LLVM_VERSION_MAJOR}/include")
endif()

link_directories(${LLVM_LIB_DIR})

find_library(CLANG_CPP_LIB NAMES clang-cpp libclang-cpp HINTS ${LLVM_LIB_DIR} REQUIRED)
find_library(LLVM_CORE_LIB NAMES LLVM libLLVM HINTS ${LLVM_LIB_DIR} REQUIRED)

add_executable(preprocessor
    preprocessor.cc
)

add_executable(cgprocessor
    cgprocessor.cc
)

# Link with clang shared library which includes everything
target_link_libraries(preprocessor
    PRIVATE
    ${CLANG_CPP_LIB}
    ${LLVM_CORE_LIB}
    z
    m
    tinfo
    dl
    pthread
)

target_link_libraries(cgprocessor
    PRIVATE
    ${CLANG_CPP_LIB}
    ${LLVM_CORE_LIB}
    z
    m
    tinfo
    dl
    pthread
)
