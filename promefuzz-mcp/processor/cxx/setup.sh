#!/bin/bash
set -e

CLANG_INSTALL_DIR=${CLANG_INSTALL_DIR:-$(llvm-config --prefix 2>/dev/null || echo "/usr/lib/llvm-18")}

echo "Building PromeFuzz MCP processors..."
echo "Using CLANG_INSTALL_DIR: $CLANG_INSTALL_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

rm -rf build
mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$CLANG_INSTALL_DIR" \
    -DLLVM_DIR="$CLANG_INSTALL_DIR/lib/cmake/llvm"

make -j$(nproc)

mkdir -p ../build/bin
ln -sf ../build/preprocessor ../build/bin/ 2>/dev/null || true
ln -sf ../build/cgprocessor ../build/bin/ 2>/dev/null || true

echo "Build complete! Binaries: processor/build/bin/{preprocessor,cgprocessor}"