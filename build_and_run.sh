#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o nounset

# The absolute path to the directory of this script.
MY_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "${MY_DIR}"

# Set up MLIR
# mkdir -p ${MY_DIR}/thirdparty
INSTALL_DIR=${MY_DIR}/thirdparty



# ## INSTALL LLVM/MLIR
LLVM_REPO=${MY_DIR}/thirdparty/llvm-project
echo "$LLVM_REPO"
LLVM_BUILD=${LLVM_REPO}/build

mkdir -p $LLVM_BUILD

cd "${LLVM_BUILD}"
cmake  -G Ninja "-H$LLVM_REPO/llvm" \
     "-B$LLVM_BUILD" \
     -DLLVM_INSTALL_UTILS=ON \
     -DLLVM_ENABLE_PROJECTS="mlir;clang" \
     -DCMAKE_BUILD_TYPE=DEBUG \
     -DLLVM_INCLUDE_TOOLS=ON \
     -DLLVM_BUILD_EXAMPLES=ON \
     -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
     -DLLVM_TARGETS_TO_BUILD="host" \
     -DCMAKE_C_COMPILER=clang \
     -DCMAKE_CXX_COMPILER=clang++

ninja && ninja check-mlir


#INSTALL
cd "${MY_DIR}"
mkdir -p build && cd build
cmake -GNinja .. \
  -DLLVM_DIR=$LLVM_REPO/build/lib/cmake/llvm \
  -DMLIR_DIR=$LLVM_REPO/build/lib/cmake/mlir \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DLLVM_ENABLE_LLD=ON \
  -DCMAKE_C_COMPILER=clang \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
  -DHETEACC_ENABLE_BINDINGS_PYTHON=OFF \
  -DCMAKE_CXX_COMPILER=clang++

# cmake --build . --target heteacc-opt  DEBUG
ninja

#BUILD hardware with sbt and run cases
cd "${MY_DIR}/hardware"
bash build.sh
