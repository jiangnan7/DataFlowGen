#!/bin/bash

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <command>  <input_file> <output_file>"
  exit 1
fi

command="$1"
input_file="$2"
output_file="$3"

if [ "$command" = "C" ]; then
  ./thirdparty/Polygeist/build/bin/cgeist -function=* -S -memref-fullrank \
  -raise-scf-to-affine  "$input_file" -o "$output_file"
elif [ "$command" = "T" ]; then
  ./thirdparty/llvm-project/build/bin/mlir-translate --mlir-to-llvmir "$input_file" -o "$output_file"
fi