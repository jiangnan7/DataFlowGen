module  {
  func.func @matrix_add(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>, %arg2: memref<1024xi32>, %arg3: memref<1024xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = affine.for %arg4 = 0 to 1024 iter_args(%arg5 = %c0_i32) -> (i32) {
      %7 = affine.load %arg0[%arg4] : memref<1024xi32>
      %8 = arith.cmpi ne, %7, %c0_i32 : i32
      %9 = scf.if %8 -> (i32) {
        %10 = arith.addi %arg5, %7 : i32
        scf.yield %10 : i32
      } else {
        scf.yield %arg5 : i32
      }
      affine.yield %9 : i32
    }
    %1 = affine.for %arg4 = 0 to 1024 iter_args(%arg5 = %c0_i32) -> (i32) {
      %7 = affine.load %arg1[%arg4] : memref<1024xi32>
      %8 = arith.cmpi ne, %7, %c0_i32 : i32
      %9 = scf.if %8 -> (i32) {
        %10 = arith.addi %arg5, %7 : i32
        scf.yield %10 : i32
      } else {
        scf.yield %arg5 : i32
      }
      affine.yield %9 : i32
    }
    %2 = affine.for %arg4 = 0 to 1024 iter_args(%arg5 = %c0_i32) -> (i32) {
      %7 = affine.load %arg2[%arg4] : memref<1024xi32>
      %8 = arith.cmpi ne, %7, %c0_i32 : i32
      %9 = scf.if %8 -> (i32) {
        %10 = arith.addi %arg5, %7 : i32
        scf.yield %10 : i32
      } else {
        scf.yield %arg5 : i32
      }
      affine.yield %9 : i32
    }
    %3 = affine.for %arg4 = 0 to 1024 iter_args(%arg5 = %c0_i32) -> (i32) {
      %7 = affine.load %arg3[%arg4] : memref<1024xi32>
      %8 = arith.cmpi ne, %7, %c0_i32 : i32
      %9 = scf.if %8 -> (i32) {
        %10 = arith.addi %arg5, %7 : i32
        scf.yield %10 : i32
      } else {
        scf.yield %arg5 : i32
      }
      affine.yield %9 : i32
    }
    %4 = arith.addi %0, %1 : i32
    %5 = arith.addi %4, %2 : i32
    %6 = arith.addi %5, %3 : i32
    return %6 : i32
  }
}