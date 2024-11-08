module {
  func.func @fir(%arg0: memref<1000xi32>, %arg1: memref<1000xi32>) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %0 = affine.for %arg2 = 0 to 1000 iter_args(%arg3 = %c0_i32) -> (i32) {
      %1 = affine.load %arg1[%arg2] : memref<1000xi32>
      %2 = affine.load %arg0[-%arg2 + 999] : memref<1000xi32>
      %3 = arith.muli %1, %2 : i32
      %4 = arith.addi %arg3, %3 : i32
      affine.yield %4 : i32
    }
    return %0 : i32
  }
}
