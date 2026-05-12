module{
  func.func @sumi3_mem(%arg0: memref<200xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = affine.for %arg1 = 0 to 200 iter_args(%arg2 = %c0_i32) -> (i32) {
      %1 = affine.load %arg0[%arg1] : memref<200xi32>
      %2 = arith.muli %1, %1 : i32
      %3 = arith.muli %2, %1 : i32
      %4 = arith.addi %arg2, %3 : i32
      affine.yield %4 : i32
    }
    return %0 : i32
  }
}
