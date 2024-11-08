module  {
  func.func @histogram(%arg0: memref<100xi32>, %arg1: memref<100xi32>, %arg2: memref<100xi32>)  {
    %c10_i32 = arith.constant 10 : i32
    affine.for %arg3 = 0 to 100 {
      %0 = affine.load %arg1[%arg3] : memref<100xi32>
      %2 = affine.load %arg0[%arg3] : memref<100xi32>
      %3 = arith.index_cast %2 : i32 to index
      %4 = memref.load %arg2[%3] : memref<100xi32>
      %5 = arith.addi %4, %0 : i32
      memref.store %5, %arg2[%3] : memref<100xi32>
    }
    return
  }
}
