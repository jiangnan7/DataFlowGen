#map = affine_map<(d0) -> (-d0 + 99)>
module {
  func.func @fir(%arg0: memref<100xi32>, %arg1: memref<100xi32>) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0 : i32
    %0 = affine.for %arg2 = 0 to 100 step 4 iter_args(%arg3 = %cst) -> (i32) {
      %c0_i32_0 = arith.constant 0 : i32
      %2 = vector.transfer_read %arg1[%arg2], %c0_i32_0 : memref<100xi32>, vector<4xi32>
      %3 = affine.apply #map(%arg2)
      %c0_i32_1 = arith.constant 0 : i32
      %4 = vector.transfer_read %arg0[%3], %c0_i32_1 : memref<100xi32>, vector<4xi32>
      %5 = arith.muli %2, %4 : vector<4xi32>
      %6 = vector.reduction <add>, %5 : vector<4xi32> into i32
      affine.yield %6 : i32
    }
    return %0 : i32
  }
}
