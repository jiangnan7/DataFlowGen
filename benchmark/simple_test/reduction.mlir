// Reduction vectorization tests: unconditional and conditional reductions.

module {
  // Test: simple sum reduction (acc += load[i])
  func.func @test_sum(%A: memref<256xi32>) -> i32 {
    %c0 = arith.constant 0 : i32
    %0 = affine.for %i = 0 to 256 iter_args(%acc = %c0) -> (i32) {
      %v = affine.load %A[%i] : memref<256xi32>
      %new_acc = arith.addi %acc, %v : i32
      affine.yield %new_acc : i32
    }
    return %0 : i32
  }

  // Test: sum of squares reduction (acc += load[i] * load[i])
  func.func @test_sum_squares(%A: memref<256xi32>) -> i32 {
    %c0 = arith.constant 0 : i32
    %0 = affine.for %i = 0 to 256 iter_args(%acc = %c0) -> (i32) {
      %v = affine.load %A[%i] : memref<256xi32>
      %sq = arith.muli %v, %v : i32
      %new_acc = arith.addi %acc, %sq : i32
      affine.yield %new_acc : i32
    }
    return %0 : i32
  }

  // Test: sum of cubes reduction (acc += v * v * v)
  func.func @test_sum_cubes(%A: memref<200xi32>) -> i32 {
    %c0 = arith.constant 0 : i32
    %0 = affine.for %i = 0 to 200 iter_args(%acc = %c0) -> (i32) {
      %v = affine.load %A[%i] : memref<200xi32>
      %sq = arith.muli %v, %v : i32
      %cube = arith.muli %sq, %v : i32
      %new_acc = arith.addi %acc, %cube : i32
      affine.yield %new_acc : i32
    }
    return %0 : i32
  }

  // Test: conditional reduction (acc += val if val > threshold)
  func.func @test_conditional_sum(%A: memref<100xi32>) -> i32 {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %0 = affine.for %i = 0 to 100 iter_args(%acc = %c0) -> (i32) {
      %v = affine.load %A[%i] : memref<100xi32>
      %cmp = arith.cmpi sgt, %v, %c10 : i32
      %res = scf.if %cmp -> (i32) {
        %add = arith.addi %v, %acc : i32
        scf.yield %add : i32
      } else {
        scf.yield %acc : i32
      }
      affine.yield %res : i32
    }
    return %0 : i32
  }

  // Test: conditional reduction with computation (acc += v*2 if v*2 > 10)
  func.func @test_conditional_compute(%A: memref<100xi32>) -> i32 {
    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i32
    %c10 = arith.constant 10 : i32
    %0 = affine.for %i = 0 to 100 iter_args(%acc = %c0) -> (i32) {
      %v = affine.load %A[%i] : memref<100xi32>
      %m = arith.muli %v, %c2 : i32
      %cmp = arith.cmpi ugt, %m, %c10 : i32
      %res = scf.if %cmp -> (i32) {
        %add = arith.addi %m, %acc : i32
        scf.yield %add : i32
      } else {
        scf.yield %acc : i32
      }
      affine.yield %res : i32
    }
    return %0 : i32
  }
}
