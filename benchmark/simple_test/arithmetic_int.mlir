// Arithmetic vectorization tests: addi, subi, muli for integer types.

module {
  // Test: vector addition (i32)
  func.func @test_addi(%A: memref<64xi32>, %B: memref<64xi32>, %C: memref<64xi32>) {
    affine.for %i = 0 to 64 {
      %a = affine.load %A[%i] : memref<64xi32>
      %b = affine.load %B[%i] : memref<64xi32>
      %c = arith.addi %a, %b : i32
      affine.store %c, %C[%i] : memref<64xi32>
    }
    return
  }

  // Test: vector subtraction (i32)
  func.func @test_subi(%A: memref<64xi32>, %B: memref<64xi32>, %C: memref<64xi32>) {
    affine.for %i = 0 to 64 {
      %a = affine.load %A[%i] : memref<64xi32>
      %b = affine.load %B[%i] : memref<64xi32>
      %c = arith.subi %a, %b : i32
      affine.store %c, %C[%i] : memref<64xi32>
    }
    return
  }

  // Test: vector multiplication (i32)
  func.func @test_muli(%A: memref<64xi32>, %B: memref<64xi32>, %C: memref<64xi32>) {
    affine.for %i = 0 to 64 {
      %a = affine.load %A[%i] : memref<64xi32>
      %b = affine.load %B[%i] : memref<64xi32>
      %c = arith.muli %a, %b : i32
      affine.store %c, %C[%i] : memref<64xi32>
    }
    return
  }

  // Test: chained operations (load -> mul -> add -> store)
  func.func @test_mul_add_chain(%A: memref<64xi32>, %B: memref<64xi32>, %C: memref<64xi32>, %D: memref<64xi32>) {
    %c3 = arith.constant 3 : i32
    affine.for %i = 0 to 64 {
      %a = affine.load %A[%i] : memref<64xi32>
      %b = affine.load %B[%i] : memref<64xi32>
      %m = arith.muli %a, %b : i32
      %c = affine.load %C[%i] : memref<64xi32>
      %r = arith.addi %m, %c : i32
      affine.store %r, %D[%i] : memref<64xi32>
    }
    return
  }
}
