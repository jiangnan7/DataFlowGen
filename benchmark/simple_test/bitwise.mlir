// Bitwise operation vectorization tests: andi, ori, xori.

module {
  // Test: vector bitwise AND (i32)
  func.func @test_andi(%A: memref<64xi32>, %B: memref<64xi32>, %C: memref<64xi32>) {
    affine.for %i = 0 to 64 {
      %a = affine.load %A[%i] : memref<64xi32>
      %b = affine.load %B[%i] : memref<64xi32>
      %c = arith.andi %a, %b : i32
      affine.store %c, %C[%i] : memref<64xi32>
    }
    return
  }

  // Test: vector bitwise OR (i32)
  func.func @test_ori(%A: memref<64xi32>, %B: memref<64xi32>, %C: memref<64xi32>) {
    affine.for %i = 0 to 64 {
      %a = affine.load %A[%i] : memref<64xi32>
      %b = affine.load %B[%i] : memref<64xi32>
      %c = arith.ori %a, %b : i32
      affine.store %c, %C[%i] : memref<64xi32>
    }
    return
  }

  // Test: vector bitwise XOR (i32)
  func.func @test_xori(%A: memref<64xi32>, %B: memref<64xi32>, %C: memref<64xi32>) {
    affine.for %i = 0 to 64 {
      %a = affine.load %A[%i] : memref<64xi32>
      %b = affine.load %B[%i] : memref<64xi32>
      %c = arith.xori %a, %b : i32
      affine.store %c, %C[%i] : memref<64xi32>
    }
    return
  }

  // Test: chained bitwise operations (A[i] & B[i]) | C[i]
  func.func @test_and_or_chain(%A: memref<64xi32>, %B: memref<64xi32>, %C: memref<64xi32>, %D: memref<64xi32>) {
    affine.for %i = 0 to 64 {
      %a = affine.load %A[%i] : memref<64xi32>
      %b = affine.load %B[%i] : memref<64xi32>
      %ab = arith.andi %a, %b : i32
      %c = affine.load %C[%i] : memref<64xi32>
      %r = arith.ori %ab, %c : i32
      affine.store %r, %D[%i] : memref<64xi32>
    }
    return
  }
}
