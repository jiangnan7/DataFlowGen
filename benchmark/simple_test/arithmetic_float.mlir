// Float arithmetic vectorization tests: addf, subf, mulf.

module {
  // Test: vector float addition
  func.func @test_addf(%A: memref<64xf32>, %B: memref<64xf32>, %C: memref<64xf32>) {
    affine.for %i = 0 to 64 {
      %a = affine.load %A[%i] : memref<64xf32>
      %b = affine.load %B[%i] : memref<64xf32>
      %c = arith.addf %a, %b : f32
      affine.store %c, %C[%i] : memref<64xf32>
    }
    return
  }

  // Test: vector float subtraction
  func.func @test_subf(%A: memref<64xf32>, %B: memref<64xf32>, %C: memref<64xf32>) {
    affine.for %i = 0 to 64 {
      %a = affine.load %A[%i] : memref<64xf32>
      %b = affine.load %B[%i] : memref<64xf32>
      %c = arith.subf %a, %b : f32
      affine.store %c, %C[%i] : memref<64xf32>
    }
    return
  }

  // Test: vector float multiplication
  func.func @test_mulf(%A: memref<64xf32>, %B: memref<64xf32>, %C: memref<64xf32>) {
    affine.for %i = 0 to 64 {
      %a = affine.load %A[%i] : memref<64xf32>
      %b = affine.load %B[%i] : memref<64xf32>
      %c = arith.mulf %a, %b : f32
      affine.store %c, %C[%i] : memref<64xf32>
    }
    return
  }

  // Test: fused multiply-add (a * b + c)
  func.func @test_fma(%A: memref<64xf32>, %B: memref<64xf32>, %C: memref<64xf32>, %D: memref<64xf32>) {
    affine.for %i = 0 to 64 {
      %a = affine.load %A[%i] : memref<64xf32>
      %b = affine.load %B[%i] : memref<64xf32>
      %m = arith.mulf %a, %b : f32
      %c = affine.load %C[%i] : memref<64xf32>
      %r = arith.addf %m, %c : f32
      affine.store %r, %D[%i] : memref<64xf32>
    }
    return
  }
}
