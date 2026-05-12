// Indirect memory access vectorization tests: index_cast + dataflow.vector.load/store.

module {
  // Test: simple indirect load (gather)
  // A[B[i]] pattern
  func.func @test_gather(%A: memref<1024xi32>, %B: memref<256xi32>, %C: memref<256xi32>) {
    affine.for %i = 0 to 256 {
      %idx_i32 = affine.load %B[%i] : memref<256xi32>
      %idx = arith.index_cast %idx_i32 : i32 to index
      %val = memref.load %A[%idx] : memref<1024xi32>
      affine.store %val, %C[%i] : memref<256xi32>
    }
    return
  }

  // Test: indirect load + compute + indirect store (histogram pattern)
  // C[B[i]] += A[i]
  func.func @test_histogram(%A: memref<1024xi32>, %B: memref<1024xi32>, %C: memref<1024xi32>) {
    affine.for %i = 0 to 1024 {
      %val = affine.load %A[%i] : memref<1024xi32>
      %idx_i32 = affine.load %B[%i] : memref<1024xi32>
      %idx = arith.index_cast %idx_i32 : i32 to index
      %old = memref.load %C[%idx] : memref<1024xi32>
      %new = arith.addi %old, %val : i32
      memref.store %new, %C[%idx] : memref<1024xi32>
    }
    return
  }

  // Test: indirect load with scaling
  // C[i] = A[B[i]] * 2
  func.func @test_gather_scale(%A: memref<1024xi32>, %B: memref<256xi32>, %C: memref<256xi32>) {
    %c2 = arith.constant 2 : i32
    affine.for %i = 0 to 256 {
      %idx_i32 = affine.load %B[%i] : memref<256xi32>
      %idx = arith.index_cast %idx_i32 : i32 to index
      %val = memref.load %A[%idx] : memref<1024xi32>
      %scaled = arith.muli %val, %c2 : i32
      affine.store %scaled, %C[%i] : memref<256xi32>
    }
    return
  }
}
