// RUN: %heteacc_opt %s --auto-vectorization 2>/dev/null | FileCheck %s

// Test that the auto-vectorization pass correctly:
// 1. Unrolls the loop by factor 4
// 2. Lowers affine to standard (scf.for + memref ops)
// 3. Applies SLP vectorization on the unrolled body

module {
  func.func @unroll_and_vectorize(%A: memref<16xi32>, %B: memref<16xi32>, %C: memref<16xi32>) {
    affine.for %i = 0 to 16 {
      %a = affine.load %A[%i] : memref<16xi32>
      %b = affine.load %B[%i] : memref<16xi32>
      %c = arith.addi %a, %b : i32
      affine.store %c, %C[%i] : memref<16xi32>
    }
    return
  }
}

// After unrolling by 4, lowering affine, and SLP vectorization:
// CHECK-LABEL: func.func @unroll_and_vectorize
// The loop has step 4:
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// Vector operations should be generated:
// CHECK: vector.transfer_read {{.*}} : memref<16xi32>, vector<4xi32>
// CHECK: vector.transfer_read {{.*}} : memref<16xi32>, vector<4xi32>
// CHECK: arith.addi {{.*}} : vector<4xi32>
// CHECK: vector.transfer_write {{.*}} : vector<4xi32>, memref<16xi32>
