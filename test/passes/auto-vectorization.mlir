// RUN: %heteacc_opt %s --auto-vectorization 2>/dev/null | FileCheck %s

// Test: Simple integer vector addition (end-to-end vectorization).
// CHECK-LABEL: func.func @simple_add
// CHECK: dataflow.launch
// CHECK: dataflow.task
// CHECK: dataflow.for
// CHECK: vector.transfer_read {{.*}} : memref<16xi32>, vector<4xi32>
// CHECK: vector.transfer_read {{.*}} : memref<16xi32>, vector<4xi32>
// CHECK: arith.addi {{.*}} : vector<4xi32>
// CHECK: vector.transfer_write {{.*}} : vector<4xi32>, memref<16xi32>

module {
  func.func @simple_add(%A: memref<16xi32>, %B: memref<16xi32>, %C: memref<16xi32>) {
    affine.for %i = 0 to 16 {
      %a = affine.load %A[%i] : memref<16xi32>
      %b = affine.load %B[%i] : memref<16xi32>
      %c = arith.addi %a, %b : i32
      affine.store %c, %C[%i] : memref<16xi32>
    }
    return
  }
}
