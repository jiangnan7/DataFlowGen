// RUN: %heteacc_opt %s --simplify-vector-memref-access | FileCheck %s

module {
  func.func @simplify_transfer_ops(%A: memref<16xf32>, %idx: index) -> vector<4xf32> {
    %pad = arith.constant 0.0 : f32
    %v = vector.transfer_read %A[%idx], %pad : memref<16xf32>, vector<4xf32>
    vector.transfer_write %v, %A[%idx] : vector<4xf32>, memref<16xf32>
    return %v : vector<4xf32>
  }
}

// CHECK-LABEL: func.func @simplify_transfer_ops
// CHECK: %[[V:.+]] = vector.load %{{.*}}[%{{.*}}] : memref<16xf32>, vector<4xf32>
// CHECK: vector.store %[[V]], %{{.*}}[%{{.*}}] : memref<16xf32>, vector<4xf32>
// CHECK-NOT: vector.transfer_read
// CHECK-NOT: vector.transfer_write
