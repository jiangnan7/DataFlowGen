// RUN: %heteacc_opt %s --optimize-dataflow | FileCheck %s

module {
  func.func @lower_affine_to_dataflow(%A: memref<16xi32>) {
    affine.for %i = 0 to 16 {
      %v = affine.load %A[%i] : memref<16xi32>
      affine.store %v, %A[%i] : memref<16xi32>
    }
    return
  }
}

// CHECK-LABEL: func.func @lower_affine_to_dataflow
// CHECK: %c0 = arith.constant 0 : index
// CHECK: %c16 = arith.constant 16 : index
// CHECK: %c1 = arith.constant 1 : index
// CHECK: dataflow.for {{.*}} = %c0 to %c16 step %c1 {
// CHECK: memref.load
// CHECK: memref.store
// CHECK-NOT: affine.for
// CHECK-NOT: affine.load
// CHECK-NOT: affine.store
