// RUN: %heteacc_opt %s --optimize-dataflow --generate-GEP | FileCheck %s

module {
  func.func @optimize_then_gep(%A: memref<16xi32>) {
    affine.for %i = 0 to 16 {
      %v = affine.load %A[%i] : memref<16xi32>
      affine.store %v, %A[%i] : memref<16xi32>
    }
    return
  }
}

// CHECK-LABEL: func.func @optimize_then_gep
// CHECK: dataflow.for
// CHECK: %[[ADDR0:.+]] = dataflow.addr {{.*}} {memShape = [16]}
// CHECK: %[[LOAD:.+]] = dataflow.load %[[ADDR0]]
// CHECK: %[[ADDR1:.+]] = dataflow.addr {{.*}} {memShape = [16]}
// CHECK: dataflow.store %[[LOAD]] %[[ADDR1]]
// CHECK-NOT: affine.for
// CHECK-NOT: memref.load
// CHECK-NOT: memref.store
