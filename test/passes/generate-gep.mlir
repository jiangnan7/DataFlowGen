// RUN: %heteacc_opt %s --generate-GEP | FileCheck %s

module {
  func.func @lower_memref_ops(%A: memref<4x8xi32>, %i: index, %j: index, %v: i32) -> i32 {
    memref.store %v, %A[%i, %j] : memref<4x8xi32>
    %0 = memref.load %A[%i, %j] : memref<4x8xi32>
    return %0 : i32
  }
}

// CHECK-LABEL: func.func @lower_memref_ops
// CHECK: %[[ADDR0:.+]] = dataflow.addr {{.*}}[%{{.*}}, %{{.*}}] {memShape = [4, 8]}
// CHECK: dataflow.store %{{.*}} %[[ADDR0]]
// CHECK: %[[ADDR1:.+]] = dataflow.addr {{.*}}[%{{.*}}, %{{.*}}] {memShape = [4, 8]}
// CHECK: %[[LOAD:.+]] = dataflow.load %[[ADDR1]]
// CHECK: return %[[LOAD]]
