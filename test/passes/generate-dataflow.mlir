// RUN: %heteacc_opt %s --generate-dataflow | FileCheck %s

module {
  func.func @partition_loop_band(%A: memref<4x8xi32>) {
    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 8 {
        %0 = affine.load %A[%i, %j] : memref<4x8xi32>
        affine.store %0, %A[%i, %j] : memref<4x8xi32>
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @partition_loop_band
// CHECK: dataflow.launch {
// CHECK: dataflow.task {
// CHECK: affine.for
// CHECK: affine.for
// CHECK: } {Loop_Band = 0 : i32, Loop_Level = 1 : i32}
// CHECK: } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
