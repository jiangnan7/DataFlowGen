// RUN: %heteacc_opt %s --generate-GEP | FileCheck %s

module {
  func.func @vector_transfer_to_dataflow(%A: memref<16xf32>) -> vector<4xf32> {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0.0 : f32
    %v = vector.transfer_read %A[%c0], %pad : memref<16xf32>, vector<4xf32>
    return %v : vector<4xf32>
  }
}

// CHECK-LABEL: func.func @vector_transfer_to_dataflow
// CHECK: %c0 = arith.constant 0 : index
// CHECK: %[[ADDR:.+]] = dataflow.addr %{{.*}}[%c0] {laneNums = 4 : i32, memShape = [16]}
// CHECK: %[[LOAD:.+]] = dataflow.load %[[ADDR]] {{.*}} laneNums = 4 : i32
// CHECK: return %[[LOAD]] : vector<4xf32>
// CHECK-NOT: vector.transfer_read
