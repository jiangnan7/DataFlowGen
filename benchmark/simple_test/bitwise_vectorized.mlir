module {
  func.func @test_andi(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    dataflow.launch {
      dataflow.task {
        dataflow.for %arg3 = %c0 to %c64 step %c4 {
          %0 = vector.transfer_read %arg0[%arg3], %c0_i32 : memref<64xi32>, vector<4xi32>
          %1 = vector.transfer_read %arg1[%arg3], %c0_i32 : memref<64xi32>, vector<4xi32>
          %2 = arith.andi %0, %1 : vector<4xi32>
          vector.transfer_write %2, %arg2[%arg3] : vector<4xi32>, memref<64xi32>
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
      }
    }
    return
  }
  func.func @test_ori(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    dataflow.launch {
      dataflow.task {
        dataflow.for %arg3 = %c0 to %c64 step %c4 {
          %0 = vector.transfer_read %arg0[%arg3], %c0_i32 : memref<64xi32>, vector<4xi32>
          %1 = vector.transfer_read %arg1[%arg3], %c0_i32 : memref<64xi32>, vector<4xi32>
          %2 = arith.ori %0, %1 : vector<4xi32>
          vector.transfer_write %2, %arg2[%arg3] : vector<4xi32>, memref<64xi32>
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
      }
    }
    return
  }
  func.func @test_xori(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    dataflow.launch {
      dataflow.task {
        dataflow.for %arg3 = %c0 to %c64 step %c4 {
          %0 = vector.transfer_read %arg0[%arg3], %c0_i32 : memref<64xi32>, vector<4xi32>
          %1 = vector.transfer_read %arg1[%arg3], %c0_i32 : memref<64xi32>, vector<4xi32>
          %2 = arith.xori %0, %1 : vector<4xi32>
          vector.transfer_write %2, %arg2[%arg3] : vector<4xi32>, memref<64xi32>
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
      }
    }
    return
  }
  func.func @test_and_or_chain(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>, %arg3: memref<64xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    dataflow.launch {
      dataflow.task {
        dataflow.for %arg4 = %c0 to %c64 step %c4 {
          %0 = vector.transfer_read %arg0[%arg4], %c0_i32 : memref<64xi32>, vector<4xi32>
          %1 = vector.transfer_read %arg1[%arg4], %c0_i32 : memref<64xi32>, vector<4xi32>
          %2 = arith.andi %0, %1 : vector<4xi32>
          %3 = vector.transfer_read %arg2[%arg4], %c0_i32 : memref<64xi32>, vector<4xi32>
          %4 = arith.ori %2, %3 : vector<4xi32>
          vector.transfer_write %4, %arg3[%arg4] : vector<4xi32>, memref<64xi32>
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
      }
    }
    return
  }
}

