module {
  func.func @test_addf(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    dataflow.launch {
      dataflow.task {
        dataflow.for %arg3 = %c0 to %c64 step %c4 {
          %0 = vector.transfer_read %arg0[%arg3], %cst : memref<64xf32>, vector<4xf32>
          %1 = vector.transfer_read %arg1[%arg3], %cst : memref<64xf32>, vector<4xf32>
          %2 = arith.addf %0, %1 : vector<4xf32>
          vector.transfer_write %2, %arg2[%arg3] : vector<4xf32>, memref<64xf32>
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
      }
    }
    return
  }
  func.func @test_subf(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    dataflow.launch {
      dataflow.task {
        dataflow.for %arg3 = %c0 to %c64 step %c4 {
          %0 = vector.transfer_read %arg0[%arg3], %cst : memref<64xf32>, vector<4xf32>
          %1 = vector.transfer_read %arg1[%arg3], %cst : memref<64xf32>, vector<4xf32>
          %2 = arith.subf %0, %1 : vector<4xf32>
          vector.transfer_write %2, %arg2[%arg3] : vector<4xf32>, memref<64xf32>
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
      }
    }
    return
  }
  func.func @test_mulf(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    dataflow.launch {
      dataflow.task {
        dataflow.for %arg3 = %c0 to %c64 step %c4 {
          %0 = vector.transfer_read %arg0[%arg3], %cst : memref<64xf32>, vector<4xf32>
          %1 = vector.transfer_read %arg1[%arg3], %cst : memref<64xf32>, vector<4xf32>
          %2 = arith.mulf %0, %1 : vector<4xf32>
          vector.transfer_write %2, %arg2[%arg3] : vector<4xf32>, memref<64xf32>
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
      }
    }
    return
  }
  func.func @test_fma(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>, %arg3: memref<64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    dataflow.launch {
      dataflow.task {
        dataflow.for %arg4 = %c0 to %c64 step %c4 {
          %0 = vector.transfer_read %arg0[%arg4], %cst : memref<64xf32>, vector<4xf32>
          %1 = vector.transfer_read %arg1[%arg4], %cst : memref<64xf32>, vector<4xf32>
          %2 = arith.mulf %0, %1 : vector<4xf32>
          %3 = vector.transfer_read %arg2[%arg4], %cst : memref<64xf32>, vector<4xf32>
          %4 = arith.addf %2, %3 : vector<4xf32>
          vector.transfer_write %4, %arg3[%arg4] : vector<4xf32>, memref<64xf32>
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
      }
    }
    return
  }
}

