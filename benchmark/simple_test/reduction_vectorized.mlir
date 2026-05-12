module {
  func.func @test_sum(%arg0: memref<256xi32>) -> i32 {
    %cst = arith.constant dense<0> : vector<4xi32>
    %c4 = arith.constant 4 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = dataflow.launch : i32 {
      %1 = dataflow.task : i32 {
        %2 = dataflow.for %arg1 = %c0 to %c256 step %c4 iter_args(%arg2 = %cst) -> (vector<4xi32>) {
          %4 = vector.transfer_read %arg0[%arg1], %c0_i32 : memref<256xi32>, vector<4xi32>
          %5 = arith.addi %4, %arg2 : vector<4xi32>
          dataflow.yield %5 : vector<4xi32>
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
        %3 = vector.reduction <add>, %2 : vector<4xi32> into i32
        dataflow.yield %3 : i32
      }
      dataflow.yield %1 : i32
    }
    return %0 : i32
  }
  func.func @test_sum_squares(%arg0: memref<256xi32>) -> i32 {
    %cst = arith.constant dense<0> : vector<4xi32>
    %c4 = arith.constant 4 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = dataflow.launch : i32 {
      %1 = dataflow.task : i32 {
        %2 = dataflow.for %arg1 = %c0 to %c256 step %c4 iter_args(%arg2 = %cst) -> (vector<4xi32>) {
          %4 = vector.transfer_read %arg0[%arg1], %c0_i32 : memref<256xi32>, vector<4xi32>
          %5 = arith.muli %4, %4 : vector<4xi32>
          %6 = arith.addi %5, %arg2 : vector<4xi32>
          dataflow.yield %6 : vector<4xi32>
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
        %3 = vector.reduction <add>, %2 : vector<4xi32> into i32
        dataflow.yield %3 : i32
      }
      dataflow.yield %1 : i32
    }
    return %0 : i32
  }
  func.func @test_sum_cubes(%arg0: memref<200xi32>) -> i32 {
    %cst = arith.constant dense<0> : vector<4xi32>
    %c4 = arith.constant 4 : index
    %c200 = arith.constant 200 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = dataflow.launch : i32 {
      %1 = dataflow.task : i32 {
        %2 = dataflow.for %arg1 = %c0 to %c200 step %c4 iter_args(%arg2 = %cst) -> (vector<4xi32>) {
          %4 = vector.transfer_read %arg0[%arg1], %c0_i32 : memref<200xi32>, vector<4xi32>
          %5 = arith.muli %4, %4 : vector<4xi32>
          %6 = arith.muli %5, %4 : vector<4xi32>
          %7 = arith.addi %6, %arg2 : vector<4xi32>
          dataflow.yield %7 : vector<4xi32>
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
        %3 = vector.reduction <add>, %2 : vector<4xi32> into i32
        dataflow.yield %3 : i32
      }
      dataflow.yield %1 : i32
    }
    return %0 : i32
  }
  func.func @test_conditional_sum(%arg0: memref<100xi32>) -> i32 {
    %cst = arith.constant dense<10> : vector<4xi32>
    %cst_0 = arith.constant dense<0> : vector<4xi32>
    %c4 = arith.constant 4 : index
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = dataflow.launch : i32 {
      %1 = dataflow.task : i32 {
        %2 = dataflow.for %arg1 = %c0 to %c100 step %c4 iter_args(%arg2 = %cst_0) -> (vector<4xi32>) {
          %4 = vector.transfer_read %arg0[%arg1], %c0_i32 : memref<100xi32>, vector<4xi32>
          %5 = arith.addi %4, %arg2 : vector<4xi32>
          %6 = arith.cmpi sgt, %4, %cst : vector<4xi32>
          %7 = dataflow.select %6, %5, %arg2 : vector<4xi1>, vector<4xi32>
          dataflow.yield %7 : vector<4xi32>
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
        %3 = vector.reduction <add>, %2 : vector<4xi32> into i32
        dataflow.yield %3 : i32
      }
      dataflow.yield %1 : i32
    }
    return %0 : i32
  }
  func.func @test_conditional_compute(%arg0: memref<100xi32>) -> i32 {
    %cst = arith.constant dense<10> : vector<4xi32>
    %cst_0 = arith.constant dense<2> : vector<4xi32>
    %cst_1 = arith.constant dense<0> : vector<4xi32>
    %c4 = arith.constant 4 : index
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = dataflow.launch : i32 {
      %1 = dataflow.task : i32 {
        %2 = dataflow.for %arg1 = %c0 to %c100 step %c4 iter_args(%arg2 = %cst_1) -> (vector<4xi32>) {
          %4 = vector.transfer_read %arg0[%arg1], %c0_i32 : memref<100xi32>, vector<4xi32>
          %5 = arith.muli %4, %cst_0 : vector<4xi32>
          %6 = arith.addi %5, %arg2 : vector<4xi32>
          %7 = arith.cmpi ugt, %5, %cst : vector<4xi32>
          %8 = dataflow.select %7, %6, %arg2 : vector<4xi1>, vector<4xi32>
          dataflow.yield %8 : vector<4xi32>
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
        %3 = vector.reduction <add>, %2 : vector<4xi32> into i32
        dataflow.yield %3 : i32
      }
      dataflow.yield %1 : i32
    }
    return %0 : i32
  }
}

