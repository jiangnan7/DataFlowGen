module {
  func.func @matrix_add(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>, %arg2: memref<1024xi32>, %arg3: memref<1024xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant dense<0> : vector<4xi32>
    %c4 = arith.constant 4 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %0 = dataflow.launch : i32 {
      %1 = dataflow.task : i32 {
        %true_0 = arith.constant true
        dataflow.state %true_0, "loop_start" or "null" {Enable = "Loop_Start"} : i1
        %6 = dataflow.for %arg4 = %c0 to %c1024 step %c4 iter_args(%arg5 = %cst) -> (vector<4xi32>) {
          %8 = dataflow.execution : vector<4xi32> {
            %9 = vector.transfer_read %arg0[%arg4], %c0_i32 : memref<1024xi32>, vector<4xi32>
            %10 = arith.addi %9, %arg5 : vector<4xi32>
            %11 = arith.cmpi ne, %9, %cst : vector<4xi32>
            %12 = dataflow.select %11, %10, %arg5 : vector<4xi1>, vector<4xi32>
            %true_1 = arith.constant true
            %13 = arith.addi %arg4, %c4 {Exe = "Loop"} : index
            dataflow.yield {execution_block = 1 : i32} %12 : vector<4xi32>
          }
          dataflow.yield %8 : vector<4xi32>
        } {Loop_Band = 3 : i32, Loop_Level = 0 : i32}
        %7 = vector.reduction <add>, %6 : vector<4xi32> into i32
        dataflow.yield %7 : i32
      }
      %2 = dataflow.task : i32 {
        %true_0 = arith.constant true
        dataflow.state %true_0, "loop_start" or "null" {Enable = "Loop_Start"} : i1
        %6 = dataflow.for %arg4 = %c0 to %c1024 step %c4 iter_args(%arg5 = %cst) -> (vector<4xi32>) {
          %8 = dataflow.execution : vector<4xi32> {
            %9 = vector.transfer_read %arg1[%arg4], %c0_i32 : memref<1024xi32>, vector<4xi32>
            %10 = arith.addi %9, %arg5 : vector<4xi32>
            %11 = arith.cmpi ne, %9, %cst : vector<4xi32>
            %12 = dataflow.select %11, %10, %arg5 : vector<4xi1>, vector<4xi32>
            %true_1 = arith.constant true
            %13 = arith.addi %arg4, %c4 {Exe = "Loop"} : index
            dataflow.yield {execution_block = 1 : i32} %12 : vector<4xi32>
          }
          dataflow.yield %8 : vector<4xi32>
        } {Loop_Band = 2 : i32, Loop_Level = 0 : i32}
        %7 = vector.reduction <add>, %6 : vector<4xi32> into i32
        dataflow.yield %7 : i32
      }
      %3 = dataflow.task : i32 {
        %true_0 = arith.constant true
        dataflow.state %true_0, "loop_start" or "null" {Enable = "Loop_Start"} : i1
        %6 = dataflow.for %arg4 = %c0 to %c1024 step %c4 iter_args(%arg5 = %cst) -> (vector<4xi32>) {
          %8 = dataflow.execution : vector<4xi32> {
            %9 = vector.transfer_read %arg2[%arg4], %c0_i32 : memref<1024xi32>, vector<4xi32>
            %10 = arith.addi %9, %arg5 : vector<4xi32>
            %11 = arith.cmpi ne, %9, %cst : vector<4xi32>
            %12 = dataflow.select %11, %10, %arg5 : vector<4xi1>, vector<4xi32>
            %true_1 = arith.constant true
            %13 = arith.addi %arg4, %c4 {Exe = "Loop"} : index
            dataflow.yield {execution_block = 1 : i32} %12 : vector<4xi32>
          }
          dataflow.yield %8 : vector<4xi32>
        } {Loop_Band = 1 : i32, Loop_Level = 0 : i32}
        %7 = vector.reduction <add>, %6 : vector<4xi32> into i32
        dataflow.yield %7 : i32
      }
      %4 = dataflow.task : i32 {
        %true_0 = arith.constant true
        dataflow.state %true_0, "loop_start" or "null" {Enable = "Loop_Start"} : i1
        %6 = dataflow.for %arg4 = %c0 to %c1024 step %c4 iter_args(%arg5 = %cst) -> (vector<4xi32>) {
          %8 = dataflow.execution : vector<4xi32> {
            %9 = vector.transfer_read %arg3[%arg4], %c0_i32 : memref<1024xi32>, vector<4xi32>
            %10 = arith.addi %9, %arg5 : vector<4xi32>
            %11 = arith.cmpi ne, %9, %cst : vector<4xi32>
            %12 = dataflow.select %11, %10, %arg5 : vector<4xi1>, vector<4xi32>
            %true_1 = arith.constant true
            %13 = arith.addi %arg4, %c4 {Exe = "Loop"} : index
            dataflow.yield {execution_block = 1 : i32} %12 : vector<4xi32>
          }
          dataflow.yield %8 : vector<4xi32>
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
        %7 = vector.reduction <add>, %6 : vector<4xi32> into i32
        dataflow.yield %7 : i32
      }
      %5 = dataflow.task : i32 {
        dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
        %6 = arith.addi %1, %2 : i32
        %7 = arith.addi %6, %3 : i32
        %8 = arith.addi %7, %4 : i32
        dataflow.yield %8 : i32
      }
      dataflow.yield %5 : i32
    }
    return %0 : i32
  }
}

