module {
  func.func @fir(%arg0: memref<100xi32>, %arg1: memref<100xi32>) -> i32 {
    %c99 = arith.constant 99 : index
    %c-1 = arith.constant -1 : index
    %c4 = arith.constant 4 : index
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = dataflow.launch : i32 {
      %1 = dataflow.task : i32 {
        %true = arith.constant true
        dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
        %2 = dataflow.for %arg2 = %c0 to %c100 step %c4 iter_args(%arg3 = %c0_i32) -> (i32) {
          %3 = dataflow.execution : i32 {
            %111 = vector.broadcast %arg3 : i32 to vector<4xi32>
            %4 = vector.transfer_read %arg1[%arg2], %c0_i32 : memref<100xi32>, vector<4xi32>
            %5 = arith.muli %arg2, %c-1 : index
            %6 = arith.addi %5, %c99 : index
            %7 = vector.transfer_read %arg0[%6], %c0_i32 : memref<100xi32>, vector<4xi32>
            %8 = arith.muli %4, %7 : vector<4xi32>
            %9 = vector.reduction <add>, %8 : vector<4xi32> into i32
            %true_0 = arith.constant true
            %10 = arith.addi %arg2, %c4 {Exe = "Loop"} : index
            %11 = arith.cmpi eq, %10, %c100 {Exe = "Loop"} : index
            dataflow.state %11, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
            dataflow.yield {execution_block = 1 : i32} %9 : i32
          }
          dataflow.yield %3 : i32
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
        dataflow.yield %2 : i32
      }
      dataflow.yield %1 : i32
    }
    return %0 : i32
  }
}