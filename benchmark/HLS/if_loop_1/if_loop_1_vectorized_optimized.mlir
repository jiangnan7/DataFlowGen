module {
  func.func @if_loop_1(%arg0: memref<100xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant dense<10> : vector<4xi32>
    %cst_0 = arith.constant dense<2> : vector<4xi32>
    %cst_1 = arith.constant dense<0> : vector<4xi32>
    %c4 = arith.constant 4 : index
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = dataflow.launch : i32 {
      %1 = dataflow.task : i32 {
        %true = arith.constant true
        dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
        %2 = dataflow.for %arg1 = %c0 to %c100 step %c4 iter_args(%arg2 = %cst_1) -> (vector<4xi32>) {
          %4 = dataflow.execution : vector<4xi32> {
            %5 = vector.transfer_read %arg0[%arg1], %c0_i32 : memref<100xi32>, vector<4xi32>
            %6 = arith.muli %5, %cst_0 {slp.group = 1 : i64, slp.lane = 3 : i64} : vector<4xi32>
            %7 = arith.addi %6, %arg2 : vector<4xi32>
            %8 = arith.cmpi ugt, %6, %cst : vector<4xi32>
            %9 = dataflow.select %8, %7, %arg2 : vector<4xi1>, vector<4xi32>
            %true_2 = arith.constant true
            %10 = arith.addi %arg1, %c4 {Exe = "Loop"} : index
            dataflow.yield {execution_block = 1 : i32} %9 : vector<4xi32>
          }
          dataflow.yield %4 : vector<4xi32>
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
        %3 = vector.reduction <add>, %2 : vector<4xi32> into i32
        dataflow.yield %3 : i32
      }
      dataflow.yield %1 : i32
    }
    return %0 : i32
  }
}

