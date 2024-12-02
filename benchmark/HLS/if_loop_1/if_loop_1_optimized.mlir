module {
  func.func @if_loop_1(%arg0: memref<100xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = dataflow.launch : i32 {
      %1 = dataflow.task : i32 {
        dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
        %2 = dataflow.for %arg1 = %c0 to %c100 step %c1 iter_args(%arg2 = %c0_i32) -> (i32) {
          %3 = dataflow.execution : i32 {
            %4 = dataflow.addr %arg0[%arg1] {memShape = [100]} : memref<100xi32>[index] -> i32
            %5 = dataflow.load %4 : i32 -> i32
            %6 = arith.muli %5, %c2_i32 : i32
            %7 = arith.cmpi ugt, %6, %c10_i32 : i32
            %8 = arith.addi %6, %arg2 : i32
            %9 = dataflow.select %7, %8, %arg2 : i32
            %10 = arith.addi %arg1, %c1 {Exe = "Loop"} : index
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