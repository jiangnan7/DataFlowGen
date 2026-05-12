#map = affine_map<(d0) -> (d0)>
module {
  func.func @getTanh(%arg0: memref<100xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c19_i32 = arith.constant 19 : i32
    %c3_i32 = arith.constant 3 : i32
    %0 = dataflow.launch : i32 {
      %1 = dataflow.task : i32 {
        %true = arith.constant true
        dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
        %2 = dataflow.for %arg1 = %c0 to %c100 step %c4 iter_args(%arg2 = %c0_i32) -> (i32) {
          %3 = dataflow.execution : i32 {
            %4 = memref.load %arg0[%arg1] {affineCoeff = [1], affineOffset = 0 : i64, map = #map, slp.group = 0 : i64, slp.lane = 0 : i64} : memref<100xi32>
            %5 = arith.cmpi slt, %4, %c1_i32 {slp.group = 1 : i64, slp.lane = 0 : i64} : i32
            %6 = arith.muli %4, %4 : i32
            %7 = arith.addi %6, %c19_i32 : i32
            %8 = arith.muli %7, %4 : i32
            %9 = arith.muli %8, %4 : i32
            %10 = arith.addi %9, %c3_i32 : i32
            %11 = arith.muli %10, %4 : i32
            %12 = dataflow.select %5, %11, %c1_i32 : i32
            %13 = arith.addi %arg2, %12 {slp.group = 3 : i64, slp.lane = 0 : i64} : i32
            %14 = arith.addi %arg1, %c1 : index
            %15 = memref.load %arg0[%14] {affineCoeff = [1], affineOffset = 0 : i64, map = #map, slp.group = 0 : i64, slp.lane = 1 : i64} : memref<100xi32>
            %16 = arith.cmpi slt, %15, %c1_i32 {slp.group = 1 : i64, slp.lane = 1 : i64} : i32
            %17 = arith.muli %15, %15 : i32
            %18 = arith.addi %17, %c19_i32 : i32
            %19 = arith.muli %18, %15 : i32
            %20 = arith.muli %19, %15 : i32
            %21 = arith.addi %20, %c3_i32 : i32
            %22 = arith.muli %21, %15 : i32
            %23 = dataflow.select %16, %22, %c1_i32 : i32
            %24 = arith.addi %13, %23 {slp.group = 3 : i64, slp.lane = 1 : i64} : i32
            %25 = arith.addi %arg1, %c2 : index
            %26 = memref.load %arg0[%25] {affineCoeff = [1], affineOffset = 0 : i64, map = #map, slp.group = 0 : i64, slp.lane = 2 : i64} : memref<100xi32>
            %27 = arith.cmpi slt, %26, %c1_i32 {slp.group = 1 : i64, slp.lane = 2 : i64} : i32
            %28 = arith.muli %26, %26 : i32
            %29 = arith.addi %28, %c19_i32 : i32
            %30 = arith.muli %29, %26 : i32
            %31 = arith.muli %30, %26 : i32
            %32 = arith.addi %31, %c3_i32 : i32
            %33 = arith.muli %32, %26 : i32
            %34 = dataflow.select %27, %33, %c1_i32 : i32
            %35 = arith.addi %24, %34 {slp.group = 3 : i64, slp.lane = 2 : i64} : i32
            %36 = arith.addi %arg1, %c3 : index
            %37 = memref.load %arg0[%36] {affineCoeff = [1], affineOffset = 0 : i64, map = #map, slp.group = 0 : i64, slp.lane = 3 : i64} : memref<100xi32>
            %38 = arith.cmpi slt, %37, %c1_i32 {slp.group = 1 : i64, slp.lane = 3 : i64} : i32
            %39 = arith.muli %37, %37 : i32
            %40 = arith.addi %39, %c19_i32 : i32
            %41 = arith.muli %40, %37 : i32
            %42 = arith.muli %41, %37 : i32
            %43 = arith.addi %42, %c3_i32 : i32
            %44 = arith.muli %43, %37 : i32
            %45 = dataflow.select %38, %44, %c1_i32 : i32
            %46 = arith.addi %35, %45 {slp.group = 3 : i64, slp.lane = 3 : i64} : i32
            %true_0 = arith.constant true
            %47 = arith.addi %arg1, %c4 {Exe = "Loop"} : index
            dataflow.yield {execution_block = 1 : i32} %46 : i32
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

