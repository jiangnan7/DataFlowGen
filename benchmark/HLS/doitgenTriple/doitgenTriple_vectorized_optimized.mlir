#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0 + d1 * 16)>
module {
  func.func @doitgenTriple(%arg0: memref<16xi32>, %arg1: memref<16xi32>, %arg2: memref<256xi32>) {
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    dataflow.launch {
      dataflow.task {
        %true = arith.constant true
        dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
        dataflow.for %arg3 = %c0 to %c16 step %c1 {
          dataflow.execution {
            %0 = dataflow.for %arg4 = %c0 to %c16 step %c4 iter_args(%arg5 = %c0_i32) -> (i32) {
              %2 = dataflow.execution : i32 {
                %3 = dataflow.merge %c0_i32 or %arg5 : i32
                %4 = memref.load %arg0[%arg4] {affineCoeff = [1], affineOffset = 0 : i64, map = #map, slp.group = 0 : i64, slp.lane = 0 : i64} : memref<16xi32>
                %5 = arith.muli %arg3, %c16 : index
                %6 = arith.addi %arg4, %5 : index
                %7 = memref.load %arg2[%6] {affineCoeff = [1], affineOffset = 0 : i64, map = #map1, slp.group = 1 : i64, slp.lane = 0 : i64} : memref<256xi32>
                %8 = arith.cmpi sgt, %4, %c0_i32 {slp.group = 2 : i64, slp.lane = 0 : i64} : i32
                %9 = arith.muli %4, %7 : i32
                %10 = arith.addi %9, %7 : i32
                %11 = arith.muli %10, %4 : i32
                %12 = arith.addi %3, %11 : i32
                %13 = dataflow.select %8, %12, %3 : i32
                %14 = arith.addi %arg4, %c1 : index
                %15 = memref.load %arg0[%14] {affineCoeff = [1], affineOffset = 0 : i64, map = #map, slp.group = 0 : i64, slp.lane = 1 : i64} : memref<16xi32>
                %16 = arith.addi %14, %5 : index
                %17 = memref.load %arg2[%16] {affineCoeff = [1], affineOffset = 0 : i64, map = #map1, slp.group = 1 : i64, slp.lane = 1 : i64} : memref<256xi32>
                %18 = arith.cmpi sgt, %15, %c0_i32 {slp.group = 2 : i64, slp.lane = 1 : i64} : i32
                %19 = arith.muli %15, %17 : i32
                %20 = arith.addi %19, %17 : i32
                %21 = arith.muli %20, %15 : i32
                %22 = arith.addi %13, %21 : i32
                %23 = dataflow.select %18, %22, %13 : i32
                %24 = arith.addi %arg4, %c2 : index
                %25 = memref.load %arg0[%24] {affineCoeff = [1], affineOffset = 0 : i64, map = #map, slp.group = 0 : i64, slp.lane = 2 : i64} : memref<16xi32>
                %26 = arith.addi %24, %5 : index
                %27 = memref.load %arg2[%26] {affineCoeff = [1], affineOffset = 0 : i64, map = #map1, slp.group = 1 : i64, slp.lane = 2 : i64} : memref<256xi32>
                %28 = arith.cmpi sgt, %25, %c0_i32 {slp.group = 2 : i64, slp.lane = 2 : i64} : i32
                %29 = arith.muli %25, %27 : i32
                %30 = arith.addi %29, %27 : i32
                %31 = arith.muli %30, %25 : i32
                %32 = arith.addi %23, %31 : i32
                %33 = dataflow.select %28, %32, %23 : i32
                %34 = arith.addi %arg4, %c3 : index
                %35 = memref.load %arg0[%34] {affineCoeff = [1], affineOffset = 0 : i64, map = #map, slp.group = 0 : i64, slp.lane = 3 : i64} : memref<16xi32>
                %36 = arith.addi %34, %5 : index
                %37 = memref.load %arg2[%36] {affineCoeff = [1], affineOffset = 0 : i64, map = #map1, slp.group = 1 : i64, slp.lane = 3 : i64} : memref<256xi32>
                %38 = arith.cmpi sgt, %35, %c0_i32 {slp.group = 2 : i64, slp.lane = 3 : i64} : i32
                %39 = arith.muli %35, %37 : i32
                %40 = arith.addi %39, %37 : i32
                %41 = arith.muli %40, %35 : i32
                %42 = arith.addi %33, %41 : i32
                %43 = dataflow.select %38, %42, %33 : i32
                %true_1 = arith.constant true
                %44 = arith.addi %arg4, %c4 {Exe = "Loop"} : index
                %45 = arith.cmpi eq, %44, %c16 {Exe = "Loop"} : index
                dataflow.state %45, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
                dataflow.yield {execution_block = 1 : i32} %43 : i32
              }
              dataflow.yield %2 : i32
            } {Loop_Band = 0 : i32, Loop_Level = 1 : i32}
            memref.store %0, %arg1[%arg3] {affineCoeff = [1], affineOffset = 0 : i64, map = #map} : memref<16xi32>
            %true_0 = arith.constant true
            %1 = arith.addi %arg3, %c1 {Exe = "Loop"} : index
            dataflow.yield {execution_block = 1 : i32}
          }
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
      }
    }
    return
  }
}

