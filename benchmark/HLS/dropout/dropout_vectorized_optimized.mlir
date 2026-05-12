#map = affine_map<(d0) -> (d0)>
module {
  func.func @dropout(%arg0: memref<1024xi32>, %arg1: memref<128xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c7 = arith.constant 7 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c8 = arith.constant 8 : index
    %0 = dataflow.launch : i32 {
      %1 = dataflow.task : i32 {
        %true = arith.constant true
        dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
        %2:2 = dataflow.for %arg2 = %c0 to %c1024 step %c4 iter_args(%arg3 = %c0_i32, %arg4 = %c0_i32) -> (i32, i32) {
          %3:2 = dataflow.execution : i32, i32 {
            %4 = arith.index_cast %arg2 {slp.group = 0 : i64, slp.lane = 0 : i64} : index to i32
            %5 = arith.andi %arg2, %c7 : index
            %6 = arith.cmpi slt, %5, %c0 {slp.group = 2 : i64, slp.lane = 0 : i64} : index
            %7 = arith.addi %5, %c8 {slp.group = 3 : i64, slp.lane = 0 : i64} : index
            %8 = arith.select %6, %7, %5 {slp.group = 4 : i64, slp.lane = 0 : i64} : index
            %9 = arith.cmpi eq, %8, %c0 {slp.group = 5 : i64, slp.lane = 0 : i64} : index
            %10 = arith.shrui %4, %c3_i32 : i32
            %11 = arith.index_cast %10 : i32 to index
            %12 = memref.load %arg1[%11] : memref<128xi32>
            %13 = dataflow.select %9, %12, %arg4 : i32
            %14 = arith.andi %13, %c1_i32 {slp.group = 7 : i64, slp.lane = 0 : i64} : i32
            %15 = arith.cmpi ne, %14, %c0_i32 {slp.group = 8 : i64, slp.lane = 0 : i64} : i32
            %16 = memref.load %arg0[%arg2] {affineCoeff = [1], affineOffset = 0 : i64, map = #map} : memref<1024xi32>
            %17 = arith.muli %16, %c2_i32 : i32
            %18 = dataflow.select %15, %17, %c0_i32 : i32
            %19 = arith.addi %arg3, %18 {slp.group = 10 : i64, slp.lane = 0 : i64} : i32
            %20 = arith.shrsi %13, %c1_i32 {slp.group = 11 : i64, slp.lane = 0 : i64} : i32
            %21 = arith.addi %arg2, %c1 : index
            %22 = arith.index_cast %21 {slp.group = 0 : i64, slp.lane = 1 : i64} : index to i32
            %23 = arith.andi %21, %c7 : index
            %24 = arith.cmpi slt, %23, %c0 {slp.group = 2 : i64, slp.lane = 1 : i64} : index
            %25 = arith.addi %23, %c8 {slp.group = 3 : i64, slp.lane = 1 : i64} : index
            %26 = arith.select %24, %25, %23 {slp.group = 4 : i64, slp.lane = 1 : i64} : index
            %27 = arith.cmpi eq, %26, %c0 {slp.group = 5 : i64, slp.lane = 1 : i64} : index
            %28 = arith.shrui %22, %c3_i32 : i32
            %29 = arith.index_cast %28 : i32 to index
            %30 = memref.load %arg1[%29] : memref<128xi32>
            %31 = dataflow.select %27, %30, %20 : i32
            %32 = arith.andi %31, %c1_i32 {slp.group = 7 : i64, slp.lane = 1 : i64} : i32
            %33 = arith.cmpi ne, %32, %c0_i32 {slp.group = 8 : i64, slp.lane = 1 : i64} : i32
            %34 = memref.load %arg0[%21] {affineCoeff = [1], affineOffset = 0 : i64, map = #map} : memref<1024xi32>
            %35 = arith.muli %34, %c2_i32 : i32
            %36 = dataflow.select %33, %35, %c0_i32 : i32
            %37 = arith.addi %19, %36 {slp.group = 10 : i64, slp.lane = 1 : i64} : i32
            %38 = arith.shrsi %31, %c1_i32 {slp.group = 11 : i64, slp.lane = 1 : i64} : i32
            %39 = arith.addi %arg2, %c2 : index
            %40 = arith.index_cast %39 {slp.group = 0 : i64, slp.lane = 2 : i64} : index to i32
            %41 = arith.andi %39, %c7 : index
            %42 = arith.cmpi slt, %41, %c0 {slp.group = 2 : i64, slp.lane = 2 : i64} : index
            %43 = arith.addi %41, %c8 {slp.group = 3 : i64, slp.lane = 2 : i64} : index
            %44 = arith.select %42, %43, %41 {slp.group = 4 : i64, slp.lane = 2 : i64} : index
            %45 = arith.cmpi eq, %44, %c0 {slp.group = 5 : i64, slp.lane = 2 : i64} : index
            %46 = arith.shrui %40, %c3_i32 : i32
            %47 = arith.index_cast %46 : i32 to index
            %48 = memref.load %arg1[%47] : memref<128xi32>
            %49 = dataflow.select %45, %48, %38 : i32
            %50 = arith.andi %49, %c1_i32 {slp.group = 7 : i64, slp.lane = 2 : i64} : i32
            %51 = arith.cmpi ne, %50, %c0_i32 {slp.group = 8 : i64, slp.lane = 2 : i64} : i32
            %52 = memref.load %arg0[%39] {affineCoeff = [1], affineOffset = 0 : i64, map = #map} : memref<1024xi32>
            %53 = arith.muli %52, %c2_i32 : i32
            %54 = dataflow.select %51, %53, %c0_i32 : i32
            %55 = arith.addi %37, %54 {slp.group = 10 : i64, slp.lane = 2 : i64} : i32
            %56 = arith.shrsi %49, %c1_i32 {slp.group = 11 : i64, slp.lane = 2 : i64} : i32
            %57 = arith.addi %arg2, %c3 : index
            %58 = arith.index_cast %57 {slp.group = 0 : i64, slp.lane = 3 : i64} : index to i32
            %59 = arith.andi %57, %c7 : index
            %60 = arith.cmpi slt, %59, %c0 {slp.group = 2 : i64, slp.lane = 3 : i64} : index
            %61 = arith.addi %59, %c8 {slp.group = 3 : i64, slp.lane = 3 : i64} : index
            %62 = arith.select %60, %61, %59 {slp.group = 4 : i64, slp.lane = 3 : i64} : index
            %63 = arith.cmpi eq, %62, %c0 {slp.group = 5 : i64, slp.lane = 3 : i64} : index
            %64 = arith.shrui %58, %c3_i32 : i32
            %65 = arith.index_cast %64 : i32 to index
            %66 = memref.load %arg1[%65] : memref<128xi32>
            %67 = dataflow.select %63, %66, %56 : i32
            %68 = arith.andi %67, %c1_i32 {slp.group = 7 : i64, slp.lane = 3 : i64} : i32
            %69 = arith.cmpi ne, %68, %c0_i32 {slp.group = 8 : i64, slp.lane = 3 : i64} : i32
            %70 = memref.load %arg0[%57] {affineCoeff = [1], affineOffset = 0 : i64, map = #map} : memref<1024xi32>
            %71 = arith.muli %70, %c2_i32 : i32
            %72 = dataflow.select %69, %71, %c0_i32 : i32
            %73 = arith.addi %55, %72 {slp.group = 10 : i64, slp.lane = 3 : i64} : i32
            %74 = arith.shrsi %67, %c1_i32 {slp.group = 11 : i64, slp.lane = 3 : i64} : i32
            %true_0 = arith.constant true
            %75 = arith.addi %arg2, %c4 {Exe = "Loop"} : index
            dataflow.yield {execution_block = 1 : i32} %73, %74 : i32, i32
          }
          dataflow.yield %3#0, %3#1 : i32, i32
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
        dataflow.yield %2#0 : i32
      }
      dataflow.yield %1 : i32
    }
    return %0 : i32
  }
}

