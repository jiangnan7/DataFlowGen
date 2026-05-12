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
        dataflow.for %arg3 = %c0 to %c16 step %c1 {
          %0 = dataflow.for %arg4 = %c0 to %c16 step %c4 iter_args(%arg5 = %c0_i32) -> (i32) {
            %1 = memref.load %arg0[%arg4] {affineCoeff = [1], affineOffset = 0 : i64, map = #map, slp.group = 0 : i64, slp.lane = 0 : i64} : memref<16xi32>
            %2 = arith.muli %arg3, %c16 : index
            %3 = arith.addi %arg4, %2 : index
            %4 = memref.load %arg2[%3] {affineCoeff = [1], affineOffset = 0 : i64, map = #map1, slp.group = 1 : i64, slp.lane = 0 : i64} : memref<256xi32>
            %5 = arith.cmpi sgt, %1, %c0_i32 {slp.group = 2 : i64, slp.lane = 0 : i64} : i32
            %6 = arith.muli %1, %4 : i32
            %7 = arith.addi %6, %4 : i32
            %8 = arith.muli %7, %1 : i32
            %9 = arith.addi %arg5, %8 : i32
            %10 = dataflow.select %5, %9, %arg5 : i32
            %11 = arith.addi %arg4, %c1 : index
            %12 = memref.load %arg0[%11] {affineCoeff = [1], affineOffset = 0 : i64, map = #map, slp.group = 0 : i64, slp.lane = 1 : i64} : memref<16xi32>
            %13 = arith.addi %11, %2 : index
            %14 = memref.load %arg2[%13] {affineCoeff = [1], affineOffset = 0 : i64, map = #map1, slp.group = 1 : i64, slp.lane = 1 : i64} : memref<256xi32>
            %15 = arith.cmpi sgt, %12, %c0_i32 {slp.group = 2 : i64, slp.lane = 1 : i64} : i32
            %16 = arith.muli %12, %14 : i32
            %17 = arith.addi %16, %14 : i32
            %18 = arith.muli %17, %12 : i32
            %19 = arith.addi %10, %18 : i32
            %20 = dataflow.select %15, %19, %10 : i32
            %21 = arith.addi %arg4, %c2 : index
            %22 = memref.load %arg0[%21] {affineCoeff = [1], affineOffset = 0 : i64, map = #map, slp.group = 0 : i64, slp.lane = 2 : i64} : memref<16xi32>
            %23 = arith.addi %21, %2 : index
            %24 = memref.load %arg2[%23] {affineCoeff = [1], affineOffset = 0 : i64, map = #map1, slp.group = 1 : i64, slp.lane = 2 : i64} : memref<256xi32>
            %25 = arith.cmpi sgt, %22, %c0_i32 {slp.group = 2 : i64, slp.lane = 2 : i64} : i32
            %26 = arith.muli %22, %24 : i32
            %27 = arith.addi %26, %24 : i32
            %28 = arith.muli %27, %22 : i32
            %29 = arith.addi %20, %28 : i32
            %30 = dataflow.select %25, %29, %20 : i32
            %31 = arith.addi %arg4, %c3 : index
            %32 = memref.load %arg0[%31] {affineCoeff = [1], affineOffset = 0 : i64, map = #map, slp.group = 0 : i64, slp.lane = 3 : i64} : memref<16xi32>
            %33 = arith.addi %31, %2 : index
            %34 = memref.load %arg2[%33] {affineCoeff = [1], affineOffset = 0 : i64, map = #map1, slp.group = 1 : i64, slp.lane = 3 : i64} : memref<256xi32>
            %35 = arith.cmpi sgt, %32, %c0_i32 {slp.group = 2 : i64, slp.lane = 3 : i64} : i32
            %36 = arith.muli %32, %34 : i32
            %37 = arith.addi %36, %34 : i32
            %38 = arith.muli %37, %32 : i32
            %39 = arith.addi %30, %38 : i32
            %40 = dataflow.select %35, %39, %30 : i32
            dataflow.yield %40 : i32
          } {Loop_Band = 0 : i32, Loop_Level = 1 : i32}
          memref.store %0, %arg1[%arg3] {affineCoeff = [1], affineOffset = 0 : i64, map = #map} : memref<16xi32>
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
      }
    }
    return
  }
}

