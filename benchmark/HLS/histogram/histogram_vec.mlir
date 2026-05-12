// module  {
//   func.func @histogram(%arg0: memref<100xi32>, %arg1: memref<100xi32>, %arg2: memref<100xi32>)  {
//     %c10_i32 = arith.constant 10 : i32
//     %c0_i32 = arith.constant 0 : i32
//     %c2_i32 = arith.constant 1 : i32
//     %c3_i32 = arith.constant 2 : i32
//     %c4_i32 = arith.constant 3 : i32
//     affine.for %arg3 = 0 to 100 step 4 {
//       %0 = vector.transfer_read %arg1[%arg3], %c0_i32 : memref<100xi32>, vector<4xi32>
//       %2 = vector.transfer_read %arg0[%arg3], %c0_i32 : memref<100xi32>, vector<4xi32>
//       %3 = arith.index_cast %2 :  vector<4xi32> to  vector<4xindex>
//       // %11 = vector.extractelement %3[%c1_i32: i32] : vector<4xindex>
//       // %12 = vector.extractelement %3[%c2_i32: i32] : vector<4xindex>
//       // %13 = vector.extractelement %3[%c3_i32: i32] : vector<4xindex>
//       // %14 = vector.extractelement %3[%c4_i32: i32] : vector<4xindex>
//       // %4 = memref.load %arg2[%11] : memref<100xi32>
//       // %5 = memref.load %arg2[%12] : memref<100xi32>
//       // %6 = memref.load %arg2[%13] : memref<100xi32>
//       // %7 = memref.load %arg2[%14] : memref<100xi32>
//       %22 = dataflow.vector.load %arg2[%3]  : memref<100xi32>, vector<4xindex> -> vector<4xi32>
//       %5 = arith.addi %22, %0 : vector<4xi32> 
//       dataflow.vector.store %5, %arg2[%3]  :  vector<4xi32>, memref<100xi32> -> vector<4xindex>
//     }
//     return
//   }
// }

module {
  func.func @histogram(%arg0: memref<100xi32>, %arg1: memref<100xi32>, %arg2: memref<100xi32>) {
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg3 = 0 to 100 step 4 {
      %0 = vector.transfer_read %arg1[%arg3], %c0_i32 : memref<100xi32>, vector<4xi32>
      %1 = vector.transfer_read %arg0[%arg3], %c0_i32 : memref<100xi32>, vector<4xi32>
      %2 = arith.index_cast %1 : vector<4xi32> to vector<4xindex>
      %3 = dataflow.vector.load %arg2[%2] : memref<100xi32>[vector<4xindex>] -> vector<4xi32>
      %4 = arith.addi %3, %0 : vector<4xi32>
      dataflow.vector.store %4, %arg2[%2] : vector<4xi32> -> memref<100xi32>[vector<4xindex>]
    }
    return
  }
}