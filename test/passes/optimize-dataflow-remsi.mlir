// RUN: %heteacc_opt %s --optimize-dataflow | FileCheck %s

module {
  func.func @remsi_to_andi(%x: i32) -> i32 {
    %c8 = arith.constant 8 : i32
    %0 = arith.remsi %x, %c8 : i32
    return %0 : i32
  }
}

// CHECK-LABEL: func.func @remsi_to_andi
// CHECK: %c7_i32 = arith.constant 7 : i32
// CHECK: %[[AND:.+]] = arith.andi %{{.*}}, %c7_i32 : i32
// CHECK: return %[[AND]] : i32
// CHECK-NOT: arith.remsi
