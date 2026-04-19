// RUN: %heteacc_opt %s --optimize-dataflow | FileCheck %s

module {
  func.func @lower_affine_if(%arg: index) {
    affine.if affine_set<(d0) : (d0 - 4 >= 0)>(%arg) {
    }
    return
  }
}

// CHECK-LABEL: func.func @lower_affine_if
// CHECK: return
// CHECK-NOT: affine.if
// CHECK-NOT: dataflow.if
