// RUN: %heteacc_opt --show-dialects | FileCheck %s

// CHECK: Available Dialects:
// CHECK-SAME: dataflow
// CHECK-SAME: vectorization

module {
}
