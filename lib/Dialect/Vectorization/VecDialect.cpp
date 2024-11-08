

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "heteacc/Dialect/Vectorization/Vectorization.h"


using namespace mlir;
using namespace heteacc;
using namespace vectorization;

//===----------------------------------------------------------------------===//
// Vectorization dialect.
//===----------------------------------------------------------------------===//

#include "heteacc/Dialect/Vectorization/VectorizationOpsDialect.cpp.inc"

void VectorizationDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "heteacc/Dialect/Vectorization/VectorizationOps.cpp.inc"
      >();
}

// void hello::ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, double value) {
//   auto dataType = RankedTensorType::get({}, builder.getF64Type());
//   auto dataAttribute = DenseElementsAttr::get(dataType, value);
//   hello::ConstantOp::build(builder, state, dataType, dataAttribute);
// }

// mlir::Operation *HelloDialect::materializeConstant(mlir::OpBuilder &builder,
//                                                  mlir::Attribute value,
//                                                  mlir::Type type,
//                                                  mlir::Location loc) {
//     return builder.create<hello::ConstantOp>(loc, type,
//                                       value.cast<mlir::DenseElementsAttr>());
// }