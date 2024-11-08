
#ifndef DIALECT_DATAFLOW_IR_TD
#define DIALECT_DATAFLOW_IR_TD

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "heteacc/Dialect/DataFlow/DataFlowOpsDialect.h.inc"
#include "heteacc/Dialect/DataFlow/DataFlowOpsEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "heteacc/Dialect/DataFlow/DataFlowOpsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "heteacc/Dialect/DataFlow/DataFlowOpsAttributes.h.inc"

namespace mlir {
namespace heteacc{
namespace dataflow{

#include "heteacc/Dialect/DataFlow/DataFlowOpsInterfaces.h.inc"

} 
}
}




#define GET_OP_CLASSES
#include "heteacc/Dialect/DataFlow/DataFlowOps.h.inc"



#endif // DIALECT_DATAFLOW_IR_TD
