
#ifndef DIALECT_VECTORIZATION_TD
#define DIALECT_VECTORIZATION_TD
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def Vectorization_Dialect : Dialect {
  let name = "vectorization";
  let summary = "An Vec out-of-tree MLIR dialect";
  let description = [{
    vec
  }];
  // let useDefaultTypePrinterParser = 1;
  // let useDefaultAttributePrinterParser = 1;
  let cppNamespace = "::mlir::heteacc::vectorization";
  // let hasConstantMaterializer = 1;
}


class Vec_Op<string mnemonic, list<Trait> traits = []> :
        Op<Vectorization_Dialect, mnemonic, traits>;

#endif // DIALECT_VECTORIZATION_TD