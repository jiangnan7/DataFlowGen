#ifndef GRAPH_UTILS_H
#define GRAPH_UTILS_H

#include "heteacc/Graph/Utils.h"
#include "heteacc/Graph/Visitor.h"
#include "heteacc/Graph/Graph.h"
#include "heteacc/Graph/Node.h"

namespace mlir {
namespace heteacc {


bool strReplace(std::string &str, const std::string &from,
                            const std::string &to);
bool strReplace(std::string &str, const std::string &from,
                            const int to);
bool strReplace(std::string &str, const std::string &from,
                            std::vector<uint32_t> to,
                            const std::string &split);
                            
bool applyGraphInit(func::FuncOp func, bool topFunc);

std::string helperScalaPrintHeader(std::string header);
}
}


#endif // GRAPH_UTILS_H