#ifndef GRAPH_UTILS_H
#define GRAPH_UTILS_H

#include "heteacc/InitAllDialects.h"
#include <string>
#include <vector>

namespace mlir {
namespace heteacc {

bool strReplace(std::string &str, const std::string &from,
                const std::string &to);
bool strReplace(std::string &str, const std::string &from, const int to);
bool strReplace(std::string &str, const std::string &from,
                std::vector<uint32_t> to, const std::string &split);

bool applyGraphInit(func::FuncOp func, bool topFunc);

std::string helperScalaPrintHeader(std::string header);
} // namespace heteacc
} // namespace mlir

#endif // GRAPH_UTILS_H
