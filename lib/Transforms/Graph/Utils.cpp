#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "heteacc/Transforms/Passes.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "heteacc/Graph/Utils.h"
#include "heteacc/Graph/Visitor.h"
#include "heteacc/Graph/GraphGen.h"


using namespace mlir;
using namespace heteacc;


bool heteacc::strReplace(std::string &str, const std::string &from,
                            const std::string &to) {
    assert(!from.compare(0, 1, "$") && "Replace string should start with $!");
    bool _ret = false;
    while (true) {
        size_t start_pos = str.find(from);
        if (start_pos == std::string::npos) break;
        str.replace(start_pos, from.length(), to);
        _ret = true;
    }
    return _ret;
}

bool heteacc::strReplace(std::string &str, const std::string &from,
                            const int to) {
    assert(!from.compare(0, 1, "$") && "Replace string should start with $!");
    bool _ret = false;
    while (true) {
        size_t start_pos = str.find(from);
        if (start_pos == std::string::npos) break;
        str.replace(start_pos, from.length(), std::to_string(to));
        _ret = true;
    }
    return _ret;
}
#include <experimental/iterator>
bool heteacc::strReplace(std::string &str, const std::string &from,
                            std::vector<uint32_t> to,
                            const std::string &split) {
    assert(!from.compare(0, 1, "$") && "Replace string should start with $!");
    bool _ret = false;
    std::stringstream test;
    std::copy(to.begin(), to.end(),
              std::experimental::make_ostream_joiner(test, split));

    while (true) {
        size_t start_pos = str.find(from);
        if (start_pos == std::string::npos) break;
        str.replace(start_pos, from.length(), test.str());
        _ret = true;
    }
    return _ret;
}

/**
 * HELPER FUNCTIONS
 * Printing header part of each section of the code
 */
std::string heteacc::helperScalaPrintHeader(std::string header) {
  std::string tmp_line = "   * "
                    "================================================================== "
                    "*/\n\n";

  uint32_t remain_space = tmp_line.length() - 2 - header.length() - 23;

  // Append space to the string
  std::string header_final = "";
  for (uint32_t i = 0; i < remain_space - 2; i++) {
    header_final.append(" ");
  }
  header_final.append("*\n");

  tmp_line = "\n\n  /* "
             "================================================================== "
             "*\n"
             "   *                   "
             + header + header_final + tmp_line;
  return tmp_line;
}