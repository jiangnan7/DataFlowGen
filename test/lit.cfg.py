# -*- Python -*-

import os

import lit.formats

from lit.llvm import llvm_config

config.name = "HETEACC"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.heteacc_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%heteacc_opt", "heteacc-opt"))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

config.excludes = ["Inputs", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.heteacc_tools_dir, append_path=True)

tool_dirs = [config.heteacc_tools_dir, config.llvm_tools_dir]
tools = ["heteacc-opt", "FileCheck", "count", "not"]

llvm_config.add_tool_substitutions(tools, tool_dirs)
