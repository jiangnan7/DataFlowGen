get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

include(AddMLIRPython)


# add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_heteacc.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(HeteaccPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT HeteaccPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE heteacc/DataFlowOps.td
  SOURCES
    heteacc/dataflow/dataflow.py
  DIALECT_NAME dataflow)

declare_mlir_python_extension(HeteaccPythonSources.Extension
  MODULE_NAME _heteacc
  ADD_TO_PARENT HeteaccPythonSources
  SOURCES
    HeteaccExtension.cpp
  PRIVATE_LINK_LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${translation_libs}
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    HeteaccCAPIDataFlow
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(HeteaccPythonCAPI
  INSTALL_COMPONENT HeteaccPythonModules
  INSTALL_DESTINATION python_packages/heteacc_core/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${HETEACC_PYTHON_PACKAGES_DIR}/heteacc_core/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    HeteaccPythonSources
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects
    MLIRPythonSources.ExecutionEngine
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(HeteaccMLIRPythonModules
  ROOT_PREFIX "${HETEACC_PYTHON_PACKAGES_DIR}/heteacc_core/mlir"
  INSTALL_PREFIX "python_packages/heteacc_core/mlir"
  DECLARED_SOURCES
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects
    MLIRPythonSources.ExecutionEngine

    HeteaccPythonSources
  COMMON_CAPI_LINK_LIBS
  HeteaccPythonCAPI
)

add_mlir_python_modules(HeteaccPythonModules
  ROOT_PREFIX "${HETEACC_PYTHON_PACKAGES_DIR}/heteacc_core/"
  INSTALL_PREFIX "python_packages/heteacc_core/"
  DECLARED_SOURCES
    HeteaccPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    HeteaccPythonCAPI
  )

