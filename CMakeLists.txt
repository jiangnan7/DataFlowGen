cmake_minimum_required(VERSION 3.13.4)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

# Add the policy check for CMP0116
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 NEW)
endif()

project(mlir-hls LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")


find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(HETEACC_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(HETEACC_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR} ) # --src-root
set(HETEACC_MAIN_INCLUDE_DIR ${HETEACC_MAIN_SRC_DIR}/include)
set(HETEACC_PYTHON_PACKAGES_DIR ${HETEACC_BINARY_DIR}/python_packages)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${HETEACC_MAIN_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

option(HETEACC_ENABLE_BINDINGS_PYTHON "Enables Heteacc Python bindings." ON)

if(HETEACC_ENABLE_BINDINGS_PYTHON)
  message(STATUS "Heteacc Python bindings are enabled.")
  
  include(MLIRDetectPythonEnv)
  find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
  message(STATUS "Found Python include dirs: ${Python3_INCLUDE_DIRS}")
  message(STATUS "Found Python libraries: ${Python3_LIBRARIES}")
  message(STATUS "Found Python executable: ${Python3_EXECUTABLE}")
  message(STATUS "Found numpy v${Python3_NumPy_VERSION}: ${Python3_NumPy_INCLUDE_DIRS}")
  include_directories(${Python3_NumPy_INCLUDE_DIRS})
  add_subdirectory(python)
  mlir_detect_pybind11_install()
  find_package(pybind11 CONFIG REQUIRED)
  message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIR}")
  message(STATUS "Python prefix = '${PYTHON_MODULE_PREFIX}', "
                 "suffix = '${PYTHON_MODULE_SUFFIX}', "
                 "extension = '${PYTHON_MODULE_EXTENSION}'")
else()
  message(STATUS "Heteacc Python bindings are disabled.")
endif()

add_subdirectory(include/heteacc)
add_subdirectory(lib)
# add_subdirectory(test)
add_subdirectory(heteacc-opt)
