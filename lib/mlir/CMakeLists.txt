cmake_minimum_required(VERSION 3.20)
project(GemmCompiler LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# 查找预编译的 MLIR/LLVM
# ============================================================================
set(MLIR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../deps/mlir/lib/cmake/mlir" CACHE PATH "MLIR CMake dir")
set(LLVM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../deps/mlir/lib/cmake/llvm" CACHE PATH "LLVM CMake dir")

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIR at: ${MLIR_DIR}")
message(STATUS "Using LLVM at: ${LLVM_DIR}")

# 引入 MLIR/LLVM 的 CMake 模块
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# LLVM/MLIR 头文件
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# 项目自身的头文件 (原始 .h 和 TableGen 生成的 .inc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

# LLVM definitions
add_definitions(${LLVM_DEFINITIONS})

# ============================================================================
# 子目录
# ============================================================================
add_subdirectory(include/Dialect)
add_subdirectory(lib/Dialect)
add_subdirectory(tools/gemm-opt)
