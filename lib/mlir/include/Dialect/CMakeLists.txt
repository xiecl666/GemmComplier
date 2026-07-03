# TableGen 生成 Dialect 和 Ops 的 .inc 文件
#
# 当前目录: include/Dialect/
# TableGen 自动添加 -I${CMAKE_CURRENT_SOURCE_DIR}
# .td 中的 include "Gemm.td" 可直接找到

set(LLVM_TARGET_DEFINITIONS GemmOps.td)

mlir_tablegen(Gemm.h.inc -gen-dialect-decls -dialect=gemm)
mlir_tablegen(Gemm.cpp.inc -gen-dialect-defs -dialect=gemm)
mlir_tablegen(GemmOps.h.inc -gen-op-decls)
mlir_tablegen(GemmOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(GemmOpsIncGen)
