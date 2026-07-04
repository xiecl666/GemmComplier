# TableGen 生成 Dialect 和 Ops 的 .inc 文件
#
# 当前目录: include/Dialect/
# TableGen 自动添加 -I${CMAKE_CURRENT_SOURCE_DIR}
# .td 中的 include "Gemm.td" 可直接找到

#告诉llvm以下输入都是GemmOps.td
set(LLVM_TARGET_DEFINITIONS GemmOps.td)

#调用mlir的宏，其实就是执行命令
mlir_tablegen(Gemm.h.inc -gen-dialect-decls -dialect=gemmc)
mlir_tablegen(Gemm.cpp.inc -gen-dialect-defs -dialect=gemmc)
mlir_tablegen(GemmOps.h.inc -gen-op-decls)
mlir_tablegen(GemmOps.cpp.inc -gen-op-defs)
mlir_tablegen(GemmTypes.h.inc -gen-typedef-decls -dialect=gemmc)
mlir_tablegen(GemmTypes.cpp.inc -gen-typedef-defs -dialect=gemmc)
mlir_tablegen(GemmAttrs.h.inc -gen-attrdef-decls -dialect=gemmc)
mlir_tablegen(GemmAttrs.cpp.inc -gen-attrdef-defs -dialect=gemmc)

#定义目标，将前边的命令收集起来
add_public_tablegen_target(GemmOpsIncGen)
