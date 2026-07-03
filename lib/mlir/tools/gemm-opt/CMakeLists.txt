get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(gemm-opt
  gemm-opt.cpp
)

target_link_libraries(gemm-opt
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  GemmDialect
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIROptLib
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRGPUDialect
  MLIRSCFDialect
)

llvm_update_compile_flags(gemm-opt)
