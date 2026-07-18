set(LLVM_Target_DEFINTIONS pass.td)
mlir_tablegen(pass.h.inc -gen-pass-decls -name Gemmc)
add_public_tablegen_target(GemmcPassIncGen)