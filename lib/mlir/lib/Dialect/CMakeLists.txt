add_mlir_library(GemmDialect
  GemmDialect.cpp

  DEPENDS
  GemmOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
