#ifndef GEMM_TD
#define GEMM_TD

#include "mlir/IR/DialectBase.td"

def GemmDialect : Dialect{
    let name = "gemm";
    let namespace = "::mlir::gemm";
    let dependencies = [];
    let hasCanonicalizer = 1;
}

#endif