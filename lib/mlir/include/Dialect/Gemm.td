#ifndef GEMM_TD
#define GEMM_TD

include "mlir/IR/OpBase.td"

def GemmDialect : Dialect{
    let name = "gemm";
    let cppNamespace = "::mlir::gemm";
    let dependentDialects = [];
    let summary = "GEMM dialect";
    let description = [{
        GEMM dialect for representing GEMM operations.
        This dialect is used to learn how to think.
    }];
    let hasCanonicalizer = 1;
}

#endif
