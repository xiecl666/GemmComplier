#ifndef GEMM_TD
#define GEMM_TD

include "mlir/IR/OpBase.td"

def GemmCDialect : Dialect{
    let name = "gemmc";
    let cppNamespace = "::mlir::gemmc";
    let dependentDialects = [];
    let summary = "GEMMC dialect";
    let description = [{
        GEMM dialect for representing GEMM operations.
        This dialect is used to learn how to think.
    }];
    let hasCanonicalizer = 1;
}

#endif
