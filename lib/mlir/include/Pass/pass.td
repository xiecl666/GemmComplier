#ifndef GEMMC_PASSES
#define GEMMC_PASSES

include "mlir/Pass/PassBase.td"

def GemmcToLinalg: InterfacePass<"gemmc-to-linalg","FuncitonOpInterface">{
    let summary="Lower Gemmc to Linalg";
    let description=[{
        !
        !
    }];

    let constructor="gemmc::createGemmcToLinalg()";
}

#endif
