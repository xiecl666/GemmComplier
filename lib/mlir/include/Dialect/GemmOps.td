// TODO: 在此定义 Gemm Dialect 的 Ops
#ifndef GEMM_OPS_TD
#define GEMM_OPS_TD

include "Gemm.td"
//include "mlir/Interfaces/SideEffectInterfaces.td" 
include "GemmTypes.td"



class GemmcOp<string mnemonic, list<Trait> traits = []> : 
    Op<GemmCDialect, mnemonic, traits>; 

def GemmOp:GemmcOp<"gemm",[]>{
    let arguments = (ins    GemmCTensor:$A, 
                            GemmCTensor:$B, 
                            GemmCTensor:$C,
                            DefaultValuedAttr<F32Attr,"1.0f">:$alpha,
                            DefaultValuedAttr<F32Attr,"0.0f">:$beta,
                            DefaultValuedAttr<TransposeAttr,"0">:$transA,
                            DefaultValuedAttr<TransposeAttr,"0">:$transB);
    let results = (outs GemmCTensor:$D);
    let summary = "GEMM operation";
    let hasVerifier = 1;
    let cppNamespace = "::mlir::gemmc";
    let extraClassDeclaration = [{
        LogicalResult verify(GemmOp op);
    }];
}


#endif
