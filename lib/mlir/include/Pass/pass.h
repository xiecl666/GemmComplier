#pragma once
#include "mlir/Pass/Pass.h"
namespace mlir::gemmc{
    #define GEN_PASS_DECL
    #include "Pass/Pass.h.inc"

    #define GEN_PASS_REGISTRATION
    #include "Pass/Pass.h.inc"
    
}