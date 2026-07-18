#include "Pass/pass.h"
#include "Dialect/GemmOps.h"

namespace mlir::gemmc{
    #deifne GEN_PASS_DEF_GEMMCTOLINALG
    #include "Pass/Pass.h.inc"
    
    namespace {
        struct GemmcTOLinalgPass : public impl::GemmcToLinalgBase<GemmcTOLinalgPass>{
            void runOpOperation override{

            }
        };
    }
    std::unique_ptr<Pass> createGemmToLinalg(){
        return std::make_unique<GemmcTOLinalgPass>();
    }
}