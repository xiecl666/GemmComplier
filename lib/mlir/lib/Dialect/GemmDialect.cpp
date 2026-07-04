// TODO: 实现 Dialect 注册和 Op 方法
// 参考:
//   #include "Dialect/Gemm.cpp.inc"  (TableGen 生成的 Dialect 定义)
//   #define GET_OP_CLASSES
//   #include "Dialect/GemmOps.cpp.inc"  (TableGen 生成的 Op 定义)
#include "Dialect/Gemm.h"
#include "Dialect/GemmOps.h"
#include "Dialect/GemmTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Dialect/Gemm.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Dialect/GemmTypes.cpp.inc"
#undef GET_TYPEDEF_CLASSES

#define GET_ATTRDEF_CLASSES
#include "Dialect/GemmAttrs.cpp.inc"
#undef GET_ATTRDEF_CLASSES

#define GET_OP_CLASSES
#include "Dialect/GemmOps.cpp.inc"
#undef GET_OP_CLASSES


namespace mlir {
namespace gemmc {
    void GemmCDialect::initialize() {
        addOperations<
            #define GET_OP_LIST
            #include "Dialect/GemmOps.cpp.inc"
            >();
        
        addTypes<
            #define GET_TYPEDEF_LIST
            #include "Dialect/GemmTypes.cpp.inc"
            >();
        
        addAttributes<
            #define GET_ATTRDEF_LIST
            #include "Dialect/GemmAttrs.cpp.inc"
            >();
    }

    LogicalResult GemmOp::verify() {
        return success();
    }
    void GemmCDialect::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results) const {
    }

}   
}
