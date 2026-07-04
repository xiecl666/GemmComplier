// TODO: gemm-opt 编译器入口
// 参考:
//   #include "mlir/Tools/mlir-opt/MlirOptMain.h"
//   #include "mlir/InitAllDialects.h"
//   #include "mlir/InitAllPasses.h"
//   注册自定义 GemmDialect, 然后调用 MlirOptMain
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/IR/DialectRegistry.h"

#include "Dialect/Gemm.h"

int main(int argc, char **argv) {
    // 1. 注册所有 MLIR 内置 Dialect（func, arith, linalg, scf, gpu 等）
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);

    // 2. 注册自定义 GemmC Dialect
    registry.insert<mlir::gemmc::GemmCDialect>();

    // 3. 注册所有内置 Pass（方便后续调用 --convert-xxx-to-yyy）
    mlir::registerAllPasses();

    // TODO: 注册自定义 Pass（如 GemmToLinalg、GemmTiling 等）
    // mlir::gemmc::registerGemmToLinalgPass();

    // 4. 启动 mlir-opt 主循环
    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "GemmC Optimizer\n", registry));
}
