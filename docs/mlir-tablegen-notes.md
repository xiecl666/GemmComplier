# MLIR TableGen 笔记

## TableGen 基本语法

### 预处理指令

TableGen 支持类 C 预处理器语法（由 TableGen 自身处理，非 C 预处理器）：

```tablegen
#ifndef GEMM_TD
#define GEMM_TD

include "mlir/IR/OpBase.td"   // 注意: 是 include 不是 #include

// ...

#endif // GEMM_TD
```

### include 规则

- TableGen 用 `include`（无 `#`）引入其他 .td 文件
- 搜索路径由 CMake 的 `-I` 参数控制，TableGen 自动添加 `CMAKE_CURRENT_SOURCE_DIR`

---

## 关键 .td 文件作用

| 文件 | 提供的基类 |
|------|-----------|
| `mlir/IR/OpBase.td` | `Dialect`、`Op`、`Attr`、`Type` 等所有基类 |
| `mlir/IR/DialectBase.td` | 仅基础辅助定义，**不含完整 Dialect 类** |
| `mlir/IR/AttrTypeBase.td` | `AttrDef`、`TypeDef` 模板基类 |
| `mlir/Interfaces/SideEffectInterfaces.td` | `Pure` 等 Trait/Interface |

**规则：只要你的 .td 中要定义 Op，include 链中必须包含 `OpBase.td`。**

---

## 各种定义的 TableGen 写法对比

### Dialect

```tablegen
def GemmCDialect : Dialect {
    let name = "gemmc";                    // IR 文本中的方言前缀
    let cppNamespace = "::mlir::gemmc";    // C++ namespace
    let summary = "...";
}
```

- 继承：直接 `: Dialect`
- C++ 类名：**= def 名本身**（`GemmCDialect`）

### Op

```tablegen
def GemmC_MatmulOp : Op<GemmCDialect, "matmul", [Pure]> {
    let arguments = (ins ...);
    let results = (outs ...);
    let assemblyFormat = [{...}];
}
```

- 继承：`: Op<Dialect, "mnemonic", [traits...]>`
- C++ 类名：**= def 名本身**（`GemmC_MatmulOp`，但通常直接用 `MatmulOp`）
- IR 文本：`gemmc.matmul`

### Attr

```tablegen
def GemmC_AlphaAttr : AttrDef<GemmCDialect, "Alpha"> {
    let mnemonic = "alpha";
    let parameters = (ins "double":$value);
    let assemblyFormat = "`<` $value `>`";
}
```

- 继承：`: AttrDef<Dialect, "Name">`
- C++ 类名：**= 第二个参数 + "Attr" 后缀**（`AlphaAttr`，不是 `GemmC_AlphaAttr`）
- def 名的 `GemmC_` 前缀仅防止 TableGen 记录名冲突，生成时被忽略
- IR 文本：`#gemmc.alpha<1.0>`

### Type

```tablegen
def GemmC_MatrixType : TypeDef<GemmCDialect, "Matrix"> {
    let mnemonic = "matrix";
    let parameters = (ins "int64_t":$rows, "int64_t":$cols);
}
```

- 继承：`: TypeDef<Dialect, "Name">`
- C++ 类名：**= 第二个参数 + "Type" 后缀**（`MatrixType`）
- IR 文本：`!gemmc.matrix<512, 768>`

---

## 命名规则总结

| 场景 | Dialect | Op | Attr | Type |
|------|---------|-----|------|------|
| .td def 名 | `GemmCDialect` | `GemmC_MatmulOp` | `GemmC_AlphaAttr` | `GemmC_MatrixType` |
| C++ 类名 | `GemmCDialect` | `GemmC_MatmulOp` | `AlphaAttr` | `MatrixType` |
| IR 文本 | `gemmc` | `gemmc.matmul` | `#gemmc.alpha<>` | `!gemmc.matrix<>` |
| 类名来自 | def 名 | def 名 | 第2参数+"Attr" | 第2参数+"Type" |

---

## Trait vs Interface

### 使用已有的

```tablegen
def GemmC_MatmulOp : Op<GemmCDialect, "matmul", [
    Pure,                                           // Trait: 无副作用
    DeclareOpInterfaceMethods<InferTypeOpInterface>  // Interface: 需要实现方法
]> { ... }
```

| | Trait | Interface |
|---|---|---|
| 本质 | 标签/标记 | 契约/能力 |
| 需要实现方法？ | 不需要 | 需要 |
| 写法 | 直接写名字 `Pure` | `DeclareOpInterfaceMethods<XXX>` |
| 类比 | Java annotation | Java interface |

### 自定义 Trait

```tablegen
// .td: 一行声明，指向 C++ 实现
def MyTrait : NativeOpTrait<"MyTrait">;
```

```cpp
// C++ 全手写
namespace OpTrait {
template <typename ConcreteType>
class MyTrait : public OpTrait::TraitBase<ConcreteType, MyTrait> {
    // 验证逻辑
};
}
```

- **TableGen 不生成代码**，C++ 全手写
- CMake 无需额外 tablegen 命令

### 自定义 Interface

```tablegen
// .td: 完整定义方法签名
def MyInterface : OpInterface<"MyInterface"> {
  let methods = [
    InterfaceMethod<"获取 foo 值", "int", "getFoo",
        (ins),           // 参数
        [{}],            // trait 约束
        [{ return 0; }]  // 默认实现
    >
  ];
}
```

- **TableGen 生成** `.h.inc` / `.cpp.inc`
- CMake 需加：
  ```cmake
  mlir_tablegen(MyInterface.h.inc -gen-op-interface-decls)
  mlir_tablegen(MyInterface.cpp.inc -gen-op-interface-defs)
  ```

---

## CMake TableGen 配置

```cmake
# 设置入口 .td 文件（所有 mlir_tablegen 命令都以此为输入）
set(LLVM_TARGET_DEFINITIONS GemmOps.td)

# Dialect 声明/定义
mlir_tablegen(Gemm.h.inc -gen-dialect-decls -dialect=gemmc)
mlir_tablegen(Gemm.cpp.inc -gen-dialect-defs -dialect=gemmc)

# Op 声明/定义
mlir_tablegen(GemmOps.h.inc -gen-op-decls)
mlir_tablegen(GemmOps.cpp.inc -gen-op-defs)

# Type 声明/定义
mlir_tablegen(GemmTypes.h.inc -gen-typedef-decls -dialect=gemmc)
mlir_tablegen(GemmTypes.cpp.inc -gen-typedef-defs -dialect=gemmc)

# Attr 声明/定义
mlir_tablegen(GemmAttrs.h.inc -gen-attrdef-decls -dialect=gemmc)
mlir_tablegen(GemmAttrs.cpp.inc -gen-attrdef-defs -dialect=gemmc)

# 收集为一个 target
add_public_tablegen_target(GemmOpsIncGen)
```

仅运行 TableGen（不编译 C++）：
```bash
ninja GemmOpsIncGen
```

---

## 常见报错

| 错误 | 原因 | 解决 |
|------|------|------|
| `The class 'Dialect' is not defined` | 入口 .td 的 include 链中没有 `OpBase.td` | 确保 include 了 `mlir/IR/OpBase.td` |
| `Couldn't find the 'Op' class!` | 同上，或入口 .td 为空 | 同上 |
| `unknown target 'GemmOpsIncGen'` | build 目录 CMake 配置过期 | 重新 `cmake ..` 或删 build 重建 |

---

## CRTP 在 MLIR 中的使用

CRTP（Curiously Recurring Template Pattern）—— 派生类把自己作为模板参数传给基类，实现编译期多态。MLIR 几乎所有可扩展基类都用了 CRTP。

### Op

```cpp
// 每个 Op 都是 CRTP
class MatmulOp : public mlir::Op<MatmulOp, Pure, ...> {
//                               ^^^^^^^^^ 自己作为模板参数
};
```

### Type

```cpp
class MatrixType : public mlir::Type::TypeBase<MatrixType, mlir::Type, MatrixTypeStorage> {
//                                             ^^^^^^^^^^ CRTP
};
```

### Attribute

```cpp
class AlphaAttr : public mlir::Attribute::AttrBase<AlphaAttr, mlir::Attribute, AlphaAttrStorage> {
//                                                 ^^^^^^^^^ CRTP
};
```

### Trait

```cpp
template <typename ConcreteType>
class Pure : public OpTrait::TraitBase<ConcreteType, Pure> {
//                                     ^^^^^^^^^^^^  ^^^^ CRTP 两层
};
```

### Interface

```cpp
class InferTypeOpInterface : public OpInterface<InferTypeOpInterface, detail::Traits> {
//                                              ^^^^^^^^^^^^^^^^^^^^ CRTP
};
```

### 总结表

| 类型 | CRTP 基类 | 目的 |
|------|-----------|------|
| Op | `Op<ConcreteOp, Traits...>` | 注入通用方法（getOperation、verify 等） |
| Type | `TypeBase<ConcreteType, BaseType, Storage>` | 类型系统注册 + 存储管理 |
| Attribute | `AttrBase<ConcreteAttr, BaseAttr, Storage>` | 同上 |
| Trait | `TraitBase<ConcreteOp, TraitClass>` | 给 Op 注入静态验证/能力 |
| Interface | `OpInterface<ConcreteInterface, Traits>` | 虚方法调度框架 |

### 为什么用 CRTP 而不是虚函数

- **零成本抽象** —— 编译期多态，无虚表开销
- **性能关键** —— 编译器处理百万级 Op 实例时不能有运行时分发
- **静态调用** —— 基类可以调用派生类的静态方法（如 `ConcreteOp::getOperationName()`）

---

## TableGen 宏与头文件包含规范

### 关键概念： `.cpp.inc` / `.h.inc` 是条件包含的

TableGen 生成的 `.inc` 文件内部全是 `#ifdef GET_XXX ... #endif` 块。**不 `#define` 对应宏就展开为空**。必须正确定义宏才能获得代码。

### 宏对应表（必记）

| 文件 | 目的 | 应定义的宏 |
|------|------|-----------|
| `Gemm.cpp.inc` | Dialect 构造函数定义 | 无需宏（无条件展开）|
| `Gemm.h.inc` | Dialect 类声明 | 无需宏 |
| `GemmOps.cpp.inc` | Op 类方法体 | `#define GET_OP_CLASSES` |
| `GemmOps.h.inc` | Op 类声明 | `#define GET_OP_CLASSES` |
| `GemmOps.cpp.inc` | Op 类名列表（用于 addOperations）| `#define GET_OP_LIST` |
| `GemmTypes.cpp.inc` | Type 类方法体 | `#define GET_TYPEDEF_CLASSES` |
| `GemmTypes.h.inc` | Type 类声明 | `#define GET_TYPEDEF_CLASSES` |
| `GemmTypes.cpp.inc` | Type 类名列表（用于 addTypes）| `#define GET_TYPEDEF_LIST` |
| `GemmAttrs.cpp.inc` | Attr 类方法体 | `#define GET_ATTRDEF_CLASSES` |
| `GemmAttrs.h.inc` | Attr 类声明 | `#define GET_ATTRDEF_CLASSES` |
| `GemmAttrs.cpp.inc` | Attr 类名列表（用于 addAttributes）| `#define GET_ATTRDEF_LIST` |

**记忆要领：**
- `_CLASSES` → 展开为完整类定义（namespace、方法体）→ 放在**全局作用域**
- `_LIST` → 展开为逗号分隔的类名列表 → 放在 `addXxx<...>` **模板参数尖括号内**

### `#define` 是粘性的

一个 `#define` 后它一直生效，多个不同 include 共享同一个宏会引发重复展开。正确做法：**每个 include 前 `#define`，后面 `#undef` 隔离**：

```cpp
#define GET_OP_CLASSES
#include "Dialect/GemmOps.cpp.inc"
#undef GET_OP_CLASSES

#define GET_TYPEDEF_CLASSES
#include "Dialect/GemmTypes.cpp.inc"
#undef GET_TYPEDEF_CLASSES

#define GET_ATTRDEF_CLASSES
#include "Dialect/GemmAttrs.cpp.inc"
#undef GET_ATTRDEF_CLASSES
```

---

## Dialect 实现文件 (`GemmDialect.cpp`) 标准骨架

```cpp
// 1. 项目头文件
#include "Dialect/Gemm.h"
#include "Dialect/GemmOps.h"
#include "Dialect/GemmTypes.h"

// 2. MLIR 支持头文件
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

// 3. 全局作用域展开 TableGen 类定义（顺序重要）
#include "Dialect/Gemm.cpp.inc"          // Dialect 构造（无需宏）

#define GET_TYPEDEF_CLASSES
#include "Dialect/GemmTypes.cpp.inc"     // Type 类
#undef GET_TYPEDEF_CLASSES

#define GET_ATTRDEF_CLASSES
#include "Dialect/GemmAttrs.cpp.inc"     // Attr 类
#undef GET_ATTRDEF_CLASSES

#define GET_OP_CLASSES
#include "Dialect/GemmOps.cpp.inc"       // Op 类（依赖上面的 Type/Attr）
#undef GET_OP_CLASSES

// 4. Dialect 初始化（在 namespace 内或用全限定名）
void mlir::gemmc::GemmCDialect::initialize() {
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

// 5. hasVerifier=1 的 Op 必须手写 verify()
::mlir::LogicalResult mlir::gemmc::GemmOp::verify() {
    return ::mlir::success();
}
```

**关键顺序：**
Type/Attr 的 `_CLASSES` 必须在 Op 的 `_CLASSES` 之前，因为 Op 的 arguments/results 中引用了 Type/Attr 类。

---

## 头文件分层职责

不要把所有 `.h.inc` 都堆到一个 `.h` 里，保持职责单一：

### `Gemm.h` —— Dialect 声明

```cpp
#pragma once
#include "mlir/IR/Dialect.h"
#include "Dialect/Gemm.h.inc"        // Dialect 类声明
```

### `GemmTypes.h` —— Type / Attr 声明

```cpp
#pragma once
#include "mlir/IR/Types.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/GemmTypes.h.inc"
#undef GET_TYPEDEF_CLASSES

#define GET_ATTRDEF_CLASSES
#include "Dialect/GemmAttrs.h.inc"
#undef GET_ATTRDEF_CLASSES
```

### `GemmOps.h` —— Op 声明（依赖 Types）

```cpp
#pragma once
#include "Dialect/Gemm.h"
#include "Dialect/GemmTypes.h"          // Op 引用了 Type/Attr
#include "mlir/IR/OpDefinition.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Dialect/GemmOps.h.inc"
#undef GET_OP_CLASSES
```

**一句话总结：哪个 `.h.inc` 就包在对应模块的 `.h` 里，不要乱包。**

---

## Attr / Type 类名的大小写陷阱

```tablegen
def TransposeAttr : GemmC_Attr<"trans", "trans"> { ... }
//                              ^^^^^^ 第一个参数直接拼接类名，不会自动首字母大写
```

生成的 C++ 类名是 `transAttr`（小写 t）而不是 `TransAttr`。

**想要大写就写大写：**

```tablegen
def TransposeAttr : GemmC_Attr<"Trans", "trans"> { ... }
//                              ^^^^^  ^^^^^
//                          类名前缀   IR文本助记符
// → 生成 TransAttr，.mlir 中写作 #gemmc.trans<...>
```

---

## 方言转换（Dialect Conversion）

把自定义方言（如 gemmc）下降到其他方言（如 linalg），需要四个核心部件。

### 1. 转换 Pattern（核心逻辑）

每个要转换的 Op 写一个 `OpConversionPattern`，实现 `matchAndRewrite`：

```cpp
struct GemmOpLowering : public OpConversionPattern<gemmc::GemmOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      gemmc::GemmOp op,
      OpAdaptor adaptor,                    // 已转换的操作数
      ConversionPatternRewriter &rewriter) const override {
    Value A = adaptor.getA();
    Value B = adaptor.getB();
    Value C = adaptor.getC();
    auto matmul = rewriter.create<linalg::MatmulOp>(
        op.getLoc(), TypeRange{op.getType()},
        ValueRange{A, B}, ValueRange{C});
    rewriter.replaceOp(op, matmul.getResults());
    return success();
  }
};
```

### 2. ConversionTarget（定义合法性）

```cpp
ConversionTarget target(getContext());
target.addLegalDialect<linalg::LinalgDialect, arith::ArithDialect,
                       tensor::TensorDialect, func::FuncDialect>();
target.addIllegalDialect<gemmc::GemmCDialect>();   // gemmc 必须被转换掉
```

### 3. Pass（驱动转换）

```cpp
void runOnOperation() override {
  ConversionTarget target(getContext());
  target.addLegalDialect<linalg::LinalgDialect, arith::ArithDialect,
                         tensor::TensorDialect, func::FuncDialect>();
  target.addIllegalDialect<gemmc::GemmCDialect>();

  RewritePatternSet patterns(&getContext());
  patterns.add<GemmOpLowering, AddOpLowering>(&getContext());

  if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
```

### Full vs Partial Conversion

| API | 语义 | 场景 |
|-----|------|------|
| `applyFullConversion` | 源方言必须全部消除 | 学习阶段推荐，能立刻发现漏转的 op |
| `applyPartialConversion` | 允许部分残留 | 渐进式 lowering |

### GEMM 语义映射决策点

- `gemmc.gemm` = `D = alpha*(A@B) + beta*C`
- `linalg.matmul` 算 `A@B`；`alpha≠1` / `beta≠0` 需额外 `linalg.generic` / `arith` 做缩放
- `transA`/`transB` 为真时先插入 `linalg.transpose`

---

## 用 TableGen 定义 Pass

### Pass 定义 (`pass.td`)

```tablegen
def GemmcToLinalg : Pass<"gemmc-to-linalg", "ModuleOp"> {
//                       ^^^^^^^^^^^^^^^ 命令行选项名      ^^^^^^^^ 作用对象
    let summary = "Lower Gemmc to Linalg";
    let constructor = "gemmc::createGemmcToLinalg()";
    let dependentDialects = ["linalg::LinalgDialect", "arith::ArithDialect"];
}
```

常见笔误：`FunctionOpInterface`（不是 FuncitonOpInterface）、`constructor`（不是 constrcutor）。
`InterfacePass<"name", "FunctionOpInterface">` 作用在 func 级别；整个 module 用 `Pass<"name", "ModuleOp">`。

### CMake 生成

```cmake
set(LLVM_TARGET_DEFINITIONS pass.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Gemmc)
add_public_tablegen_target(GemmcPassIncGen)
```

### Pass 声明头 `Passes.h`

```cpp
#pragma once
#include "mlir/Pass/Pass.h"
namespace mlir::gemmc {
#define GEN_PASS_DECL
#include "Pass/Passes.h.inc"
#define GEN_PASS_REGISTRATION
#include "Pass/Passes.h.inc"
}
```

### Pass 实现 `.cpp`

```cpp
#include "Pass/Passes.h"
namespace mlir::gemmc {
#define GEN_PASS_DEF_GEMMCTOLINALG
#include "Pass/Passes.h.inc"

namespace {
struct GemmcToLinalgPass
    : public impl::GemmcToLinalgBase<GemmcToLinalgPass> {   // 继承基类
  void runOnOperation() override { /* 转换逻辑 */ }        // 填空钩子
};
}

std::unique_ptr<Pass> createGemmcToLinalg() {
  return std::make_unique<GemmcToLinalgPass>();
}
}
```

### 注册 + 调用

```cpp
// gemm-opt.cpp main() 里
mlir::gemmc::registerPasses();   // TableGen 生成的批量注册
```

```bash
# 命令行用 pass 名（td 里第一个字符串）调用
./tools/gemm-opt/gemm-opt input.mlir --gemmc-to-linalg

# 串联多个 pass
./tools/gemm-opt/gemm-opt input.mlir --gemmc-to-linalg --convert-linalg-to-loops --canonicalize
```

### Pass 相关宏

| 宏 | 生成什么 | 用在哪 |
|----|---------|--------|
| `-gen-pass-decls -name Gemmc` | 所有声明 | CMake tablegen 命令 |
| `GEN_PASS_DECL` | `createXxx()` 声明 | Passes.h |
| `GEN_PASS_REGISTRATION` | `registerPasses()` | Passes.h |
| `GEN_PASS_DEF_GEMMCTOLINALG` | `GemmcToLinalgBase` 基类 | 实现的 .cpp |

**命令行选项名 = td 里第一个字符串参数，不是 def 名。**

---

## 为什么 Pass 需要继承，而 Op/Type/Attr 不需要

本质区别：**Op/Type/Attr 是“纯声明性数据”，而 Pass 包含“任意命令式逻辑”。**

### Op/Type/Attr —— TableGen 生成**完整类**

它们的全部行为都能用 TableGen 声明清楚（operands、results、assemblyFormat），TableGen 能生成可直接用的完整类。你顶多补几个方法体（verify、initialize），不需要再继承。

### Pass —— TableGen 只生成**基类骨架**

Pass 的核心是 `runOnOperation()` 里的转换算法——任意 C++ 逻辑，TableGen 无法描述。它只知道 Pass 的元数据（名字、描述、选项、依赖方言），所以生成一个基类把元数据样板代码填好，留个 `runOnOperation()` 空钩子让你继承实现。

### 对比

| | TableGen 生成什么 | 你要做什么 |
|---|---|---|
| Op/Type/Attr | **完整类** | 直接用（补个别方法体）|
| Pass | **基类骨架**（含元数据）| **继承 + 实现 `runOnOperation()`** |

**一句话：能被声明式完全描述的（Op/Type/Attr）就生成完整类；含有无法声明的命令式逻辑的（Pass）就只能生成基类留钩子让你继承。**
