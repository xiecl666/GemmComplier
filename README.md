# GemmCompiler —— 面向 GEMM 算子的个人 AI 编译器

## 项目概述

GemmCompiler 是一个专注于 GEMM（General Matrix Multiply）算子的端到端 AI 编译器，实现从 ONNX 模型 IR 到 GPU 可执行程序的完整编译流程。基于 MLIR 多层级 IR 基础设施，逐层下降（Lowering），最终生成高效的 GPU Kernel。

## 编译流水线总览

```
ONNX Model (.onnx)
       │
       ▼
┌─────────────────────┐
│  Stage 1: Frontend  │  解析 ONNX，提取 Gemm 算子
│  ONNX IR → Custom   │
│  High-Level Dialect │
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  Stage 2: High-Level│  图级别优化（常量折叠、形状推断）
│  Optimization       │
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  Stage 3: Lowering  │  Gemm Dialect → Linalg Dialect
│  to Linalg          │
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  Stage 4: Tiling &  │  分块、向量化、循环变换
│  Optimization       │
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  Stage 5: Lowering  │  Linalg → SCF/Affine → GPU Dialect
│  to GPU Dialect     │
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  Stage 6: GPU to    │  GPU Dialect → NVVM IR (LLVM NVPTX)
│  NVVM/PTX           │
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  Stage 7: CodeGen   │  LLVM IR → PTX → cubin
│  & Runtime          │  Host 端启动 Kernel
└─────────────────────┘
       │
       ▼
  GPU Executable (cubin + host launcher)
```

---

## 各阶段详细说明

### Stage 1: Frontend —— ONNX 解析与 Gemm 算子提取

**输入：** `.onnx` 模型文件  
**输出：** GemmCompiler High-Level Dialect（自定义 MLIR Dialect）

**流程：**
1. 使用 ONNX protobuf 接口（或 onnx-mlir 的 parser）加载 `.onnx` 文件
2. 遍历 ONNX 计算图，识别 `op_type == "Gemm"` 的节点
3. 提取 Gemm 算子的属性：
   - `alpha`、`beta`：缩放因子
   - `transA`、`transB`：转置标记
   - 输入 tensor 的形状信息 `[M, K] × [K, N] + [M, N]`
4. 将 ONNX Gemm 节点转换为自定义的 `gemm.matmul` op

```mlir
// 自定义 Gemm Dialect IR 示例
func.func @gemm(%A: tensor<512x768xf32>, %B: tensor<768x1024xf32>, %C: tensor<512x1024xf32>) -> tensor<512x1024xf32> {
  %result = gemm.matmul alpha(1.0) beta(1.0) transA(false) transB(false)
      ins(%A, %B : tensor<512x768xf32>, tensor<768x1024xf32>)
      outs(%C : tensor<512x1024xf32>) -> tensor<512x1024xf32>
  return %result : tensor<512x1024xf32>
}
```

**关键实现：**
- 定义 `GemmDialect`，包含 `gemm.matmul` Operation
- 使用 ODS（Operation Definition Specification）在 TableGen 中描述 Op 的 operand/result/attribute

---

### Stage 2: High-Level Optimization —— 图级优化

**输入：** Gemm High-Level Dialect IR  
**输出：** 优化后的 High-Level Dialect IR

**优化 Pass：**
1. **常量折叠（Constant Folding）**：若权重矩阵 B 为常量，预计算 `alpha * B`
2. **形状推断（Shape Inference）**：静态推导输出 tensor 形状
3. **转置消除（Transpose Elimination）**：若 `transB=true`，尝试在编译期重排权重布局，消除运行时转置
4. **算子融合（Op Fusion）**：将 `Gemm + BiasAdd + ReLU` 融合为单个 fused op（如存在后续激活算子）

---

### Stage 3: Lowering to Linalg —— 降级到 Linalg Dialect

**输入：** Gemm Dialect IR  
**输出：** Linalg Dialect IR（`linalg.matmul` + `linalg.generic`）

**转换规则：**

`gemm.matmul` → `linalg.matmul` + 可选的 `linalg.generic`（用于 alpha/beta 缩放和 bias 加法）

```mlir
// Lowering 后的 Linalg IR
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @gemm(%A: tensor<512x768xf32>, %B: tensor<768x1024xf32>, %C: tensor<512x1024xf32>) -> tensor<512x1024xf32> {
  // D = A × B
  %D = linalg.matmul ins(%A, %B : tensor<512x768xf32>, tensor<768x1024xf32>)
                      outs(%C : tensor<512x1024xf32>) -> tensor<512x1024xf32>
  // result = alpha * D + beta * C  (alpha=1.0, beta=1.0 时简化为 D + C)
  return %D : tensor<512x1024xf32>
}
```

**关键实现：**
- 编写 `GemmToLinalgLoweringPass`，使用 MLIR 的 `ConversionPattern` 框架
- 注册 `TypeConverter` 处理 tensor 类型映射

---

### Stage 4: Tiling & Optimization —— 分块与循环优化

**输入：** Linalg Dialect IR  
**输出：** 分块后的 SCF（Structured Control Flow）+ Linalg IR

**核心变换：**

1. **Tiling（分块）**：将 `[M, N, K]` 循环空间按 GPU 层次结构分块
   - Block-level tile: `[BM=128, BN=128, BK=32]`（映射到 GPU Block）
   - Thread-level tile: `[TM=8, TN=8]`（映射到 GPU Thread）
   - Warp-level tile:（可选，映射到 Warp 级别 Tensor Core）

2. **数据搬运优化**：
   - 插入 Shared Memory 的 `memref.alloc` + 显式 copy（Global → Shared）
   - 双缓冲（Double Buffering）隐藏访存延迟

3. **循环变换**：
   - 循环置换（Loop Permutation）改善数据局部性
   - 循环展开（Loop Unrolling）暴露指令级并行

```mlir
// Tiling 后的 IR（简化示意）
scf.forall (%bx, %by) in (4, 8) {  // Block-level grid
  // 分配 shared memory
  %shmA = memref.alloc() : memref<128x32xf32, #gpu.address_space<workgroup>>
  %shmB = memref.alloc() : memref<32x128xf32, #gpu.address_space<workgroup>>

  scf.for %k = 0 to 768 step 32 {  // K-loop tiling
    // Load tile from global to shared memory
    // ...
    gpu.barrier  // __syncthreads()

    scf.forall (%tx, %ty) in (16, 16) {  // Thread-level
      // 每个线程计算 8x8 子块
      // 寄存器级别的微内核计算
    }
    gpu.barrier
  }
}
```

**关键实现：**
- 使用 `transform dialect` 或自定义 pass 驱动 tiling 策略
- 利用 `linalg.tile_using_forall` / `linalg.tile_using_for` 接口

---

### Stage 5: Lowering to GPU Dialect —— 映射到 GPU 执行模型

**输入：** Tiled SCF + Linalg IR  
**输出：** GPU Dialect IR（`gpu.launch_func`, `gpu.module`）

**映射关系：**
| 计算层级 | GPU 硬件 | MLIR 表示 |
|---------|---------|----------|
| Block-level tile | Thread Block (CTA) | `gpu.block_id` |
| Thread-level tile | Thread | `gpu.thread_id` |
| K-loop iteration | Sequential | `scf.for` |
| Shared Memory | SMEM | `memref<...x...xf32, #gpu.address_space<workgroup>>` |
| Register | Register File | `memref<...x...xf32, #gpu.address_space<private>>` |

**转换步骤：**
1. `scf.forall` → `gpu.launch` with block/grid 维度
2. 插入 `gpu.module` 包装 kernel 函数
3. Host 侧生成 `gpu.launch_func` 调用
4. 内存分配：`memref.alloc` → `gpu.alloc`（device memory）

```mlir
gpu.module @gemm_kernels {
  gpu.func @gemm_kernel(%A: memref<512x768xf32>, %B: memref<768x1024xf32>, %C: memref<512x1024xf32>)
      kernel attributes {gpu.known_block_size = array<i32: 16, 16, 1>} {
    %bx = gpu.block_id x
    %by = gpu.block_id y
    %tx = gpu.thread_id x
    %ty = gpu.thread_id y
    // ... tiled matmul computation ...
    gpu.return
  }
}

func.func @main(%A: memref<512x768xf32>, %B: memref<768x1024xf32>, %C: memref<512x1024xf32>) {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  gpu.launch_func @gemm_kernels::@gemm_kernel
      blocks in (%c4, %c8, %c1)
      threads in (%c16, %c16, %c1)
      args(%A : memref<512x768xf32>, %B : memref<768x1024xf32>, %C : memref<512x1024xf32>)
  return
}
```

---

### Stage 6: GPU Dialect → NVVM/PTX —— 目标代码生成

**输入：** GPU Dialect IR  
**输出：** LLVM NVPTX IR（即 NVVM IR）

**Lowering 链：**
```
GPU Dialect
    │
    ├── gpu.func → llvm.func (with NVVM metadata)
    ├── gpu.block_id → nvvm.read.ptx.sreg.ctaid.{x,y,z}
    ├── gpu.thread_id → nvvm.read.ptx.sreg.tid.{x,y,z}
    ├── gpu.barrier → nvvm.barrier0
    ├── gpu.shuffle → nvvm.shfl.sync
    │
    ▼
NVVM Dialect (LLVM IR for NVPTX)
    │
    ▼
LLVM IR (NVPTX target)
```

**关键 Pass 序列：**
1. `convert-gpu-to-nvvm`：GPU Dialect → NVVM Dialect
2. `convert-arith-to-llvm`：算术运算 → LLVM IR
3. `convert-memref-to-llvm`：内存操作 → LLVM IR
4. `convert-scf-to-cf` → `convert-cf-to-llvm`：控制流 → LLVM IR
5. `gpu-to-llvm`：Host 侧 GPU runtime 调用 → LLVM IR

---

### Stage 7: CodeGen & Runtime —— 最终代码生成与执行

**输入：** LLVM NVPTX IR  
**输出：** PTX 汇编 → cubin 二进制 + Host 启动程序

**步骤：**

1. **Device 侧编译：**
   ```
   LLVM NVPTX IR → (llc -mcpu=sm_80) → PTX Assembly
   PTX Assembly → (ptxas / CUDA Driver API) → cubin (SASS)
   ```

2. **Host 侧编译：**
   - Host LLVM IR 包含 CUDA Runtime API 调用：
     - `cuModuleLoadData` — 加载 cubin
     - `cuModuleGetFunction` — 获取 kernel 函数指针
     - `cuLaunchKernel` — 启动 kernel
     - `cuMemAlloc` / `cuMemcpyHtoD` / `cuMemcpyDtoH` — 设备内存管理

3. **链接与打包：**
   ```
   Host LLVM IR → (llc) → Host .o
   Host .o + libcuda.so → (linker) → Final Executable
   cubin 内嵌于可执行文件或运行时加载
   ```

**执行流程：**
```
Host Program 启动
    │
    ├── 1. 分配 GPU 内存 (cuMemAlloc)
    ├── 2. 拷贝输入矩阵 A, B 到 GPU (cuMemcpyHtoD)
    ├── 3. 加载 cubin，获取 kernel 指针
    ├── 4. 配置 grid/block 维度，启动 kernel (cuLaunchKernel)
    ├── 5. 等待 kernel 完成 (cuStreamSynchronize)
    ├── 6. 拷贝结果矩阵 C 回 Host (cuMemcpyDtoH)
    └── 7. 释放资源
```

---

## 项目结构（规划）

```
GemmCompiler/
├── README.md                    # 本文档
├── CMakeLists.txt               # 项目构建配置
├── deps/
│   └── mlir/                    # MLIR/LLVM 预编译依赖
├── include/
│   └── GemmCompiler/
│       ├── Dialect/
│       │   └── Gemm/
│       │       ├── GemmOps.td       # TableGen Op 定义
│       │       ├── GemmOps.h        # 生成的 Op 声明
│       │       └── GemmDialect.h    # Dialect 声明
│       ├── Conversion/
│       │   ├── GemmToLinalg.h       # Gemm → Linalg 转换 Pass
│       │   └── Passes.h            # 所有 Pass 注册
│       └── Transforms/
│           └── TilingStrategy.h     # 分块策略接口
├── lib/
│   ├── Dialect/
│   │   └── Gemm/
│   │       ├── GemmOps.cpp          # Op 实现
│   │       └── GemmDialect.cpp      # Dialect 注册
│   ├── Conversion/
│   │   ├── GemmToLinalg.cpp         # Gemm → Linalg Lowering
│   │   ├── LinalgToGPU.cpp          # Tiling + 映射到 GPU
│   │   └── GPUToNVVM.cpp            # GPU → NVVM Pipeline
│   └── Transforms/
│       └── GemmTiling.cpp           # GEMM 分块策略实现
├── tools/
│   └── gemm-compiler/
│       └── main.cpp                 # 编译器 driver（命令行入口）
├── test/
│   ├── models/
│   │   └── simple_gemm.onnx        # 测试用 ONNX 模型
│   └── lit/
│       ├── frontend.mlir            # Frontend 输出验证
│       ├── lowering.mlir            # Lowering 正确性测试
│       └── gpu-codegen.mlir         # GPU 代码生成测试
└── runtime/
    └── gpu_runtime.cpp              # 轻量 GPU Runtime 封装
```

---

## MLIR 依赖构建

```bash
cd llvm-project
git checkout llvmorg-20.0.1
mkdir build && cd build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="/home/cangluo.xcl/ssd2/xcl/GemmComplier/deps/mlir" \
  -DLLVM_INSTALL_UTILS=ON
ninja -j$(nproc)
ninja install
```

---

## 核心 Lowering Pass 示例

### gemm-compiler 命令行调用（目标形态）

```bash
# 完整编译流水线
gemm-compiler input.onnx \
  --tile-sizes=128,128,32 \
  --target=nvptx64 \
  --gpu-arch=sm_80 \
  -o gemm_kernel

# 分步调试：仅输出某阶段 IR
gemm-compiler input.onnx --emit=gemm-dialect    # 查看 Gemm Dialect IR
gemm-compiler input.onnx --emit=linalg          # 查看 Linalg IR
gemm-compiler input.onnx --emit=gpu             # 查看 GPU Dialect IR
gemm-compiler input.onnx --emit=nvvm            # 查看 NVVM IR
gemm-compiler input.onnx --emit=ptx             # 查看 PTX 汇编
```

### 使用 mlir-opt 手动 Lowering（开发调试）

```bash
# Gemm Dialect → Linalg
mlir-opt input.mlir --convert-gemm-to-linalg

# Linalg → Tiled + GPU mapping
mlir-opt input.mlir \
  --linalg-tile-using-forall="tile-sizes=128,128,32" \
  --convert-forall-to-gpu

# GPU → NVVM
mlir-opt input.mlir \
  --convert-gpu-to-nvvm \
  --convert-arith-to-llvm \
  --convert-memref-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts

# NVVM → PTX
mlir-translate --mlir-to-llvmir output.mlir | \
  llc -mcpu=sm_80 -o kernel.ptx
```

---

## 参考资料

- [MLIR Official Documentation](https://mlir.llvm.org/)
- [MLIR Linalg Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)
- [MLIR GPU Dialect](https://mlir.llvm.org/docs/Dialects/GPU/)
- [ONNX Specification - Gemm Op](https://onnx.ai/onnx/operators/onnx__Gemm.html)
- [CUTLASS: CUDA Templates for Linear Algebra](https://github.com/NVIDIA/cutlass)（Tiling 策略参考）
