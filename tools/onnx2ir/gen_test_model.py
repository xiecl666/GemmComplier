#!/usr/bin/env python3
"""
生成一个包含 Gemm 算子的测试 ONNX 模型

用法:
    python gen_test_model.py [-o output.onnx] [--M 512] [--K 768] [--N 1024] [--bias]
"""

import argparse
import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
except ImportError:
    import sys
    print("错误: 需要安装 onnx 包。运行: pip install onnx", file=sys.stderr)
    sys.exit(1)


def create_gemm_model(M=512, K=768, N=1024, has_bias=True, alpha=1.0, beta=1.0,
                       transA=0, transB=0):
    """创建一个简单的 Gemm ONNX 模型"""

    np.random.seed(42)  # 固定种子，保证可复现

    # 权重 B
    if transB:
        weight_shape = (N, K)
    else:
        weight_shape = (K, N)
    weight_B = np.random.randn(*weight_shape).astype(np.float32) * 0.01

    # 输入 A 的形状
    if transA:
        input_shape_a = (K, M)
    else:
        input_shape_a = (M, K)

    # 构建 inputs
    input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, list(input_shape_a))
    output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])

    # initializers
    initializers = [
        numpy_helper.from_array(weight_B, name="B")
    ]

    # Gemm inputs
    gemm_inputs = ["A", "B"]
    attrs = {
        "alpha": alpha,
        "beta": beta,
        "transA": transA,
        "transB": transB,
    }

    if has_bias:
        bias_C = np.random.randn(N).astype(np.float32) * 0.01
        initializers.append(numpy_helper.from_array(bias_C, name="C"))
        gemm_inputs.append("C")

    # 创建 Gemm 节点
    gemm_node = helper.make_node(
        "Gemm",
        inputs=gemm_inputs,
        outputs=["Y"],
        name="gemm_0",
        **attrs
    )

    # 构建 graph
    graph = helper.make_graph(
        [gemm_node],
        "gemm_test_graph",
        [input_a],
        [output],
        initializer=initializers,
    )

    # 构建 model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7

    onnx.checker.check_model(model)
    return model


def main():
    parser = argparse.ArgumentParser(description="生成包含 Gemm 算子的测试 ONNX 模型")
    parser.add_argument("-o", "--output", default="test_gemm.onnx", help="输出文件路径")
    parser.add_argument("--M", type=int, default=512, help="矩阵 M 维度")
    parser.add_argument("--K", type=int, default=768, help="矩阵 K 维度")
    parser.add_argument("--N", type=int, default=1024, help="矩阵 N 维度")
    parser.add_argument("--no-bias", action="store_true", help="不添加 bias")
    parser.add_argument("--alpha", type=float, default=1.0, help="alpha 缩放因子")
    parser.add_argument("--beta", type=float, default=1.0, help="beta 缩放因子")
    parser.add_argument("--transA", type=int, default=0, help="转置 A (0 或 1)")
    parser.add_argument("--transB", type=int, default=0, help="转置 B (0 或 1)")

    args = parser.parse_args()

    model = create_gemm_model(
        M=args.M, K=args.K, N=args.N,
        has_bias=not args.no_bias,
        alpha=args.alpha, beta=args.beta,
        transA=args.transA, transB=args.transB,
    )

    onnx.save(model, args.output)
    print(f"模型已保存: {args.output}")
    print(f"  Gemm: A[{args.M}x{args.K}] @ B[{args.K}x{args.N}]"
          f"{' + bias' if not args.no_bias else ''}")
    print(f"  alpha={args.alpha}, beta={args.beta}, transA={args.transA}, transB={args.transB}")


if __name__ == "__main__":
    main()
