#!/usr/bin/env python3
"""
onnx2ir: ONNX Gemm 算子 → 自定义 Gemm 方言 MLIR 文本

用法:
    python onnx2ir.py input.onnx -o output.mlir

输出格式 (紧凑风格):
    %0 = gemm.matmul(%A, %B) {alpha=1.0, beta=1.0, transA=false, transB=false, bias=dense<...>}
         : (tensor<MxKxf32>, tensor<KxNxf32>) -> tensor<MxNxf32>
"""

import argparse
import sys
import numpy as np

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    print("错误: 需要安装 onnx 包。运行: pip install onnx", file=sys.stderr)
    sys.exit(1)


def get_tensor_shape(graph, name):
    """从 graph inputs/outputs/value_info 中获取 tensor 形状"""
    # 先从 input 中找
    for inp in graph.input:
        if inp.name == name:
            dims = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    dims.append(dim.dim_value)
                else:
                    dims.append(-1)  # 动态维度
            return dims
    # 从 value_info 中找
    for vi in graph.value_info:
        if vi.name == name:
            dims = []
            for dim in vi.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    dims.append(dim.dim_value)
                else:
                    dims.append(-1)
            return dims
    return None


def get_initializer(graph, name):
    """从 graph initializer 中获取 numpy 数组"""
    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    return None


def format_dense_attr(arr):
    """将 numpy 数组格式化为 MLIR dense attribute 字符串"""
    if arr.ndim == 1:
        values = ", ".join(f"{v:.6e}" for v in arr.flat)
        return f"dense<[{values}]>"
    elif arr.ndim == 2:
        rows = []
        for i in range(arr.shape[0]):
            row_vals = ", ".join(f"{v:.6e}" for v in arr[i])
            rows.append(f"[{row_vals}]")
        inner = ", ".join(rows)
        return f"dense<[{inner}]>"
    else:
        # 高维 fallback: flatten
        values = ", ".join(f"{v:.6e}" for v in arr.flat)
        return f"dense<[{values}]>"


def format_tensor_type(shape, elem_type="f32"):
    """格式化 tensor 类型字符串, 如 tensor<512x768xf32>"""
    dims = "x".join(str(d) if d > 0 else "?" for d in shape)
    return f"tensor<{dims}x{elem_type}>"


def get_elem_type_str(onnx_elem_type):
    """ONNX element type enum → MLIR type string"""
    mapping = {
        1: "f32",   # FLOAT
        2: "ui8",   # UINT8
        3: "i8",    # INT8
        5: "i16",   # INT16
        6: "i32",   # INT32
        7: "i64",   # INT64
        10: "f16",  # FLOAT16
        11: "f64",  # DOUBLE
        16: "bf16", # BFLOAT16
    }
    return mapping.get(onnx_elem_type, "f32")


def convert_gemm_node(graph, node):
    """
    将一个 ONNX Gemm 节点转换为 Gemm 方言 MLIR 文本

    返回: (func_str, func_name)
    """
    # 提取属性
    alpha = 1.0
    beta = 1.0
    transA = False
    transB = False

    for attr in node.attribute:
        if attr.name == "alpha":
            alpha = attr.f
        elif attr.name == "beta":
            beta = attr.f
        elif attr.name == "transA":
            transA = bool(attr.i)
        elif attr.name == "transB":
            transB = bool(attr.i)

    # 输入名
    input_a_name = node.input[0]
    input_b_name = node.input[1]
    has_bias = len(node.input) > 2 and node.input[2] != ""
    input_c_name = node.input[2] if has_bias else None

    # 获取权重 B 数据
    weight_b = get_initializer(graph, input_b_name)

    # 获取形状信息
    shape_a = get_tensor_shape(graph, input_a_name)
    if weight_b is not None:
        shape_b = list(weight_b.shape)
    else:
        shape_b = get_tensor_shape(graph, input_b_name)

    if shape_a is None or shape_b is None:
        print(f"警告: 无法推断 Gemm 节点 '{node.name}' 的形状信息", file=sys.stderr)
        return None, None

    # 计算 output shape, 考虑转置
    M = shape_a[1] if transA else shape_a[0]
    K_a = shape_a[0] if transA else shape_a[1]
    K_b = shape_b[1] if transB else shape_b[0]
    N = shape_b[0] if transB else shape_b[1]

    if K_a != K_b:
        print(f"警告: Gemm 节点 '{node.name}' K 维度不匹配: {K_a} vs {K_b}", file=sys.stderr)

    shape_out = [M, N]

    # 获取 bias 数据
    bias_data = None
    if has_bias:
        bias_data = get_initializer(graph, input_c_name)

    # 确定元素类型
    elem_type = "f32"

    # 生成 MLIR 文本
    type_a = format_tensor_type(shape_a, elem_type)
    type_b = format_tensor_type(shape_b, elem_type)
    type_out = format_tensor_type(shape_out, elem_type)

    # 函数名
    func_name = node.name if node.name else f"gemm_{node.output[0]}"
    func_name = func_name.replace("/", "_").replace(".", "_").replace("-", "_")

    lines = []
    lines.append(f'func.func @{func_name}(%arg0: {type_a}) -> {type_out} {{')

    # 权重 B 作为 arith.constant
    if weight_b is not None:
        dense_b = format_dense_attr(weight_b)
        lines.append(f'  %B = arith.constant {dense_b} : {type_b}')
    else:
        # 没有 initializer 时，B 也作为函数参数（退化情况）
        # 重新生成函数签名
        lines = []
        lines.append(f'func.func @{func_name}(%arg0: {type_a}, %arg1: {type_b}) -> {type_out} {{')

    # 构建 attributes 字符串
    attrs = []
    attrs.append(f"alpha = {alpha:.1e}" if alpha != 1.0 else "alpha = 1.0")
    attrs.append(f"beta = {beta:.1e}" if beta != 1.0 else "beta = 0.0" if beta == 0.0 else f"beta = {beta}")
    attrs.append(f"transA = {'true' if transA else 'false'}")
    attrs.append(f"transB = {'true' if transB else 'false'}")

    # bias 作为可选 attribute
    if has_bias and bias_data is not None:
        bias_type = format_tensor_type(list(bias_data.shape), elem_type)
        dense_c = format_dense_attr(bias_data)
        attrs.append(f"bias = {dense_c} : {bias_type}")

    attrs_str = ", ".join(attrs)

    # 生成 gemm.matmul op
    if weight_b is not None:
        # B 是常量
        operand_str = "%arg0, %B"
        type_sig = f"({type_a}, {type_b}) -> {type_out}"
    else:
        # B 是函数参数
        operand_str = "%arg0, %arg1"
        type_sig = f"({type_a}, {type_b}) -> {type_out}"

    lines.append(f'  %result = gemm.matmul({operand_str}) {{{attrs_str}}}')
    lines.append(f'       : {type_sig}')
    lines.append(f'  return %result : {type_out}')
    lines.append(f'}}')

    return "\n".join(lines), func_name


def convert_onnx_to_gemm_ir(model_path):
    """
    加载 ONNX 模型，提取所有 Gemm 节点，生成 Gemm 方言 MLIR 文本
    """
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    graph = model.graph

    # 收集所有 Gemm 节点
    gemm_nodes = [node for node in graph.node if node.op_type == "Gemm"]

    if not gemm_nodes:
        print("警告: 模型中未找到 Gemm 算子", file=sys.stderr)
        return ""

    print(f"找到 {len(gemm_nodes)} 个 Gemm 算子", file=sys.stderr)

    # 生成 module header
    mlir_parts = []
    mlir_parts.append('// Auto-generated by onnx2ir')
    mlir_parts.append(f'// Source: {model_path}')
    mlir_parts.append(f'// Gemm nodes: {len(gemm_nodes)}')
    mlir_parts.append('')
    mlir_parts.append('module {')

    for i, node in enumerate(gemm_nodes):
        func_str, func_name = convert_gemm_node(graph, node)
        if func_str:
            # indent inside module
            indented = "\n".join("  " + line if line else "" for line in func_str.split("\n"))
            mlir_parts.append(indented)
            mlir_parts.append('')
            print(f"  [{i+1}/{len(gemm_nodes)}] 转换: {func_name}", file=sys.stderr)

    mlir_parts.append('}')

    return "\n".join(mlir_parts)


def main():
    parser = argparse.ArgumentParser(
        description="onnx2ir: 将 ONNX 模型中的 Gemm 算子转换为自定义 Gemm 方言 MLIR 文本"
    )
    parser.add_argument("input", help="输入 ONNX 模型路径 (.onnx)")
    parser.add_argument("-o", "--output", default=None,
                        help="输出 MLIR 文件路径 (默认输出到 stdout)")

    args = parser.parse_args()

    # 转换
    mlir_text = convert_onnx_to_gemm_ir(args.input)

    if not mlir_text:
        print("错误: 转换失败", file=sys.stderr)
        sys.exit(1)

    # 输出
    if args.output:
        with open(args.output, "w") as f:
            f.write(mlir_text)
            f.write("\n")
        print(f"输出已写入: {args.output}", file=sys.stderr)
    else:
        print(mlir_text)


if __name__ == "__main__":
    main()
