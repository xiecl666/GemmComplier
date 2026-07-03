import onnx
from onnx import helper, TensorProto
import numpy as np
import onnxruntime as ort

def create_single_gemm_model():
    # 1. 定义矩阵维度 (M=2, K=3, N=4)
    # 公式: Y = alpha * A * B + beta * C
    M, K, N = 2, 3, 4

    # 2. 定义输入和输出的张量信息 (Value Info)
    # 输入 A: [M, K]
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [M, K])
    # 输入 B: [K, N]
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [K, N])
    # 输入 C (偏置): [N] (ONNX Gemm 支持 1D 偏置自动广播)
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [N])
    
    # 输出 Y: [M, N]
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [M, N])

    # 3. 创建 GEMM 节点
    # transA=0, transB=0 表示不转置 A 和 B
    # alpha=1.0, beta=1.0 是缩放系数
    gemm_node = helper.make_node(
        op_type='Gemm',
        inputs=['A', 'B', 'C'],
        outputs=['Y'],
        name='gemm_node_0',
        transA=0,
        transB=0,
        alpha=1.0,
        beta=1.0
    )

    # 4. 创建计算图 (Graph)
    graph = helper.make_graph(
        nodes=[gemm_node],
        name='single_gemm_graph',
        inputs=[A, B, C],
        outputs=[Y]
    )

    # 5. 创建模型 (Model)
    # opset_version=13 是比较稳定且常用的版本
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7

    # 6. 检查模型合法性并保存
    onnx.checker.check_model(model)
    onnx.save(model, 'single_gemm.onnx')
    print("✅ 成功生成模型: single_gemm.onnx")

    return model

def verify_model():
    """使用 ONNX Runtime 验证模型是否正确"""
    print("\n--- 开始验证模型 ---")
    M, K, N = 2, 3, 4
    
    # 生成随机输入数据
    np.random.seed(42)
    A_data = np.random.randn(M, K).astype(np.float32)
    B_data = np.random.randn(K, N).astype(np.float32)
    C_data = np.random.randn(N).astype(np.float32)

    # 使用 ONNX Runtime 推理
    session = ort.InferenceSession("single_gemm.onnx")
    ort_inputs = {'A': A_data, 'B': B_data, 'C': C_data}
    ort_outputs = session.run(None, ort_inputs)
    y_ort = ort_outputs[0]

    # 使用 NumPy 计算标准答案
    y_numpy = np.dot(A_data, B_data) + C_data

    # 对比结果
    if np.allclose(y_ort, y_numpy, rtol=1e-5, atol=1e-5):
        print("✅ 验证通过！ONNX Runtime 结果与 NumPy 结果一致。")
    else:
        print("❌ 验证失败！结果不一致。")
        print("ORT 结果:", y_ort)
        print("NumPy 结果:", y_numpy)

if __name__ == '__main__':
    create_single_gemm_model()
    verify_model()