import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

def lp_normalization(x, axis=1, p=2):
    """使用ONNX ReferenceEvaluator实现LpNormalization"""
    # 创建一个ONNX模型
    node = onnx.helper.make_node(
        'LpNormalization',
        inputs=['x'],
        outputs=['y'],
        axis=axis,
        p=p
    )
    
    # 创建输入和输出信息
    x_shape = x.shape
    y_shape = x.shape  # 输出形状与输入相同
    
    # 创建图形
    graph_def = onnx.helper.make_graph(
        [node],
        'lp_normalization_test',
        [onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, x_shape)],
        [onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, y_shape)]
    )
    
    # 创建模型
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-test')
    model_def.opset_import[0].version = 12  # 使用ONNX opset版本12
    
    # 使用ReferenceEvaluator运行模型
    ref = ReferenceEvaluator(model_def)
    result = ref.run(None, {'x': x})
    
    return result[0]  # 返回第一个输出，即归一化后的结果

def print_array_info(arr, name="输出"):
    """打印数组信息，包括形状和数据"""
    print(f"{name}形状: {arr.shape}")
    if arr.ndim <= 2:
        print(f"{name}数据:")
        print(arr)
    else:
        print(f"{name}数据 [前几个元素]:")
        flat_arr = arr.flatten()
        print(flat_arr[:min(10, len(flat_arr))])
    print()

def test_2d_lp_normalization():
    """测试2D输入的LP归一化"""
    print("\n===== 2D LpNormalization测试 =====")
    
    # 测试用例1: 基本的2D输入 (axis=1, p=2)
    print("测试用例1: 基本2D输入 (axis=1, p=2)")
    data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ], dtype=np.float32)
    
    print_array_info(data, "输入")
    
    # L2范数测试 (p=2)
    y_l2 = lp_normalization(data, axis=1, p=2)
    print_array_info(y_l2, "L2范数输出")
    
    # 手动计算验证
    norm1_l2 = np.sqrt(1**2 + 2**2 + 3**2)
    norm2_l2 = np.sqrt(4**2 + 5**2 + 6**2)
    expected_l2 = np.array([
        [1/norm1_l2, 2/norm1_l2, 3/norm1_l2],
        [4/norm2_l2, 5/norm2_l2, 6/norm2_l2]
    ])
    print("验证L2归一化 (第一行):", np.allclose(y_l2[0], expected_l2[0]))
    print("验证L2归一化 (第二行):", np.allclose(y_l2[1], expected_l2[1]))
    
    # 测试用例2: 基本的2D输入 (axis=1, p=1)
    print("\n测试用例2: 基本2D输入 (axis=1, p=1)")
    # L1范数测试 (p=1)
    y_l1 = lp_normalization(data, axis=1, p=1)
    print_array_info(y_l1, "L1范数输出")
    
    # 手动计算验证
    norm1_l1 = 1 + 2 + 3
    norm2_l1 = 4 + 5 + 6
    expected_l1 = np.array([
        [1/norm1_l1, 2/norm1_l1, 3/norm1_l1],
        [4/norm2_l1, 5/norm2_l1, 6/norm2_l1]
    ])
    print("验证L1归一化 (第一行):", np.allclose(y_l1[0], expected_l1[0]))
    print("验证L1归一化 (第二行):", np.allclose(y_l1[1], expected_l1[1]))
    
    # 测试用例3: 2D输入沿axis=0归一化
    print("\n测试用例3: 2D输入 (axis=0, p=2)")
    y_axis0 = lp_normalization(data, axis=0, p=2)
    print_array_info(y_axis0, "沿axis=0的L2范数输出")
    
    # 手动计算验证 (沿axis=0)
    col1_norm = np.sqrt(1**2 + 4**2)
    col2_norm = np.sqrt(2**2 + 5**2)
    col3_norm = np.sqrt(3**2 + 6**2)
    expected_axis0 = np.array([
        [1/col1_norm, 2/col2_norm, 3/col3_norm],
        [4/col1_norm, 5/col2_norm, 6/col3_norm]
    ])
    print("验证axis=0归一化:", np.allclose(y_axis0, expected_axis0))
    
    return y_l2, y_l1, y_axis0

def test_3d_lp_normalization():
    """测试3D输入的LP归一化"""
    print("\n===== 3D LpNormalization测试 =====")
    
    # 3D输入测试: 2x3x2
    data = np.array([
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ],
        [
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0]
        ]
    ], dtype=np.float32)
    
    print_array_info(data, "输入")
    
    # 测试用例1: 沿axis=1归一化 (p=2)
    print("测试用例1: 3D输入 (axis=1, p=2)")
    y_l2_axis1 = lp_normalization(data, axis=1, p=2)
    print_array_info(y_l2_axis1, "沿axis=1的L2范数输出")
    
    # 验证几个特定点
    # 批次0，第0个空间位置
    norm_b0_s0 = np.sqrt(1**2 + 3**2 + 5**2)
    print(f"批次0，位置0的L2范数: {norm_b0_s0}")
    print(f"归一化值 (1/{norm_b0_s0}): {1/norm_b0_s0}")
    print(f"Python计算值: {y_l2_axis1[0,0,0]}")
    print(f"一致性检查: {np.isclose(y_l2_axis1[0,0,0], 1/norm_b0_s0)}")
    
    # 测试用例2: 沿axis=0归一化 (p=2)
    print("\n测试用例2: 3D输入 (axis=0, p=2)")
    y_l2_axis0 = lp_normalization(data, axis=0, p=2)
    print_array_info(y_l2_axis0, "沿axis=0的L2范数输出")
    
    # 测试用例3: 沿axis=2归一化 (p=2)
    print("\n测试用例3: 3D输入 (axis=2, p=2)")
    y_l2_axis2 = lp_normalization(data, axis=2, p=2)
    print_array_info(y_l2_axis2, "沿axis=2的L2范数输出")
    
    # 测试用例4: 沿axis=1归一化 (p=1)
    print("\n测试用例4: 3D输入 (axis=1, p=1)")
    y_l1 = lp_normalization(data, axis=1, p=1)
    print_array_info(y_l1, "沿axis=1的L1范数输出")
    
    return y_l2_axis1, y_l2_axis0, y_l2_axis2, y_l1

def test_4d_lp_normalization():
    """测试4D输入的LP归一化"""
    print("\n===== 4D LpNormalization测试 =====")
    
    # 创建一个简单的4D输入
    data = np.array([
        # batch 0
        [
            # channel 0
            [
                [1.0, 2.0],
                [3.0, 4.0]
            ],
            # channel 1
            [
                [5.0, 6.0],
                [7.0, 8.0]
            ]
        ],
        # batch 1
        [
            # channel 0
            [
                [9.0, 10.0],
                [11.0, 12.0]
            ],
            # channel 1
            [
                [13.0, 14.0],
                [15.0, 16.0]
            ]
        ]
    ], dtype=np.float32)
    
    print_array_info(data, "输入")
    
    # 测试用例1: 沿axis=1归一化 (channel维度, p=2)
    print("测试用例1: 4D输入 (axis=1, p=2)")
    y_l2_axis1 = lp_normalization(data, axis=1, p=2)
    print_array_info(y_l2_axis1, "沿axis=1的L2范数输出")
    
    # 测试用例2: 沿axis=2归一化 (height维度, p=2)
    print("\n测试用例2: 4D输入 (axis=2, p=2)")
    y_l2_axis2 = lp_normalization(data, axis=2, p=2)
    print_array_info(y_l2_axis2, "沿axis=2的L2范数输出")
    
    # 测试用例3: 沿axis=3归一化 (width维度, p=2)
    print("\n测试用例3: 4D输入 (axis=3, p=2)")
    y_l2_axis3 = lp_normalization(data, axis=3, p=2)
    print_array_info(y_l2_axis3, "沿axis=3的L2范数输出")
    
    # 验证特定点
    # 计算批次0,通道0,行0上的L2范数: [1, 2]
    norm_b0_c0_h0 = np.sqrt(1**2 + 2**2)
    expected_val = 1 / norm_b0_c0_h0
    print(f"批次0,通道0,行0,列0的L2范数: {norm_b0_c0_h0}")
    print(f"期望值: {expected_val}")
    print(f"实际值: {y_l2_axis3[0,0,0,0]}")
    print(f"一致性检查: {np.isclose(y_l2_axis3[0,0,0,0], expected_val)}")
    
    return y_l2_axis1, y_l2_axis2, y_l2_axis3

def test_5d_lp_normalization():
    """测试5D输入的LP归一化，使用1-48的连续数字"""
    print("\n===== 5D LpNormalization测试 (1-48数据) =====")
    # 创建5D张量 (2,3,2,2,2)，使用1-48填充
    data = np.arange(1, 49, dtype=np.float32).reshape(2, 3, 2, 2, 2)
    print_array_info(data, "输入")
    
    # 测试用例1: 沿axis=1归一化 (p=2)
    print("测试用例1: 5D输入 (axis=1, p=2)")
    y_l2 = lp_normalization(data, axis=1, p=2)
    print_array_info(y_l2, "L2范数输出")
    
    # 测试用例2: 沿axis=2归一化 (p=2)
    print("\n测试用例2: 5D输入 (axis=2, p=2)")
    y_l2_axis2 = lp_normalization(data, axis=2, p=2)
    print_array_info(y_l2_axis2, "沿axis=2的L2范数输出")
    
    # 测试用例3: 沿axis=1归一化 (p=1)
    print("\n测试用例3: 5D输入 (axis=1, p=1)")
    y_l1 = lp_normalization(data, axis=1, p=1)
    print_array_info(y_l1, "L1范数输出")
    
    # 验证第一个切片 [1, 9, 17]
    print("\n验证第一个切片 [1, 9, 17]:")
    l2_norm_first = np.sqrt(1**2 + 9**2 + 17**2)
    l1_norm_first = 1 + 9 + 17
    
    print(f"L2范数: {l2_norm_first}")
    print(f"L1范数: {l1_norm_first}")
    
    print(f"L2归一化值 [1/{l2_norm_first}, 9/{l2_norm_first}, 17/{l2_norm_first}]:")
    print(f"  计算值: [{1/l2_norm_first}, {9/l2_norm_first}, {17/l2_norm_first}]")
    print(f"  实际值: [{y_l2[0,0,0,0,0]}, {y_l2[0,1,0,0,0]}, {y_l2[0,2,0,0,0]}]")
    
    print(f"L1归一化值 [1/{l1_norm_first}, 9/{l1_norm_first}, 17/{l1_norm_first}]:")
    print(f"  计算值: [{1/l1_norm_first}, {9/l1_norm_first}, {17/l1_norm_first}]")
    print(f"  实际值: [{y_l1[0,0,0,0,0]}, {y_l1[0,1,0,0,0]}, {y_l1[0,2,0,0,0]}]")
    
    return y_l2, y_l2_axis2, y_l1


if __name__ == "__main__":
    print("===== LpNormalization 测试 =====")
    # 2D测试
    test_2d_lp_normalization()
    # 3D测试
    test_3d_lp_normalization()
    
    # 5D测试 (1-48数据)
    test_5d_lp_normalization()
