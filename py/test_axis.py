import numpy as np
from onnx.reference.ops.op_scatter_elements import scatter_elements

def test_scatter_elements():
    # 准备数据：2x5 的矩阵
    data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                     [6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32)
    # 每行有两个更新位置
    indices = np.array([[1, 3],
                       [0, 4]], dtype=np.int64)
    # 对应的更新值
    updates = np.array([[1.1, 2.1],
                       [5.5, 8.8]], dtype=np.float32)
    axis = 1
    # e[0][1] = 1.1, e[0][3] = 4.0
    # e[1][0] = 6.0, e[1][4] = 10.0

    # 使用 max reduction 计算结果
    result = scatter_elements(data, indices, updates, axis, reduction="max")
    
    # 期望值：
    # 第一行：在位置1用1.1（比2.0小），在位置3用2.1（比4.0小）
    # 第二行：在位置0用5.5（比6.0小），在位置4用8.8（比10.0小）
    expected = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                        [6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32)

    print("原始数据:\n", data)
    print("索引位置:\n", indices)
    print("更新值:\n", updates)
    print("\n使用 max reduction 的实际结果:\n", result)
    print("期望的结果:\n", expected)

    # 这个断言会失败，因为 max reduction 的结果与直接替换的结果不同
    assert np.array_equal(result, expected), f"测试失败！\n期望:\n{expected}\n实际:\n{result}"



def test_scatter_elements_3d():
    # 准备数据：2x2x2 的矩阵
    data = np.array([
        [
          [1.0, 2.0], 
          [3.0, 4.0], 
        ],
        [
            [5.0, 6.0]
            [7.0, 8.0], 
        ]
    ], dtype=np.float32)
    indices = np.array([
        [
            [0, 3], 
            [1, 0]
        ],
        [
            [1, 0], 
                      {{1, 0}, {0, 1}}}, dtype=np.int64)
    # 对应的更新值
    updates = np.array([
        [
          [0.1, 0.2], 
          [0.3, 0.4], 
          [0.5, 0.6]
        ],
        [
          [0.7, 0.8], 
          [0.9, 1.0], 
          [1.1, 1.2]
        ]
      ], dtype=np.float32)
    axis = 1
    # e[0][1] = 1.1, e[0][3] = 4.0
    # e[1][0] = 6.0, e[1][4] = 10.0


    result = scatter_elements(data, indices, updates, axis, reduction="add")
    
    expected = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                        [6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32)

    print("原始数据:\n", data)
    print("索引位置:\n", indices)
    print("更新值:\n", updates)
    print("\n使用 add reduction 的实际结果:\n", result)
    print("期望的结果:\n", expected)

    # 这个断言会失败，因为 max reduction 的结果与直接替换的结果不同
    assert np.array_equal(result, expected), f"测试失败！\n期望:\n{expected}\n实际:\n{result}"

test_scatter_elements_3d()