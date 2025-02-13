import onnx;
import numpy as np;
from onnx.reference.ops.op_scatter_elements import scatter_elements



def test_duplicate_indices():
    # 3维数据
    data = np.array([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]
    ], dtype=np.float32)
    
    # 重复的索引，指向多个位置
    indices = np.array([
        [[1, 0, 1], [0, 1, 1], [1, 1, 0]],
        [[0, 1, 0], [1, 1, 0], [0, 0, 1]]
    ], dtype=np.int64)
    
    # data[0][0][1] += 0.1, data[0][0][0] += 0.2, data[0][0][1] += 0.3
    #        2.1                  1.2                  2.4
    # 更新值
    updates = np.array([
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        [[1.0, 1.1, 1.2], [1.3, 1.4, 1.5], [1.6, 1.7, 1.8]]
    ], dtype=np.float32)
    
    # 在最内层维度上进行scatter
    axis = 2
    y = scatter_elements(data, indices, updates, axis, reduction="add")
    print("Complex scatter result:")
    print(y)


test_duplicate_indices()