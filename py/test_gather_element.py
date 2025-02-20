import numpy as np
from onnx.reference.ops.op_gather_elements import gather_numpy

# 数据张量 (2,3,2)
data = np.array([
    # 第一层
    [1.0, 2.0,    # 第一行
     3.0, 4.0,    # 第二行
     5.0, 6.0],   # 第三行
    
    # 第二层
    [7.0, 8.0,    # 第一行
     9.0, 10.0,   # 第二行
     11.0, 12.0]  # 第三行
]).reshape(2, 3, 2).astype(np.float32)

# 索引张量 (2,3,2)
indices = np.array([
    # 第一层
    [1, 0, 2,     # 三行
     0, 2, 1],
    
    # 第二层
    [2, 1, 0,     # 三行
     1, 0, 2]
]).reshape(2, 3, 2).astype(np.int64)
axis = 1
output = [
    [1, 1],
    [4, 3],
]

print(gather_numpy(data, axis, indices))
