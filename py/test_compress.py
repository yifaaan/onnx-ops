import numpy as np
import onnx.reference.ops.op_compress

# 原始数据
data = np.array([
    # batch 0
    [  # 第一个2D矩阵
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    ],
    # batch 1
    [  # 第二个2D矩阵
        [[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]],
        [[19.0, 20.0], [21.0, 22.0], [23.0, 24.0]]
    ]
]).astype(np.float32)  # shape: (2, 2, 3, 2)

# 索引数据
indices = np.array([
    # batch 0
    [  # 第一个2D矩阵
        [[1, 2], [0, 1], [2, 0]],
        [[0, 1], [2, 0], [1, 2]]
    ],
    # batch 1
    [  # 第二个2D矩阵
        [[2, 1], [0, 2], [1, 0]],
        [[2, 1], [0, 2], [1, 0]]
    ]
]).astype(np.int64)  # shape: (2, 2, 3, 2)

# 更新数据
updates = np.array([
    # batch 0
    [  # 第一个2D矩阵
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]
    ],
    # batch 1
    [  # 第二个2D矩阵
        [[1.3, 1.4], [1.5, 1.6], [1.7, 1.8]],
        [[1.9, 2.0], [2.1, 2.2], [2.3, 2.4]]
    ]
]).astype(np.float32)  # shape: (2, 2, 3, 2)

print(np.compress(indices, data, axis=2))

