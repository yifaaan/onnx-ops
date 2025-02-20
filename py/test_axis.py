import numpy as np
from onnx.reference.ops.op_scatter_elements import scatter_elements


def test_scatter_elements_3d():

    data = np.array([
        # batch 0
        [  
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        ],
        # batch 1
        [  
            [[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]],
            [[19.0, 20.0], [21.0, 22.0], [23.0, 24.0]]
        ]
    ]).astype(np.float32)  # shape: (2, 2, 3, 2)

    indices = np.array([
        # batch 0
        [ 
            [[1, 2], [0, 1], [2, 0]],
            [[0, 1], [2, 0], [1, 2]]
        ],
        # batch 1
        [ 
            [[2, 1], [0, 2], [1, 0]],
            [[2, 1], [0, 2], [1, 0]]
        ]
    ]).astype(np.int64)  # shape: (2, 2, 3, 2)

    updates = np.array([
        # batch 0
        [  
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]
        ],
        # batch 1
        [  
            [[1.3, 1.4], [1.5, 1.6], [1.7, 1.8]],
            [[1.9, 2.0], [2.1, 2.2], [2.3, 2.4]]
        ]
    ]).astype(np.float32)  # shape: (2, 2, 3, 2)
    axis = 2
    # e[0][1] = 1.1, e[0][3] = 4.0
    # e[1][0] = 6.0, e[1][4] = 10.0

    result = scatter_elements(data, indices, updates, axis, reduction="add")
    

    print("原始数据:\n", data)
    print("索引位置:\n", indices)
    print("更新值:\n", updates)
    print("\n使用 add reduction 的实际结果:\n", result)


test_scatter_elements_3d()