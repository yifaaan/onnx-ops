import numpy as np
from onnx.reference.ops.op_gather_elements import gather_numpy

data = np.array([
    [1.0, 2.0,
     3.0, 4.0,
     5.0, 6.0],
    
    [7.0, 8.0,
     9.0, 10.0,
     11.0, 12.0] 
]).reshape(2, 3, 2).astype(np.float32)

indices = np.array([
    [1, 0, 2, 0, 2, 1],
    
    [2, 1, 0, 1, 0, 2]
]).reshape(2, 3, 2).astype(np.int64)
axis = 1

# print(gather_numpy(data, axis, indices).shape)




data = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0, 16.0],
    [17.0, 18.0, 19.0, 20.0],
    [21.0, 22.0, 23.0, 24.0]
]).reshape(2, 3, 2, 2).astype(np.float32)

indices = np.array([
    1, 0, 2,  
    0, 2, 1,   
    2, 1, 0,   
    1, 0, 2,
    1, 0, 2,
    2
]).reshape(2, 2, 2, 2).astype(np.int64)
axis = 1

print(gather_numpy(data, axis, indices))