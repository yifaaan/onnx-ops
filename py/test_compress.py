import numpy as np
import onnx.reference.ops.op_compress

a = np.array([[1, 2], [3, 4], [5, 6]])
print(a)

# Select rows where the condition is True
print(np.compress([False, True, True], a, axis=0))
# Select columns where the condition is True
print(np.compress([False, True], a, axis=1))
# Apply compress to a flattened array
print(np.compress([False, True], a))


