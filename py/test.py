import numpy as np

# 创建一个4维数组：2批次，3个样本，4个特征组，每组5个特征
array_4d = np.arange(120).reshape(2, 3, 4, 5)

print("原始数组形状:", array_4d.shape)
print("第一个批次的第一个样本:\n", array_4d[0, 0])

# 沿着轴0求和 (合并批次)
sum_axis0 = np.sum(array_4d, axis=0)
print("沿轴0求和后形状:", sum_axis0.shape)  # (3, 4, 5)
print("第一个样本的结果:\n", sum_axis0[0])

# 沿着轴1求和 (合并样本)
sum_axis1 = np.sum(array_4d, axis=1)
print("沿轴1求和后形状:", sum_axis1.shape)  # (2, 4, 5)
print("第一个批次的结果:\n", sum_axis1[0])

# 沿着轴2求和 (合并特征组)
sum_axis2 = np.sum(array_4d, axis=2)
print("沿轴2求和后形状:", sum_axis2.shape)  # (2, 3, 5)
print("第一个批次的第一个样本结果:", sum_axis2[0, 0])

# 沿着轴3求和 (合并特征)
sum_axis3 = np.sum(array_4d, axis=3)
print("沿轴3求和后形状:", sum_axis3.shape)  # (2, 3, 4)
print("第一个批次的第一个样本结果:", sum_axis3[0, 0])



# 沿着轴(0,1)求和
sum_axis01 = np.sum(array_4d, axis=(0, 1))
print("沿轴(0,1)求和后形状:", sum_axis01.shape)  # (4, 5)
print("结果:\n", sum_axis01)

# 沿着轴(1,2)求和
sum_axis12 = np.sum(array_4d, axis=(1, 2))
print("沿轴(1,2)求和后形状:", sum_axis12.shape)  # (2, 5)
print("结果:\n", sum_axis12)

# 沿着轴(0,2,3)求和
sum_axis023 = np.sum(array_4d, axis=(0, 2, 3))
print("沿轴(0,2,3)求和后形状:", sum_axis023.shape)  # (3,)
print("结果:", sum_axis023)



print("FDSFSDFSDFSDFSDFSDF")


# # 创建一个3D数组
# array_3d = np.array([
#     # 第一个矩阵(第0个元素)
#     [
#         [1, 2, 3, 4],    # 第0行
#         [5, 6, 7, 8],    # 第1行
#         [9, 10, 11, 12]  # 第2行
#     ],
#     # 第二个矩阵(第1个元素)
#     [
#         [13, 14, 15, 16],  # 第0行
#         [17, 18, 19, 20],  # 第1行
#         [21, 22, 23, 24]   # 第2行
#     ]
# ])

# print("原始数组形状:", array_3d.shape)  # (2, 3, 4)

# sum_axis0 = np.sum(array_3d, axis=1)
# print(sum_axis0)