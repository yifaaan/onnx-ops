import onnx
import numpy as np



def global_max_pool(x: np.ndarray) -> np.ndarray:
    spatial_shape = np.ndim(x) - 2
    y = x.max(axis=tuple(range(spatial_shape, spatial_shape + 2)))
    for _ in range(spatial_shape):
        y = np.expand_dims(y, -1)
    return y  # type: ignore

def test_basic_global_max_pool():
    # 创建一个 GlobalMaxPool 节点
    node = onnx.helper.make_node(
        "GlobalMaxPool",
        inputs=["x"],
        outputs=["y"],
    )
    # 生成随机输入数据：1x3x5x5
    x = np.random.randn(1, 3, 5, 5).astype(np.float32)
    # 计算全局最大池化
    y = global_max_pool(x)
    print("基本测试结果:")
    print(y)

def test_simple_3x3():
    # 测试简单的 3x3 输入
    node = onnx.helper.make_node(
        "GlobalMaxPool",
        inputs=["x"],
        outputs=["y"],
    )
    # 输入数据：1x1x3x3 的张量
    x = np.array([
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        ]
    ]).astype(np.float32)
    # 输出结果：1x1x1x1 的张量，值为 9（最大值）
    y = global_max_pool(x)
    print("\n3x3 输入测试结果:")
    print("输入形状:", x.shape)
    print("输入数据:\n", x)
    print("输出形状:", y.shape)
    print("输出数据:\n", y)


def test_simple_2x3():
    # 测试简单的 2x3 输入
    node = onnx.helper.make_node(
        "GlobalMaxPool",
        inputs=["x"],
        outputs=["y"],
    )
    # 输入数据：2x3 的张量
    x = np.array([

                [1, 2, 3],
                [4, 5, 6],

    ]).astype(np.float32)
    # 输出结果：1x1x1x1 的张量，值为 9（最大值）
    y = np.array(6).astype(np.float32)
    print("\n2x3 输入测试结果:")
    print("输入形状:", x.shape)
    print("输入数据:\n", x)
    print("输出形状:", y.shape)
    print("输出数据:\n", y)
    print(global_max_pool(x))
def test_multichannel():
    # 测试多通道输入
    node = onnx.helper.make_node(
        "GlobalMaxPool",
        inputs=["x"],
        outputs=["y"],
    )
    # 输入数据：1x2x3x3 的张量（2个通道）
    x = np.array([
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            [
                [9, 8, 7],
                [6, 5, 4],
                [3, 2, 1],
            ]
        ]
    ]).astype(np.float32)
    # 计算全局最大池化
    y = global_max_pool(x)
    print("\n多通道测试结果:")
    print("输入形状:", x.shape)
    print("输入数据:\n", x)
    print("输出形状:", y.shape)
    print("输出数据:\n", y)

def test_5d_tensor():
    # 测试5维张量输入
    node = onnx.helper.make_node(
        "GlobalMaxPool",
        inputs=["x"],
        outputs=["y"],
    )
    # 输入数据：1x2x3x2x2 的张量
    # 1: batch size
    # 2: channels
    # 3x2x2: 3D空间维度
    x = np.array([  # N=1
        [  # C=2 (两个通道)
            [  # D=3 (第一个通道的3个平面)
                [[1, 2],  # H=2, W=2
                 [3, 4]],
                [[5, 6],
                 [7, 8]],
                [[9, 10],
                 [11, 12]]
            ],
            [  # 第二个通道的3个平面
                [[13, 14],
                 [15, 16]],
                [[17, 18],
                 [19, 20]],
                [[21, 22],
                 [23, 24]]
            ]
        ]
    ]).astype(np.float32)

    # 计算全局最大池化
    # 在最后三个维度(D,H,W)上进行最大值计算
    y = global_max_pool(x)

    print("\n5维张量测试结果:")
    print("输入形状:", x.shape)
    print("输入数据第一个通道:\n", x[0,0])
    print("输入数据第二个通道:\n", x[0,1])
    print("输出形状:", y.shape)
    print("输出数据:\n", y)

def test_batch_processing():
    # 测试批处理
    node = onnx.helper.make_node(
        "GlobalMaxPool",
        inputs=["x"],
        outputs=["y"],
    )
    # 输入数据：2x1x2x2 的张量（批量大小为2）
    x = np.array([
        [[[1, 2],
          [3, 4]]],
        [[[5, 6],
          [7, 8]]]
    ]).astype(np.float32)
    # 计算全局最大池化
    y = global_max_pool(x)
    print("\n批处理测试结果:")
    print("输入形状:", x.shape)
    print("输入数据:\n", x)
    print("输出形状:", y.shape)
    print("输出数据:\n", y)

if __name__ == "__main__":
    print("===== GlobalMaxPool 测试用例 =====")
    test_basic_global_max_pool()
    test_simple_3x3()
    test_5d_tensor()
    test_multichannel()
    test_batch_processing()
    test_simple_2x3()
    print(global_max_pool(np.array([[1], [2]])))