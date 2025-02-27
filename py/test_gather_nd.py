import numpy as np
import onnxruntime as ort
import onnx
from onnx import helper, TensorProto, numpy_helper

def test_gather_nd(data, indices, batch_dims=0, print_result=True):
    """
    使用ONNX Runtime测试GatherND操作
    
    参数:
        data: numpy数组，输入数据
        indices: numpy数组，索引数据
        batch_dims: 批处理维度数量
        print_result: 是否打印结果
    
    返回:
        输出结果
    """
    # 创建ONNX模型
    node = helper.make_node(
        'GatherND',
        inputs=['data', 'indices'],
        outputs=['output'],
        batch_dims=batch_dims
    )
    
    # 创建输入
    data_tensor = numpy_helper.from_array(data)
    indices_tensor = numpy_helper.from_array(indices)
    
    # 创建输出
    output_tensor = helper.make_tensor_value_info(
        'output',
        data_tensor.data_type,
        None  # 动态形状
    )
    
    # 创建图和模型
    graph = helper.make_graph(
        [node],
        'gather_nd_test',
        [
            helper.make_tensor_value_info('data', data_tensor.data_type, data.shape),
            helper.make_tensor_value_info('indices', indices_tensor.data_type, indices.shape)
        ],
        [output_tensor],
    )
    
    # 使用opset 13（GatherND在opset 13中已经支持batch_dims）
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    
    # 运行模型
    session = ort.InferenceSession(model.SerializeToString())
    output = session.run(
        None,
        {
            'data': data,
            'indices': indices
        }
    )[0]
    
    if print_result:
        print(f"Data shape: {data.shape}")
        print(f"Indices shape: {indices.shape}")
        print(f"Batch dims: {batch_dims}")
        print(f"Output shape: {output.shape}")
        print(f"Output: {output}")
    
    return output

# 测试用例1: 2D数据，2D索引，last_dim=1
def test_case_1():
    print("\n测试用例1: 2D数据，2D索引，last_dim=1")
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype(np.float32)
    indices = np.array([[0], [1]]).astype(np.int64)
    return test_gather_nd(data, indices)

# 测试用例2: 2D数据，2D索引，last_dim=2
def test_case_2():
    print("\n测试用例2: 2D数据，2D索引，last_dim=2")
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype(np.float32)
    indices = np.array([[0, 0], [1, 1]]).astype(np.int64)
    return test_gather_nd(data, indices)

# 测试用例3: 3D数据，2D索引，last_dim=2
def test_case_3():
    print("\n测试用例3: 3D数据，2D索引，last_dim=2")
    data = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).astype(np.float32)
    indices = np.array([[0, 0], [1, 1]]).astype(np.int64)
    return test_gather_nd(data, indices)

# 测试用例4: 3D数据，2D索引，last_dim=3
def test_case_4():
    print("\n测试用例4: 3D数据，2D索引，last_dim=3")
    data = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).astype(np.float32)
    indices = np.array([[0, 1, 0]]).astype(np.int64)
    return test_gather_nd(data, indices)

# 测试用例5: 3D数据，3D索引，last_dim=2
def test_case_5():
    print("\n测试用例5: 3D数据，3D索引，last_dim=2")
    data = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).astype(np.float32)
    indices = np.array([[[0, 0]], [[1, 1]]]).astype(np.int64)
    return test_gather_nd(data, indices)

# 测试用例6: 带有负索引
def test_case_6():
    print("\n测试用例6: 带有负索引")
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype(np.float32)
    indices = np.array([[-3], [-2]]).astype(np.int64)
    return test_gather_nd(data, indices)

# 测试用例7: 带有batch_dims=1
def test_case_7():
    print("\n测试用例7: 带有batch_dims=1")
    data = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 
                     [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]).astype(np.float32)
    indices = np.array([[[0]], [[1]]]).astype(np.int64)
    return test_gather_nd(data, indices, batch_dims=1)

if __name__ == "__main__":
    test_case_1()
    test_case_2()
    test_case_3()
    test_case_4()
    test_case_5()
    test_case_6()
    test_case_7() 