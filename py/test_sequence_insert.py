import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import time
import json
import os


"""
生成SequenceInsert测试用例

参数:
test_name: 测试用例名称
seed: 随机种子，如果为None则使用时间戳
seq_length: 序列长度
tensor_dims: 张量维度
tensor_shape: 张量形状
position: 位置索引
"""
def generate_sequence_insert_test(test_name="test", seed=None, seq_length=20, tensor_dims=3, tensor_shape=[3, 20, 50], insert_tensor_shape=[3, 20, 50], position=-1):
    if seed is None:
        seed = int(time.time())
    np.random.seed(seed)
    
    # 生成插入的张量
    insert_tensor_data = np.random.uniform(-10, 10, insert_tensor_shape).tolist()
    # 生成序列中的张量
    sequence = []
    for _ in range(seq_length):
        tensor_data = np.random.uniform(-10, 10, tensor_shape).tolist()
        sequence.append({
            'shape': tensor_shape,
            'data': tensor_data
        })
    
    
    # 创建测试数据
    test_data = {
        "name": test_name,
        "seed": seed,
        "input": {
            "sequence": sequence,
            "position": int(position),
            "insert_tensor": {
                "shape": insert_tensor_shape,
                "data": insert_tensor_data
            }
        },
        "params": {
            "sequence_length": seq_length,
            "tensor_dims": tensor_dims,
            "tensor_shape": tensor_shape
        },
        "output": {
            "len": []
        }
    }
    
    return test_data

def generate_multiple_tests(num_tests=3):
    """
    生成多个测试用例
    """
    # 创建测试数据目录
    test_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "jsons", "sequence_insert")
    os.makedirs(test_dir, exist_ok=True)
    
    files = []
    all_test_data = []
    for i in range(num_tests):
        test_name = f"test_{i+1}"
        test_data = generate_sequence_insert_test(test_name, seq_length=3, tensor_dims=1, tensor_shape=[1], insert_tensor_shape=[1])
        # test_data = generate_sequence_insert_test(test_name)
        all_test_data.append(test_data)
        # 保存测试数据到JSON文件
        file_path = os.path.join(test_dir, f"{test_name}.json")
        with open(file_path, "w") as f:
            json.dump(test_data, f, indent=2)
        print(f"Generated test file: {file_path}")
        files.append(file_path)
    return files, all_test_data
    

def run_sequence_insert_test(test_data):
    """
    使用ONNX Runtime运行SequenceAt测试
    
    参数:
    test_data: 测试数据字典
    """
    # 创建序列输入
    sequence = test_data["input"]["sequence"]
    tensor_to_insert = test_data["input"]["insert_tensor"]
    
    # 确定是否有位置参数
    has_position = "position" in test_data["input"]
    position = test_data["input"].get("position", None)
    
    # 创建ONNX模型
    sequence_type = helper.make_sequence_type_proto(
        helper.make_tensor_type_proto(TensorProto.FLOAT, sequence[0]["shape"])
    )
    
    # 创建节点
    inputs = ['input_sequence', 'tensor_to_insert']
    if has_position:
        inputs.append('position')
    
    node = helper.make_node(
        'SequenceInsert',
        inputs=inputs,
        outputs=['output']
    )
    
    # 创建输入
    input_sequence = helper.make_value_info('input_sequence', sequence_type, None)
    tensor_to_insert_info = helper.make_tensor_value_info(
        'tensor_to_insert',
        TensorProto.FLOAT,
        tensor_to_insert["shape"]
    )
    
    input_value_infos = [input_sequence, tensor_to_insert_info]
    
    if has_position:
        position_tensor = helper.make_tensor_value_info(
            'position',
            TensorProto.INT32,
            None
        )
        input_value_infos.append(position_tensor)
    
    # 创建输出
    output_sequence = helper.make_value_info('output', sequence_type, None)
    
    # 创建图和模型
    graph = helper.make_graph(
        [node],
        'sequence_insert_test',
        input_value_infos,
        [output_sequence]
    )
    
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    
    # 运行模型
    session = ort.InferenceSession(model.SerializeToString())
    
    # 准备输入数据
    input_sequence_data = [np.array(tensor["data"], dtype=np.float32) for tensor in sequence]
    tensor_to_insert_data = np.array(tensor_to_insert["data"], dtype=np.float32)
    
    feed_dict = {
        'input_sequence': input_sequence_data,
        'tensor_to_insert': tensor_to_insert_data
    }
    
    if has_position:
        feed_dict['position'] = np.array(position, dtype=np.int32)
    
    # 运行推理
    output = session.run(None, feed_dict)[0]
    
    # 创建深拷贝避免意外修改
    import copy
    output_copy = copy.deepcopy(output)
    
    return output_copy

# 运行测试生成
if __name__ == "__main__":
    files, all_test_data = generate_multiple_tests(3)
    
    # 运行一个示例测试
    # test_data = generate_sequence_at_test("Example_Test")
    # print("\n运行示例测试:")
    # print(f"序列长度: {test_data['params']['sequence_length']}")
    # print(f"张量维度: {test_data['params']['tensor_dims']}")
    # print(f"张量形状: {test_data['params']['tensor_shape']}")
    # print(f"位置索引: {test_data['input']['position']}")
    for file, test_data in zip(files, all_test_data):
        out = run_sequence_insert_test(test_data)
        o = []
        for tensor in out:
            o.append({"data": tensor.tolist(), "shape": tensor.shape})
        s = len(out)
        # 读取现有的JSON文件
        with open(file, "r") as f:
            existing_data = json.load(f)
        existing_data["output"]["sequence"] = o
        existing_data["output"]["len"] = s
        # 保存更新后的JSON文件
        with open(file, "w") as f:
            json.dump(existing_data, f, indent=2)
