import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import time
import json
import os

def generate_sequence_at_test(test_name="test", seed=None):
    """
    生成SequenceAt测试用例
    
    参数:
    test_name: 测试用例名称
    seed: 随机种子，如果为None则使用时间戳
    """
    if seed is None:
        seed = int(time.time())
    np.random.seed(seed)
    
    # 生成随机数据
    seq_length = np.random.randint(50, 120)  # 序列长度在3-9之间
    tensor_dims = np.random.randint(2, 5)  # 张量维度在1-3之间
    tensor_shape = [np.random.randint(40, 80) for _ in range(tensor_dims)]  # 每个维度的大小在2-4之间
    
    # 生成序列中的张量
    sequence = []
    for _ in range(seq_length):
        tensor_data = np.random.uniform(-10, 10, tensor_shape).tolist()
        sequence.append({
            'shape': tensor_shape,
            'data': tensor_data
        })
    
    # 生成position参数
    # 有50%的概率生成负索引
    if np.random.random() < 0.5:
        position = np.random.randint(-seq_length, 0)
    else:
        position = np.random.randint(0, seq_length)
    
    # 创建测试数据
    test_data = {
        "name": test_name,
        "seed": seed,
        "input": {
            "sequence": sequence,
            "position": int(position)
        },
        "params": {
            "sequence_length": seq_length,
            "tensor_dims": tensor_dims,
            "tensor_shape": tensor_shape
        }
    }
    
    return test_data

def generate_multiple_tests(num_tests=3):
    """
    生成多个测试用例
    """
    # 创建测试数据目录
    test_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "py", "sequence_at_test")
    os.makedirs(test_dir, exist_ok=True)
    
    all_test_data = []
    
    for i in range(num_tests):
        test_name = f"SequenceAt_Test_{i+1}"
        test_data = generate_sequence_at_test(test_name)
        all_test_data.append(test_data)
        
        # 保存测试数据到JSON文件
        file_path = os.path.join(test_dir, f"sequence_at_test_{test_name}.json")
        with open(file_path, "w") as f:
            json.dump(test_data, f, indent=2)
        print(f"Generated test file: {file_path}")
    
    # 生成示例测试
    test_name = "Example_Test"
    test_data = generate_sequence_at_test(test_name)
    file_path = os.path.join(test_dir, f"sequence_at_test_{test_name}.json")
    with open(file_path, "w") as f:
        json.dump(test_data, f, indent=2)
    print(f"Generated example test file: {file_path}")

def run_sequence_at_test(test_data):
    """
    使用ONNX Runtime运行SequenceAt测试
    
    参数:
    test_data: 测试数据字典
    """
    # 创建序列输入
    sequence = test_data["input"]["sequence"]
    position = test_data["input"]["position"]
    
    # 创建ONNX模型
    sequence_type = helper.make_sequence_type_proto(
        helper.make_tensor_type_proto(TensorProto.FLOAT, sequence[0]["shape"])
    )
    
    # 创建节点
    node = helper.make_node(
        'SequenceAt',
        inputs=['input_sequence', 'position'],
        outputs=['output']
    )
    
    # 创建输入
    input_sequence = helper.make_value_info('input_sequence', sequence_type, None)
    position_tensor = helper.make_tensor_value_info(
        'position',
        TensorProto.INT32,
        None
    )
    
    # 创建输出
    output_tensor = helper.make_tensor_value_info(
        'output',
        TensorProto.FLOAT,
        sequence[0]["shape"]
    )
    
    # 创建图和模型
    graph = helper.make_graph(
        [node],
        'sequence_at_test',
        [input_sequence, position_tensor],
        [output_tensor]
    )
    
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    
    # 运行模型
    session = ort.InferenceSession(model.SerializeToString())
    
    # 准备输入数据
    input_sequence_data = [np.array(tensor["data"], dtype=np.float32) for tensor in sequence]
    position_data = np.array(position, dtype=np.int32)
    
    # 运行推理
    output = session.run(
        None,
        {
            'input_sequence': input_sequence_data,
            'position': position_data
        }
    )[0]
    
    return output

# 运行测试生成
if __name__ == "__main__":
    generate_multiple_tests(3)
    
    # 运行一个示例测试
    test_data = generate_sequence_at_test("Example_Test")
    print("\n运行示例测试:")
    print(f"序列长度: {test_data['params']['sequence_length']}")
    print(f"张量维度: {test_data['params']['tensor_dims']}")
    print(f"张量形状: {test_data['params']['tensor_shape']}")
    print(f"位置索引: {test_data['input']['position']}")
    
    output = run_sequence_at_test(test_data)
    print(f"\n输出结果:")
    print(output)
