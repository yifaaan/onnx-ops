import numpy as np
import json
import os
import time

def generate_tensor():
    """生成随机张量数据"""
    tensor_dims = np.random.randint(2, 5)  # 张量维度在2-4之间
    tensor_shape = [np.random.randint(2, 10) for _ in range(tensor_dims)]
    tensor_data = np.random.uniform(-10, 10, tensor_shape).tolist()
    return {
        'shape': tensor_shape,
        'data': tensor_data
    }

def generate_sequence_construct_test(test_name="test", seed=None):
    """生成SequenceConstruct测试用例"""
    if seed is None:
        seed = int(time.time())
    np.random.seed(seed)
    
    # 生成2-5个张量
    num_tensors = np.random.randint(2, 6)
    tensors = [generate_tensor() for _ in range(num_tensors)]
    
    test_data = {
        "name": test_name,
        "seed": seed,
        "input": {
            "tensors": tensors
        },
        "params": {
            "num_tensors": num_tensors
        }
    }
    
    return test_data

def generate_sequence_insert_test(test_name="test", seed=None):
    """生成SequenceInsert测试用例"""
    if seed is None:
        seed = int(time.time())
    np.random.seed(seed)
    
    # 生成原始序列
    seq_length = np.random.randint(3, 8)
    sequence = [generate_tensor() for _ in range(seq_length)]
    
    # 生成要插入的张量
    tensor_to_insert = generate_tensor()
    
    # 生成插入位置（包括负索引和末尾插入的情况）
    # 有20%的概率不指定位置（末尾插入）
    if np.random.random() < 0.2:
        position = None
    else:
        # 有50%的概率使用负索引
        if np.random.random() < 0.5:
            position = np.random.randint(-seq_length, 0)
        else:
            position = np.random.randint(0, seq_length + 1)
    
    test_data = {
        "name": test_name,
        "seed": seed,
        "input": {
            "sequence": sequence,
            "tensor": tensor_to_insert,
            "position": position
        },
        "params": {
            "sequence_length": seq_length
        }
    }
    
    return test_data

def generate_sequence_erase_test(test_name="test", seed=None):
    """生成SequenceErase测试用例"""
    if seed is None:
        seed = int(time.time())
    np.random.seed(seed)
    
    # 生成原始序列
    seq_length = np.random.randint(3, 8)
    sequence = [generate_tensor() for _ in range(seq_length)]
    
    # 生成删除位置（包括负索引和末尾删除的情况）
    # 有20%的概率不指定位置（删除末尾元素）
    if np.random.random() < 0.2:
        position = None
    else:
        # 有50%的概率使用负索引
        if np.random.random() < 0.5:
            position = np.random.randint(-seq_length, 0)
        else:
            position = np.random.randint(0, seq_length)
    
    test_data = {
        "name": test_name,
        "seed": seed,
        "input": {
            "sequence": sequence,
            "position": position
        },
        "params": {
            "sequence_length": seq_length
        }
    }
    
    return test_data

def generate_test_files(num_tests=3):
    """为每个操作生成测试文件"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 定义测试生成函数和它们的目录
    test_configs = [
        {
            "generator": generate_sequence_construct_test,
            "dir": "sequence_construct_test",
            "prefix": "sequence_construct_test_SequenceConstruct_Test_"
        },
        {
            "generator": generate_sequence_insert_test,
            "dir": "sequence_insert_test",
            "prefix": "sequence_insert_test_SequenceInsert_Test_"
        },
        {
            "generator": generate_sequence_erase_test,
            "dir": "sequence_erase_test",
            "prefix": "sequence_erase_test_SequenceErase_Test_"
        }
    ]
    
    for config in test_configs:
        # 创建测试目录
        test_dir = os.path.join(base_dir, "py", config["dir"])
        os.makedirs(test_dir, exist_ok=True)
        
        # 生成测试文件
        for i in range(num_tests):
            test_name = f"{config['prefix']}{i+1}"
            test_data = config["generator"](test_name)
            
            file_path = os.path.join(test_dir, f"{test_name}.json")
            with open(file_path, "w") as f:
                json.dump(test_data, f, indent=2)
            print(f"Generated test file: {file_path}")

if __name__ == "__main__":
    generate_test_files(3) 