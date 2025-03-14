import numpy as np
import onnx
import onnxruntime as ort
import json
import os
from onnx import helper, TensorProto
import random

# 添加辅助函数，确保所有NumPy对象都转换为JSON可序列化的Python原生类型
def make_json_serializable(obj):
    """递归地将NumPy类型转换为Python原生类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.number):
        return obj.item()  # 将NumPy数值类型转换为Python原生数值类型
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif hasattr(obj, 'tolist'):  # 处理其他可能的NumPy对象
        return obj.tolist()
    else:
        return obj

# 验证生成的JSON文件是否有效
def verify_json_file(file_path):
    try:
        with open(file_path, 'r') as f:
            json_data = json.load(f)
            file_size = os.path.getsize(file_path)
            print(f"  ✓ 验证成功: {file_path}, 文件大小: {file_size} 字节")
            return True
    except Exception as e:
        print(f"  ✗ 验证失败: {file_path}, 错误: {e}")
        return False

# 设置输出目录
output_dir = "py"




def generate_nms_test(test_name, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # 生成随机数据
    batch_size = np.random.randint(4, 5)
    num_boxes = np.random.randint(100, 101)
    num_classes = np.random.randint(1, 5)
    
    # 生成随机边界框坐标和分数
    boxes = []
    
    for class_id in range(num_classes):
        for _ in range(num_boxes):
            x1 = np.random.uniform(0, 0.5)
            y1 = np.random.uniform(0, 0.5)
            x2 = np.random.uniform(x1 + 0.1, 1.0)  # 确保x2 > x1
            y2 = np.random.uniform(y1 + 0.1, 1.0)  # 确保y2 > y1
            score = np.random.uniform(0.1, 1.0)
            
            boxes.append({
                "class_id": class_id,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "score": float(score)
            })
    
    # 设置NMS参数
    iou_threshold = float(np.random.uniform(0.1, 0.9))
    
    # 准备用于ONNX模型的输入数据 - 修改为3D张量
    total_boxes = num_boxes * num_classes  # 总框数
    boxes_per_batch = total_boxes  # 每个batch中的框数量
    
    # 重新组织boxes为正确的形状
    boxes_array = np.zeros((batch_size, boxes_per_batch, 4), dtype=np.float32)
    scores_array = np.zeros((batch_size, num_classes, boxes_per_batch), dtype=np.float32)
    
    # 填充数据
    class_box_map = {}  # 用于跟踪每个类别框的索引
    for i, box in enumerate(boxes):
        class_id = box["class_id"]
        if class_id not in class_box_map:
            class_box_map[class_id] = 0
            
        # 对于每个batch，使用相同的box数据
        for batch_idx in range(batch_size):
            box_idx = i
            boxes_array[batch_idx, box_idx] = [box["x1"], box["y1"], box["x2"], box["y2"]]
            
            # 只在对应类别上设置分数，其他类别为0
            scores_array[batch_idx, class_id, box_idx] = box["score"]
    
    # 创建ONNX模型
    max_output_boxes_per_class = np.array([num_boxes], dtype=np.int64)
    iou_threshold_tensor = np.array([iou_threshold], dtype=np.float32)
    score_threshold = np.array([0.0], dtype=np.float32)  # 不设置分数阈值
    
    # 创建ONNX模型
    boxes_tensor = helper.make_tensor_value_info('boxes', TensorProto.FLOAT, [batch_size, boxes_per_batch, 4])
    scores_tensor = helper.make_tensor_value_info('scores', TensorProto.FLOAT, [batch_size, num_classes, boxes_per_batch])
    max_output_boxes_tensor = helper.make_tensor_value_info('max_output_boxes_per_class', TensorProto.INT64, [1])
    iou_threshold_info = helper.make_tensor_value_info('iou_threshold', TensorProto.FLOAT, [1])
    score_threshold_info = helper.make_tensor_value_info('score_threshold', TensorProto.FLOAT, [1])
    selected_indices = helper.make_tensor_value_info('selected_indices', TensorProto.INT64, [None, 3])
    
    node = helper.make_node(
        'NonMaxSuppression',
        inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
        outputs=['selected_indices'],
    )
    
    graph = helper.make_graph(
        [node],
        'nms_test',
        [boxes_tensor, scores_tensor, max_output_boxes_tensor, iou_threshold_info, score_threshold_info],
        [selected_indices]
    )
    
    model = helper.make_model(graph, producer_name='onnx-nms')
    onnx.checker.check_model(model)
    
    # 使用ONNX Runtime运行模型
    sess = ort.InferenceSession(model.SerializeToString())
    inputs = {
        'boxes': boxes_array,
        'scores': scores_array,
        'max_output_boxes_per_class': max_output_boxes_per_class,
        'iou_threshold': iou_threshold_tensor,
        'score_threshold': score_threshold
    }
    
    onnx_output = sess.run(None, inputs)
    selected_indices_onnx = onnx_output[0]
    
    # 基于ONNX输出创建结果列表
    onnx_results = []
    for idx in selected_indices_onnx:
        batch_idx = idx[0]
        class_idx = idx[1]
        box_idx = idx[2]
        
        # 从boxes_array中提取对应的box
        box_coords = boxes_array[batch_idx, box_idx]
        score_val = scores_array[batch_idx, class_idx, box_idx]
        
        onnx_results.append({
            "class_id": int(class_idx),
            "x1": float(box_coords[0]),
            "y1": float(box_coords[1]),
            "x2": float(box_coords[2]),
            "y2": float(box_coords[3]),
            "score": float(score_val)
        })
    
    # 准备保存到JSON的数据 - 转换为C++代码可用的格式
    # 为了与你的C++代码兼容，我们只保存原始单个batch的数据
    # 但添加ONNX的结果用于比较
    test_data = {
        "input": {
            "boxes": boxes,
            "iou_threshold": iou_threshold
        },
        "output": {
            "onnx_result": onnx_results  # 添加ONNX的结果
        }
    }
    
    # 使用辅助函数确保所有数据都是JSON可序列化的
    test_data = make_json_serializable(test_data)
    
    # 确保目录存在
    os.makedirs("py/nms_test", exist_ok=True)
    
    # 保存为JSON文件
    file_path = f"py/nms_test/nms_test_{test_name}.json"
    with open(file_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    return file_path

def generate_sequence_at_test(test_name, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # 生成随机序列
    seq_length = np.random.randint(1000, 1001)
    sequence = []
    
    for i in range(seq_length):
        shape = [np.random.randint(1, 5), np.random.randint(1, 5), np.random.randint(1, 5), np.random.randint(1, 5), np.random.randint(1, 5)]
        data = np.random.rand(*shape).astype(np.float32)
        
        sequence.append({
            "shape": shape,
            "data": data.flatten().tolist()
        })
    
    # 随机选择一个位置
    position = np.random.randint(-seq_length, seq_length)
    
    # 创建ONNX序列输入
    tensors = []
    for seq_item in sequence:
        shape = seq_item["shape"]
        data = np.array(seq_item["data"], dtype=np.float32).reshape(shape)
        tensors.append(data)
    
    # 使用 onnxruntime 执行 SequenceAt 操作
    # 构建一个简单的 ONNX 模型
    # 创建序列类型的ValueInfoProto
    tensor_type_proto = onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [])  # 使用空列表而不是None
    sequence_type_proto = onnx.helper.make_sequence_type_proto(tensor_type_proto)
    
    # 创建序列输入和输出
    sequence_input = onnx.helper.make_value_info('sequence_input', sequence_type_proto)
    position_input = onnx.helper.make_tensor_value_info('position', onnx.TensorProto.INT64, [])
    tensor_output = onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [])  # 使用空列表
    
    node = onnx.helper.make_node(
        'SequenceAt',
        inputs=['sequence_input', 'position'],
        outputs=['output']
    )
    
    graph = onnx.helper.make_graph(
        [node],
        'sequence_at_test',
        [sequence_input, position_input],
        [tensor_output]
    )
    
    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model)
    
    # 创建会话并运行
    sess = ort.InferenceSession(model.SerializeToString())
    
    # 准备输入
    pos_tensor = np.array(position, dtype=np.int64)
    
    # 运行模型
    onnx_outputs = sess.run(['output'], {
        'sequence_input': tensors,
        'position': pos_tensor
    })
    
    # 获取输出并转换为JSON格式
    output_tensor = onnx_outputs[0]
    
    # 确保转换为Python内置类型
    onnx_result = {
        "shape": output_tensor.shape.tolist() if hasattr(output_tensor.shape, 'tolist') else list(output_tensor.shape),
        "data": output_tensor.flatten().tolist()
    }
    
    # 准备保存到JSON的数据
    test_data = {
        "input": {
            "sequence": sequence,
            "position": int(position) if isinstance(position, np.number) else position
        },
        "output": {
            "onnx_result": onnx_result
        }
    }
    
    # 使用辅助函数确保所有数据都是JSON可序列化的
    test_data = make_json_serializable(test_data)
    
    # 确保目录存在
    os.makedirs("py/sequence_at_test", exist_ok=True)
    
    # 保存为JSON文件
    file_path = f"py/sequence_at_test/sequence_at_test_{test_name}.json"
    with open(file_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    return file_path

def generate_sequence_construct_test(test_name, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # 生成随机张量
    num_tensors = np.random.randint(1000, 1001)
    tensors = []
    
    for i in range(num_tensors):
        shape = [np.random.randint(1, 5), np.random.randint(1, 5), np.random.randint(1, 5), np.random.randint(1, 5), np.random.randint(1, 5)]
        data = np.random.rand(*shape).astype(np.float32)
        
        tensors.append({
            "shape": shape,
            "data": data.flatten().tolist()
        })
    
    # 构建 ONNX 输入
    tensor_inputs = []
    for tensor_data in tensors:
        shape = tensor_data["shape"]
        data = np.array(tensor_data["data"], dtype=np.float32).reshape(shape)
        tensor_inputs.append(data)
    
    # 创建 ONNX 模型
    inputs = []
    input_names = []
    for i in range(num_tensors):
        input_name = f'input_{i}'
        input_names.append(input_name)
        inputs.append(onnx.helper.make_tensor_value_info(
            input_name, onnx.TensorProto.FLOAT, []))  # 使用空列表
    
    # 创建序列类型的输出
    tensor_type_proto = onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [])  # 使用空列表
    sequence_type_proto = onnx.helper.make_sequence_type_proto(tensor_type_proto)
    sequence_output = onnx.helper.make_value_info('output_sequence', sequence_type_proto)
    
    node = onnx.helper.make_node(
        'SequenceConstruct',
        inputs=input_names,
        outputs=['output_sequence']
    )
    
    graph = onnx.helper.make_graph(
        [node],
        'sequence_construct_test',
        inputs,
        [sequence_output]
    )
    
    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model)
    
    # 创建会话并运行
    sess = ort.InferenceSession(model.SerializeToString())
    
    # 准备输入
    input_feed = {}
    for i, tensor in enumerate(tensor_inputs):
        input_feed[f'input_{i}'] = tensor
    
    # 运行模型
    onnx_outputs = sess.run(['output_sequence'], input_feed)
    
    # 获取输出并转换为JSON格式
    output_sequence = onnx_outputs[0]
    
    onnx_result = []
    for tensor in output_sequence:
        # 确保转换为Python内置类型
        onnx_result.append({
            "shape": tensor.shape.tolist() if hasattr(tensor.shape, 'tolist') else list(tensor.shape),
            "data": tensor.flatten().tolist()
        })
    
    # 准备保存到JSON的数据
    test_data = {
        "input": {
            "tensors": tensors
        },
        "output": {
            "onnx_result": onnx_result
        }
    }
    
    # 使用辅助函数确保所有数据都是JSON可序列化的
    test_data = make_json_serializable(test_data)
    
    # 确保目录存在
    os.makedirs("py/sequence_construct_test", exist_ok=True)
    
    # 保存为JSON文件
    file_path = f"py/sequence_construct_test/sequence_construct_test_{test_name}.json"
    with open(file_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    return file_path

def generate_sequence_insert_test(idx=1, num_tensors=3, tensor_size=5):
    """生成SequenceInsert算子的测试数据，使用ONNX官方实现"""
    test_name = f"sequence_insert_test_SequenceInsert_Test_{idx}"
    
    # 创建一个序列（包含多个张量）
    sequence = []
    for i in range(num_tensors):
        # 创建一个张量
        shape = [np.random.randint(1, 5), np.random.randint(1, 5), np.random.randint(1, 5), np.random.randint(1, 5), np.random.randint(1, 5)]
        data = [float(i + j * 0.1) for j in range(shape[0] * shape[1] * shape[2] * shape[3] * shape[4])]
        sequence.append({
            "shape": shape,
            "data": data
        })
    
    # 创建要插入的张量
    insert_tensor = {
        "shape": [2, tensor_size],
        "data": [float(100 + j * 0.1) for j in range(2 * tensor_size)]
    }
    
    # 随机选择一个位置插入
    position = np.random.randint(-num_tensors, num_tensors)
    
    # 创建测试数据
    test_data = {
        "input": {
            "sequence": sequence,
            "tensor": insert_tensor,
            "position": position
        },
        "output": {
            "expected_sequence_length": len(sequence) + 1
        }
    }
    
    # 使用ONNX Runtime执行SequenceInsert操作
    try:
        print(f"生成SequenceInsert测试数据 {test_name}...")
        
        # 准备输入序列
        input_sequence = []
        for tensor_data in sequence:
            shape = tensor_data["shape"]
            data = tensor_data["data"]
            tensor = np.array(data, dtype=np.float32).reshape(shape)
            input_sequence.append(tensor)
        
        # 创建要插入的张量
        insert_shape = insert_tensor["shape"]
        insert_data = insert_tensor["data"]
        tensor_to_insert = np.array(insert_data, dtype=np.float32).reshape(insert_shape)
        
        # 创建ONNX模型
        # 定义输入类型
        tensor_type_proto = onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [])
        sequence_type_proto = onnx.helper.make_sequence_type_proto(tensor_type_proto)
        
        # 创建输入
        sequence_input = onnx.helper.make_value_info('sequence_input', sequence_type_proto)
        tensor_input = onnx.helper.make_tensor_value_info('tensor_input', onnx.TensorProto.FLOAT, [])
        position_input = onnx.helper.make_tensor_value_info('position', onnx.TensorProto.INT64, [])
        
        # 创建输出
        sequence_output = onnx.helper.make_value_info('output_sequence', sequence_type_proto)
        
        # 创建SequenceInsert节点
        node = onnx.helper.make_node(
            'SequenceInsert',
            inputs=['sequence_input', 'tensor_input', 'position'],
            outputs=['output_sequence']
        )
        
        # 创建图
        graph = onnx.helper.make_graph(
            [node],
            'sequence_insert_test',
            [sequence_input, tensor_input, position_input],
            [sequence_output]
        )
        
        # 创建模型
        model = onnx.helper.make_model(graph)
        onnx.checker.check_model(model)
        
        # 创建会话并运行
        sess = ort.InferenceSession(model.SerializeToString())
        
        # 准备输入
        pos_tensor = np.array(position, dtype=np.int64)
        
        # 运行模型
        onnx_outputs = sess.run(['output_sequence'], {
            'sequence_input': input_sequence,
            'tensor_input': tensor_to_insert,
            'position': pos_tensor
        })
        
        # 获取输出序列
        output_sequence = onnx_outputs[0]
        
        # 验证结果长度
        if len(output_sequence) != len(input_sequence) + 1:
            print(f"  警告: 结果长度不符合预期，期望 {len(input_sequence) + 1}，实际 {len(output_sequence)}")
        else:
            print(f"  结果长度检查通过: {len(output_sequence)}")
        
        # 将结果转换为可序列化格式
        onnx_result = []
        for tensor in output_sequence:
            tensor_shape = tensor.shape
            tensor_data = tensor.flatten().tolist()
            onnx_result.append({
                "shape": list(tensor_shape),
                "data": tensor_data
            })
        
        # 添加ONNX结果到输出
        test_data["output"]["onnx_result"] = onnx_result
        print(f"  成功生成ONNX官方结果，包含 {len(onnx_result)} 个张量")
    except Exception as e:
        print(f"  错误: 运行ONNX SequenceInsert时出错: {e}")
        print(f"  异常详情: {type(e).__name__}")
        import traceback
        print(traceback.format_exc())
    
    # 保存到文件
    os.makedirs(f"{output_dir}/sequence_insert_test", exist_ok=True)
    file_path = f"{output_dir}/sequence_insert_test/{test_name}.json"
    
    try:
        with open(file_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        print(f"  ✓ 已保存测试数据到: {file_path}")
        
        # 验证生成的JSON
        if verify_json_file(file_path):
            print(f"  ✓ JSON文件验证通过")
        else:
            print(f"  ✗ JSON文件验证失败")
    except Exception as e:
        print(f"  ✗ 保存文件失败: {e}")
    
    return file_path

def generate_sequence_erase_test(test_name, seed=None):
    """生成SequenceErase算子的测试数据，使用ONNX官方实现"""
    if seed is not None:
        np.random.seed(seed)
    
    print(f"生成SequenceErase测试数据 {test_name}...")
    
    # 生成随机序列
    seq_length = np.random.randint(1000, 1001)
    sequence = []
    
    for i in range(seq_length):
        shape = [np.random.randint(1, 5), np.random.randint(1, 5), np.random.randint(1, 5), np.random.randint(1, 5), np.random.randint(1, 5)]
        data = np.random.rand(*shape).astype(np.float32)
        
        sequence.append({
            "shape": shape,
            "data": data.flatten().tolist()
        })
    
    # 随机选择一个位置或不指定位置（删除最后一个元素）
    use_position = np.random.choice([True, False])
    position = np.random.randint(-seq_length, seq_length) if use_position else None
    
    print(f"  序列长度: {seq_length}, 删除位置: {position if position is not None else '默认（末尾）'}")
    
    # 创建 ONNX 序列输入
    tensor_sequence = []
    for seq_item in sequence:
        shape = seq_item["shape"]
        data = np.array(seq_item["data"], dtype=np.float32).reshape(shape)
        tensor_sequence.append(data)
    
    try:
        # 创建 ONNX 模型
        # 创建序列类型的ValueInfoProto
        tensor_type_proto = onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [])
        sequence_type_proto = onnx.helper.make_sequence_type_proto(tensor_type_proto)
        
        if position is not None:
            # 创建带位置参数的输入
            sequence_input = onnx.helper.make_value_info('input_sequence', sequence_type_proto)
            position_input = onnx.helper.make_tensor_value_info('position', onnx.TensorProto.INT64, [])
            
            node = onnx.helper.make_node(
                'SequenceErase',
                inputs=['input_sequence', 'position'],
                outputs=['output_sequence']
            )
            
            inputs = [sequence_input, position_input]
        else:
            # 创建不带位置参数的输入
            sequence_input = onnx.helper.make_value_info('input_sequence', sequence_type_proto)
            
            node = onnx.helper.make_node(
                'SequenceErase',
                inputs=['input_sequence'],
                outputs=['output_sequence']
            )
            
            inputs = [sequence_input]
        
        # 创建序列类型的输出
        sequence_output = onnx.helper.make_value_info('output_sequence', sequence_type_proto)
        
        graph = onnx.helper.make_graph(
            [node],
            'sequence_erase_test',
            inputs,
            [sequence_output]
        )
        
        model = onnx.helper.make_model(graph)
        onnx.checker.check_model(model)
        
        # 创建会话并运行
        sess = ort.InferenceSession(model.SerializeToString())
        
        # 准备输入
        input_feed = {
            'input_sequence': tensor_sequence
        }
        
        if position is not None:
            input_feed['position'] = np.array(position, dtype=np.int64)
        
        # 运行模型
        onnx_outputs = sess.run(['output_sequence'], input_feed)
        
        # 获取输出并转换为JSON格式
        output_sequence = onnx_outputs[0]
        
        # 验证结果长度
        if len(output_sequence) != len(tensor_sequence) - 1:
            print(f"  警告: 结果长度不符合预期，期望 {len(tensor_sequence) - 1}，实际 {len(output_sequence)}")
        else:
            print(f"  结果长度检查通过: {len(output_sequence)}")
        
        onnx_result = []
        for tensor in output_sequence:
            # 确保转换为Python内置类型
            onnx_result.append({
                "shape": tensor.shape.tolist() if hasattr(tensor.shape, 'tolist') else list(tensor.shape),
                "data": tensor.flatten().tolist()
            })
        
        print(f"  成功生成ONNX官方结果，包含 {len(onnx_result)} 个张量")
    except Exception as e:
        print(f"  错误: 运行ONNX SequenceErase时出错: {e}")
        print(f"  异常详情: {type(e).__name__}")
        import traceback
        print(traceback.format_exc())
        onnx_result = []
    
    # 修正JSON结构 - 确保与其他测试保持一致
    # 准备保存到JSON的数据 - 正确匹配C++代码期望的结构
    test_data = {
        "input": {
            "sequence": sequence,
            # 始终包含position键，如果没有指定则为null
            "position": int(position) if isinstance(position, np.number) and position is not None else position
        },
        "output": {
            "onnx_result": onnx_result,
            "expected_sequence_length": seq_length - 1
        }
    }
    
    # 使用辅助函数确保所有数据都是JSON可序列化的
    test_data = make_json_serializable(test_data)
    
    # 确保目录存在
    os.makedirs("py/sequence_erase_test", exist_ok=True)
    
    # 保存为JSON文件
    file_path = f"py/sequence_erase_test/sequence_erase_test_{test_name}.json"
    
    try:
        with open(file_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        print(f"  ✓ 已保存测试数据到: {file_path}")
        
        # 验证生成的JSON
        if verify_json_file(file_path):
            print(f"  ✓ JSON文件验证通过")
        else:
            print(f"  ✗ JSON文件验证失败")
    except Exception as e:
        print(f"  ✗ 保存文件失败: {e}")
    
    return file_path

# 生成多个测试
def generate_multiple_tests(num_tests=1):
    print("生成测试数据...")
    print(f"当前工作目录: {os.getcwd()}")
    
    # 确保测试目录存在 - 添加上层目录检查
    for dir_path in ["py", "py/nms_test", "py/sequence_at_test", "py/sequence_construct_test", 
                     "py/sequence_insert_test", "py/sequence_erase_test"]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 生成一个简单的测试样例文件，方便排除C++端读取问题
    sample_test_data = {
        "input": {
            "boxes": [
                {
                    "class_id": 0,
                    "x1": 0.1,
                    "y1": 0.1,
                    "x2": 0.2,
                    "y2": 0.2,
                    "score": 0.9
                }
            ],
            "iou_threshold": 0.5
        },
        "output": {
            "onnx_result": [
                {
                    "class_id": 0,
                    "x1": 0.1,
                    "y1": 0.1,
                    "x2": 0.2,
                    "y2": 0.2,
                    "score": 0.9
                }
            ]
        }
    }
    
    # 保存简单测试样例
    sample_file_path = "py/sample_test.json"
    with open(sample_file_path, 'w') as f:
        json.dump(sample_test_data, f, indent=2)
    print(f"已生成样例测试文件: {sample_file_path}")
    
    # 生成一个超简单的SequenceInsert测试样例
    simple_seq_insert_test = {
        "input": {
            "sequence": [
                {
                    "shape": [2, 2],
                    "data": [0.1, 0.2, 0.3, 0.4]
                },
                {
                    "shape": [2, 2],
                    "data": [0.5, 0.6, 0.7, 0.8]
                }
            ],
            "tensor": {
                "shape": [2, 2],
                "data": [0.9, 1.0, 1.1, 1.2]
            },
            "position": 1
        },
        "output": {
            "onnx_result": [
                {
                    "shape": [2, 2],
                    "data": [0.1, 0.2, 0.3, 0.4]
                },
                {
                    "shape": [2, 2],
                    "data": [0.9, 1.0, 1.1, 1.2]
                },
                {
                    "shape": [2, 2],
                    "data": [0.5, 0.6, 0.7, 0.8]
                }
            ]
        }
    }
    
    # 保存简单SequenceInsert测试样例
    simple_seq_insert_path = "py/simple_sequence_insert_test.json"
    with open(simple_seq_insert_path, 'w') as f:
        json.dump(simple_seq_insert_test, f, indent=2)
    print(f"已生成简化的SequenceInsert测试文件: {simple_seq_insert_path}")
    
    # 生成NonMaxSuppression测试
    nms_test_paths = []
    for i in range(1, num_tests + 1):
        test_name = f"NonMaxSuppression_Test_{i}"
        filepath = generate_nms_test(test_name, seed=i*100)
        nms_test_paths.append(filepath)
        print(f"已生成 NMS 测试: {filepath}")
        # 验证JSON文件是否有效
        try:
            with open(filepath, 'r') as f:
                test_data = json.load(f)
                print(f"  ✓ 验证成功: 文件大小 {os.path.getsize(filepath)} 字节")
        except Exception as e:
            print(f"  ✗ 验证失败: {e}")
    
    # 生成SequenceAt测试
    seq_at_test_paths = []
    for i in range(1, num_tests + 1):
        test_name = f"SequenceAt_Test_{i}"
        filepath = generate_sequence_at_test(test_name, seed=i*100)
        seq_at_test_paths.append(filepath)
        print(f"已生成 SequenceAt 测试: {filepath}")
        # 验证JSON文件是否有效
        try:
            with open(filepath, 'r') as f:
                test_data = json.load(f)
                print(f"  ✓ 验证成功: 文件大小 {os.path.getsize(filepath)} 字节")
        except Exception as e:
            print(f"  ✗ 验证失败: {e}")
    
    # 生成SequenceConstruct测试
    seq_construct_test_paths = []
    for i in range(1, num_tests + 1):
        test_name = f"SequenceConstruct_Test_{i}"
        filepath = generate_sequence_construct_test(test_name, seed=i*100)
        seq_construct_test_paths.append(filepath)
        print(f"已生成 SequenceConstruct 测试: {filepath}")
        # 验证JSON文件是否有效
        try:
            with open(filepath, 'r') as f:
                test_data = json.load(f)
                print(f"  ✓ 验证成功: 文件大小 {os.path.getsize(filepath)} 字节")
        except Exception as e:
            print(f"  ✗ 验证失败: {e}")
    
    # 生成SequenceInsert测试
    seq_insert_test_paths = []
    for i in range(1, num_tests + 1):
        test_name = f"SequenceInsert_Test_{i}"
        filepath = generate_sequence_insert_test(i, 1000, 5)
        seq_insert_test_paths.append(filepath)
        print(f"已生成 SequenceInsert 测试: {filepath}")
        # 验证JSON文件是否有效
        try:
            with open(filepath, 'r') as f:
                test_data = json.load(f)
                print(f"  ✓ 验证成功: 文件大小 {os.path.getsize(filepath)} 字节")
                # 打印JSON文件的第一级键
                print(f"  JSON根级键: {list(test_data.keys())}")
                # 打印input下的键
                if "input" in test_data:
                    print(f"  input键下的字段: {list(test_data['input'].keys())}")
        except Exception as e:
            print(f"  ✗ 验证失败: {e}")
    
    # 生成SequenceErase测试
    seq_erase_test_paths = []
    for i in range(1, num_tests + 1):
        test_name = f"SequenceErase_Test_{i}"
        filepath = generate_sequence_erase_test(test_name, seed=i*100)
        seq_erase_test_paths.append(filepath)
        print(f"已生成 SequenceErase 测试: {filepath}")
        # 验证JSON文件是否有效
        try:
            with open(filepath, 'r') as f:
                test_data = json.load(f)
                print(f"  ✓ 验证成功: 文件大小 {os.path.getsize(filepath)} 字节")
        except Exception as e:
            print(f"  ✗ 验证失败: {e}")
    
    # 生成路径映射文件，帮助C++程序找到测试文件
    path_map = {
        "sample": "py/sample_test.json",
        "simple_sequence_insert": simple_seq_insert_path,
        "nms_tests": nms_test_paths,
        "sequence_at_tests": seq_at_test_paths,
        "sequence_construct_tests": seq_construct_test_paths,
        "sequence_insert_tests": seq_insert_test_paths,
        "sequence_erase_tests": seq_erase_test_paths
    }
    
    # 保存路径映射文件
    with open("test_paths.json", "w") as f:
        json.dump(path_map, f, indent=2)
    print(f"已生成路径映射文件: test_paths.json")
    
    

    
    print("所有测试数据生成完成！")
    


# 执行测试数据生成
if __name__ == "__main__":
    generate_multiple_tests()
    # 打印当前工作目录和目录内容，便于检查
    print(f"当前工作目录: {os.getcwd()}")
    print("目录结构:")
    os.system("find py -type f -name '*.json' | sort")
    print("所有测试数据生成完成！")
    
    print("\n调试帮助:")
    print("1. 测试单个样例文件: ./src/main py/sample_test.json")
    print("2. 测试路径映射: ./src/main test_paths.json")
    print("3. 完整测试: ./src/main")
    print("如果还有问题，请检查C++代码中的JSON解析逻辑与生成的JSON结构是否匹配") 