import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import time
import json
import os
import glob

def generate_nms_test(test_name="test", seed=None):
    """
    生成NonMaxSuppression测试用例，使用更简单和可控的数据
    """
    if seed is None:
        seed = int(time.time())
    np.random.seed(seed)
    
    # 使用更小的测试数据
    num_classes = 48  
    num_boxes = 20    # 每个类别的框数量固定为个
    score_threshold = 0.5
    
    iou_threshold = 0.5
    max_output_boxes_per_class = 2
    center_point_box = 0  # 0表示框格式为[y1, x1, y2, x2]，1表示[x_center, y_center, width, height]
    
    # 生成随机框
    boxes = []
    for class_id in range(num_classes):
        # 为每个类别生成一组框，确保有重叠的框
        for i in range(num_boxes):
            # 基础坐标
            base_x = class_id * 100  # 不同类别的框在不同区域
            base_y = i * 100
            
            # 生成有一定重叠的框
            x1 = base_x + np.random.uniform(0, 20)
            y1 = base_y + np.random.uniform(0, 20)
            x2 = x1 + 80 + np.random.uniform(0, 20)  # 确保框足够大
            y2 = y1 + 80 + np.random.uniform(0, 20)
            
            # 分数按照索引递减，确保排序可预测
            score = 1.0 - (i * 0.15)  # 从1.0开始递减
            
            boxes.append({
                'class_id': class_id,
                'x1': float(x1),
                'y1': float(y1),
                'x2': float(x2),
                'y2': float(y2),
                'score': float(score)
            })
    
    # 创建测试数据
    test_data = {
        "name": test_name,
        "seed": seed,
        "input": {
            "boxes": boxes,
            "iou_threshold": iou_threshold,
            "score_threshold": score_threshold,
            "max_output_boxes_per_class": max_output_boxes_per_class,
            "center_point_box": center_point_box
        },
        "params": {
            "num_classes": num_classes,
            "num_boxes": num_boxes
        }
    }
    
    return test_data

def generate_multiple_tests(num_tests=3):
    """
    生成多个测试用例
    """
    all_test_data = []
    test_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "jsons", "non_max_suppression")
    os.makedirs(test_dir, exist_ok=True)
    for i in range(num_tests):
        test_name = f"{i+1}"
        test_data = generate_nms_test(test_name)
        all_test_data.append(test_data)
        
        # 保存测试数据到JSON文件
        with open(os.path.join(test_dir, f"test_{test_name}.json"), "w") as f:
            json.dump(test_data, f, indent=2)
    
    print("生成的C++测试代码：")
    print("\n测试数据已保存到JSON文件")

def create_nms_model():
    """创建包含NonMaxSuppression算子的ONNX模型"""
    # 输入定义
    boxes = helper.make_tensor_value_info('boxes', TensorProto.FLOAT, ['num_batches', 'spatial_dimension', 4])
    scores = helper.make_tensor_value_info('scores', TensorProto.FLOAT, ['num_batches', 'num_classes', 'spatial_dimension'])
    max_output_boxes = helper.make_tensor_value_info('max_output_boxes_per_class', TensorProto.INT64, [])
    iou_threshold = helper.make_tensor_value_info('iou_threshold', TensorProto.FLOAT, [])
    score_threshold = helper.make_tensor_value_info('score_threshold', TensorProto.FLOAT, [])
    
    # 输出定义
    selected_indices = helper.make_tensor_value_info('selected_indices', TensorProto.INT64, [None, 3])
    
    # NonMaxSuppression节点
    nms_node = helper.make_node(
        'NonMaxSuppression',
        inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
        outputs=['selected_indices'],
        center_point_box=0
    )
    
    # 创建图和模型
    graph = helper.make_graph(
        [nms_node],
        'nms_test_model',
        [boxes, scores, max_output_boxes, iou_threshold, score_threshold],
        [selected_indices]
    )
    
    model = helper.make_model(graph, producer_name='nms_test')
    
    # 设置opset版本
    model.opset_import[0].version = 18
    
    return model

def generate_reference_output(input_json_path):
    """处理输入JSON文件，生成参考输出"""
    # 加载输入数据
    with open(input_json_path, 'r') as f:
        test_data = json.load(f)
    
    # 构建ONNX模型
    model = create_nms_model()
    session = ort.InferenceSession(model.SerializeToString())
    
    num_classes = test_data["params"]["num_classes"]
    
    # 准备输入数据
    boxes_list = []
    scores_list = [[[] for _ in range(num_classes)] for _ in range(1)]  # batch_size = 1
    
    # 打印原始数据
    print("\n=== 原始输入数据 ===")
    print(f"类别数: {num_classes}")
    print(f"框数量: {len(test_data['input']['boxes'])}")
    print(f"IoU阈值: {test_data['input']['iou_threshold']}")
    
    # 填充数据并打印一些示例
    print("\n=== 框和分数示例 ===")
    for i, box in enumerate(test_data["input"]["boxes"]):
        box_coords = [
            float(box["y1"]),
            float(box["x1"]),
            float(box["y2"]),
            float(box["x2"])
        ]
        boxes_list.append(box_coords)
        
        class_id = int(box["class_id"])
        score = float(box["score"])
        
        if i < 5:  # 打印前5个框的信息
            print(f"框 {i}:")
            print(f"  坐标: {box_coords}")
            print(f"  类别: {class_id}")
            print(f"  分数: {score}")
        
        # 填充分数
        for j in range(num_classes):
            if j == class_id:
                scores_list[0][j].append(score)
            else:
                scores_list[0][j].append(0.0)
    
    # 转换为numpy数组
    boxes_np = np.array([boxes_list], dtype=np.float32)
    scores_np = np.array(scores_list, dtype=np.float32)
    
    print("\n=== 输入张量形状 ===")
    print(f"boxes shape: {boxes_np.shape}")
    print(f"scores shape: {scores_np.shape}")
    
    # 打印分数矩阵的一部分
    print("\n=== 分数矩阵示例（每个类别的前3个分数）===")
    for i in range(num_classes):
        print(f"类别 {i}: {scores_np[0][i][:3]}")
    
    # 设置参数
    max_output_boxes_per_class = np.array(3, dtype=np.int64)  # 限制每类最多3个框
    iou_threshold = np.array(test_data["input"]["iou_threshold"], dtype=np.float32)
    score_threshold = np.array(0.1, dtype=np.float32)  # 设置一个较低的分数阈值
    
    print("\n=== NMS参数 ===")
    print(f"每类最大输出框数: {max_output_boxes_per_class}")
    print(f"IoU阈值: {iou_threshold}")
    print(f"分数阈值: {score_threshold}")
    
    # 打印每个类别的分数排序情况
    print("\n=== 每个类别的分数排序 ===")
    for class_id in range(num_classes):
        class_scores = [(i, scores_np[0][class_id][i]) for i in range(len(boxes_list))]
        class_scores.sort(key=lambda x: x[1], reverse=True)
        print(f"\n类别 {class_id} 的分数排序:")
        for box_idx, score in class_scores[:5]:  # 只打印前5个
            box = boxes_list[box_idx]
            print(f"  框 {box_idx}: score={score:.4f}, 坐标={box}")
    
    # 运行推理
    result = session.run(
        ['selected_indices'],
        {
            'boxes': boxes_np,
            'scores': scores_np,
            'max_output_boxes_per_class': max_output_boxes_per_class,
            'iou_threshold': iou_threshold,
            'score_threshold': score_threshold
        }
    )
    
    selected_indices = result[0].tolist()
    
    print("\n=== NMS结果说明 ===")
    print(f"选中框数量: {len(selected_indices)}")
    if selected_indices:
        print("\n每个类别选中的框:")
        for class_id in range(num_classes):
            class_boxes = [idx for idx in selected_indices if idx[1] == class_id]
            print(f"\n类别 {class_id}:")
            for idx in class_boxes:
                batch_idx, _, box_idx = idx
                score = scores_np[batch_idx][class_id][box_idx]
                box = boxes_list[box_idx]
                print(f"  框 {box_idx}: score={score:.4f}, 坐标={box}")
                
                # 打印与其他未选中框的IoU
                print("  与其他框的IoU:")
                for other_idx in range(len(boxes_list)):
                    if other_idx != box_idx and scores_np[0][class_id][other_idx] > 0:
                        iou = calculate_iou(box, boxes_list[other_idx])
                        if iou > 0.1:  # 只打印有明显重叠的框
                            print(f"    与框 {other_idx} 的IoU: {iou:.4f}")
    
    # 更新JSON
    test_data["output"] = {
        "selected_indices": selected_indices,
        "explanation": {
            "max_output_boxes_per_class": int(max_output_boxes_per_class),
            "iou_threshold": float(iou_threshold),
            "score_threshold": float(score_threshold)
        }
    }
    
    # 保存更新后的JSON
    with open(input_json_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    return selected_indices

def calculate_iou(box1, box2):
    """计算两个框的IoU"""
    # 计算交集
    y1 = max(box1[0], box2[0])
    x1 = max(box1[1], box2[1])
    y2 = min(box1[2], box2[2])
    x2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # 计算并集
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union

# 处理所有测试文件
def process_all_test_files(test_dir="../jsons/non_max_suppression"):
    """处理指定目录下的所有JSON测试文件"""
    json_files = glob.glob(os.path.join(test_dir, "*.json"))
    
    for json_file in json_files:
        generate_reference_output(json_file)
    
    print(f"已处理{len(json_files)}个测试文件")

# 运行测试生成
if __name__ == "__main__":
    # 重新生成测试用例
    generate_multiple_tests(1)  # 生成一个测试用例
    
    # 处理测试文件
    process_all_test_files()