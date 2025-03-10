import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import time
import json

def generate_nms_test(test_name="test", seed=None):
    """
    生成NonMaxSuppression测试用例
    
    参数:
    test_name: 测试用例名称
    seed: 随机种子，如果为None则使用时间戳
    """
    if seed is None:
        seed = int(time.time())
    np.random.seed(seed)
    
    # 生成随机数据
    num_classes = np.random.randint(2, 5)        # 类别数量 2-4
    num_boxes = np.random.randint(10, 20)        # 每个类别的框数量
    
    # 生成随机框
    boxes = []
    scores = []
    for class_id in range(num_classes):
        # 为每个类别生成一组框
        for _ in range(num_boxes):
            # 确保x2>x1, y2>y1
            x1 = np.random.uniform(0, 300)
            y1 = np.random.uniform(0, 300)
            w = np.random.uniform(20, 100)  # 宽度
            h = np.random.uniform(20, 100)  # 高度
            x2 = x1 + w
            y2 = y1 + h
            score = np.random.uniform(0.1, 1.0)
            
            boxes.append({
                'class_id': class_id,
                'x1': float(x1),
                'y1': float(y1),
                'x2': float(x2),
                'y2': float(y2),
                'score': float(score)
            })
    
    # 设置IoU阈值
    iou_threshold = float(np.random.uniform(0.3, 0.7))
    
    # 创建测试数据
    test_data = {
        "name": test_name,
        "seed": seed,
        "input": {
            "boxes": boxes,
            "iou_threshold": iou_threshold
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
    
    for i in range(num_tests):
        test_name = f"NonMaxSuppression_Test_{i+1}"
        test_data, cpp_code = generate_nms_test(test_name)
        all_test_data.append(test_data)
        
        # 保存测试数据到JSON文件
        with open(f"./nms_test/nms_test_{test_name}.json", "w") as f:
            json.dump(test_data, f, indent=2)
    
    print("生成的C++测试代码：")
    print("\n测试数据已保存到JSON文件")

# 运行测试生成
if __name__ == "__main__":
    # 可以设置固定种子以获得可重现的结果
    # test_data, cpp_code = generate_nms_test(seed=42)
    
    # 或者生成多个随机测试用例
    generate_multiple_tests(3)