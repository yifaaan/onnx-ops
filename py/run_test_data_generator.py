#!/usr/bin/env python3
"""
运行测试数据生成器
"""

import os
import sys
from generate_test_data import generate_multiple_tests

def main():
    print("运行ONNX算子测试数据生成...")
    
    # 确保当前工作目录正确
    print(f"当前工作目录: {os.getcwd()}")
    
    # 设置生成测试数量
    num_tests = 1
    if len(sys.argv) > 1:
        num_tests = int(sys.argv[1])
        print(f"将生成每种类型的 {num_tests} 个测试")
    
    # 生成测试数据
    generate_multiple_tests(num_tests)
    
    print("测试数据生成完成!")

if __name__ == "__main__":
    main() 