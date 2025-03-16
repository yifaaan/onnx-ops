import pandas as pd
import numpy as np

# 创建数据框
data = [
    # SequenceAt
    ['SequenceAt', 'sequence_length', '0, 1', '3-20', '测试空序列和单元素序列'],
    ['SequenceAt', 'position', '-seq_length, -1, 0, seq_length-1, seq_length', '随机位置', '测试边界索引和越界索引'],
    ['SequenceAt', 'tensor_shape', '[1], [1,1]', '[3,20,50]', '测试不同维度的张量'],
    ['SequenceAt', 'tensor_dtype', 'float32, int32, bool', 'float32', '测试不同数据类型的张量'],
    
    # SequenceConstruct
    ['SequenceConstruct', 'sequence_length', '0, 1', '3-20', '测试空序列和单元素序列构造'],
    ['SequenceConstruct', 'tensor_shape', '[1], [1,1]', '[3,20,50]', '测试不同维度张量的序列构造'],
    ['SequenceConstruct', 'tensor_dtype', 'float32, int32, bool', 'float32', '测试不同数据类型的张量'],
    ['SequenceConstruct', '输入张量一致性', '形状不一致, 类型不一致', '完全一致', '测试输入张量不完全一致的情况'],
    
    # SequenceErase
    ['SequenceErase', 'sequence_length', '1, 2', '3-20', '测试最小长度序列的擦除'],
    ['SequenceErase', 'position', '-seq_length, -1, 0, seq_length-1, seq_length', '随机位置', '测试边界位置和越界位置的擦除'],
    ['SequenceErase', 'tensor_shape', '[1], [1,1]', '[3,20,50]', '测试不同维度张量序列的擦除'],
    ['SequenceErase', 'tensor_dtype', 'float32, int32, bool', 'float32', '测试不同数据类型的张量序列'],
    
    # SequenceInsert
    ['SequenceInsert', 'sequence_length', '0, 1', '3-20', '测试空序列和单元素序列的插入'],
    ['SequenceInsert', 'position', '-seq_length-1, -1, 0, seq_length, None', '随机位置', '测试边界位置、越界位置和不提供位置的插入'],
    ['SequenceInsert', 'tensor_shape', '[1], [1,1]', '[3,20,50]', '测试不同维度张量的插入'],
    ['SequenceInsert', 'tensor_dtype', 'float32, int32, bool', 'float32', '测试不同数据类型的张量'],
    ['SequenceInsert', '插入张量一致性', '与序列中张量形状不同', '与序列中张量形状相同', '测试插入形状不一致的张量'],
    
    # 序列操作组合
    ['组合操作', 'Construct+At', '构造后立即获取', '随机操作', '测试序列构造后立即获取元素'],
    ['组合操作', 'Insert+Erase', '插入后立即删除同一位置', '随机操作', '测试序列插入后立即删除同一位置'],
    ['组合操作', 'Construct+Insert+At+Erase', '完整序列操作链', '随机组合', '测试完整的序列操作链']
]

# 创建DataFrame
df = pd.DataFrame(data, columns=['算子', '参数', '边界值测试', '普通测试', '说明'])

# 保存到Excel
df.to_excel('onnx_sequence_ops_tests.xlsx', index=False)
print('已创建Excel文件: onnx_sequence_ops_tests.xlsx') 