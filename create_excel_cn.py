import pandas as pd
import numpy as np

# 创建数据框
data = [
    # SequenceAt
    ['SequenceAt', 'sequence_length (序列长度)', '0 (空序列)', '正常长度 (3-20)', '测试空序列边界情况，预期应该报错'],
    ['SequenceAt', 'sequence_length (序列长度)', '1 (单元素序列)', '正常长度 (3-20)', '测试单元素序列'],
    ['SequenceAt', 'position (索引位置)', '-seq_length (最小负索引)', '随机负索引', '测试负索引边界，等价于位置0'],
    ['SequenceAt', 'position (索引位置)', '-1 (负索引)', '随机负索引', '测试常见负索引，指向最后一个元素'],
    ['SequenceAt', 'position (索引位置)', '0 (起始位置)', '随机正索引', '测试起始位置索引'],
    ['SequenceAt', 'position (索引位置)', 'seq_length-1 (最后位置)', '随机正索引', '测试末尾位置索引'],
    ['SequenceAt', 'position (索引位置)', 'seq_length (越界索引)', '随机正索引', '测试越界索引，预期应该报错'],
    ['SequenceAt', 'tensor_shape (张量形状)', '[1] (标量)', '多维张量', '测试标量元素的序列'],
    ['SequenceAt', 'tensor_shape (张量形状)', '[1,1] (最小矩阵)', '多维张量', '测试最小矩阵元素的序列'],
    ['SequenceAt', 'tensor_dtype (数据类型)', 'float32, int32, bool', 'float32', '测试不同数据类型的张量序列'],
    
    # SequenceConstruct
    ['SequenceConstruct', 'sequence_length (序列长度)', '0 (空序列)', '正常长度 (3-20)', '测试构造空序列'],
    ['SequenceConstruct', 'sequence_length (序列长度)', '1 (单元素序列)', '正常长度 (3-20)', '测试构造单元素序列'],
    ['SequenceConstruct', 'tensor_shape (张量形状)', '[1] (标量)', '多维张量', '测试构造标量元素的序列'],
    ['SequenceConstruct', 'tensor_shape (张量形状)', '[1,1] (最小矩阵)', '多维张量', '测试构造最小矩阵元素的序列'],
    ['SequenceConstruct', 'tensor_dtype (数据类型)', 'float32, int32, bool', 'float32', '测试构造不同数据类型的张量序列'],
    ['SequenceConstruct', '输入张量一致性', '形状不一致', '形状一致', '测试构造元素形状不一致的序列，ONNX规范对此不明确'],
    ['SequenceConstruct', '输入张量一致性', '类型不一致', '类型一致', '测试构造元素类型不一致的序列，ONNX规范对此不明确'],
    ['SequenceConstruct', '大数据量', '大量输入张量 (>100)', '适量输入张量', '测试大数据量的序列构造性能'],
    
    # SequenceErase
    ['SequenceErase', 'sequence_length (序列长度)', '1 (单元素序列)', '正常长度 (3-20)', '测试擦除单元素序列后得到空序列'],
    ['SequenceErase', 'sequence_length (序列长度)', '2 (双元素序列)', '正常长度 (3-20)', '测试最小有意义序列的擦除'],
    ['SequenceErase', 'position (索引位置)', '-seq_length (最小负索引)', '随机负索引', '测试负索引边界，等价于位置0'],
    ['SequenceErase', 'position (索引位置)', '-1 (负索引)', '随机负索引', '测试常见负索引，指向最后一个元素'],
    ['SequenceErase', 'position (索引位置)', '0 (起始位置)', '随机正索引', '测试擦除起始位置元素'],
    ['SequenceErase', 'position (索引位置)', 'seq_length-1 (最后位置)', '随机正索引', '测试擦除末尾位置元素'],
    ['SequenceErase', 'position (索引位置)', 'seq_length (越界索引)', '随机正索引', '测试越界索引，预期应该报错'],
    ['SequenceErase', 'tensor_shape (张量形状)', '[1] (标量)', '多维张量', '测试擦除标量元素的序列'],
    ['SequenceErase', 'tensor_shape (张量形状)', '[1,1] (最小矩阵)', '多维张量', '测试擦除最小矩阵元素的序列'],
    ['SequenceErase', 'tensor_dtype (数据类型)', 'float32, int32, bool', 'float32', '测试擦除不同数据类型的张量序列'],
    
    # SequenceInsert
    ['SequenceInsert', 'sequence_length (序列长度)', '0 (空序列)', '正常长度 (3-20)', '测试向空序列插入元素'],
    ['SequenceInsert', 'sequence_length (序列长度)', '1 (单元素序列)', '正常长度 (3-20)', '测试向单元素序列插入元素'],
    ['SequenceInsert', 'position (索引位置)', 'None (不指定位置)', '随机位置', '测试不指定位置，应默认在末尾插入'],
    ['SequenceInsert', 'position (索引位置)', '-seq_length-1 (越界负索引)', '随机负索引', '测试越界负索引，预期应该报错'],
    ['SequenceInsert', 'position (索引位置)', '-seq_length (最小负索引)', '随机负索引', '测试负索引边界，等价于位置0'],
    ['SequenceInsert', 'position (索引位置)', '-1 (负索引)', '随机负索引', '测试常见负索引，在最后一个元素前插入'],
    ['SequenceInsert', 'position (索引位置)', '0 (起始位置)', '随机正索引', '测试在起始位置插入'],
    ['SequenceInsert', 'position (索引位置)', 'seq_length (末尾之后)', '随机正索引', '测试在末尾之后插入，等价于追加'],
    ['SequenceInsert', 'position (索引位置)', 'seq_length+1 (越界索引)', '随机正索引', '测试越界正索引，预期应该报错'],
    ['SequenceInsert', 'tensor_shape (张量形状)', '[1] (标量)', '多维张量', '测试插入标量张量'],
    ['SequenceInsert', 'tensor_shape (张量形状)', '[1,1] (最小矩阵)', '多维张量', '测试插入最小矩阵张量'],
    ['SequenceInsert', 'tensor_dtype (数据类型)', 'float32, int32, bool', 'float32', '测试插入不同数据类型的张量'],
    ['SequenceInsert', '插入张量一致性', '与序列中张量形状不同', '与序列中张量形状相同', '测试插入形状不一致的张量，ONNX规范对此不明确'],
    ['SequenceInsert', '插入张量一致性', '与序列中张量类型不同', '与序列中张量类型相同', '测试插入类型不一致的张量，ONNX规范对此不明确'],
    
    # 序列操作组合测试
    ['组合操作', 'Construct+At', '构造空序列后获取元素', '正常序列操作', '测试边界情况组合，预期应该报错'],
    ['组合操作', 'Construct+At', '构造单元素序列后获取越界元素', '正常序列操作', '测试边界情况组合，预期应该报错'],
    ['组合操作', 'Insert+Erase', '向空序列插入后立即删除', '正常序列操作', '测试边界情况组合'],
    ['组合操作', 'Insert+Erase', '插入后立即删除同一位置的元素', '正常序列操作', '测试序列操作的顺序执行'],
    ['组合操作', 'Construct+Insert+At+Erase', '完整序列操作链-极限情况', '正常序列操作链', '测试复杂的边界情况序列操作组合'],
    ['组合操作', '性能测试', '大规模序列操作 (>1000次)', '适量序列操作', '测试序列操作的性能和内存使用']
]

# 创建DataFrame
df = pd.DataFrame(data, columns=['算子', '参数', '边界值测试', '普通测试', '说明'])

# 保存到Excel
df.to_excel('onnx_sequence_ops_tests_cn.xlsx', index=False)
print('已创建中文版Excel文件: onnx_sequence_ops_tests_cn.xlsx') 