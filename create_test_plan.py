import pandas as pd
import numpy as np

# 创建测试计划数据
data = [
    # SequenceAt 测试计划
    ['AT-001', 'SequenceAt', '正常索引场景', 'sequence_length=5, position=2, tensor_shape=[3,4,5]', 
     '1. 构造长度为5的序列\n2. 对位置2执行SequenceAt操作', '成功获取位置2的张量', 'P0', '未开始'],
    
    ['AT-002', 'SequenceAt', '空序列测试', 'sequence_length=0, position=0', 
     '1. 构造空序列\n2. 尝试获取元素', '应产生运行时错误', 'P1', '未开始'],
    
    ['AT-003', 'SequenceAt', '负索引测试', 'sequence_length=5, position=-1', 
     '1. 构造长度为5的序列\n2. 使用-1索引获取最后一个元素', '成功获取最后一个元素', 'P1', '未开始'],
    
    ['AT-004', 'SequenceAt', '索引边界测试', 'sequence_length=5, position=4', 
     '1. 构造长度为5的序列\n2. 获取最后一个有效索引位置', '成功获取最后一个元素', 'P1', '未开始'],
    
    ['AT-005', 'SequenceAt', '索引越界测试', 'sequence_length=5, position=5', 
     '1. 构造长度为5的序列\n2. 尝试获取超出范围的元素', '应产生运行时错误', 'P1', '未开始'],
    
    ['AT-006', 'SequenceAt', '特殊张量形状测试', 'sequence_length=3, tensor_shape=[1]', 
     '1. 构造包含标量tensor的序列\n2. 获取元素', '成功获取标量元素', 'P2', '未开始'],
    
    # SequenceConstruct 测试计划
    ['CS-001', 'SequenceConstruct', '正常构造测试', 'sequence_length=5, tensor_shape=[3,4,5]', 
     '1. 创建5个形状为[3,4,5]的张量\n2. 构造序列', '成功创建包含5个元素的序列', 'P0', '未开始'],
    
    ['CS-002', 'SequenceConstruct', '空序列构造', 'sequence_length=0', 
     '1. 不提供任何张量\n2. 构造序列', '成功创建空序列', 'P1', '未开始'],
    
    ['CS-003', 'SequenceConstruct', '单元素序列', 'sequence_length=1, tensor_shape=[3,4,5]', 
     '1. 创建1个张量\n2. 构造序列', '成功创建包含单个元素的序列', 'P1', '未开始'],
    
    ['CS-004', 'SequenceConstruct', '形状不一致测试', 'tensor_shapes=[[1], [2,2], [3,3,3]]', 
     '1. 创建不同形状的张量\n2. 尝试构造序列', '根据ONNX规范行为执行', 'P2', '未开始'],
    
    ['CS-005', 'SequenceConstruct', '类型不一致测试', 'tensor_dtypes=[float32, int32, bool]', 
     '1. 创建不同类型的张量\n2. 尝试构造序列', '根据ONNX规范行为执行', 'P2', '未开始'],
    
    # SequenceErase 测试计划
    ['ER-001', 'SequenceErase', '正常擦除测试', 'sequence_length=5, position=2', 
     '1. 构造长度为5的序列\n2. 从位置2删除元素', '成功删除位置2的元素，序列长度变为4', 'P0', '未开始'],
    
    ['ER-002', 'SequenceErase', '单元素序列擦除', 'sequence_length=1, position=0', 
     '1. 构造单元素序列\n2. 删除唯一的元素', '成功删除元素，得到空序列', 'P1', '未开始'],
    
    ['ER-003', 'SequenceErase', '负索引擦除', 'sequence_length=5, position=-1', 
     '1. 构造长度为5的序列\n2. 使用负索引删除最后一个元素', '成功删除最后一个元素', 'P1', '未开始'],
    
    ['ER-004', 'SequenceErase', '边界索引擦除', 'sequence_length=5, position=0', 
     '1. 构造长度为5的序列\n2. 删除第一个元素', '成功删除第一个元素', 'P1', '未开始'],
    
    ['ER-005', 'SequenceErase', '索引越界测试', 'sequence_length=5, position=5', 
     '1. 构造长度为5的序列\n2. 尝试删除越界位置的元素', '应产生运行时错误', 'P1', '未开始'],
    
    # SequenceInsert 测试计划
    ['IN-001', 'SequenceInsert', '正常插入测试', 'sequence_length=5, position=2', 
     '1. 构造长度为5的序列\n2. 在位置2插入新元素', '成功在位置2插入元素，序列长度变为6', 'P0', '未开始'],
    
    ['IN-002', 'SequenceInsert', '空序列插入', 'sequence_length=0, position=0', 
     '1. 构造空序列\n2. 插入一个元素', '成功插入元素，得到单元素序列', 'P1', '未开始'],
    
    ['IN-003', 'SequenceInsert', '末尾插入(无位置参数)', 'sequence_length=5, position=None', 
     '1. 构造长度为5的序列\n2. 不指定位置插入元素', '元素应被追加到序列末尾', 'P1', '未开始'],
    
    ['IN-004', 'SequenceInsert', '负索引插入', 'sequence_length=5, position=-1', 
     '1. 构造长度为5的序列\n2. 在-1位置插入元素', '元素应被插入到最后一个元素之前', 'P1', '未开始'],
    
    ['IN-005', 'SequenceInsert', '越界索引插入', 'sequence_length=5, position=6', 
     '1. 构造长度为5的序列\n2. 尝试在越界位置插入元素', '应产生运行时错误', 'P2', '未开始'],
    
    ['IN-006', 'SequenceInsert', '形状不一致插入', 'sequence_tensor_shape=[3,4,5], insert_tensor_shape=[2,2]', 
     '1. 构造包含形状[3,4,5]张量的序列\n2. 插入形状为[2,2]的张量', '根据ONNX规范行为执行', 'P2', '未开始'],
    
    # 组合操作测试计划
    ['CO-001', '组合操作', 'Construct+At组合', 'sequence_length=5', 
     '1. 构造序列\n2. 立即获取元素', '操作应正确执行', 'P2', '未开始'],
    
    ['CO-002', '组合操作', 'Insert+Erase组合', 'sequence_length=5, position=2', 
     '1. 构造序列\n2. 插入元素\n3. 立即删除同一位置的元素', '最终序列应与原始序列相同', 'P2', '未开始'],
    
    ['CO-003', '组合操作', '完整操作链', 'sequence_length=0', 
     '1. 构造空序列\n2. 插入多个元素\n3. 获取元素\n4. 删除元素', '每步操作都应正确执行', 'P3', '未开始'],
    
    ['CO-004', '组合操作', '边界情况组合测试', 'sequence_length=1', 
     '1. 构造单元素序列\n2. 删除元素得到空序列\n3. 尝试获取元素', '前两步应成功，第三步应产生错误', 'P2', '未开始'],
    
    # 性能测试计划
    ['PF-001', '性能测试', '大规模序列操作', 'sequence_length=1000, operations=2000', 
     '1. 构造大型序列\n2. 执行大量序列操作', '操作应在合理时间内完成且内存使用可控', 'P3', '未开始']
]

# 创建DataFrame
df = pd.DataFrame(data, columns=['测试ID', '算子名称', '测试场景', '测试参数', '测试步骤', '预期结果', '优先级', '状态'])

# 保存到Excel
df.to_excel('onnx_sequence_ops_test_plan.xlsx', index=False)
print('已创建测试计划表格: onnx_sequence_ops_test_plan.xlsx') 