import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import json
import os
import random
import time
def generate_lppool_test(dims, p=2, kernel_shape=None, strides=None, pads=None, auto_pad="NOTSET", dilations=None, ceil_mode=False, test_name="test"):
    """
    生成LpPool测试用例
    
    参数:
    dims: 维度数量(3-5)
    p: p范数值，默认为2
    kernel_shape: 池化窗口大小，默认为每个维度为2的数组
    strides: 步长，默认与kernel_shape相同
    pads: 填充，默认为0
    auto_pad: 填充模式，默认为"NOTSET"
    dilations: 膨胀率，默认为1
    ceil_mode: 是否使用向上取整模式，默认为False
    test_name: 测试用例名称
    """
    # 根据维度数量生成输入形状
    # 对于3维: [batch, channels, seq_len]
    # 对于4维: [batch, channels, height, width]
    # 对于5维: [batch, channels, depth, height, width]
    
    
    if dims == 3:
        batch = random.randint(2, 10)
        channels = random.randint(2, 4)
        seq_len = random.randint(80, 120)
        input_shape = [batch, channels, seq_len]
        if kernel_shape is None:
            kernel_shape = [3]
        if dilations is None:
            dilations = [1]
    elif dims == 4:
        batch = random.randint(2, 10)
        channels = random.randint(2, 4)
        height = random.randint(80, 120)
        width = random.randint(80, 120)
        input_shape = [batch, channels, height, width]
        if kernel_shape is None:
            kernel_shape = [3, 3]
        if dilations is None:
            dilations = [1, 1]
    else:  # dims == 5
        batch = random.randint(2, 10)
        channels = random.randint(2, 4)
        depth = random.randint(80, 120)
        height = random.randint(80, 120)
        width = random.randint(80, 120)
        input_shape = [batch, channels, depth, height, width]
        if kernel_shape is None:
            kernel_shape = [2, 2, 2]
        if dilations is None:
            dilations = [1, 1, 1]
    
    # 设置默认步长和填充
    spatial_dims = dims - 2  # 减去batch和channel维度
    if strides is None:
        strides = kernel_shape.copy()
    if pads is None:
        pads = [0] * (spatial_dims * 2)  # 前后各维度的填充
    
    # 生成随机输入数据
    # np.random.seed(0)  # 固定随机种子以便复现
    random.seed(int(time.time()))
    input_data = np.random.uniform(-1.0, 1.0, input_shape).astype(np.float32)
    
    # 构建ONNX模型
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_shape)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)  # 输出形状由推理确定
    
    node = helper.make_node(
        'LpPool',
        inputs=['X'],
        outputs=['Y'],
        kernel_shape=kernel_shape,
        strides=strides,
        pads=pads,
        p=p,
        dilations=dilations,
        auto_pad=auto_pad,
        ceil_mode=1 if ceil_mode else 0  # ONNX使用整数0/1表示bool值
    )
    
    graph = helper.make_graph(
        [node],
        'lppool_test',
        [X],
        [Y]
    )
    
    model = helper.make_model(graph, producer_name='lppool_test')
    # 显式设置opset版本为21，确保兼容性
    model.opset_import[0].version = 21
    
    # 用ONNX Runtime运行模型
    session = ort.InferenceSession(model.SerializeToString())
    ort_outputs = session.run(None, {'X': input_data})
    ort_output = ort_outputs[0]
    
    # 准备测试数据
    test_data = {
        "name": test_name,
        "input": {
            "data": input_data.flatten().tolist(),
            "shape": input_shape
        },
        "output": {
            "data": ort_output.flatten().tolist(),
            "shape": list(ort_output.shape)
        },
        "params": {
            "kernel_shape": kernel_shape,
            "strides": strides,
            "pads": pads,
            "p": p,
            "auto_pad": auto_pad,
            "dilations": dilations,
            "ceil_mode": ceil_mode
        }
    }
    
    # 生成C++测试代码
    cpp_code = f"""
// Test case for {test_name}
// Input shape: {input_shape}
// Output shape: {list(ort_output.shape)}
// Parameters: kernel_shape={kernel_shape}, strides={strides}, pads={pads}, p={p}, auto_pad="{auto_pad}", dilations={dilations}, ceil_mode={ceil_mode}
TEST(LpPoolTest, {test_name.replace(" ", "_")}) {{
    // 从文件读取测试数据
    std::ifstream file("lppool_test_{test_name.replace(" ", "_")}.json");
    nlohmann::json test_data;
    file >> test_data;
    
    // 提取测试数据
    std::vector<float> input_data = test_data["input"]["data"].get<std::vector<float>>();
    std::vector<int64_t> input_shape = test_data["input"]["shape"].get<std::vector<int64_t>>();
    std::vector<int64_t> kernel_shape = test_data["params"]["kernel_shape"].get<std::vector<int64_t>>();
    std::vector<int64_t> strides = test_data["params"]["strides"].get<std::vector<int64_t>>();
    std::vector<int64_t> pads = test_data["params"]["pads"].get<std::vector<int64_t>>();
    int64_t p = test_data["params"]["p"];
    std::string auto_pad = test_data["params"]["auto_pad"];
    std::vector<int64_t> dilations = test_data["params"]["dilations"].get<std::vector<int64_t>>();
    bool ceil_mode = test_data["params"]["ceil_mode"].get<bool>();
    
    // 调用LpPool实现
    auto [output_data, output_shape] = onnx::LpPool<float>::Compute(
        input_data, input_shape, kernel_shape, strides, pads, auto_pad, p, dilations, ceil_mode);
    
    // 验证输出形状
    std::vector<int64_t> expected_shape = test_data["output"]["shape"].get<std::vector<int64_t>>();
    ASSERT_EQ(output_shape.size(), expected_shape.size());
    for (size_t i = 0; i < output_shape.size(); ++i) {{
        EXPECT_EQ(output_shape[i], expected_shape[i]);
    }}
    
    // 验证输出数据
    std::vector<float> expected_data = test_data["output"]["data"].get<std::vector<float>>();
    ASSERT_EQ(output_data.size(), expected_data.size());
    
    // 验证数据值是否接近，考虑浮点误差
    for (size_t i = 0; i < output_data.size(); ++i) {{
        EXPECT_NEAR(output_data[i], expected_data[i], 1e-4f);
    }}
}}
"""
    
    # 如果数据过大，可以选择只验证一部分
    if len(test_data["output"]["data"]) > 1000:
        # 仅验证随机抽样的值
        cpp_code = f"""
// Test case for {test_name}
// Input shape: {input_shape}
// Output shape: {list(ort_output.shape)}
// Parameters: kernel_shape={kernel_shape}, strides={strides}, pads={pads}, p={p}, auto_pad="{auto_pad}", dilations={dilations}, ceil_mode={ceil_mode}
TEST(LpPoolTest, {test_name.replace(" ", "_")}) {{
    // 从文件读取测试数据
    std::ifstream file("lppool_test_{test_name.replace(" ", "_")}.json");
    nlohmann::json test_data;
    file >> test_data;
    
    // 提取测试数据
    std::vector<float> input_data = test_data["input"]["data"].get<std::vector<float>>();
    std::vector<int64_t> input_shape = test_data["input"]["shape"].get<std::vector<int64_t>>();
    std::vector<int64_t> kernel_shape = test_data["params"]["kernel_shape"].get<std::vector<int64_t>>();
    std::vector<int64_t> strides = test_data["params"]["strides"].get<std::vector<int64_t>>();
    std::vector<int64_t> pads = test_data["params"]["pads"].get<std::vector<int64_t>>();
    int64_t p = test_data["params"]["p"];
    std::string auto_pad = test_data["params"]["auto_pad"];
    std::vector<int64_t> dilations = test_data["params"]["dilations"].get<std::vector<int64_t>>();
    bool ceil_mode = test_data["params"]["ceil_mode"].get<bool>();
    
    // 调用LpPool实现
    auto [output_data, output_shape] = onnx::LpPool<float>::Compute(
        input_data, input_shape, kernel_shape, strides, pads, auto_pad, p, dilations, ceil_mode);
    
    // 验证输出形状
    std::vector<int64_t> expected_shape = test_data["output"]["shape"].get<std::vector<int64_t>>();
    ASSERT_EQ(output_shape.size(), expected_shape.size());
    for (size_t i = 0; i < output_shape.size(); ++i) {{
        EXPECT_EQ(output_shape[i], expected_shape[i]);
    }}
    
    // 验证输出数据
    std::vector<float> expected_data = test_data["output"]["data"].get<std::vector<float>>();
    ASSERT_EQ(output_data.size(), expected_data.size());
    
    // 随机检查100个值，避免测试时间过长
    std::vector<size_t> indices;
    for (size_t i = 0; i < 100; ++i) {{
        indices.push_back(rand() % output_data.size());
    }}
    
    for (size_t idx : indices) {{
        EXPECT_NEAR(output_data[idx], expected_data[idx], 1e-4f);
    }}
}}
"""
    
    return test_data, cpp_code

# 生成3维测试用例
test_3d_1, cpp_3d_1 = generate_lppool_test(
    dims=3,
    p=2,
    kernel_shape=[3],
    strides=[1],
    test_name="LpPool_3D_1"
)

test_3d_2, cpp_3d_2 = generate_lppool_test(
    dims=3,
    p=2,
    kernel_shape=[3],
    strides=[2],
    test_name="LpPool_3D_2"
)

test_3d_3, cpp_3d_3 = generate_lppool_test(
    dims=3,
    p=2,
    kernel_shape=[3],
    strides=[3],
    test_name="LpPool_3D_3"
)

# 生成4维测试用例
test_4d_1, cpp_4d_1 = generate_lppool_test(
    dims=4,
    p=2,
    kernel_shape=[3, 3],
    strides=[2, 2],
    test_name="LpPool_4D_1"
)

test_4d_2, cpp_4d_2 = generate_lppool_test(
    dims=4,
    p=2,
    kernel_shape=[3, 3],
    strides=[1, 2],
    test_name="LpPool_4D_2"
)

test_4d_3, cpp_4d_3 = generate_lppool_test(
    dims=4,
    p=2,
    kernel_shape=[3, 3],
    strides=[2, 3],
    test_name="LpPool_4D_3"
)

# 生成5维测试用例1
test_5d_1, cpp_5d_1 = generate_lppool_test(
    dims=5,
    p=1,  # 使用L1范数
    kernel_shape=[2, 2, 2],
    strides=[1, 2, 3],
    test_name="LpPool_5D_1"
)

# 生成5维测试用例2
test_5d_2, cpp_5d_2 = generate_lppool_test(
    dims=5,
    p=2,
    kernel_shape=[2, 2, 2],
    strides=[2, 1, 3],
    test_name="LpPool_5D_2"
)

test_5d_3, cpp_5d_3 = generate_lppool_test(
    dims=5,
    p=2,
    kernel_shape=[2, 2, 2],
    strides=[2, 2, 3],
    test_name="LpPool_5D_3"
)

# 添加带有ceil_mode=True的测试用例
test_3d_ceil, cpp_3d_ceil = generate_lppool_test(
    dims=3,
    p=2,
    kernel_shape=[3],
    strides=[2],
    ceil_mode=True,
    test_name="LpPool_3D_Ceil"
)

test_4d_ceil, cpp_4d_ceil = generate_lppool_test(
    dims=4,
    p=2,
    kernel_shape=[3, 3],
    strides=[2, 2],
    ceil_mode=True,
    test_name="LpPool_4D_Ceil"
)

test_4d_dilation_ceil, cpp_4d_dilation_ceil = generate_lppool_test(
    dims=4,
    p=2,
    kernel_shape=[3, 3],
    strides=[2, 2],
    dilations=[2, 2],
    ceil_mode=True,
    test_name="LpPool_4D_Dilation_Ceil"
)

# 添加更多测试用例，特别关注 ceil_mode 和 dilations 参数

# 添加带有不同 dilations 值的测试用例
test_4d_dilation_1, cpp_4d_dilation_1 = generate_lppool_test(
    dims=4,
    p=2,
    kernel_shape=[3, 3],
    strides=[2, 2],
    dilations=[2, 2],
    ceil_mode=False,
    test_name="LpPool_4D_Dilation_1"
)

test_4d_dilation_2, cpp_4d_dilation_2 = generate_lppool_test(
    dims=4,
    p=2,
    kernel_shape=[3, 3],
    strides=[1, 1],
    dilations=[3, 3],
    ceil_mode=False,
    test_name="LpPool_4D_Dilation_2"
)

# 添加带有不同 ceil_mode 值的测试用例
test_3d_ceil_dilation, cpp_3d_ceil_dilation = generate_lppool_test(
    dims=3,
    p=2,
    kernel_shape=[3],
    strides=[2],
    dilations=[2],
    ceil_mode=True,
    test_name="LpPool_3D_Ceil_Dilation"
)

test_5d_ceil_dilation, cpp_5d_ceil_dilation = generate_lppool_test(
    dims=5,
    p=3,  # 使用L3范数
    kernel_shape=[2, 2, 2],
    strides=[2, 2, 2],
    dilations=[2, 2, 2],
    ceil_mode=True,
    test_name="LpPool_5D_Ceil_Dilation"
)

# 输出C++测试代码
print("// LpPool 3D 测试")
print(cpp_3d_1)
print(cpp_3d_2)
print(cpp_3d_3)
print("\n// LpPool 4D 测试")
print(cpp_4d_1)
print(cpp_4d_2)
print(cpp_4d_3)
print("\n// LpPool 5D 测试")
print(cpp_5d_1)
print(cpp_5d_2)
print(cpp_5d_3)
print("\n// LpPool Ceil测试")
print(cpp_3d_ceil)
print(cpp_4d_ceil)
print(cpp_4d_dilation_ceil)

# 打印新增测试用例的C++代码
print("\n// LpPool Dilation测试")
print(cpp_4d_dilation_1)
print(cpp_4d_dilation_2)
print("\n// LpPool Ceil和Dilation组合测试")
print(cpp_3d_ceil_dilation)
print(cpp_5d_ceil_dilation)

# 保存测试数据到文件
with open("./lppool_test/lppool_test_LpPool_3D_1.json", "w") as f:
    json.dump(test_3d_1, f)

with open("./lppool_test/lppool_test_LpPool_3D_2.json", "w") as f:
    json.dump(test_3d_2, f)

with open("./lppool_test/lppool_test_LpPool_3D_3.json", "w") as f:
    json.dump(test_3d_3, f)

with open("./lppool_test/lppool_test_LpPool_4D_1.json", "w") as f:
    json.dump(test_4d_1, f)
with open("./lppool_test/lppool_test_LpPool_4D_2.json", "w") as f:
    json.dump(test_4d_2, f)
with open("./lppool_test/lppool_test_LpPool_4D_3.json", "w") as f:
    json.dump(test_4d_3, f)


with open("./lppool_test/lppool_test_LpPool_5D_1.json", "w") as f:
    json.dump(test_5d_1, f)
with open("./lppool_test/lppool_test_LpPool_5D_2.json", "w") as f:
    json.dump(test_5d_2, f)
with open("./lppool_test/lppool_test_LpPool_5D_3.json", "w") as f:
    json.dump(test_5d_3, f)

with open("./lppool_test/lppool_test_LpPool_3D_Ceil.json", "w") as f:
    json.dump(test_3d_ceil, f)

with open("./lppool_test/lppool_test_LpPool_4D_Ceil.json", "w") as f:
    json.dump(test_4d_ceil, f)

with open("./lppool_test/lppool_test_LpPool_4D_Dilation_Ceil.json", "w") as f:
    json.dump(test_4d_dilation_ceil, f)

with open("./lppool_test/lppool_test_LpPool_4D_Dilation_1.json", "w") as f:
    json.dump(test_4d_dilation_1, f)

with open("./lppool_test/lppool_test_LpPool_4D_Dilation_2.json", "w") as f:
    json.dump(test_4d_dilation_2, f)

with open("./lppool_test/lppool_test_LpPool_3D_Ceil_Dilation.json", "w") as f:
    json.dump(test_3d_ceil_dilation, f)

with open("./lppool_test/lppool_test_LpPool_5D_Ceil_Dilation.json", "w") as f:
    json.dump(test_5d_ceil_dilation, f)

print("\n测试数据已保存到json文件，可用于C++测试")