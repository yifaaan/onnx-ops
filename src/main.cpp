#include <fstream>
#include <iostream>
#include <json.hpp>
#include <ops/lppool.h>
#include <vector>

int main()
{
    // 从文件读取测试数据
    std::ifstream file("/home/smooth/dev/onnx-ops/src/lppool_test_LpPool_5D_2.json");
    nlohmann::json test_data;
    file >> test_data;

    // 提取测试数据
    std::vector<float> input_data = test_data["input"]["data"].get<std::vector<float>>();
    std::vector<int64_t> input_shape = test_data["input"]["shape"].get<std::vector<int64_t>>();
    std::vector<int64_t> kernel_shape =
        test_data["params"]["kernel_shape"].get<std::vector<int64_t>>();
    std::vector<int64_t> strides = test_data["params"]["strides"].get<std::vector<int64_t>>();
    std::vector<int64_t> pads = test_data["params"]["pads"].get<std::vector<int64_t>>();
    int64_t p = test_data["params"]["p"];
    std::string auto_pad = test_data["params"]["auto_pad"];

    // 调用LpPool实现
    auto [output_data, output_shape] = onnx::LpPool<float>::Compute(
        input_data, input_shape, kernel_shape, strides, pads, auto_pad, p);

    // 验证输出形状
    std::vector<int64_t> expected_shape = test_data["output"]["shape"].get<std::vector<int64_t>>();
    assert(output_shape.size() == expected_shape.size());
    for (size_t i = 0; i < output_shape.size(); ++i)
    {
        {
            (output_shape[i], expected_shape[i]);
        }
    }

    // 验证输出数据
    std::vector<float> expected_data = test_data["output"]["data"].get<std::vector<float>>();
    assert(output_data.size() == expected_data.size());

    // 验证数据值是否接近，考虑浮点误差
    for (size_t i = 0; i < output_data.size(); ++i)
    {
        {
            assert(std::abs(output_data[i] - expected_data[i]) < 1e-4f);
        }
    }
}