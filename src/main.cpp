#include <cassert>
#include <fstream>
#include <iostream>
#include <json.hpp>
#include <ops/lppool.h>
#include <vector>

#include "inc.h"

void test_lppool(std::string_view file_name)
{
    // 从文件读取测试数据
    std::ifstream file(file_name.data());
    nlohmann::json test_data;
    file >> test_data;

    // 提取测试数据
    std::vector<float> input_data = test_data["input"]["data"].get<std::vector<float>>();
    std::vector<int64_t> input_shape = test_data["input"]["shape"].get<std::vector<int64_t>>();
    std::vector<int64_t> kernel_shape =
        test_data["params"]["kernel_shape"].get<std::vector<int64_t>>();
    std::vector<int64_t> strides = test_data["params"]["strides"].get<std::vector<int64_t>>();
    std::vector<int64_t> pads = test_data["params"]["pads"].get<std::vector<int64_t>>();
    std::vector<int64_t> dilations = test_data["params"]["dilations"].get<std::vector<int64_t>>();
    int64_t p = test_data["params"]["p"];
    std::string auto_pad = test_data["params"]["auto_pad"];

    // 调用LpPool实现
    auto [output_data, output_shape] = onnx::LpPool<float>::Compute(
        input_data, input_shape, kernel_shape, strides, pads, auto_pad, p, dilations);

    // 验证输出形状
    std::vector<int64_t> expected_shape = test_data["output"]["shape"].get<std::vector<int64_t>>();
    assert(output_shape.size() == expected_shape.size());
    for (size_t i = 0; i < output_shape.size(); ++i)
    {
        assert(output_shape[i] == expected_shape[i]);
    }

    // 验证输出数据
    std::vector<float> expected_data = test_data["output"]["data"].get<std::vector<float>>();
    assert(output_data.size() == expected_data.size());

    // 验证数据值是否接近，考虑浮点误差
    for (size_t i = 0; i < output_data.size(); ++i)
    {
        assert(std::abs(output_data[i] - expected_data[i]) < 1e-4f);
    }
}

class Box;

void test_non_max_suppression(std::string_view file_name)
{
    // 从文件读取测试数据
    std::ifstream file(file_name.data());
    nlohmann::json test_data;
    file >> test_data;

    // 创建输入boxes
    std::vector<Box> input_boxes;
    auto boxes_data = test_data["input"]["boxes"];
    for (const auto& box : boxes_data)
    {
        {
            input_boxes.emplace_back(box["class_id"].get<int>(), box["x1"].get<float>(),
                                     box["y1"].get<float>(), box["x2"].get<float>(),
                                     box["y2"].get<float>(), box["score"].get<float>());
        }
    }

    float iou_threshold = test_data["input"]["iou_threshold"].get<float>();

    // 调用NMS实现
    auto result_boxes = multiClassNMS(input_boxes, iou_threshold);

    // 验证结果
    // 1. 验证分数是否按降序排列
    for (size_t i = 1; i < result_boxes.size(); ++i)
    {
        {
            assert(result_boxes[i - 1].score >= result_boxes[i].score);
        }
    }

    // 2. 验证同类别框之间的IoU是否都小于阈值
    for (size_t i = 0; i < result_boxes.size(); ++i)
    {
        {
            for (size_t j = i + 1; j < result_boxes.size(); ++j)
            {
                {
                    if (result_boxes[i].class_id == result_boxes[j].class_id)
                    {
                        {
                            assert(computeIoU(result_boxes[i], result_boxes[j]) <= iou_threshold);
                        }
                    }
                }
            }
        }
    }

    // 3. 验证每个输出框的坐标是否合法
    for (const auto& box : result_boxes)
    {
        {
            assert(box.x1 <= box.x2);
            assert(box.y1 <= box.y2);
            assert(box.score >= 0.0f);
            assert(box.score <= 1.0f);
        }
    }

    // 4. 打印结果（可选）
    // std::cout << "Test case {test_name} results:" << std::endl;
    // for (const auto& box : result_boxes)
    // {
    //     {
    //         std::cout << "Class: " << box.class_id << ", Score: " << box.score << ", Box: ["
    //                   << box.x1 << ", " << box.y1 << ", " << box.x2 << ", " << box.y2 << "]"
    //                   << std::endl;
    //     }
    // }
}

void test_sequence_at(std::string_view file_name)
{
    // 从文件读取测试数据
    std::ifstream file(file_name.data());
    if (!file.is_open())
    {
        std::cerr << "无法打开文件: " << file_name << std::endl;
        return;
    }

    nlohmann::json test_data;
    file >> test_data;

    // 打印调试信息
    std::cout << "Reading test data from: " << file_name << std::endl;

    // 验证JSON结构
    if (!test_data.contains("input") || !test_data["input"].contains("sequence") ||
        !test_data["input"].contains("position"))
    {
        std::cerr << "JSON格式错误：缺少必要的字段" << std::endl;
        return;
    }

    // 创建输入序列
    std::vector<Tensor<float>> input_sequence;
    auto& sequence_array = test_data["input"]["sequence"];

    for (const auto& tensor_json : sequence_array)
    {
        if (!tensor_json.contains("shape") || !tensor_json.contains("data"))
        {
            std::cerr << "Tensor JSON格式错误" << std::endl;
            continue;
        }

        auto shape_array = tensor_json["shape"];
        auto data_array = tensor_json["data"];

        std::vector<int64_t> shape;
        std::vector<float> data;

        // 确保shape是数组并且包含数字
        if (shape_array.is_array())
        {
            for (const auto& dim : shape_array)
            {
                if (dim.is_number())
                {
                    shape.push_back(dim.get<int64_t>());
                }
            }
        }

        // 确保data是数组并且包含数字
        if (data_array.is_array())
        {
            for (const auto& val : data_array)
            {
                if (val.is_number())
                {
                    data.push_back(val.get<float>());
                }
            }
        }

        Tensor<float> tensor(shape, data);
        input_sequence.push_back(tensor);
    }

    // 获取position
    if (!test_data["input"]["position"].is_number())
    {
        std::cerr << "Position必须是数字" << std::endl;
        return;
    }
    int pos = test_data["input"]["position"].get<int>();

    // 调用SequenceAt实现
    auto output = SequenceAt(input_sequence, pos);

    // 获取预期输出数据
    int actual_pos = pos;
    if (pos < 0)
    {
        actual_pos += input_sequence.size();
    }

    if (actual_pos < 0 || actual_pos >= input_sequence.size())
    {
        std::cerr << "Invalid position: " << actual_pos << std::endl;
        return;
    }

    // 直接从input_sequence中获取预期的tensor
    const auto& expected_tensor = input_sequence[actual_pos];
    const auto& expected_shape = expected_tensor.getDims();
    const auto& expected_data = expected_tensor.getData();

    // 验证输出形状
    assert(output.getDims().size() == expected_shape.size());
    for (size_t i = 0; i < output.getDims().size(); ++i)
    {
        assert(output.getDims()[i] == expected_shape[i]);
    }

    // 验证输出数据
    const auto& output_data = output.getData();
    assert(output_data.size() == expected_data.size());

    // 验证数据值是否接近，考虑浮点误差
    for (size_t i = 0; i < output_data.size(); ++i)
    {
        assert(std::abs(output_data[i] - expected_data[i]) < 1e-4f);
    }

    std::cout << "SequenceAt test passed: " << file_name << std::endl;
}

void test_sequence_construct(std::string_view file_name)
{
    // 从文件读取测试数据
    std::ifstream file(file_name.data());
    if (!file.is_open())
    {
        std::cerr << "无法打开文件: " << file_name << std::endl;
        return;
    }

    nlohmann::json test_data;
    file >> test_data;

    std::cout << "Reading test data from: " << file_name << std::endl;

    // 验证JSON结构
    if (!test_data.contains("input") || !test_data["input"].contains("tensors"))
    {
        std::cerr << "JSON格式错误：缺少必要的字段" << std::endl;
        return;
    }

    // 创建输入张量
    std::vector<Tensor<float>> tensors;
    auto& tensors_array = test_data["input"]["tensors"];

    for (const auto& tensor_json : tensors_array)
    {
        if (!tensor_json.contains("shape") || !tensor_json.contains("data"))
        {
            std::cerr << "Tensor JSON格式错误" << std::endl;
            continue;
        }

        auto shape_array = tensor_json["shape"];
        auto data_array = tensor_json["data"];

        std::vector<int64_t> shape;
        std::vector<float> data;

        // 确保shape是数组并且包含数字
        if (shape_array.is_array())
        {
            for (const auto& dim : shape_array)
            {
                if (dim.is_number())
                {
                    shape.push_back(dim.get<int64_t>());
                }
            }
        }

        // 确保data是数组并且包含数字
        if (data_array.is_array())
        {
            for (const auto& val : data_array)
            {
                if (val.is_number())
                {
                    data.push_back(val.get<float>());
                }
            }
        }

        tensors.emplace_back(shape, data);
    }

    // 调用SequenceConstruct实现
    std::vector<Tensor<float>> output_sequence;
    if (tensors.size() >= 2)
    {
        output_sequence = SequenceConstruct(tensors[0], tensors[1]);
        for (size_t i = 2; i < tensors.size(); ++i)
        {
            output_sequence.push_back(tensors[i]);
        }
    }

    // 验证输出序列
    assert(output_sequence.size() == tensors.size());
    for (size_t i = 0; i < output_sequence.size(); ++i)
    {
        const auto& output_tensor = output_sequence[i];
        const auto& input_tensor = tensors[i];

        // 验证形状
        assert(output_tensor.getDims() == input_tensor.getDims());

        // 验证数据
        const auto& output_data = output_tensor.getData();
        const auto& input_data = input_tensor.getData();
        assert(output_data.size() == input_data.size());

        for (size_t j = 0; j < output_data.size(); ++j)
        {
            assert(std::abs(output_data[j] - input_data[j]) < 1e-4f);
        }
    }

    std::cout << "SequenceConstruct test passed: " << file_name << std::endl;
}

void test_sequence_insert(std::string_view file_name)
{
    // 从文件读取测试数据
    std::ifstream file(file_name.data());
    if (!file.is_open())
    {
        std::cerr << "无法打开文件: " << file_name << std::endl;
        return;
    }

    nlohmann::json test_data;
    file >> test_data;

    std::cout << "Reading test data from: " << file_name << std::endl;

    // 验证JSON结构
    if (!test_data.contains("input") || !test_data["input"].contains("sequence") ||
        !test_data["input"].contains("tensor"))
    {
        std::cerr << "JSON格式错误：缺少必要的字段" << std::endl;
        return;
    }

    // 创建输入序列
    std::vector<Tensor<float>> input_sequence;
    auto& sequence_array = test_data["input"]["sequence"];

    for (const auto& tensor_json : sequence_array)
    {
        if (!tensor_json.contains("shape") || !tensor_json.contains("data"))
        {
            std::cerr << "Tensor JSON格式错误" << std::endl;
            continue;
        }

        auto shape_array = tensor_json["shape"];
        auto data_array = tensor_json["data"];

        std::vector<int64_t> shape;
        std::vector<float> data;

        // 确保shape是数组并且包含数字
        if (shape_array.is_array())
        {
            for (const auto& dim : shape_array)
            {
                if (dim.is_number())
                {
                    shape.push_back(dim.get<int64_t>());
                }
            }
        }

        // 确保data是数组并且包含数字
        if (data_array.is_array())
        {
            for (const auto& val : data_array)
            {
                if (val.is_number())
                {
                    data.push_back(val.get<float>());
                }
            }
        }

        input_sequence.emplace_back(shape, data);
    }

    // 创建要插入的张量
    auto tensor_json = test_data["input"]["tensor"];
    std::vector<int64_t> shape;
    std::vector<float> data;

    auto shape_array = tensor_json["shape"];
    auto data_array = tensor_json["data"];

    // 确保shape是数组并且包含数字
    if (shape_array.is_array())
    {
        for (const auto& dim : shape_array)
        {
            if (dim.is_number())
            {
                shape.push_back(dim.get<int64_t>());
            }
        }
    }

    // 确保data是数组并且包含数字
    if (data_array.is_array())
    {
        for (const auto& val : data_array)
        {
            if (val.is_number())
            {
                data.push_back(val.get<float>());
            }
        }
    }

    Tensor<float> insert_tensor(shape, data);

    // 获取插入位置（如果有）
    auto sequence_before = input_sequence; // 保存原始序列副本
    if (test_data["input"].contains("position") && !test_data["input"]["position"].is_null())
    {
        int position = test_data["input"]["position"].get<int>();
        SequenceInsert(input_sequence, insert_tensor, position);
    }
    else
    {
        SequenceInsert(input_sequence, insert_tensor);
    }

    // 验证序列长度增加了1
    assert(input_sequence.size() == sequence_before.size() + 1);

    std::cout << "SequenceInsert test passed: " << file_name << std::endl;
}

void test_sequence_erase(std::string_view file_name)
{
    // 从文件读取测试数据
    std::ifstream file(file_name.data());
    if (!file.is_open())
    {
        std::cerr << "无法打开文件: " << file_name << std::endl;
        return;
    }

    nlohmann::json test_data;
    file >> test_data;

    std::cout << "Reading test data from: " << file_name << std::endl;

    // 验证JSON结构
    if (!test_data.contains("input") || !test_data["input"].contains("sequence"))
    {
        std::cerr << "JSON格式错误：缺少必要的字段" << std::endl;
        return;
    }

    // 创建输入序列
    std::vector<Tensor<float>> input_sequence;
    auto& sequence_array = test_data["input"]["sequence"];

    for (const auto& tensor_json : sequence_array)
    {
        if (!tensor_json.contains("shape") || !tensor_json.contains("data"))
        {
            std::cerr << "Tensor JSON格式错误" << std::endl;
            continue;
        }

        auto shape_array = tensor_json["shape"];
        auto data_array = tensor_json["data"];

        std::vector<int64_t> shape;
        std::vector<float> data;

        // 确保shape是数组并且包含数字
        if (shape_array.is_array())
        {
            for (const auto& dim : shape_array)
            {
                if (dim.is_number())
                {
                    shape.push_back(dim.get<int64_t>());
                }
            }
        }

        // 确保data是数组并且包含数字
        if (data_array.is_array())
        {
            for (const auto& val : data_array)
            {
                if (val.is_number())
                {
                    data.push_back(val.get<float>());
                }
            }
        }

        input_sequence.emplace_back(shape, data);
    }

    // 保存原始序列副本
    auto sequence_before = input_sequence;

    // 获取删除位置（如果有）
    if (test_data["input"].contains("position") && !test_data["input"]["position"].is_null())
    {
        int position = test_data["input"]["position"].get<int>();
        SequenceErase(input_sequence, position);
    }
    else
    {
        SequenceErase(input_sequence);
    }

    // 验证序列长度减少了1
    assert(input_sequence.size() == sequence_before.size() - 1);

    std::cout << "SequenceErase test passed: " << file_name << std::endl;
}

int main()
{
    test_lppool("/home/smooth/dev/onnx-ops/py/lppool_test/lppool_test_LpPool_3D_1.json");
    test_lppool("/home/smooth/dev/onnx-ops/py/lppool_test/lppool_test_LpPool_3D_2.json");
    test_lppool("/home/smooth/dev/onnx-ops/py/lppool_test/lppool_test_LpPool_3D_3.json");
    test_lppool("/home/smooth/dev/onnx-ops/py/lppool_test/lppool_test_LpPool_4D_1.json");
    test_lppool("/home/smooth/dev/onnx-ops/py/lppool_test/lppool_test_LpPool_4D_2.json");
    test_lppool("/home/smooth/dev/onnx-ops/py/lppool_test/lppool_test_LpPool_4D_3.json");
    test_lppool("/home/smooth/dev/onnx-ops/py/lppool_test/lppool_test_LpPool_5D_1.json");
    test_lppool("/home/smooth/dev/onnx-ops/py/lppool_test/lppool_test_LpPool_5D_2.json");
    test_lppool("/home/smooth/dev/onnx-ops/py/lppool_test/lppool_test_LpPool_5D_3.json");

    // test_non_max_suppression(
    //     "/home/smooth/dev/onnx-ops/py/nms_test/nms_test_NonMaxSuppression_Test_1.json");
    // test_non_max_suppression(
    //     "/home/smooth/dev/onnx-ops/py/nms_test/nms_test_NonMaxSuppression_Test_2.json");
    // test_non_max_suppression(
    //     "/home/smooth/dev/onnx-ops/py/nms_test/nms_test_NonMaxSuppression_Test_3.json");

    // 运行SequenceAt测试
    // std::cout << "\nTesting SequenceAt operator..." << std::endl;
    // for (int i = 1; i <= 3; ++i)
    // {
    //     std::string filename =
    //         std::string(
    //             "../py/sequence_at_test/sequence_at_test_SequenceAt_Test_") +
    //         std::to_string(i) + ".json";
    //     test_sequence_at(filename);
    // }

    // // 运行SequenceConstruct测试
    // std::cout << "\nTesting SequenceConstruct operator..." << std::endl;
    // for (int i = 1; i <= 3; ++i) {
    //     std::string filename =
    //     std::string("../py/sequence_construct_test/sequence_construct_test_SequenceConstruct_Test_")
    //     +
    //                          std::to_string(i) + ".json";
    //     test_sequence_construct(filename);
    // }

    // // 运行SequenceInsert测试
    // std::cout << "\nTesting SequenceInsert operator..." << std::endl;
    // for (int i = 1; i <= 3; ++i) {
    //     std::string filename =
    //     std::string("../py/sequence_insert_test/sequence_insert_test_SequenceInsert_Test_") +
    //                          std::to_string(i) + ".json";
    //     test_sequence_insert(filename);
    // }

    // // 运行SequenceErase测试
    // std::cout << "\nTesting SequenceErase operator..." << std::endl;
    // for (int i = 1; i <= 3; ++i) {
    //     std::string filename =
    //     std::string("../py/sequence_erase_test/sequence_erase_test_SequenceErase_Test_") +
    //                          std::to_string(i) + ".json";
    //     test_sequence_erase(filename);
    // }

    return 0;
}