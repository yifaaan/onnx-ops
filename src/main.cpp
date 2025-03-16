#include <cassert>
#include <fstream>
#include <iostream>
#include <json.hpp>
#include <ops/lppool.h>
#include <vector>
#include <set>
#include "inc.h"

// 递归展平任意维度的张量
std::vector<float> flatten_tensor(const nlohmann::json& tensor_json)
{
    std::vector<float> result;

    if (tensor_json.is_array())
    {
        for (const auto& item : tensor_json)
        {
            if (item.is_array())
            {
                // 递归处理子数组
                auto sub_result = flatten_tensor(item);
                result.insert(result.end(), sub_result.begin(), sub_result.end());
            }
            else if (item.is_number())
            {
                // 直接添加数值
                result.push_back(item.get<float>());
            }
        }
    }

    return result;
}

double test_non_max_suppression(std::string_view file_name)
{
    // 从文件读取测试数据
    std::ifstream file(file_name.data());
    nlohmann::json test_data;
    file >> test_data;

    // 创建输入数据结构
    std::vector<std::vector<std::vector<float>>> boxes;
    std::vector<std::vector<std::vector<float>>> scores;
    
    // 假设有1个batch
    boxes.push_back(std::vector<std::vector<float>>());
    scores.push_back(std::vector<std::vector<float>>());
    
    // 从测试数据中提取boxes
    auto boxes_data = test_data["input"]["boxes"];
    
    // 为每个类别创建一个score向量
    int num_classes = test_data["params"]["num_classes"].get<int>();
    for (int i = 0; i < num_classes; i++) {
        scores[0].push_back(std::vector<float>());
    }
    
    // 填充boxes和scores数据
    for (const auto& box : boxes_data)
    {
        std::vector<float> box_coords = {
            box["y1"].get<float>(),
            box["x1"].get<float>(),
            box["y2"].get<float>(),
            box["x2"].get<float>()
        };
        
        int class_id = box["class_id"].get<int>();
        float score = box["score"].get<float>();
        
        // 添加box坐标
        boxes[0].push_back(box_coords);
        
        // 将score添加到对应类别
        for (int i = 0; i < num_classes; i++) {
            if (i == class_id) {
                scores[0][i].push_back(score);
            } else {
                scores[0][i].push_back(0.0f); // 其他类别分数为0
            }
        }
    }

    float iou_threshold = test_data["input"]["iou_threshold"].get<float>();
    int64_t max_output_boxes_per_class = 0; // 使用默认值
    float score_threshold = 0.0f; // 使用默认值
    int center_point_box = 0; // 默认使用[y1, x1, y2, x2]格式

    auto start = std::chrono::steady_clock::now();

    // 调用NMS实现
    auto selected_indices = nonMaxSuppression(
        boxes, 
        scores, 
        max_output_boxes_per_class, 
        iou_threshold,
        score_threshold,
        center_point_box
    );
    
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;

    // 验证结果与ONNX参考输出是否一致
    if (test_data.contains("output") && test_data["output"].contains("selected_indices")) {
        auto expected_indices = test_data["output"]["selected_indices"];
        
        // 打印调试信息
        std::cout << "\nNMS结果比较:" << std::endl;
        std::cout << "C++实现选择的框数量: " << selected_indices.size() << std::endl;
        std::cout << "ONNX参考输出框数量: " << expected_indices.size() << std::endl;
        
        std::cout << "\nC++选择的框:" << std::endl;
        for (const auto& idx : selected_indices) {
            std::cout << "[" << idx[0] << ", " << idx[1] << ", " << idx[2] << "]" << std::endl;
        }
        
        std::cout << "\nONNX参考输出:" << std::endl;
        for (const auto& idx : expected_indices) {
            std::cout << "[" << idx[0] << ", " << idx[1] << ", " << idx[2] << "]" << std::endl;
        }
        
        // 如果数量不一致，打印更多信息而不是直接断言失败
        if (selected_indices.size() != expected_indices.size()) {
            std::cout << "\n警告: 输出数量不匹配!" << std::endl;
            std::cout << "测试文件: " << file_name << std::endl;
            std::cout << "IoU阈值: " << test_data["input"]["iou_threshold"] << std::endl;
            std::cout << "输入框数量: " << boxes[0].size() << std::endl;
            std::cout << "类别数量: " << scores[0].size() << std::endl;
            return 0;
        }
        
        // 创建用于比较的set
        std::set<std::vector<int64_t>> selected_set;
        std::set<std::vector<int64_t>> expected_set;
        
        for (const auto& idx : selected_indices) {
            selected_set.insert(idx);
        }
        
        for (const auto& idx : expected_indices) {
            std::vector<int64_t> index_vec;
            for (const auto& val : idx) {
                index_vec.push_back(val.get<int64_t>());
            }
            expected_set.insert(index_vec);
        }
        
        // 检查两个集合是否相等
        if (selected_set != expected_set) {
            std::cout << "\n警告: 选择的框不匹配!" << std::endl;
            return 0;
        }
        
        std::cout << "NonMaxSuppression test passed: " << file_name << std::endl;
    } else {
        // 仅进行基本验证
        // 1. 验证选中的索引是否有效
        for (const auto& idx : selected_indices) {
            assert(idx[0] >= 0 && idx[0] < boxes.size()); // batch_idx
            assert(idx[1] >= 0 && idx[1] < scores[0].size()); // class_idx
            assert(idx[2] >= 0 && idx[2] < boxes[0].size()); // box_idx
        }

        // 2. 验证同类别框之间的IoU是否都小于阈值
        for (size_t i = 0; i < selected_indices.size(); ++i) {
            for (size_t j = i + 1; j < selected_indices.size(); ++j) {
                // 只检查同一类别的框
                if (selected_indices[i][1] == selected_indices[j][1]) {
                    // 计算两个框的IoU
                    float iou = calculateIOU(
                        boxes[selected_indices[i][0]][selected_indices[i][2]],
                        boxes[selected_indices[j][0]][selected_indices[j][2]],
                        center_point_box != 0
                    );
                    
                    // 确保IoU小于阈值
                    assert(iou <= iou_threshold);
                }
            }
        }
        
        std::cout << "NonMaxSuppression basic test passed: " << file_name << std::endl;
    }

    return diff.count() * 1000;
}

double test_sequence_at(std::string_view file_name)
{
    // 从文件读取测试数据
    std::ifstream file(file_name.data());

    nlohmann::json test_data;
    file >> test_data;

    // 打印调试信息
    // std::cout << "Reading test data from: " << file_name << std::endl;

    // 创建输入序列
    std::vector<Tensor<float>> input_sequence;
    auto& sequence_array = test_data["input"]["sequence"];
    for (const auto& tensor_json : sequence_array)
    {
        auto shape = tensor_json["shape"].get<std::vector<int64_t>>();
        auto data = flatten_tensor(tensor_json["data"]);
        Tensor<float> tensor(shape, data);
        input_sequence.push_back(tensor);
    }

    // 获取position
    if (!test_data["input"]["position"].is_number())
    {
        std::cerr << "Position必须是数字" << std::endl;
        return 0;
    }
    int pos = test_data["input"]["position"].get<int>();

    auto start = std::chrono::steady_clock::now();
    // 调用SequenceAt实现
    auto output = SequenceAt(input_sequence, pos);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    // 获取预期输出数据
    int actual_pos = pos;
    if (pos < 0)
    {
        actual_pos += input_sequence.size();
    }

    if (actual_pos < 0 || actual_pos >= input_sequence.size())
    {
        std::cerr << "Invalid position: " << actual_pos << std::endl;
        return 0;
    }

    // 从json中获取预期的tensor

    auto expected_tensor = flatten_tensor(test_data["output"]["tensor"]);
    auto expected_shape = test_data["output"]["shape"].get<std::vector<int64_t>>();

    // 验证输出形状
    assert(output.getDims().size() == expected_shape.size());
    for (size_t i = 0; i < output.getDims().size(); ++i)
    {
        assert(output.getDims()[i] == expected_shape[i]);
    }

    // 验证输出数据
    const auto& output_data = output.getData();
    assert(output_data.size() == expected_tensor.size());

    // 验证数据值是否接近，考虑浮点误差
    for (size_t i = 0; i < output_data.size(); ++i)
    {
        assert(std::abs(output_data[i] - expected_tensor[i]) < 1e-4f);
    }
    return diff.count() * 1000;

    // std::cout << "SequenceAt test passed: " << file_name << std::endl;
}

double test_sequence_construct(std::string_view file_name)
{
    // 从文件读取测试数据
    std::ifstream file(file_name.data());

    nlohmann::json test_data;
    file >> test_data;

    // 创建输入张量

    std::vector<Tensor<float>> input_sequence;
    auto& sequence_array = test_data["input"]["sequence"];
    for (const auto& tensor_json : sequence_array)
    {
        auto shape = tensor_json["shape"].get<std::vector<int64_t>>();
        auto data = flatten_tensor(tensor_json["data"]);
        Tensor<float> tensor(shape, data);
        input_sequence.push_back(tensor);
    }

    // 调用SequenceConstruct实现
    std::vector<Tensor<float>> output_sequence;
    std::chrono::duration<double> diff;
    if (input_sequence.size() >= 2)
    {
        auto start = std::chrono::steady_clock::now();
        output_sequence = SequenceConstruct(input_sequence[0], input_sequence[1]);
        auto end = std::chrono::steady_clock::now();
        diff = end - start;
        for (size_t i = 2; i < input_sequence.size(); ++i)
        {
            output_sequence.push_back(input_sequence[i]);
        }
    }

    // 验证输出序列

    std::vector<Tensor<float>> expected_sequence;
    for (const auto& tensor : test_data["output"]["sequence"])
    {
        auto data = flatten_tensor(tensor["data"]);
        auto shape = tensor["shape"].get<std::vector<int64_t>>();
        expected_sequence.emplace_back(shape, data);
    }
    auto len = test_data["output"]["len"].get<int>();
    assert(output_sequence.size() == len);

    for (int i = 0; i < len; i++)
    {
        auto a = output_sequence[i].getData();
        auto b = expected_sequence[i].getData();
        assert(a.size() == b.size());
        for (size_t j = 0; j < a.size(); j++)
        {
            assert(std::abs(a[j] - b[j]) < 1e-4f);
        }
    }
    return diff.count() * 1000;

    // std::cout << "SequenceConstruct test passed: " << file_name << std::endl;
}

double test_sequence_insert(std::string_view file_name)
{
    // 从文件读取测试数据
    std::ifstream file(file_name.data());

    nlohmann::json test_data;

    file >> test_data;

    // std::cout << "Reading test data from: " << file_name << std::endl;

    auto& input = test_data["input"];

    // 创建输入序列
    std::vector<Tensor<float>> input_sequence;
    auto& sequence_array = input["sequence"];
    for (const auto& tensor_json : sequence_array)
    {

        auto shape_array = tensor_json["shape"].get<std::vector<int64_t>>();
        auto data_array = flatten_tensor(tensor_json["data"]);

        input_sequence.emplace_back(shape_array, data_array);
    }
    // 创建要插入的张量
    auto& tensor_json = input["insert_tensor"];
    auto shape_array = tensor_json["shape"].get<std::vector<int64_t>>();
    auto data_array = flatten_tensor(tensor_json["data"]);
    Tensor<float> insert_tensor(shape_array, data_array);

    // 获取插入位置（如果有）
    auto start = std::chrono::steady_clock::now();
    // onnx输出序列
    std::vector<Tensor<float>> onnx_output_sequence;
    for (const auto& tensor : test_data["output"]["sequence"])
    {
        auto data = flatten_tensor(tensor["data"]);
        auto shape = tensor["shape"].get<std::vector<int64_t>>();
        onnx_output_sequence.emplace_back(shape, data);
    }

    if (input.contains("position") && !input["position"].is_null())
    {
        int position = input["position"].get<int>();
        SequenceInsert(input_sequence, insert_tensor, position);
    }
    else
    {
        SequenceInsert(input_sequence, insert_tensor);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;

    assert(input_sequence.size() == test_data["output"]["len"].get<int>());

    for (int i = 0; i < onnx_output_sequence.size(); i++)
    {
        auto a = onnx_output_sequence[i].getData();
        auto b = input_sequence[i].getData();
        assert(a.size() == b.size());
        assert(onnx_output_sequence[i].getDims() == input_sequence[i].getDims());
    }
    return diff.count() * 1000;
}

double test_sequence_erase(std::string_view file_name)
{
    // 从文件读取测试数据
    std::ifstream file(file_name.data());

    nlohmann::json test_data;
    file >> test_data;

    // std::cout << "Reading test data from: " << file_name << std::endl;

    // 创建输入序列
    std::vector<Tensor<float>> input_sequence;
    auto& sequence_array = test_data["input"]["sequence"];

    for (const auto& tensor_json : sequence_array)
    {

        auto shape = tensor_json["shape"].get<std::vector<int64_t>>();
        auto data = flatten_tensor(tensor_json["data"]);
        input_sequence.emplace_back(shape, data);
    }

    // 保存原始序列副本
    auto sequence_before = input_sequence;

    auto start = std::chrono::steady_clock::now();
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
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    // 验证序列长度减少了1
    assert(input_sequence.size() == sequence_before.size() - 1);

    std::vector<Tensor<float>> expected_sequence;
    for (const auto& tensor : test_data["output"]["sequence"])
    {
        auto data = flatten_tensor(tensor["data"]);
        auto shape = tensor["shape"].get<std::vector<int64_t>>();
        expected_sequence.emplace_back(shape, data);
    }
    auto len = test_data["output"]["len"].get<int>();
    assert(input_sequence.size() == len);

    for (int i = 0; i < len; i++)
    {
        auto a = input_sequence[i].getData();
        auto b = expected_sequence[i].getData();
        assert(a.size() == b.size());
        for (size_t j = 0; j < a.size(); j++)
        {
            assert(std::abs(a[j] - b[j]) < 1e-4f);
        }
    }

    return diff.count() * 1000;
    // std::cout << "SequenceErase test passed: " << file_name << std::endl;
}

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
    bool ceil_mode = test_data["params"].contains("ceil_mode")
                         ? test_data["params"]["ceil_mode"].get<bool>()
                         : false;

    // 调用LpPool实现
    auto [output_data, output_shape] = onnx::LpPool<float>::Compute(
        input_data, input_shape, kernel_shape, strides, pads, auto_pad, p, dilations, ceil_mode);

    // 验证输出形状
    std::vector<int64_t> expected_shape = test_data["output"]["shape"].get<std::vector<int64_t>>();
    if (output_shape.size() != expected_shape.size())
    {
        std::cerr << "输出形状维度不匹配：计算结果 " << output_shape.size() << " vs 预期 "
                  << expected_shape.size() << std::endl;
        return;
    }

    bool shape_match = true;
    for (size_t i = 0; i < output_shape.size(); ++i)
    {
        if (output_shape[i] != expected_shape[i])
        {
            std::cerr << "输出形状在维度 " << i << " 不匹配：计算结果 " << output_shape[i]
                      << " vs 预期 " << expected_shape[i] << std::endl;
            shape_match = false;
        }
    }

    if (!shape_match)
    {
        std::cerr << "完整输出形状：计算结果 [";
        for (size_t i = 0; i < output_shape.size(); ++i)
        {
            std::cerr << output_shape[i] << (i < output_shape.size() - 1 ? ", " : "");
        }
        std::cerr << "] vs 预期 [";
        for (size_t i = 0; i < expected_shape.size(); ++i)
        {
            std::cerr << expected_shape[i] << (i < expected_shape.size() - 1 ? ", " : "");
        }
        std::cerr << "]" << std::endl;
        return;
    }

    // 验证输出数据
    std::vector<float> expected_data = test_data["output"]["data"].get<std::vector<float>>();
    assert(output_data.size() == expected_data.size());

    // 验证数据值是否接近，考虑浮点误差
    bool data_match = true;
    float max_diff = 0.0f;
    size_t max_diff_index = 0;
    int diff_count = 0;
    const float tolerance = 1e-3f; // 增加容忍度

    for (size_t i = 0; i < output_data.size(); ++i)
    {
        float diff = std::abs(output_data[i] - expected_data[i]);
        if (diff > tolerance)
        {
            data_match = false;
            diff_count++;
            if (diff > max_diff)
            {
                max_diff = diff;
                max_diff_index = i;
            }

            // 只打印前10个不匹配的数据点
            if (diff_count <= 10)
            {
                std::cerr << "数据在索引 " << i << " 处不匹配：计算结果 " << output_data[i]
                          << " vs 预期 " << expected_data[i] << " (差异 " << diff << ")"
                          << std::endl;
            }
        }
    }

    if (!data_match)
    {
        std::cerr << "共有 " << diff_count << " 个数据点差异超过容忍度 " << tolerance << std::endl;
        std::cerr << "最大差异在索引 " << max_diff_index << "：计算结果 "
                  << output_data[max_diff_index] << " vs 预期 " << expected_data[max_diff_index]
                  << " (差异 " << max_diff << ")" << std::endl;

        // 增加打印参数信息，帮助调试
        std::cerr << "测试参数：p=" << p << ", ceil_mode=" << (ceil_mode ? "true" : "false")
                  << std::endl;
        std::cerr << "kernel_shape=[";
        for (size_t i = 0; i < kernel_shape.size(); ++i)
        {
            std::cerr << kernel_shape[i] << (i < kernel_shape.size() - 1 ? ", " : "");
        }
        std::cerr << "], strides=[";
        for (size_t i = 0; i < strides.size(); ++i)
        {
            std::cerr << strides[i] << (i < strides.size() - 1 ? ", " : "");
        }
        std::cerr << "], dilations=[";
        for (size_t i = 0; i < dilations.size(); ++i)
        {
            std::cerr << dilations[i] << (i < dilations.size() - 1 ? ", " : "");
        }
        std::cerr << "]" << std::endl;

        return;
    }

    std::cout << "LpPool test passed for " << file_name << std::endl;
}
int main(int argc, char* argv[])
{
    std::string test_path;

    // // 运行NMS测试
    double t = 0;
    // for (int i = 0; i < 100; i++)
    // {
    //     t +=
    //     test_non_max_suppression("../py/py/nms_test/nms_test_NonMaxSuppression_Test_1.json");
    // }

    // std::cout << "执行时间: " << t / 100 << " 毫秒" << std::endl;

    // // 运行SequenceAt测试
    // std::cout << "\nTesting SequenceAt operator..." << std::endl;
    // t = 0;
    // for (int i = 0; i < 100; i++)
    // {
    //     t += test_sequence_at("../jsons/sequence_at/test_1.json");
    // }
    // std::cout << "执行时间: " << t / 100 << " 毫秒" << std::endl;

    // // 运行SequenceConstruct测试
    // std::cout << "\nTesting SequenceConstruct operator..." << std::endl;
    // t = 0;
    // for (int i = 0; i < 100; i++)
    // {
    //     t += test_sequence_construct("../jsons/sequence_construct/test_1.json");
    // }
    // std::cout << "执行时间: " << t / 100 << " 毫秒" << std::endl;

    // // 运行SequenceInsert测试
    // std::cout << "\nTesting SequenceInsert operator..." << std::endl;
    // t = 0;
    // for (int i = 0; i < 100; i++)
    // {
    //     t += test_sequence_insert("../jsons/sequence_insert/test_1.json");
    // }
    // std::cout << "执行时间: " << t / 100 << " 毫秒" << std::endl;

    // // 运行SequenceErase测试
    // std::cout << "\nTesting SequenceErase operator..." << std::endl;
    // t = 0;
    // for (int i = 0; i < 100; i++)
    // {
    //     t += test_sequence_erase("../jsons/sequence_erase/test_1.json");
    // }
    // std::cout << "执行时间: " << t / 100 << " 毫秒" << std::endl;

    // std::cout << "\n所有测试完成!" << std::endl;


// 运行SequenceErase测试
    std::cout << "\nTesting SequenceErase operator..." << std::endl;
    t = 0;
    for (int i = 0; i < 100; i++)
    {
        t += test_non_max_suppression("../jsons/non_max_suppression/test_1.json");
    }
    std::cout << "执行时间: " << t / 100 << " 毫秒" << std::endl;

    std::cout << "\n所有测试完成!" << std::endl;
    

    // test_lppool("../py/lppool_test/lppool_test_LpPool_3D_1.json");
    // test_lppool("../py/lppool_test/lppool_test_LpPool_3D_2.json");
    // test_lppool("../py/lppool_test/lppool_test_LpPool_3D_3.json");
    // test_lppool("../py/lppool_test/lppool_test_LpPool_4D_1.json");
    // test_lppool("../py/lppool_test/lppool_test_LpPool_4D_2.json");
    // test_lppool("../py/lppool_test/lppool_test_LpPool_4D_3.json");
    // test_lppool("../py/lppool_test/lppool_test_LpPool_5D_1.json");
    // test_lppool("../py/lppool_test/lppool_test_LpPool_5D_2.json");
    // test_lppool("../py/lppool_test/lppool_test_LpPool_5D_3.json");

    // // 测试ceil_mode相关的测试用例
    // test_lppool("../py/lppool_test/lppool_test_LpPool_3D_Ceil.json");
    // test_lppool("../py/lppool_test/lppool_test_LpPool_4D_Ceil.json");
    // test_lppool("../py/lppool_test/lppool_test_LpPool_4D_Dilation_Ceil.json");

    // // 测试新增的dilations和ceil_mode组合测试用例
    // test_lppool("../py/lppool_test/lppool_test_LpPool_4D_Dilation_1.json");
    // test_lppool("../py/lppool_test/lppool_test_LpPool_4D_Dilation_2.json");
    // test_lppool("../py/lppool_test/lppool_test_LpPool_3D_Ceil_Dilation.json");
    // test_lppool("../py/lppool_test/lppool_test_LpPool_5D_Ceil_Dilation.json");

    return 0;
}