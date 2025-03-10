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
            assert(output_shape[i] == expected_shape[i]);
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

int main()
{
    // test_lppool("/home/smooth/dev/onnx-ops/py/lppool_test/lppool_test_LpPool_3D_1.json");
    // test_lppool("/home/smooth/dev/onnx-ops/py/lppool_test/lppool_test_LpPool_3D_2.json");
    // test_lppool("/home/smooth/dev/onnx-ops/py/lppool_test/lppool_test_LpPool_3D_3.json");
    // test_lppool("/home/smooth/dev/onnx-ops/py/lppool_test/lppool_test_LpPool_4D_1.json");
    // test_lppool("/home/smooth/dev/onnx-ops/py/lppool_test/lppool_test_LpPool_4D_2.json");
    // test_lppool("/home/smooth/dev/onnx-ops/py/lppool_test/lppool_test_LpPool_4D_3.json");
    // test_lppool("/home/smooth/dev/onnx-ops/py/lppool_test/lppool_test_LpPool_5D_1.json");
    // test_lppool("/home/smooth/dev/onnx-ops/py/lppool_test/lppool_test_LpPool_5D_2.json");
    // test_lppool("/home/smooth/dev/onnx-ops/py/lppool_test/lppool_test_LpPool_5D_3.json");

    test_non_max_suppression(
        "/home/smooth/dev/onnx-ops/py/nms_test/nms_test_NonMaxSuppression_Test_1.json");
    test_non_max_suppression(
        "/home/smooth/dev/onnx-ops/py/nms_test/nms_test_NonMaxSuppression_Test_2.json");
    test_non_max_suppression(
        "/home/smooth/dev/onnx-ops/py/nms_test/nms_test_NonMaxSuppression_Test_3.json");
}