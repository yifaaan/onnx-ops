#pragma once

#include "utils.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace onnx
{

template <typename T>
class LpPool
{
public:
    static std::pair<std::vector<T>, std::vector<int64_t>>
    Compute(const std::vector<T>& data, const std::vector<int64_t>& shape,
            const std::vector<int64_t>& kernel_shape, const std::vector<int64_t>& strides = {},
            const std::vector<int64_t>& pads = {}, const std::string& auto_pad = "NOTSET",
            int64_t p = 2, const std::vector<int64_t>& dilations = {}, bool ceil_mode = false)
    {
        // 验证输入参数
        if (shape.size() < 2)
        {
            throw std::invalid_argument("LpPool: input tensor must have at least 2 dimensions");
        }
        if (kernel_shape.empty())
        {
            throw std::invalid_argument("LpPool: kernel_shape must be specified");
        }
        if (p < 1)
        {
            throw std::invalid_argument("LpPool: p must be greater than or equal to 1");
        }

        // 空间维度数量
        size_t spatial_dims = shape.size() - 2;

        // 验证kernel_shape维度匹配空间维度
        if (kernel_shape.size() != spatial_dims)
        {
            throw std::invalid_argument(
                "LpPool: kernel_shape must match the number of spatial dimensions");
        }

        // 设置默认strides
        std::vector<int64_t> working_strides = strides;
        if (working_strides.empty())
        {
            working_strides.resize(spatial_dims, 1);
        }
        else if (working_strides.size() != spatial_dims)
        {
            throw std::invalid_argument(
                "LpPool: strides must match the number of spatial dimensions");
        }

        // 设置默认dilations
        std::vector<int64_t> working_dilations = dilations;
        if (working_dilations.empty())
        {
            working_dilations.resize(spatial_dims, 1);
        }
        else if (working_dilations.size() != spatial_dims)
        {
            throw std::invalid_argument(
                "LpPool: dilations must match the number of spatial dimensions");
        }

        // 设置默认pads
        std::vector<int64_t> working_pads = pads;
        if (working_pads.empty() && auto_pad == "NOTSET")
        {
            working_pads.resize(spatial_dims * 2, 0);
        }
        else if (working_pads.size() != spatial_dims * 2 && auto_pad == "NOTSET")
        {
            throw std::invalid_argument(
                "LpPool: pads must be twice the number of spatial dimensions");
        }

        // 计算输出形状
        std::vector<int64_t> output_shape = {shape[0], shape[1]}; // batch_size和channels保持不变

        // 计算空间维度的输出大小
        for (size_t i = 0; i < spatial_dims; ++i)
        {
            int64_t input_dim = shape[i + 2];
            int64_t kernel_dim = kernel_shape[i];
            int64_t stride = working_strides[i];
            int64_t dilation = working_dilations[i];
            int64_t pad_head = 0;
            int64_t pad_tail = 0;

            if (auto_pad == "NOTSET")
            {
                pad_head = working_pads[i];
                pad_tail = working_pads[i + spatial_dims];
            }
            else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER")
            {
                int64_t output_dim = (input_dim + stride - 1) / stride;
                int64_t padding_needed =
                    (output_dim - 1) * stride + (dilation * (kernel_dim - 1) + 1) - input_dim;
                padding_needed = std::max<int64_t>(0, padding_needed);

                if (auto_pad == "SAME_UPPER")
                {
                    pad_head = padding_needed / 2;
                    pad_tail = padding_needed - pad_head;
                }
                else // "SAME_LOWER"
                {
                    pad_tail = padding_needed / 2;
                    pad_head = padding_needed - pad_tail;
                }

                // 更新working_pads用于后续计算
                working_pads.resize(spatial_dims * 2, 0);
                working_pads[i] = pad_head;
                working_pads[i + spatial_dims] = pad_tail;
            }

            // 考虑dilation的输出维度计算
            int64_t dilated_kernel_size = dilation * (kernel_dim - 1) + 1;
            int64_t output_dim;

            // 根据ONNX规范计算输出维度
            if (ceil_mode)
            {
                // 使用向上取整 (ceil) 函数
                double temp =
                    static_cast<double>(input_dim + pad_head + pad_tail - dilated_kernel_size) /
                        stride +
                    1;
                output_dim = static_cast<int64_t>(std::ceil(temp));

                // 关键修复：确保最后一个池化窗口的起始位置不会超出输入边界
                // 参考ONNX规范：如果最后一个窗口超出了边界，则减少输出维度
                if ((output_dim - 1) * stride >= input_dim + pad_head)
                {
                    output_dim -= 1;
                }
            }
            else
            {
                // 使用向下取整 (floor) 函数（默认行为）
                double temp =
                    static_cast<double>(input_dim + pad_head + pad_tail - dilated_kernel_size) /
                        stride +
                    1;
                output_dim = static_cast<int64_t>(std::floor(temp));
            }

            // 确保输出维度至少为1
            output_dim = std::max<int64_t>(1, output_dim);

            output_shape.push_back(output_dim);
        }

        // 计算kernel中的元素个数
        int64_t kernel_size = 1;
        for (auto k : kernel_shape)
        {
            kernel_size *= k;
        }

        // 创建输出数据
        size_t output_size = 1;
        for (auto dim : output_shape)
        {
            output_size *= dim;
        }
        std::vector<T> output_data(output_size, 0);

        // 计算每个维度的步长(stride)
        std::vector<int64_t> input_strides(shape.size());
        input_strides[shape.size() - 1] = 1;
        for (int64_t i = shape.size() - 2; i >= 0; --i)
        {
            input_strides[i] = input_strides[i + 1] * shape[i + 1];
        }

        std::vector<int64_t> output_strides(output_shape.size());
        output_strides[output_shape.size() - 1] = 1;
        for (int64_t i = output_shape.size() - 2; i >= 0; --i)
        {
            output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
        }

        // 为每个batch和channel并行处理
        for (int64_t n = 0; n < output_shape[0]; ++n)
        {
            for (int64_t c = 0; c < output_shape[1]; ++c)
            {
                // 批次和通道偏移量
                int64_t batch_channel_offset = n * input_strides[0] + c * input_strides[1];
                int64_t out_batch_channel_offset = n * output_strides[0] + c * output_strides[1];

                // 遍历输出的空间位置
                std::vector<int64_t> out_spatial_coords(spatial_dims, 0);
                bool done = false;

                while (!done)
                {
                    // 计算当前输出位置的线性索引
                    int64_t out_idx = out_batch_channel_offset;
                    for (size_t i = 0; i < spatial_dims; ++i)
                    {
                        out_idx += out_spatial_coords[i] * output_strides[i + 2];
                    }

                    // 计算池化窗口在输入中的起始位置
                    std::vector<int64_t> in_window_start(spatial_dims);
                    for (size_t i = 0; i < spatial_dims; ++i)
                    {
                        in_window_start[i] =
                            out_spatial_coords[i] * working_strides[i] - working_pads[i];
                    }

                    // 执行池化操作
                    T sum = 0;
                    int64_t count = 0;

                    // 遍历池化窗口
                    std::vector<int64_t> window_coords(spatial_dims, 0);
                    bool window_done = false;

                    while (!window_done)
                    {
                        // 检查当前窗口位置是否在有效范围内
                        bool valid = true;
                        int64_t in_idx = batch_channel_offset;

                        for (size_t i = 0; i < spatial_dims; ++i)
                        {
                            // 考虑dilation的位置计算
                            int64_t in_pos =
                                in_window_start[i] + window_coords[i] * working_dilations[i];
                            if (in_pos < 0 || in_pos >= shape[i + 2])
                            {
                                valid = false;
                                break;
                            }
                            in_idx += in_pos * input_strides[i + 2];
                        }

                        // 如果位置有效，则累加p次方
                        if (valid)
                        {
                            T val = std::abs(data[in_idx]);
                            // 避免可能的数值不稳定性
                            if (p == 1)
                            {
                                sum += val; // 对于 p=1 直接加绝对值
                            }
                            else if (p == 2)
                            {
                                sum += val * val; // 对于 p=2 使用平方
                            }
                            else
                            {
                                sum += std::pow(val, static_cast<T>(p));
                            }
                            count++;
                        }

                        // 更新窗口坐标
                        for (int i = spatial_dims - 1; i >= 0; --i)
                        {
                            window_coords[i]++;
                            if (window_coords[i] < kernel_shape[i])
                            {
                                break;
                            }
                            window_coords[i] = 0;
                            if (i == 0)
                            {
                                window_done = true;
                            }
                        }
                    }

                    // 计算当前位置的LpPool结果
                    if (count > 0)
                    {
                        /*
                         * 严格按照 ONNX 实现的计算方式：
                         * 参考
                         * https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_pool_common.py
                         *
                         * LpPool 实现的核心思路：
                         * 1. 对输入数据取绝对值的 p 次方: |x|^p
                         * 2. 对结果求和得到 sum(|x|^p)
                         * 3. 对结果取 p 根: (sum(|x|^p))^(1/p)
                         */

                        // 计算步骤3：取 p 根
                        if (p == 1)
                        {
                            output_data[out_idx] = sum; // p=1 时无需开根
                        }
                        else if (p == 2)
                        {
                            output_data[out_idx] = std::sqrt(sum); // p=2 时使用平方根
                        }
                        else
                        {
                            output_data[out_idx] = std::pow(sum, static_cast<T>(1.0 / p));
                        }
                    }

                    // 更新输出空间坐标
                    for (int i = spatial_dims - 1; i >= 0; --i)
                    {
                        out_spatial_coords[i]++;
                        if (out_spatial_coords[i] < output_shape[i + 2])
                        {
                            break;
                        }
                        out_spatial_coords[i] = 0;
                        if (i == 0)
                        {
                            done = true;
                        }
                    }
                }
            }
        }

        return {output_data, output_shape};
    }
};

} // namespace onnx

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
