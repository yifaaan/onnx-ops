#pragma once

#include "utils.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
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

        // 输入验证
        if (shape.size() < 2 || kernel_shape.empty() || p < 1)
        {
            throw std::invalid_argument("LpPool: invalid input parameters");
        }

        size_t spatial_dims = shape.size() - 2;
        if (kernel_shape.size() != spatial_dims)
        {
            throw std::invalid_argument("LpPool: kernel_shape dimension mismatch");
        }

        if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER" || auto_pad == "VALID")
        {
            if (ceil_mode)
            {
                throw std::invalid_argument("LpPool: ceil_mode must be false when auto_pad is "
                                            "SAME_UPPER, SAME_LOWER or VALID");
            }
        }

        // 处理默认参数
        std::vector<int64_t> working_strides =
            strides.empty() ? std::vector<int64_t>(spatial_dims, 1) : strides;
        // 1表示不扩张核
        std::vector<int64_t> working_dilations =
            dilations.empty() ? std::vector<int64_t>(spatial_dims, 1) : dilations;
        // 长度必须是空间维度数 n 的两倍（2n），因为每个空间维度需要指定起始填充和结束填充。
        std::vector<int64_t> working_pads =
            pads.empty() && auto_pad == "NOTSET" ? std::vector<int64_t>(spatial_dims * 2, 0) : pads;

        if (working_strides.size() != spatial_dims || working_dilations.size() != spatial_dims ||
            (auto_pad == "NOTSET" && working_pads.size() != spatial_dims * 2))
        {
            throw std::invalid_argument("LpPool: parameter dimension mismatch");
        }

        // 计算输出形状
        auto output_shape = compute_output_shape(shape, kernel_shape, working_strides, working_pads,
                                                 working_dilations, auto_pad, ceil_mode);

        // 计算输入和输出的步长
        std::vector<int64_t> input_strides = compute_strides(shape);
        std::vector<int64_t> output_strides = compute_strides(output_shape);

        // 输出数据的元素总个数，并初始化
        size_t output_size = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                             std::multiplies<int64_t>());
        std::vector<T> output_data(output_size, 0);

        // 执行池化
        compute_pooling(data, output_data, shape, output_shape, kernel_shape, working_strides,
                        working_pads, working_dilations, p, input_strides, output_strides);

        return {output_data, output_shape};
    }

private:
    // 计算张量步长
    static std::vector<int64_t> compute_strides(const std::vector<int64_t>& shape)
    {
        std::vector<int64_t> strides(shape.size());
        strides.back() = 1;
        for (int64_t i = shape.size() - 2; i >= 0; --i)
        {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }

    // 计算输出形状
    static std::vector<int64_t> compute_output_shape(const std::vector<int64_t>& shape,
                                                     const std::vector<int64_t>& kernel_shape,
                                                     const std::vector<int64_t>& strides,
                                                     std::vector<int64_t>& pads,
                                                     const std::vector<int64_t>& dilations,
                                                     const std::string& auto_pad, bool ceil_mode)
    {

        size_t spatial_dims = shape.size() - 2;
        std::vector<int64_t> output_shape = {shape[0], shape[1]}; // batch_size 和 channels

        for (size_t i = 0; i < spatial_dims; ++i)
        {
            int64_t input_dim = shape[i + 2];
            int64_t kernel_dim = kernel_shape[i];
            int64_t stride = strides[i];
            int64_t dilation = dilations[i];
            int64_t pad_head = pads[i];
            int64_t pad_tail = pads[i + spatial_dims];

            // 需要对输入数据进行填充，使得输出的维度尽量接近input_dim/stride
            if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER")
            {
                // 计算输出维度
                int64_t output_dim = (input_dim + stride - 1) / stride;
                // 计算需要填充的数目，使得核移动的步长为stride时，输出维度为output_dim
                int64_t padding_needed =
                    (output_dim - 1) * stride + (dilation * (kernel_dim - 1) + 1) - input_dim;
                padding_needed = std::max<int64_t>(0, padding_needed);
                // 前面填充padding_needed / 2，后面填充padding_needed - padding_needed / 2
                pad_head = auto_pad == "SAME_UPPER" ? padding_needed / 2
                                                    : padding_needed - padding_needed / 2;
                pad_tail = padding_needed - pad_head;
                // 该轴的开始和结束填充
                pads[i] = pad_head;
                pads[i + spatial_dims] = pad_tail;
            }
            // 计算扩张后的窗口核大小
            int64_t dilated_kernel_size = dilation * (kernel_dim - 1) + 1;
            // 计算输出维度
            double temp =
                static_cast<double>(input_dim + pad_head + pad_tail - dilated_kernel_size) /
                    stride +
                1;
            int64_t output_dim = ceil_mode ? static_cast<int64_t>(std::ceil(temp))
                                           : static_cast<int64_t>(std::floor(temp));
            // 修正边界越界
            if (ceil_mode && (output_dim - 1) * stride >= input_dim + pad_head)
            {
                output_dim -= 1;
            }
            output_shape.push_back(std::max<int64_t>(1, output_dim));
        }
        return output_shape;
    }

    // 执行池化计算
    static void compute_pooling(
        const std::vector<T>& data, std::vector<T>& output_data,
        const std::vector<int64_t>& input_shape, const std::vector<int64_t>& output_shape,
        const std::vector<int64_t>& kernel_shape, const std::vector<int64_t>& strides,
        const std::vector<int64_t>& pads, const std::vector<int64_t>& dilations, int64_t p,
        const std::vector<int64_t>& input_strides, const std::vector<int64_t>& output_strides)
    {

        size_t spatial_dims = input_shape.size() - 2;
        int64_t batch_size = output_shape[0];
        int64_t channels = output_shape[1];

        for (int64_t n = 0; n < batch_size; ++n)
        {
            for (int64_t c = 0; c < channels; ++c)
            {
                // 获取data[n, c, ...]首元素的偏移量
                int64_t base_offset = n * input_strides[0] + c * input_strides[1];
                // 获取output_data[n, c, ...]首元素的偏移量
                int64_t out_base_offset = n * output_strides[0] + c * output_strides[1];

                // len = C1 * C2 * ... * Cn,表示子切片output_data[n, c, ...]的元素个数
                // 遍历子切片output_data[n, c, ...]的每个元素
                int len = output_data.size() / (batch_size * channels);
                for (int out_idx = 0; out_idx < len; ++out_idx)
                {
                    // 每个元素的坐标
                    std::vector<int64_t> out_coords(spatial_dims);
                    size_t temp_idx = out_idx;
                    // 从后往前遍历，计算当前元素在输出结果中每个维度的坐标
                    for (int i = spatial_dims - 1; i >= 0; --i)
                    {
                        // output_shape前两个为batch_size和channels，从第三个开始为空间维度
                        out_coords[i] = temp_idx % output_shape[i + 2];
                        temp_idx /= output_shape[i + 2];
                    }

                    // 计算该输出数据对应的池化窗口在输入数据中的起始位置
                    std::vector<int64_t> in_start(spatial_dims);
                    for (size_t i = 0; i < spatial_dims; ++i)
                    {
                        // 第i个空间维度上的左边
                        in_start[i] = out_coords[i] * strides[i] - pads[i];
                    }

                    // 池化窗口的元素个数
                    int kernel_num_size = std::accumulate(kernel_shape.begin(), kernel_shape.end(),
                                                          1, std::multiplies<int64_t>());
                    // 计算 Lp 范数
                    T sum = 0;
                    for (int k = 0; k < kernel_num_size; ++k)
                    {
                        // 计算池化窗口中每个元素的相对坐标
                        std::vector<int64_t> k_coords(spatial_dims);
                        size_t temp_k = k;
                        for (int i = spatial_dims - 1; i >= 0; --i)
                        {
                            k_coords[i] = temp_k % kernel_shape[i];
                            temp_k /= kernel_shape[i];
                        }

                        // 初始化池化窗口中每个元素在输入数据中的基础偏移量
                        int64_t in_idx = base_offset;
                        // 判断池化窗口中每个元素是否在输入数据中有效
                        bool valid = true;
                        for (int i = 0; i < spatial_dims; ++i)
                        {
                            // 元素在输入数据中该维度的绝对坐标
                            int64_t pos = in_start[i] + k_coords[i] * dilations[i];
                            // 超出维度的范围
                            if (pos < 0 || pos >= input_shape[i + 2])
                            {
                                valid = false;
                                break;
                            }
                            // 更新池化窗口中每个元素在输入数据中的偏移量
                            in_idx += pos * input_strides[i + 2];
                        }

                        if (valid)
                        {
                            T val = std::abs(data[in_idx]);
                            sum += (p == 1)   ? val
                                   : (p == 2) ? val * val
                                              : std::pow(val, static_cast<T>(p));
                        }
                    }

                    // 输出元素的偏移量
                    size_t out_pos = out_base_offset + out_idx;
                    T result = 0;
                    if (sum > 0)
                    {
                        if (p == 1)
                        {
                            result = sum; // L1范数
                        }
                        else if (p == 2)
                        {
                            result = std::sqrt(sum); // L2范数
                        }
                        else
                        {
                            result = std::pow(sum, static_cast<T>(1.0 / p)); // 一般Lp范数
                        }
                    }
                    output_data[out_pos] = result;
                }
            }
        }
    }
};

} // namespace onnx