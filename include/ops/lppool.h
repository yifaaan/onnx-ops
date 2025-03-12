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
            int64_t p = 2, const std::vector<int64_t>& dilations = {})
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
            int64_t output_dim =
                (input_dim + pad_head + pad_tail - dilated_kernel_size) / stride + 1;
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
                            sum += std::pow(val, static_cast<T>(p));
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
                        // 按照ONNX标准: (kernel_size * avg(|x|^p))^(1/p)
                        T avg = sum / static_cast<T>(count);
                        T scaled_avg = avg * static_cast<T>(kernel_size);
                        output_data[out_idx] = std::pow(scaled_avg, static_cast<T>(1.0 / p));
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