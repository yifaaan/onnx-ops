#pragma once

#include "utils.h"
#include <algorithm>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>

namespace onnx
{

template <typename T>
class GlobalMaxPool
{
public:
    static std::pair<std::vector<T>, std::vector<int64_t>>
    Compute(const std::vector<T>& data, const std::vector<int64_t>& shape)
    {
        // 检查输入维度
        if (shape.size() < 2)
        {
            throw std::runtime_error("GlobalMaxPool: Input tensor must have at least 2 dimensions");
        }
        if (shape.size() == 2)
        {
            T max_val = std::numeric_limits<T>::lowest();
            for (size_t i = 0; i < data.size(); ++i)
            {
                max_val = std::max(max_val, data[i]);
            }
            return std::make_pair<std::vector<T>, std::vector<int64_t>>({max_val}, {1});
        }

        // 计算输出形状：保持N和C维度不变，其他维度变为1
        std::vector<int64_t> output_shape = shape;
        for (size_t i = 2; i < shape.size(); ++i)
        {
            output_shape[i] = 1;
        }

        // 计算输出大小
        int64_t batch_size = shape[0];
        int64_t channels = shape[1];
        int64_t output_size = batch_size * channels;

        // 计算每个通道需要处理的元素数量
        int64_t elements_per_channel = 1;
        for (size_t i = 2; i < shape.size(); ++i)
        {
            elements_per_channel *= shape[i];
        }

        // 创建输出数据
        std::vector<T> result(output_size);

        // 对每个批次和通道进行处理
        for (int64_t n = 0; n < batch_size; ++n)
        {
            for (int64_t c = 0; c < channels; ++c)
            {
                // 计算当前通道的最大值
                T max_val = std::numeric_limits<T>::lowest();

                // 计算当前批次和通道的基础偏移
                int64_t base_offset =
                    n * channels * elements_per_channel + c * elements_per_channel;

                // 在所有空间维度上查找最大值
                for (int64_t i = 0; i < elements_per_channel; ++i)
                {
                    max_val = std::max(max_val, data[base_offset + i]);
                }

                // 设置输出值
                result[n * channels + c] = max_val;
            }
        }

        return {result, output_shape};
    }
};

} // namespace onnx