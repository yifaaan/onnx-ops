#pragma once

#include "utils.h"
#include <cassert>
#include <stdexcept>
#include <vector>

namespace onnx
{
template <typename T>
class GatherND
{
public:
    static std::pair<std::vector<T>, std::vector<int64_t>>
    Compute(const std::vector<T>& data, const std::vector<int64_t>& data_shape,
            const std::vector<int64_t>& indices, const std::vector<int64_t>& indices_shape,
            int64_t batch_dims = 0)
    {
        // 检查输入有效性
        int64_t r = data_shape.size();    // data的秩
        int64_t q = indices_shape.size(); // indices的秩

        assert(r >= 1 && q >= 1);

        // 获取indices最后一维的大小
        int64_t last_indices_dim = indices_shape[q - 1];

        // 检查indices最后一维的大小是否在有效范围[1, r - batch_dims]内
        if (last_indices_dim < 1 || last_indices_dim > r - batch_dims)
        {
            throw std::invalid_argument("indices_shape[-1] should be in range [1, r - batch_dims]");
        }

        // 处理batch_dims
        if (batch_dims < 0)
        {
            batch_dims += std::min(r, q);
        }

        if (batch_dims < 0 || batch_dims >= std::min(r, q))
        {
            throw std::invalid_argument(
                "batch_dims must be in range [0, min(rank(data), rank(indices)))");
        }

        // 检查batch维度的形状是否匹配, 前batch_dims维度的形状必须相同
        for (int64_t i = 0; i < batch_dims; ++i)
        {
            if (data_shape[i] != indices_shape[i])
            {
                throw std::invalid_argument("batch dimensions must have the same size");
            }
        }

        // 添加batch维度
        // 输出形状：indices_shape[:-1] + data_shape[last_indices_dim+batch_dims:]
        // 假设我们有以下输入张量和参数：
        // data 张量: 形状 data_shape = [3, 4, 10, 20, 30] (五维张量)
        // indices 张量: 形状 indices_shape = [3, 4, 5, 2] (四维张量)
        // batch_dims: 2
        // indices[-1]=last_indices_dim=2，表示data的第2维和第3维将被索引
        // 输出形状：indices_shape[:-1] + data_shape[last_indices_dim+batch_dims:] = [3, 4, 5] +
        // [30] = [3, 4, 5, 30]

        // 计算输出形状
        std::vector<int64_t> output_shape;
        // 添加indices除了最后一维的其他维度
        for (int64_t i = 0; i < q - 1; ++i)
        {
            output_shape.push_back(indices_shape[i]);
        }
        // 添加data中未被索引的维度
        for (int64_t i = batch_dims + last_indices_dim; i < r; ++i)
        {
            output_shape.push_back(data_shape[i]);
        }

        // 计算输出元素的总数：3 * 4 * 5 * 30 = 1800
        int64_t output_size = 1;
        for (int64_t dim : output_shape)
        {
            output_size *= dim;
        }

        std::vector<T> output(output_size);

        // 计算data和indices的步长
        std::vector<int64_t> data_strides = get_strides(data_shape);
        std::vector<int64_t> indices_strides = get_strides(indices_shape);

        // 计算一共多少个batch：3 * 4 = 12
        int64_t batch_size = 1;
        for (int64_t i = 0; i < batch_dims; ++i)
        {
            batch_size *= indices_shape[i];
        }

        // 计算indices除了batch和最后一维的部分的总大小：5
        int64_t indices_mid_size = 1;
        for (int64_t i = batch_dims; i < q - 1; ++i)
        {
            indices_mid_size *= indices_shape[i];
        }

        // 计算data未被索引部分的总大小：30
        int64_t data_tail_size = 1;
        for (int64_t i = batch_dims + last_indices_dim; i < r; ++i)
        {
            data_tail_size *= data_shape[i];
        }

        // data 张量: 形状 data_shape = [3, 4, 10, 20, 30] (五维张量)
        // indices 张量: 形状 indices_shape = [3, 4, 5, 2] (四维张量)
        // 对每个batch进行处理共3 * 4 = 12个batch
        for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx)
        {
            // 计算当前batch的坐标
            std::vector<int64_t> batch_coords = offset_to_coords(
                batch_idx,
                std::vector<int64_t>(indices_shape.begin(), indices_shape.begin() + batch_dims));

            // 计算batch部分在data和indices中的偏移
            // 例如，当 batch_coords = [1,2] 时：
            // data_strides = [24000, 6000, 200, 10, 1]
            // int64_t batch_offset_data = 0;
            // batch_offset_data += batch_coords[0] * data_strides[0];    // 1 * 24000
            // batch_offset_data += batch_coords[1] * data_strides[1];    // 2 * 6000
            // batch_offset_data = 24000 + 12000 = 36000

            // indices_strides = [40, 10, 2, 1]
            // int64_t batch_offset_indices = 0;
            // batch_offset_indices += batch_coords[0] * indices_strides[0];    // 1 * 40
            // batch_offset_indices += batch_coords[1] * indices_strides[1];    // 2 * 10
            // batch_offset_indices = 40 + 20 = 60
            int64_t batch_offset_data = 0;
            int64_t batch_offset_indices = 0;
            for (int64_t i = 0; i < batch_dims; ++i)
            {
                batch_offset_data += batch_coords[i] * data_strides[i];
                batch_offset_indices += batch_coords[i] * indices_strides[i];
            }

            // 对indices中间部分的每个元素进行处理
            // data_shape = [3, 4, 10, 20, 30]
            // indices_shape = [3, 4, 5, 2]
            for (int64_t mid_idx = 0; mid_idx < indices_mid_size; ++mid_idx)
            {
                // 计算中间部分的坐标
                std::vector<int64_t> mid_coords = offset_to_coords(
                    mid_idx, std::vector<int64_t>(indices_shape.begin() + batch_dims,
                                                  indices_shape.end() - 1));
                // mid_idx 从 0 到 4，mid_coords 会依次是：
                // mid_idx = 0 -> mid_coords = [0]
                // mid_idx = 1 -> mid_coords = [1]
                // mid_idx = 2 -> mid_coords = [2]
                // mid_idx = 3 -> mid_coords = [3]
                // mid_idx = 4 -> mid_coords = [4]

                // 计算中间部分在indices中的偏移, 以batch_offset_indices为基准
                int64_t mid_offset_indices = 0;
                for (int64_t i = 0; i < mid_coords.size(); ++i)
                {
                    mid_offset_indices += mid_coords[i] * indices_strides[batch_dims + i];
                }

                // 批次偏移 batch_offset_indices（比如当 batch_coords = [1,2] 时是60）
                // 访问 indices[1,2,3] 这个索引元组(x, y)
                // 访问第一个元素 (x)
                // indices[batch_offset_indices + mid_offset_indices + 0]
                // = indices[60 + 6 + 0]
                // = indices[66]
                // = x

                // // 访问第二个元素 (y)
                // indices[batch_offset_indices + mid_offset_indices + 1]
                // = indices[60 + 6 + 1]
                // = indices[67]
                // = y

                // 获取当前索引元组
                std::vector<int64_t> current_indices(last_indices_dim);
                for (int64_t i = 0; i < last_indices_dim; ++i)
                {
                    int64_t idx_offset =
                        batch_offset_indices + mid_offset_indices + i * indices_strides[q - 1];
                    current_indices[i] = indices[idx_offset];

                    // 处理负索引
                    if (current_indices[i] < 0)
                    {
                        current_indices[i] += data_shape[batch_dims + i];
                    }

                    // 检查索引是否越界
                    if (current_indices[i] < 0 || current_indices[i] >= data_shape[batch_dims + i])
                    {
                        throw std::out_of_range("index out of range");
                    }
                }

                // 计算索引在data中的偏移, 以batch_offset_data为基准
                int64_t indices_offset_data = batch_offset_data;
                for (int64_t i = 0; i < last_indices_dim; ++i)
                {
                    indices_offset_data += current_indices[i] * data_strides[batch_dims + i];
                }

                // 对data未被索引的部分进行处理
                for (int64_t tail_idx = 0; tail_idx < data_tail_size; ++tail_idx)
                {
                    // 计算未被索引部分的坐标
                    std::vector<int64_t> tail_coords = offset_to_coords(
                        tail_idx,
                        std::vector<int64_t>(data_shape.begin() + batch_dims + last_indices_dim,
                                             data_shape.end()));

                    // 计算未被索引部分在data中的偏移, 以batch_offset_data为基准
                    int64_t tail_offset_data = 0;
                    for (int64_t i = 0; i < tail_coords.size(); ++i)
                    {
                        tail_offset_data +=
                            tail_coords[i] * data_strides[batch_dims + last_indices_dim + i];
                    }

                    // 计算输出索引
                    int64_t output_idx = 0;
                    int64_t output_stride = 1;

                    // 添加tail部分的索引
                    for (int64_t i = tail_coords.size() - 1; i >= 0; --i)
                    {
                        output_idx += tail_coords[i] * output_stride;
                        output_stride *= data_shape[batch_dims + last_indices_dim + i];
                    }

                    // 添加mid部分的索引
                    for (int64_t i = mid_coords.size() - 1; i >= 0; --i)
                    {
                        output_idx += mid_coords[i] * output_stride;
                        output_stride *= indices_shape[batch_dims + i];
                    }

                    // 添加batch部分的索引
                    for (int64_t i = batch_coords.size() - 1; i >= 0; --i)
                    {
                        output_idx += batch_coords[i] * output_stride;
                        output_stride *= indices_shape[i];
                    }

                    // 复制数据
                    output[output_idx] = data[indices_offset_data + tail_offset_data];
                }
            }
        }

        return {output, output_shape};
    }
};
} // namespace onnx
