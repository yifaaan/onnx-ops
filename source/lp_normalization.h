#pragma once

#include "utils.h"
#include <cmath>
#include <stdexcept>
#include <vector>

template <typename T>
std::pair<std::vector<T>, std::vector<int64_t>> LpNormalization(const std::vector<T>& data,
                                                                const std::vector<int64_t>& shape,
                                                                int64_t axis = -1, int64_t p = 2)
{
    if (p != 1 && p != 2)
    {
        throw std::invalid_argument("LpNormalization: p must be 1 or 2");
    }
    if (axis < 0)
    {
        axis += shape.size();
    }

    if (axis < 0 || axis >= static_cast<int64_t>(shape.size()))
    {
        throw std::invalid_argument("LpNormalization: axis out of range");
    }

    std::vector<int64_t> output_shape = shape;

    int64_t axis_size = shape[axis];

    // 计算轴前所有维度的乘积
    int64_t outer_size = 1;
    for (int64_t i = 0; i < axis; ++i)
    {
        outer_size *= shape[i];
    }

    // 计算轴后的所有维度的乘积
    int64_t inner_size = 1;
    for (int64_t i = axis + 1; i < static_cast<int64_t>(shape.size()); ++i)
    {
        inner_size *= shape[i];
    }

    std::vector<T> result(data.size());

    // axis = 1
    // 2 * 3 * 2 * 2 * 2
    // outer_size = 2
    // axis_size = 3
    // inner_size = 8

    for (int64_t outer = 0; outer < outer_size; ++outer)
    {
        for (int64_t inner = 0; inner < inner_size; ++inner)
        {
            // 计算当前切片的范数
            T norm = 0;

            for (int64_t a = 0; a < axis_size; ++a)
            {
                // 计算数据索引
                // 2 * 3 * 4 * 5
                int64_t idx = outer * axis_size * inner_size + a * inner_size + inner;

                if (p == 1)
                {
                    // L1范数: 绝对值之和
                    norm += std::abs(data[idx]);
                }
                else
                {
                    // L2范数: 平方和的平方根
                    norm += data[idx] * data[idx];
                }
            }

            // 对L2范数计算平方根
            if (p == 2 && norm > 0)
            {
                norm = std::sqrt(norm);
            }

            // 归一化当前切片的元素
            for (int64_t a = 0; a < axis_size; ++a)
            {
                int64_t idx = outer * axis_size * inner_size + a * inner_size + inner;

                if (norm > 0)
                {
                    // 归一化
                    result[idx] = data[idx] / norm;
                }
                else
                {
                    // 如果范数为0，则保持原值
                    result[idx] = data[idx];
                }
            }
        }
    }

    return {result, output_shape};
}