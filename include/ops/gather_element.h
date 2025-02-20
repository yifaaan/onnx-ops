#pragma once

#include <cassert>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include "utils.h"

namespace onnx {
template <typename T>
class GatherElements {
 public:
  static std::pair<std::vector<T>, std::vector<int64_t>> Compute(const std::vector<T>& data,
                                const std::vector<int64_t>& data_shape,
                                const std::vector<int64_t>& indices,
                                const std::vector<int64_t>& indices_shape,
                                int64_t axis = 0) {
    assert(data_shape.size() >= 1 && data_shape.size() == indices_shape.size());
    if (indices.empty()) {
      return {};
    }
    
    // 检查除了axis维度外的其他维度是否相同
    for (int d = 0; d < indices_shape.size(); d++) {
      if (d != axis && data_shape[d] != indices_shape[d]) {
        throw std::invalid_argument("indices_shape must be the same as data_shape except for the axis dimension");
      }
    }

    // 处理负数axis
    if (axis < 0) axis += data_shape.size();
    if (axis >= data_shape.size()) {
      throw std::invalid_argument("axis must be in range [0, " +
                                  std::to_string(data_shape.size()) + ")");
    }

    // 计算输出元素的个数
    auto len = std::accumulate(std::begin(indices_shape), std::end(indices_shape), 
                             1, std::multiplies<int64_t>());
    std::vector<T> result(len);
    
    // 计算步长
    std::vector<int64_t> strides = get_strides(data_shape);
    
    // 对每个索引进行处理
    for (int64_t i = 0; i < len; ++i) {
      // 计算当前位置的多维坐标
      std::vector<int64_t> curr_coords = offset_to_coords(i, indices_shape);
      
      // 对应轴的索引
      int64_t target_idx = indices[i];
      // 处理负数索引
      if (target_idx < 0) target_idx += data_shape[axis];
      // 检查索引范围
      if (target_idx < 0 || target_idx >= data_shape[axis]) {
        throw std::out_of_range("index out of range");
      }
      
      

      // 构建源数据的坐标
      std::vector<int64_t> src_coords = curr_coords;
      // result[i][j][k] = data[index[i][j][k]][j][k] if axis = 0,
      // result[i][j][k] = data[i][index[i][j][k]][k] if axis = 1,
      // result[i][j][k] = data[i][j][index[i][j][k]] if axis = 2,
      // 将目标索引赋值给对应轴
      src_coords[axis] = target_idx;
      
      // 计算源数据的对应的一维偏移
      int64_t src_offset = 0;
      for (size_t d = 0; d < data_shape.size(); ++d) {
        src_offset += src_coords[d] * strides[d];
      }
      
      // 收集元素
      result[i] = data[src_offset];
    }
    
    return {result, indices_shape};
  }
};
}  // namespace onnx