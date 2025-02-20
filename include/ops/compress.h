#pragma once

#include <numeric>
#include <stdexcept>
#include <optional>
#include <vector>
#include <algorithm>

namespace onnx {

template <typename T>
class Compress {
 public:
  static std::pair<std::vector<T>, std::vector<int64_t>> Compute(
      const std::vector<T>& data, const std::vector<int64_t>& shape,
      const std::vector<bool>& condition, std::optional<int64_t> axis_opt) {

    if (!axis_opt.has_value()) {
      auto len = std::min(data.size(), condition.size());
      std::vector<T> result;
      for (int i = 0; i < len; i++) {
        if (condition[i]) {
          result.push_back(data[i]);
        }
      }
      return {result, std::vector<int64_t>{result.size()}};
    }
    auto axis = axis_opt.value();


    auto dim = shape.size();
    if (axis < 0) axis += dim;
    if (axis < 0 || axis >= dim) {
      throw std::invalid_argument("axis must be in range [0, " +
                                  std::to_string(dim) + ")");
    }

    if (condition.size() != shape[axis]) {
      throw std::invalid_argument("condition size must match dimension size");
    }


    auto output_shape = shape;
    int64_t compressed_size =
        std::count(std::begin(condition), std::end(condition), true);
    output_shape[axis] = compressed_size;


    size_t total_size =
        std::accumulate(output_shape.begin(), output_shape.end(),
                        static_cast<size_t>(1), std::multiplies<size_t>());
    std::vector<T> result(total_size);


    std::vector<int64_t> strides(shape.size());
    strides.back() = 1;
    for (int i = shape.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }


    std::vector<int64_t> output_strides(output_shape.size());
    output_strides.back() = 1;
    for (int i = output_shape.size() - 2; i >= 0; --i) {
      output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
    }

    // 创建索引映射数组
    // condition [true, false, true]，-> [0, -1, 1]
    std::vector<int64_t> condition_indices(condition.size());
    int64_t curr_idx = 0;
    for (size_t i = 0; i < condition.size(); ++i) {
      if (condition[i]) {
        condition_indices[i] = curr_idx++;
      } else {
        condition_indices[i] = -1;
      }
    }


    std::vector<int64_t> curr_indices(shape.size());
    for (size_t i = 0; i < data.size(); ++i) {
      // 计算当前位置的多维索引
      size_t temp = i;
      for (int d = shape.size() - 1; d >= 0; --d) {
        curr_indices[d] = temp % shape[d];
        temp /= shape[d];
      }

      if (!condition[curr_indices[axis]]) {
        continue;
      }

      size_t output_idx = 0;
      for (size_t d = 0; d < shape.size(); ++d) {
        if (d == axis) {
          output_idx += condition_indices[curr_indices[d]] * output_strides[d];
        } else {
          output_idx += curr_indices[d] * output_strides[d];
        }
      }

      result[output_idx] = data[i];
    }

    return {result, output_shape};
  }
};

}  // namespace onnx