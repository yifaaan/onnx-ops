#pragma once

#include <cassert>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace onnx {
template <typename T>
class ScatterElements {
 public:
  static std::vector<T> Compute(const std::vector<T>& data,
                                const std::vector<int64_t>& data_shape,
                                const std::vector<int64_t>& indices,
                                const std::vector<int64_t>& indices_shape,
                                const std::vector<T>& updates,
                                const std::vector<int64_t>& updates_shape,
                                int64_t axis = 0,
                                const std::string& reduction = "none") {
    assert(data_shape.size() >= 1);
    assert(indices_shape.size() >= 1);
    assert(updates_shape.size() >= 1);
    // default is assign
    std::function<T(T, T)> f = [](T a, T b) { return b; };
    if (reduction == "add") {
      f = [](T a, T b) { return a + b; };
    } else if (reduction == "sub") {
      f = [](T a, T b) { return a - b; };
    } else if (reduction == "mul") {
      f = [](T a, T b) { return a * b; };
    } else if (reduction == "max") {
      f = [](T a, T b) { return std::max(a, b); };
    } else if (reduction == "min") {
      f = [](T a, T b) { return std::min(a, b); };
    }

    if (indices_shape != updates_shape) {
      throw std::invalid_argument(
          "indices and updates must have the same shape");
    }
    if (axis < 0) axis += data_shape.size();
    if (axis >= data_shape.size()) {
      throw std::invalid_argument("axis must be in range [0, " +
                                  std::to_string(data_shape.size()) + ")");
    }

    auto output = data;

    std::vector<int64_t> strides(data_shape.size());
    strides.back() = 1;
    for (int64_t i = data_shape.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * data_shape[i + 1];
    }

    auto total_idx = indices.size();
    for (int64_t i = 0; i < total_idx; ++i) {
      // 计算i的多维索引
      std::vector<int64_t> curr_indices(indices_shape.size());
      int64_t remaining = i;
      for (int64_t d = indices_shape.size() - 1; d >= 0; --d) {
        curr_indices[d] = remaining % indices_shape[d];
        remaining /= indices_shape[d];
      }

      int64_t target_idx = indices[i];
      if (target_idx < 0) target_idx += data_shape[axis];
      if (target_idx < 0 || target_idx >= data_shape[axis]) {
        throw std::out_of_range("index out of range");
      }

      std::vector<int64_t> output_indices = curr_indices;
      output_indices[axis] = target_idx;

      int64_t flat_output_idx = 0;
      int64_t flat_input_idx = 0;
      for (size_t d = 0; d < data_shape.size(); ++d) {
        flat_output_idx += output_indices[d] * strides[d];
        flat_input_idx += curr_indices[d] * strides[d];
      }

      output[flat_output_idx] = f(output[flat_output_idx], updates[i]);
    }

    return output;
  }
};
}  // namespace onnx