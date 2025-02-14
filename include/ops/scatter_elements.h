#pragma once

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>

namespace onnx {
template <typename Scalar, int Dim>
class ScatterElements {
 public:
  static Eigen::Tensor<Scalar, Dim> Compute(
      const Eigen::Tensor<Scalar, Dim>& data,
      const Eigen::Tensor<int64_t, Dim>& indices,
      const Eigen::Tensor<Scalar, Dim>& updates, int64_t axis = 0,
      const std::string& reduction = "none") {
    static_assert(Dim >= 1, "Dim must be at least 1");
    // default is assign
    std::function<Scalar(Scalar, Scalar)> f = [](Scalar a, Scalar b) {
      return b;
    };
    if (reduction == "add") {
      f = [](Scalar a, Scalar b) { return a + b; };
    } else if (reduction == "sub") {
      f = [](Scalar a, Scalar b) { return a - b; };
    } else if (reduction == "mul") {
      f = [](Scalar a, Scalar b) { return a * b; };
    } else if (reduction == "max") {
      f = [](Scalar a, Scalar b) { return std::max(a, b); };
    } else if (reduction == "min") {
      f = [](Scalar a, Scalar b) { return std::min(a, b); };
    }

    auto data_dims = data.dimensions();
    auto indices_dims = indices.dimensions();
    if (indices_dims != updates.dimensions()) {
      throw std::invalid_argument(
          "indices and updates must have the same shape");
    }
    if (axis < 0) axis += Dim;
    if (axis >= Dim) {
      throw std::invalid_argument("axis must be in range [0, " +
                                  std::to_string(Dim) + ")");
    }

    auto output = data;

    // 计算索引的总数
    auto total_elements = indices.size();
    // 遍历所有索引
    for (Eigen::Index i = 0; i < total_elements; ++i) {
      // 计算多维索引
      Eigen::array<Eigen::Index, Dim> idx;
      Eigen::Index remaining = i;
      /*
        IndicesAndUpdatesDim = 2
        (2 , 3)
        1, 2, 3
        4, 5, 6
      */

      for (int d = Dim - 1; d >= 0; --d) {
        // 得到在对应维度的索引
        idx[d] = remaining % indices_dims[d];
        remaining /= indices_dims[d];
      }
      // 获取目标索引
      int64_t target_idx = indices(idx);
      if (!(target_idx >= 0 && target_idx < data_dims[axis])) {
        throw std::out_of_range("index out of range");
      }

      // 构建输出索引
      auto output_idx = idx;
      output_idx[axis] = target_idx;

      if (reduction == "add") {
        output(output_idx) += updates(idx);
      } else if (reduction == "sub") {
        output(output_idx) -= updates(idx);
      } else if (reduction == "mul") {
        output(output_idx) *= updates(idx);
      } else {
        output(output_idx) = updates(idx);
      }
    }

    return output;
  }
};
}  // namespace onnx