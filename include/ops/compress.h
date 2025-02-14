#pragma once

#include <iostream>
#include <stdexcept>
#include <unsupported/Eigen/CXX11/Tensor>

namespace onnx {

template <typename Scalar, int Dim>
class Compress {
 public:
  static auto Compute(const Eigen::Tensor<Scalar, Dim>& data,
                      const Eigen::Tensor<bool, 1>& condition,
                      int64_t axis = -1) {
    auto data_dims = data.dimensions();

    // 计算总元素数
    Eigen::Index total_elements = data.size();
    int length = std::min(total_elements, condition.size());
    Eigen::Tensor<bool, 1> condition_truncated(length);
    condition_truncated =
        condition.slice(Eigen::array<Eigen::Index, 1>{0},
                        Eigen::array<Eigen::Index, 1>{length});

    if (axis == -1) {
      // 计算输出大小
      int64_t output_size = 0;
      for (Eigen::Index i = 0; i < length; ++i) {
        if (condition_truncated(i)) output_size++;
      }

      // 创建一维输出tensor
      Eigen::Tensor<Scalar, 1> result(output_size);

      // 将数据展平
      Eigen::array<Eigen::Index, Dim> shuffle_dims;
      for (int i = 0; i < Dim; ++i) {
        shuffle_dims[i] = Dim - 1 - i;
      }
      auto shuffled_data = data.shuffle(shuffle_dims);
      Eigen::array<Eigen::Index, 1> flat_dims{{total_elements}};
      auto flattened_data = shuffled_data.reshape(flat_dims);

      // 填充数据
      int64_t out_idx = 0;
      for (Eigen::Index i = 0; i < length; ++i) {
        if (condition_truncated(i)) {
          result(out_idx++) = flattened_data(i);
        }
      }
      return result;
    } else {
      // 按指定轴压缩
      if (axis < 0) axis += Dim;
      if (axis < 0 || axis >= Dim) {
        throw std::invalid_argument("axis must be in range [0, " +
                                    std::to_string(Dim) + ")");
      }

      if (condition.size() != data_dims[axis]) {
        throw std::invalid_argument("condition size must match dimension size");
      }

      // 计算输出维度
      Eigen::array<Eigen::Index, Dim> output_dims = data_dims;
      int64_t compressed_size = 0;
      for (int i = 0; i < condition.size(); ++i) {
        if (condition(i)) compressed_size++;
      }
      output_dims[axis] = compressed_size;

      // 创建输出tensor
      Eigen::Tensor<Scalar, Dim> output(output_dims);

      // 计算每个维度的步长
      Eigen::array<Eigen::Index, Dim> strides;
      strides[Dim - 1] = 1;
      for (int i = Dim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * data_dims[i + 1];
      }

      // 遍历并压缩
      Eigen::array<Eigen::Index, Dim> in_idx;
      Eigen::array<Eigen::Index, Dim> out_idx;
      for (Eigen::Index i = 0; i < total_elements; ++i) {
        // 计算多维索引
        Eigen::Index remaining = i;
        for (int d = Dim - 1; d >= 0; --d) {
          in_idx[d] = remaining % data_dims[d];
          remaining /= data_dims[d];
        }

        // 检查当前轴的条件
        if (!condition(in_idx[axis])) continue;

        // 计算输出索引
        int compressed_idx = 0;
        for (int j = 0; j < in_idx[axis]; ++j) {
          if (condition(j)) compressed_idx++;
        }

        for (int d = 0; d < Dim; ++d) {
          out_idx[d] = d == axis ? compressed_idx : in_idx[d];
        }

        output(out_idx) = data(in_idx);
      }

      return output;
    }
  }
};
}  // namespace onnx