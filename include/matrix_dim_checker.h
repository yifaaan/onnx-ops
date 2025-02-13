#pragma once

#include <Eigen/Dense>
#include <sstream>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace onnx {

template <typename TensorType>
struct TensorDimChecker {
  using RawType = typename std::remove_cv<
      typename std::remove_reference<TensorType>::type>::type;

  // 获取维度数
  static constexpr int GetNumDimensions() { return RawType::NumDimensions; }

  // 检查是否为1维张量
  static constexpr bool is_1D() { return RawType::NumDimensions == 1; }

  // 检查是否为2维张量
  static constexpr bool is_2D() { return RawType::NumDimensions == 2; }

  // 检查是否为3维张量
  static constexpr bool is_3D() { return RawType::NumDimensions == 3; }

  // 检查是否为4维张量
  static constexpr bool is_4D() { return RawType::NumDimensions == 4; }

  // 获取张量的具体维度大小
  static std::vector<Eigen::Index> get_shape(const TensorType& tensor) {
    std::vector<Eigen::Index> shape;
    auto dims = tensor.dimensions();
    for (int i = 0; i < RawType::NumDimensions; ++i) {
      shape.push_back(dims[i]);
    }
    return shape;
  }

  // 获取格式化的维度字符串，类似 Python 的 shape
  static std::string get_shape_str(const TensorType& tensor) {
    std::ostringstream oss;
    auto shape = get_shape(tensor);
    oss << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << shape[i];
    }
    oss << ")";
    return oss.str();
  }
};

}  // namespace onnx
