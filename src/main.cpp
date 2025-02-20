#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

// #include "ops/compress.h"
#include "ops/scatter_elements.h"
int main() {
  // 创建长度为4的布尔型Tensor
  // Eigen::Tensor<bool, 1> x(4);
  // x.setValues({true, false, true, false});

  // 打印原始Tensor
  // std::cout << "原始Tensor：\n" << x << std::endl;

  // Eigen::Tensor<float, 2> data(4, 3);
  // data.setValues({{1.0f, 2.0f, 3.0f},
  //                 {4.0f, 5.0f, 6.0f},
  //                 {7.0f, 8.0f, 9.0f},
  //                 {10.0f, 11.0f, 12.0f}});

  // Eigen::Tensor<bool, 1> condition(4);
  // condition.setValues({true, false, true, false});

  // auto result = onnx::Compress<float, 2>::Compute(data, condition, -1);
  // std::cout << "result: " << result << std::endl;

  return 0;
}