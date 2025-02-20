#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "ops/scatter_elements.h"

using Catch::Approx;

TEST_CASE("2D Scatter Elements Basic Operations", "[scatter][2d]") {
  SECTION("Column-wise scatter (axis=1)") {
    // 创建4x3的测试数据
    Eigen::Tensor<float, 2> data(4, 3);
    data.setValues({{1.0f, 2.0f, 3.0f},
                    {4.0f, 5.0f, 6.0f},
                    {7.0f, 8.0f, 9.0f},
                    {10.0f, 11.0f, 12.0f}});

    Eigen::Tensor<int64_t, 2> indices(4, 3);
    indices.setValues({{1, 0, 2}, {2, 1, 0}, {0, 2, 1}, {1, 0, 2}});

    Eigen::Tensor<float, 2> updates(4, 3);
    updates.setValues({{0.1f, 0.2f, 0.3f},
                       {0.4f, 0.5f, 0.6f},
                       {0.7f, 0.8f, 0.9f},
                       {1.0f, 1.1f, 1.2f}});

    // 测试加法操作
    auto result_add = onnx::ScatterElements<float, 2>::Compute(
        data, indices, updates, 1, "add");
    /*
    {{1.2f, 2.1f, 3.3f},
    {4.6f, 5.5f, 6.4f},
    {7.7f, 8.9f, 9.8f},
    {11.1f, 12f, 13.2f}}
    */

    // 验证部分结果
    REQUIRE(result_add(0, 1) == Approx(2.1f));
    REQUIRE(result_add(1, 2) == Approx(6.4f));
    REQUIRE(result_add(2, 0) == Approx(7.7f));

    // 测试替换操作
    auto result_replace = onnx::ScatterElements<float, 2>::Compute(
        data, indices, updates, 1, "none");
    /*
    {{0.2f, 0.1f, 0.3f},
    {0.6f, 0.5f, 0.4f},
    {0.7f, 0.9f, 0.8f},
    {1.1f, 1.0f, 1.2f}}
    */
    REQUIRE(result_replace(0, 1) == Approx(0.1f));
    REQUIRE(result_replace(1, 2) == Approx(0.4f));
    REQUIRE(result_replace(2, 0) == Approx(0.7f));
  }

  SECTION("Row-wise scatter (axis=0)") {
    // 创建3x4的测试数据
    Eigen::Tensor<float, 2> data(3, 4);
    data.setValues({{1.0f, 2.0f, 3.0f, 4.0f},
                    {5.0f, 6.0f, 7.0f, 8.0f},
                    {9.0f, 10.0f, 11.0f, 12.0f}});

    Eigen::Tensor<int64_t, 2> indices(2, 4);
    indices.setValues({{2, 0, 1, 0}, {1, 2, 0, 1}});

    Eigen::Tensor<float, 2> updates(2, 4);
    updates.setValues({{0.1f, 0.2f, 0.3f, 0.4f}, {0.5f, 0.6f, 0.7f, 0.8f}});

    // 测试乘法操作
    auto result_mul = onnx::ScatterElements<float, 2>::Compute(
        data, indices, updates, 0, "mul");

    // 验证部分结果
    REQUIRE(result_mul(2, 0) == Approx(0.9f));  // 9.0 * 0.1
    REQUIRE(result_mul(0, 1) == Approx(0.4f));  // 2.0 * 0.2
    REQUIRE(result_mul(1, 2) == Approx(2.1f));  // 7.0 * 0.3
  }
}

TEST_CASE("2D Scatter Elements Error Handling", "[scatter][2d][error]") {
  SECTION("Index out of bounds") {
    Eigen::Tensor<float, 2> data(2, 2);
    data.setValues({{1.0f, 2.0f}, {3.0f, 4.0f}});

    Eigen::Tensor<int64_t, 2> indices(2, 2);
    indices.setValues({{0, 2}, {1, 0}});  // 2超出了列范围

    Eigen::Tensor<float, 2> updates(2, 2);
    updates.setValues({{0.1f, 0.2f}, {0.3f, 0.4f}});

    bool exception_caught = false;
    try {
      auto result = onnx::ScatterElements<float, 2>::Compute(data, indices,
                                                             updates, 1, "add");
    } catch (const std::out_of_range&) {
      exception_caught = true;
    }
    REQUIRE(exception_caught);
  }

  SECTION("Shape mismatch") {
    Eigen::Tensor<float, 2> data(2, 2);
    data.setValues({{1.0f, 2.0f}, {3.0f, 4.0f}});

    Eigen::Tensor<int64_t, 2> indices(1, 2);  // 形状不匹配
    indices.setValues({{0, 1}});

    Eigen::Tensor<float, 2> updates(2, 2);
    updates.setValues({{0.1f, 0.2f}, {0.3f, 0.4f}});

    bool exception_caught = false;
    try {
      auto result = onnx::ScatterElements<float, 2>::Compute(data, indices,
                                                             updates, 1, "add");
    } catch (const std::invalid_argument&) {
      exception_caught = true;
    }
    REQUIRE(exception_caught);
  }
}

TEST_CASE("2D Scatter Elements Special Cases", "[scatter][2d][special]") {
  SECTION("Zero updates") {
    // 测试更新值为0的情况
    Eigen::Tensor<float, 2> data(2, 3);
    data.setValues({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});

    Eigen::Tensor<int64_t, 2> indices(2, 3);
    indices.setValues({{1, 0, 1}, {0, 1, 0}});

    Eigen::Tensor<float, 2> updates(2, 3);
    updates.setValues({{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}});

    auto result = onnx::ScatterElements<float, 2>::Compute(data, indices,
                                                           updates, 1, "add");

    // 验证加0后的值保持不变
    REQUIRE(result(0, 0) == Approx(1.0f));
    REQUIRE(result(0, 1) == Approx(2.0f));
    REQUIRE(result(1, 2) == Approx(6.0f));
  }

  SECTION("Identity mapping") {
    // 测试索引为原位置的情况
    Eigen::Tensor<float, 2> data(2, 2);
    data.setValues({{1.0f, 2.0f}, {3.0f, 4.0f}});

    Eigen::Tensor<int64_t, 2> indices(2, 2);
    indices.setValues({{0, 1}, {0, 1}});

    Eigen::Tensor<float, 2> updates(2, 2);
    updates.setValues({{0.1f, 0.2f}, {0.3f, 0.4f}});

    auto result = onnx::ScatterElements<float, 2>::Compute(data, indices,
                                                           updates, 1, "add");

    // 验证结果
    REQUIRE(result(0, 0) == Approx(1.1f));  // 1.0 + 0.1
    REQUIRE(result(0, 1) == Approx(2.2f));  // 2.0 + 0.2
    REQUIRE(result(1, 0) == Approx(3.3f));  // 3.0 + 0.3
    REQUIRE(result(1, 1) == Approx(4.4f));  // 4.0 + 0.4
  }
}

TEST_CASE("3D Scatter Elements Operations", "[scatter][3d]") {
  SECTION("Basic 3D scatter (axis=1)") {
    // 创建2x3x2的测试数据
    Eigen::Tensor<float, 3> data(2, 3, 2);
    data.setValues(
      {
        {
          {1.0f, 2.0f}, 
          {3.0f, 4.0f}, 
          {5.0f, 6.0f}
        },
        {
          {7.0f, 8.0f}, 
          {9.0f, 10.0f},
          {11.0f, 12.0f}
        }
      });

    Eigen::Tensor<int64_t, 3> indices(2, 3, 2);
    indices.setValues(
      {
        {
          {1, 0}, 
          {0, 2}, 
          {2, 1}
        },
        {
          {0, 1}, 
          {2, 0}, 
          {1, 2}}
      });

    /*
    {
        {
          {1.3f, 2.4f}, 
          {3.1f, 4.6f}, 
          {5.5f, 6.4f}
        },
        {
          {7.7f, 9.0f}, 
          {10.1f, 10.8f},
          {11.9f, 13.2f}
        }
      }
    */

    Eigen::Tensor<float, 3> updates(2, 3, 2);
    updates.setValues(
      {
        {
          {0.1f, 0.2f}, 
          {0.3f, 0.4f}, 
          {0.5f, 0.6f}
        },
        {
            {0.7f, 0.8f}, 
            {0.9f, 1.0f}, 
            {1.1f, 1.2f}
        }
      });

    // 测试加法操作
    auto result_add = onnx::ScatterElements<float, 3>::Compute(
        data, indices, updates, 1, "add");

    // 验证部分结果
    REQUIRE(result_add(0, 0, 0) == Approx(1.3f)); 
    REQUIRE(result_add(0, 1, 1) == Approx(4.6f)); 
    REQUIRE(result_add(1, 2, 0) == Approx(10.1f));

    // 测试乘法操作
    auto result_mul = onnx::ScatterElements<float, 3>::Compute(
        data, indices, updates, 1, "mul");

    // 验证部分结果
    REQUIRE(result_mul(0, 0, 0) == Approx(0.3f));  // 1.0 * 0.3
    REQUIRE(result_mul(0, 1, 1) == Approx(0.8f));  // 4.0 * 0.2
    REQUIRE(result_mul(1, 2, 0) == Approx(9.9f));  // 11.0 * 0.9
  }

  SECTION("3D scatter along axis 2") {
    // 创建2x2x3的测试数据
    Eigen::Tensor<float, 3> data(2, 2, 3);
    data.setValues({{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},
                    {{7.0f, 8.0f, 9.0f}, {10.0f, 11.0f, 12.0f}}});

    Eigen::Tensor<int64_t, 3> indices(2, 2, 3);
    indices.setValues({{{2, 0, 1}, {1, 2, 0}},
                      {{0, 1, 2}, {2, 1, 0}}});

    Eigen::Tensor<float, 3> updates(2, 2, 3);
    updates.setValues({{{0.1f, 0.2f, 0.3f}, {0.4f, 0.5f, 0.6f}},
                      {{0.7f, 0.8f, 0.9f}, {1.0f, 1.1f, 1.2f}}});

    // 测试替换操作
    auto result_none = onnx::ScatterElements<float, 3>::Compute(
        data, indices, updates, 2, "none");

    // 验证部分结果
    REQUIRE(result_none(0, 0, 0) == Approx(0.2f));
    REQUIRE(result_none(0, 1, 1) == Approx(0.5f));
    REQUIRE(result_none(1, 0, 2) == Approx(0.9f));
  }

  SECTION("3D scatter error handling") {
    // 创建2x2x2的测试数据
    Eigen::Tensor<float, 3> data(2, 2, 2);
    data.setValues({{{1.0f, 2.0f}, {3.0f, 4.0f}},
                    {{5.0f, 6.0f}, {7.0f, 8.0f}}});

    // 索引超出范围的情况
    Eigen::Tensor<int64_t, 3> indices(2, 2, 2);
    indices.setValues({{{0, 3}, {1, 0}},  // 3超出了范围
                      {{1, 0}, {0, 1}}});

    Eigen::Tensor<float, 3> updates(2, 2, 2);
    updates.setValues({{{0.1f, 0.2f}, {0.3f, 0.4f}},
                      {{0.5f, 0.6f}, {0.7f, 0.8f}}});

    bool exception_caught = false;
    try {
      auto result = onnx::ScatterElements<float, 3>::Compute(
          data, indices, updates, 1, "add");
    } catch (const std::out_of_range&) {
      exception_caught = true;
    }
    REQUIRE(exception_caught);
  }
}