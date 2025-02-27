#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "ops/gather_nd.h"

TEST_CASE("GatherND Basic Operations", "[gather_nd]")
{
    SECTION("2D data, 2D indices, last_dim=1 (Python Test Case 1)")
    {
        // data shape: [3, 2]
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<int64_t> data_shape = {3, 2};

        // indices shape: [2, 1]
        std::vector<int64_t> indices = {0, 1};
        std::vector<int64_t> indices_shape = {2, 1};

        auto [result, shape] =
            onnx::GatherND<float>::Compute(data, data_shape, indices, indices_shape);

        // 输出形状应该是 [2, 2]
        REQUIRE(shape == std::vector<int64_t>{2, 2});
        // 结果应该是 [[1.0, 2.0], [3.0, 4.0]]
        // Python结果: [[1. 2.], [3. 4.]]
        REQUIRE(result == std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
    }

    SECTION("2D data, 2D indices, last_dim=2 (Python Test Case 2)")
    {
        // data shape: [3, 2]
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<int64_t> data_shape = {3, 2};

        // indices shape: [2, 2]
        std::vector<int64_t> indices = {0, 0, 1, 1};
        std::vector<int64_t> indices_shape = {2, 2};

        auto [result, shape] =
            onnx::GatherND<float>::Compute(data, data_shape, indices, indices_shape);

        // 输出形状应该是 [2]
        REQUIRE(shape == std::vector<int64_t>{2});
        // 结果应该是 [1.0, 4.0]
        // Python结果: [1. 4.]
        REQUIRE(result == std::vector<float>{1.0f, 4.0f});
    }

    SECTION("3D data, 2D indices, last_dim=2 (Python Test Case 3)")
    {
        // data shape: [2, 2, 2]
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        std::vector<int64_t> data_shape = {2, 2, 2};

        // indices shape: [2, 2]
        std::vector<int64_t> indices = {0, 0, 1, 1};
        std::vector<int64_t> indices_shape = {2, 2};

        auto [result, shape] =
            onnx::GatherND<float>::Compute(data, data_shape, indices, indices_shape);

        // 输出形状应该是 [2, 2]
        REQUIRE(shape == std::vector<int64_t>{2, 2});
        // 结果应该是 [[1.0, 2.0], [7.0, 8.0]]
        // Python结果: [[1. 2.], [7. 8.]]
        REQUIRE(result == std::vector<float>{1.0f, 2.0f, 7.0f, 8.0f});
    }

    SECTION("3D data, 2D indices, last_dim=3 (Python Test Case 4)")
    {
        // data shape: [2, 2, 2]
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        std::vector<int64_t> data_shape = {2, 2, 2};

        // indices shape: [1, 3]
        std::vector<int64_t> indices = {0, 1, 0};
        std::vector<int64_t> indices_shape = {1, 3};

        auto [result, shape] =
            onnx::GatherND<float>::Compute(data, data_shape, indices, indices_shape);

        // 输出形状应该是 [1]
        REQUIRE(shape == std::vector<int64_t>{1});
        // 结果应该是 [3.0]
        // Python结果: [3.]
        REQUIRE(result == std::vector<float>{3.0f});
    }

    SECTION("3D data, 3D indices, last_dim=2 (Python Test Case 5)")
    {
        // data shape: [2, 2, 2]
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        std::vector<int64_t> data_shape = {2, 2, 2};

        // indices shape: [2, 1, 2]
        std::vector<int64_t> indices = {0, 0, 1, 1};
        std::vector<int64_t> indices_shape = {2, 1, 2};

        auto [result, shape] =
            onnx::GatherND<float>::Compute(data, data_shape, indices, indices_shape);

        // 输出形状应该是 [2, 1, 2]
        REQUIRE(shape == std::vector<int64_t>{2, 1, 2});
        // 结果应该是 [[[1.0, 2.0]], [[7.0, 8.0]]]
        // Python结果: [[[1. 2.]], [[7. 8.]]]
        REQUIRE(result == std::vector<float>{1.0f, 2.0f, 7.0f, 8.0f});
    }

    SECTION("With negative indices (Python Test Case 6)")
    {
        // data shape: [3, 2]
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<int64_t> data_shape = {3, 2};

        // indices shape: [2, 1]
        std::vector<int64_t> indices = {-3, -2};
        std::vector<int64_t> indices_shape = {2, 1};

        auto [result, shape] =
            onnx::GatherND<float>::Compute(data, data_shape, indices, indices_shape);

        // 输出形状应该是 [2, 2]
        REQUIRE(shape == std::vector<int64_t>{2, 2});
        // 结果应该是 [[1.0, 2.0], [3.0, 4.0]]
        // Python结果: [[1. 2.], [3. 4.]]
        REQUIRE(result == std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
    }

    SECTION("With batch_dims=1 (Python Test Case 7)")
    {
        // data shape: [2, 2, 3]
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                   7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
        std::vector<int64_t> data_shape = {2, 2, 3};

        // indices shape: [2, 1, 1]
        std::vector<int64_t> indices = {0, 1};
        std::vector<int64_t> indices_shape = {2, 1, 1};

        auto [result, shape] =
            onnx::GatherND<float>::Compute(data, data_shape, indices, indices_shape, 1);

        // 输出形状应该是 [2, 1, 3]
        REQUIRE(shape == std::vector<int64_t>{2, 1, 3});
        // 结果应该是 [[[1.0, 2.0, 3.0]], [[10.0, 11.0, 12.0]]]
        // Python结果: [[[1. 2. 3.]], [[10. 11. 12.]]]
        REQUIRE(result == std::vector<float>{1.0f, 2.0f, 3.0f, 10.0f, 11.0f, 12.0f});
    }
}

TEST_CASE("GatherND Error Handling", "[gather_nd][error]")
{
    SECTION("Invalid indices last dimension")
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<int64_t> data_shape = {2, 2};
        std::vector<int64_t> indices = {0, 1, 2, 3};
        std::vector<int64_t> indices_shape = {2, 2};

        REQUIRE_THROWS_AS(onnx::GatherND<float>::Compute(data, data_shape, indices, indices_shape),
                          std::out_of_range);
    }

    SECTION("Index out of bounds")
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<int64_t> data_shape = {2, 2};
        std::vector<int64_t> indices = {0, 3};
        std::vector<int64_t> indices_shape = {1, 2};

        REQUIRE_THROWS_AS(onnx::GatherND<float>::Compute(data, data_shape, indices, indices_shape),
                          std::out_of_range);
    }

    SECTION("Invalid batch_dims")
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<int64_t> data_shape = {2, 2};
        std::vector<int64_t> indices = {0, 1};
        std::vector<int64_t> indices_shape = {2, 1};

        REQUIRE_THROWS_AS(
            onnx::GatherND<float>::Compute(data, data_shape, indices, indices_shape, 2),
            std::invalid_argument);
    }

    SECTION("Mismatched batch dimensions")
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<int64_t> data_shape = {3, 2};
        std::vector<int64_t> indices = {0, 1};
        std::vector<int64_t> indices_shape = {2, 1};

        REQUIRE_THROWS_AS(
            onnx::GatherND<float>::Compute(data, data_shape, indices, indices_shape, 1),
            std::invalid_argument);
    }
}