#include "ops/global_max_pool.h"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <vector>

TEST_CASE("GlobalMaxPool Operation", "[global_max_pool]")
{
    SECTION("Basic 4D input (1x1x5x5)")
    {
        std::vector<float> data = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,
                                   10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                                   19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f};
        std::vector<int64_t> shape = {1, 1, 5, 5};

        auto result = onnx::GlobalMaxPool<float>::Compute(data, shape);

        std::vector<int64_t> expected_shape = {1, 1, 1, 1};
        REQUIRE(result.second == expected_shape);
        REQUIRE(result.first.size() == 1);
        REQUIRE(result.first[0] == Catch::Approx(25.0f));
    }

    SECTION("Simple 3x3 input (1x1x3x3)")
    {
        std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        std::vector<int64_t> shape = {1, 1, 3, 3};

        auto result = onnx::GlobalMaxPool<float>::Compute(data, shape);

        std::vector<int64_t> expected_shape = {1, 1, 1, 1};
        REQUIRE(result.second == expected_shape);
        REQUIRE(result.first[0] == Catch::Approx(9.0f));
    }

    SECTION("Multi-channel input (1x2x3x3)")
    {
        std::vector<float> data = {
            1, 2, 3, 4, 5, 6, 7, 8, 9, // First channel
            9, 8, 7, 6, 5, 4, 3, 2, 1  // Second channel
        };
        std::vector<int64_t> shape = {1, 2, 3, 3};

        auto result = onnx::GlobalMaxPool<float>::Compute(data, shape);

        std::vector<int64_t> expected_shape = {1, 2, 1, 1};
        REQUIRE(result.second == expected_shape);
        REQUIRE(result.first.size() == 2);
        REQUIRE(result.first[0] == Catch::Approx(9.0f)); // Max of first channel
        REQUIRE(result.first[1] == Catch::Approx(9.0f)); // Max of second channel
    }

    SECTION("Batch processing (2x1x2x2)")
    {
        std::vector<float> data = {
            1, 2, 3, 4, // First sample
            5, 6, 7, 8  // Second sample
        };
        std::vector<int64_t> shape = {2, 1, 2, 2};

        auto result = onnx::GlobalMaxPool<float>::Compute(data, shape);

        std::vector<int64_t> expected_shape = {2, 1, 1, 1};
        REQUIRE(result.second == expected_shape);
        REQUIRE(result.first.size() == 2);
        REQUIRE(result.first[0] == Catch::Approx(4.0f)); // Max of first sample
        REQUIRE(result.first[1] == Catch::Approx(8.0f)); // Max of second sample
    }

    SECTION("5D input (1x2x3x2x2)")
    {
        std::vector<float> data = {
            // First channel
            1, 2, 3, 4,    // D1
            5, 6, 7, 8,    // D2
            9, 10, 11, 12, // D3
            // Second channel
            13, 14, 15, 16, // D1
            17, 18, 19, 20, // D2
            21, 22, 23, 24  // D3
        };
        std::vector<int64_t> shape = {1, 2, 3, 2, 2};

        auto result = onnx::GlobalMaxPool<float>::Compute(data, shape);

        std::vector<int64_t> expected_shape = {1, 2, 1, 1, 1};
        REQUIRE(result.second == expected_shape);
        REQUIRE(result.first.size() == 2);
        REQUIRE(result.first[0] == Catch::Approx(12.0f)); // Max of first channel
        REQUIRE(result.first[1] == Catch::Approx(24.0f)); // Max of second channel
    }

    SECTION("5D input (2x2x3x2x2)")
    {
        std::vector<float> data = {
            /* batch 0 */
            // First channel
            1, 2, 3, 4,    // D1
            5, 6, 7, 8,    // D2
            9, 10, 11, 12, // D3
            // Second channel
            13, 14, 15, 16, // D1
            17, 18, 19, 20, // D2
            21, 22, 23, 24, // D3

            /* batch 1 */
            // First channel
            1, 2, 3, 4,    // D1
            5, 6, 7, 8,    // D2
            9, 10, 11, 12, // D3
            // Second channel
            13, 14, 15, 16, // D1
            17, 18, 19, 20, // D2
            21, 22, 23, 24  // D3
        };
        std::vector<int64_t> shape = {2, 2, 3, 2, 2};

        auto result = onnx::GlobalMaxPool<float>::Compute(data, shape);

        std::vector<int64_t> expected_shape = {2, 2, 1, 1, 1};
        REQUIRE(result.second == expected_shape);
        REQUIRE(result.first.size() == 4);
        // batch 0
        REQUIRE(result.first[0] == Catch::Approx(12.0f)); // Max of first channel
        REQUIRE(result.first[1] == Catch::Approx(24.0f)); // Max of second channel
        // batch 1
        REQUIRE(result.first[2] == Catch::Approx(12.0f)); // Max of first channel
        REQUIRE(result.first[3] == Catch::Approx(24.0f)); // Max of second channel
    }

    SECTION("2D input (2x3)")
    {
        std::vector<float> data = {1, 2, 3, 4, 5, 6};
        std::vector<int64_t> shape = {2, 3};

        auto result = onnx::GlobalMaxPool<float>::Compute(data, shape);

        std::vector<int64_t> expected_shape = {1};
        REQUIRE(result.second == expected_shape);
        REQUIRE(result.first[0] == Catch::Approx(6.0f));
    }

    SECTION("Invalid 1D input")
    {
        std::vector<float> data = {1, 2, 3};
        std::vector<int64_t> shape = {3};

        REQUIRE_THROWS_AS(onnx::GlobalMaxPool<float>::Compute(data, shape), std::runtime_error);
    }
}