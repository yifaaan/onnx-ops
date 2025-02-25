#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

#include "ops/scatter_elements.h"

using Catch::Approx;

TEST_CASE("2D Scatter Elements Basic Operations", "[scatter][2d]")
{
    SECTION("Column-wise scatter (axis=1)")
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                   7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
        std::vector<int64_t> data_shape = {4, 3};

        std::vector<int64_t> indices = {1, 0, 2, 2, 1, 0, 0, 2, 1, 1, 0, 2};
        std::vector<int64_t> indices_shape = {4, 3};

        std::vector<float> updates = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                                      0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};
        std::vector<int64_t> updates_shape = {4, 3};

        auto result_add = onnx::ScatterElements<float>::Compute(
            data, data_shape, indices, indices_shape, updates, updates_shape, 1, "add");

        REQUIRE(result_add[1] == Approx(2.1f));
        REQUIRE(result_add[5] == Approx(6.4f));
        REQUIRE(result_add[6] == Approx(7.7f));

        auto result_replace = onnx::ScatterElements<float>::Compute(
            data, data_shape, indices, indices_shape, updates, updates_shape, 1, "none");

        REQUIRE(result_replace[1] == Approx(0.1f));
        REQUIRE(result_replace[5] == Approx(0.4f));
        REQUIRE(result_replace[6] == Approx(0.7f));
    }

    SECTION("Row-wise scatter (axis=0)")
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                   7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
        std::vector<int64_t> data_shape = {3, 4};

        std::vector<int64_t> indices = {2, 0, 1, 0, 1, 2, 0, 1};
        std::vector<int64_t> indices_shape = {2, 4};

        std::vector<float> updates = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
        std::vector<int64_t> updates_shape = {2, 4};

        auto result_mul = onnx::ScatterElements<float>::Compute(
            data, data_shape, indices, indices_shape, updates, updates_shape, 0, "mul");

        REQUIRE(result_mul[8] == Approx(0.9f)); // (2,0)位置: 9.0 * 0.1
        REQUIRE(result_mul[1] == Approx(0.4f)); // (0,1)位置: 2.0 * 0.2
        REQUIRE(result_mul[6] == Approx(2.1f)); // (1,2)位置: 7.0 * 0.3
    }
}

TEST_CASE("3D Scatter Elements Operations", "[scatter][3d]")
{
    SECTION("3D scatter (axis=1)")
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                   7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
        std::vector<int64_t> data_shape = {2, 3, 2};
        /*
         [
          [
            [1, 2],
            [3, 4],
            [5, 6]
          ],

          [
            [7, 8],
            [9, 10],
            [11, 12]
          ]
         ]
        */

        std::vector<int64_t> indices = {1, 0, 0, 1, 0, 1, 1, 0};
        /*
         [
          [
            [1, 0],
            [0, 1],
          ],
          [
            [0, 1],
            [1, 0],
          ]
         ]
        */
        std::vector<int64_t> indices_shape = {2, 2, 2};

        std::vector<float> updates = {0.1f, 0.2f, 0.3f, 0.4f, 0.7f, 0.8f, 0.9f, 1.0f};

        /*
         [
          [
            [0.1, 0.2],
            [0.3, 0.4],
          ],
          [
            [0.7, 0.8],
            [0.9, 1.0],
          ]
         ]
        */
        std::vector<int64_t> updates_shape = {2, 2, 2};

        auto result_add = onnx::ScatterElements<float>::Compute(
            data, data_shape, indices, indices_shape, updates, updates_shape, 1, "add");

        REQUIRE(result_add == std::vector<float>({1.3f, 2.2f, 3.1f, 4.4f, 5.0f, 6.0f, 7.7f, 9.0f,
                                                  9.9f, 10.8f, 11.0f, 12.0f}));
    }

    SECTION("3D scatter axis 2")
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                   7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
        std::vector<int64_t> data_shape = {2, 2, 3};

        std::vector<int64_t> indices = {2, 0, 1, 1, 2, 0, 0, 1, 2, 2, 1, 0};
        std::vector<int64_t> indices_shape = {2, 2, 3};

        std::vector<float> updates = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                                      0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};
        std::vector<int64_t> updates_shape = {2, 2, 3};

        auto result_add = onnx::ScatterElements<float>::Compute(
            data, data_shape, indices, indices_shape, updates, updates_shape, 2, "add");

        REQUIRE(result_add ==
                std::vector<float>{1.2, 2.3, 3.1, 4.6, 5.4, 6.5, 7.7, 8.8, 9.9, 11.2, 12.1, 13.0});
    }
}

TEST_CASE("4D Scatter Elements Operations", "[scatter][4d]")
{
    SECTION("4D scatter (axis=2)")
    {
        std::vector<float> data = {
            // batch 0
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
            // batch 1
            13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f};
        std::vector<int64_t> data_shape = {2, 2, 3, 2};

        std::vector<int64_t> indices = {// batch 0
                                        1, 2, 0, 1, 2, 0, 0, 1, 2, 0, 1, 2,
                                        // batch 1
                                        2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0};
        std::vector<int64_t> indices_shape = {2, 2, 3, 2};

        std::vector<float> updates = {
            // batch 0
            0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f,
            // batch 1
            1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.1f, 2.2f, 2.3f, 2.4f};
        std::vector<int64_t> updates_shape = {2, 2, 3, 2};

        auto result_add = onnx::ScatterElements<float>::Compute(
            data, data_shape, indices, indices_shape, updates, updates_shape, 2, "add");
        REQUIRE(result_add == std::vector<float>{1.3f, 2.6f, 3.1f, 4.4f, 5.5f, 6.2f,

                                                 7.7f, 9.0f, 10.1f, 10.8f, 11.9f, 13.2f,

                                                 // batch 1
                                                 14.5f, 15.8f, 16.7f, 17.4f, 18.3f, 19.6f,

                                                 21.1f, 22.4f, 23.3f, 24.0f, 24.9f, 26.2f});
    }
}