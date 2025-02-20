#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "ops/gather_element.h"


TEST_CASE("GatherElements Basic Operations", "[gather_elements]") {
  SECTION("1D Basic Gather") {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<int64_t> data_shape = {5};
    
    std::vector<int64_t> indices = {4, 3, 2, 1, 0};
    std::vector<int64_t> indices_shape = {5};
    
    auto result = onnx::GatherElements<float>::Compute(
        data, data_shape, indices, indices_shape, 0);
    
    REQUIRE(result == std::vector<float>{5.0f, 4.0f, 3.0f, 2.0f, 1.0f});
  }

  SECTION("2D Gather along axis 0") {
    std::vector<float> data = {1.0f, 2.0f, 3.0f,
                              4.0f, 5.0f, 6.0f};
    std::vector<int64_t> data_shape = {2, 3};
    
    std::vector<int64_t> indices = {1, 0, 1,
                                   0, 1, 0};
    std::vector<int64_t> indices_shape = {2, 3};
    
    auto result = onnx::GatherElements<float>::Compute(
        data, data_shape, indices, indices_shape, 0);
    
    REQUIRE(result.first == std::vector<float>{4.0f, 2.0f, 6.0f,
                                       1.0f, 5.0f, 3.0f});
    REQUIRE(result.second == std::vector<int64_t>{2, 3});
  }

  SECTION("2D Gather along axis 1") {
    std::vector<float> data = {1.0f, 2.0f, 3.0f,
                              4.0f, 5.0f, 6.0f};
    std::vector<int64_t> data_shape = {2, 3};
    
    std::vector<int64_t> indices = {2, 1, 0,
                                   0, 2, 1};
    std::vector<int64_t> indices_shape = {2, 3};
    
    auto result = onnx::GatherElements<float>::Compute(
        data, data_shape, indices, indices_shape, 1);
    
    REQUIRE(result.first == std::vector<float>{3.0f, 2.0f, 1.0f,
                                       4.0f, 6.0f, 5.0f});
    REQUIRE(result.second == std::vector<int64_t>{2, 3});
  }

  SECTION("3D Gather along axis 1") {
    // 输入数据 (2,3,2)
    std::vector<float> data = {1.0f, 2.0f,   
                              3.0f, 4.0f,    
                              5.0f, 6.0f,    
                              
                              7.0f, 8.0f,    
                              9.0f, 10.0f,   
                              11.0f, 12.0f}; 
    std::vector<int64_t> data_shape = {2, 3, 2};
    
    // indices (2,3,2)
    std::vector<int64_t> indices = {1, 0, 2,  
                                   0, 2, 1,   
                                   
                                   2, 1, 0,   
                                   1, 0, 2}; 
    std::vector<int64_t> indices_shape = {2, 3, 2};
    
    auto result = onnx::GatherElements<float>::Compute(
        data, data_shape, indices, indices_shape, 1);
    

    
    std::vector<float> expected = {
        3.0f, 2.0f,    
        5.0f, 2.0f,   
        5.0f, 4.0f,    
        
        11.0f, 10.0f, 
        7.0f, 10.0f,
        7.0f, 12.0f   
    };
    
    REQUIRE(result.first == expected);
    REQUIRE(result.second == std::vector<int64_t>{2, 3});
  }
}

TEST_CASE("GatherElements Error Handling", "[gather_elements][error]") {
  SECTION("Invalid axis") {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int64_t> data_shape = {2, 2};
    std::vector<int64_t> indices = {0, 1, 1, 0};
    std::vector<int64_t> indices_shape = {2, 2};
    
    REQUIRE_THROWS_AS(
        onnx::GatherElements<float>::Compute(
            data, data_shape, indices, indices_shape, 2),
        std::invalid_argument);
  }

  SECTION("Index out of bounds") {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int64_t> data_shape = {2, 2};
    std::vector<int64_t> indices = {0, 3, 1, 0};  // 3 is out of bounds
    std::vector<int64_t> indices_shape = {2, 2};
    
    REQUIRE_THROWS_AS(
        onnx::GatherElements<float>::Compute(
            data, data_shape, indices, indices_shape, 1),
        std::out_of_range);
  }


  SECTION("Empty input") {
    std::vector<float> data = {};
    std::vector<int64_t> data_shape = {0};
    std::vector<int64_t> indices = {};
    std::vector<int64_t> indices_shape = {0};
    
    auto result = onnx::GatherElements<float>::Compute(
        data, data_shape, indices, indices_shape, 0);
    
    REQUIRE(result.empty());
  }
}
