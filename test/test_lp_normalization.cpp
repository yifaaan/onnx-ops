#include "ops/lp_normalization.h"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iostream>
#include <vector>

using namespace onnx;
using Catch::Approx;

TEST_CASE("2D LpNormalization基本操作", "[lp_normalization][2d]")
{
    SECTION("基本2D归一化 (axis=1, p=2)")
    {
        // 创建一个简单的2D输入
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<int64_t> shape = {2, 3};
        int64_t axis = 1; // 默认轴

        // 计算LpNormalization
        auto result = LpNormalization<float>::Compute(data, shape, axis, 2);

        // 验证输出形状与输入相同
        REQUIRE(result.second == shape);

        // 验证输出值符合预期
        float norm1 = std::sqrt(1 * 1 + 2 * 2 + 3 * 3);
        float norm2 = std::sqrt(4 * 4 + 5 * 5 + 6 * 6);

        REQUIRE(result.first[0] == Approx(1.0f / norm1));
        REQUIRE(result.first[1] == Approx(2.0f / norm1));
        REQUIRE(result.first[2] == Approx(3.0f / norm1));
        REQUIRE(result.first[3] == Approx(4.0f / norm2));
        REQUIRE(result.first[4] == Approx(5.0f / norm2));
        REQUIRE(result.first[5] == Approx(6.0f / norm2));
    }

    SECTION("基本2D归一化 (axis=1, p=1)")
    {
        // 创建一个简单的2D输入
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<int64_t> shape = {2, 3};
        int64_t axis = 1;

        // 计算LpNormalization使用L1范数
        auto result = LpNormalization<float>::Compute(data, shape, axis, 1);

        // 验证输出形状与输入相同
        REQUIRE(result.second == shape);

        // 验证输出值符合预期
        float norm1 = 1.0f + 2.0f + 3.0f;
        float norm2 = 4.0f + 5.0f + 6.0f;

        REQUIRE(result.first[0] == Approx(1.0f / norm1));
        REQUIRE(result.first[1] == Approx(2.0f / norm1));
        REQUIRE(result.first[2] == Approx(3.0f / norm1));
        REQUIRE(result.first[3] == Approx(4.0f / norm2));
        REQUIRE(result.first[4] == Approx(5.0f / norm2));
        REQUIRE(result.first[5] == Approx(6.0f / norm2));
    }

    SECTION("2D归一化 (axis=0, p=2)")
    {
        // 创建一个简单的2D输入
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<int64_t> shape = {2, 3};
        int64_t axis = 0;

        // 计算LpNormalization沿axis=0
        auto result = LpNormalization<float>::Compute(data, shape, axis, 2);

        // 验证输出形状与输入相同
        REQUIRE(result.second == shape);

        // 验证输出值符合预期
        float col1_norm = std::sqrt(1 * 1 + 4 * 4);
        float col2_norm = std::sqrt(2 * 2 + 5 * 5);
        float col3_norm = std::sqrt(3 * 3 + 6 * 6);

        REQUIRE(result.first[0] == Approx(1.0f / col1_norm));
        REQUIRE(result.first[1] == Approx(2.0f / col2_norm));
        REQUIRE(result.first[2] == Approx(3.0f / col3_norm));
        REQUIRE(result.first[3] == Approx(4.0f / col1_norm));
        REQUIRE(result.first[4] == Approx(5.0f / col2_norm));
        REQUIRE(result.first[5] == Approx(6.0f / col3_norm));
    }
}

TEST_CASE("3D LpNormalization操作", "[lp_normalization][3d]")
{
    SECTION("3D归一化 (axis=1, p=2)")
    {
        // 创建一个3D输入: 2x3x2
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                   7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
        std::vector<int64_t> shape = {2, 3, 2};
        int64_t axis = 1; // 第2个轴(从0开始计数)

        // 计算LpNormalization
        auto result = LpNormalization<float>::Compute(data, shape, axis, 2);

        // 验证输出形状与输入相同
        REQUIRE(result.second == shape);

        // 批次0，第0个空间位置
        float norm_b0_s0 = std::sqrt(1 * 1 + 3 * 3 + 5 * 5);
        // 批次0，第1个空间位置
        float norm_b0_s1 = std::sqrt(2 * 2 + 4 * 4 + 6 * 6);
        // 批次1，第0个空间位置
        float norm_b1_s0 = std::sqrt(7 * 7 + 9 * 9 + 11 * 11);
        // 批次1，第1个空间位置
        float norm_b1_s1 = std::sqrt(8 * 8 + 10 * 10 + 12 * 12);

        // 验证几个关键点的值
        REQUIRE(result.first[0] == Approx(1.0f / norm_b0_s0));
        REQUIRE(result.first[1] == Approx(2.0f / norm_b0_s1));
        REQUIRE(result.first[6] == Approx(7.0f / norm_b1_s0));
    }

    SECTION("3D归一化 (axis=0, p=2)")
    {
        // 创建一个3D输入: 2x3x2
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                   7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
        std::vector<int64_t> shape = {2, 3, 2};
        int64_t axis = 0; // 第1个轴(从0开始计数)

        // 计算LpNormalization
        auto result = LpNormalization<float>::Compute(data, shape, axis, 2);

        // 验证输出形状与输入相同
        REQUIRE(result.second == shape);

        // 对特定点进行验证
        float norm_c0_s0 = std::sqrt(1 * 1 + 7 * 7);
        REQUIRE(result.first[0] == Approx(1.0f / norm_c0_s0));
        REQUIRE(result.first[6] == Approx(7.0f / norm_c0_s0));
    }

    SECTION("3D归一化 (axis=2, p=2)")
    {
        // 创建一个3D输入: 2x3x2
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                   7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
        std::vector<int64_t> shape = {2, 3, 2};
        int64_t axis = 2; // 第3个轴(从0开始计数)

        // 计算LpNormalization
        auto result = LpNormalization<float>::Compute(data, shape, axis, 2);

        // 验证输出形状与输入相同
        REQUIRE(result.second == shape);

        // 对特定点进行验证
        float norm_b0_c0 = std::sqrt(1 * 1 + 2 * 2);
        REQUIRE(result.first[0] == Approx(1.0f / norm_b0_c0));
        REQUIRE(result.first[1] == Approx(2.0f / norm_b0_c0));
    }

    SECTION("3D归一化 (axis=1, p=1)")
    {
        // 创建一个3D输入: 2x3x2
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                   7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
        std::vector<int64_t> shape = {2, 3, 2};
        int64_t axis = 1;

        // 计算LpNormalization使用L1范数
        auto result = LpNormalization<float>::Compute(data, shape, axis, 1);

        // 验证输出形状与输入相同
        REQUIRE(result.second == shape);

        // 批次0，第0个空间位置
        float norm_b0_s0 = 1.0f + 3.0f + 5.0f;
        REQUIRE(result.first[0] == Approx(1.0f / norm_b0_s0));
    }
}

TEST_CASE("4D LpNormalization操作", "[lp_normalization][4d]")
{
    SECTION("4D归一化 (axis=1, p=2)")
    {
        // 创建一个4D输入: 2x2x2x2
        std::vector<float> data = {// batch 0
                                   1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                   // batch 1
                                   9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
        std::vector<int64_t> shape = {2, 2, 2, 2};
        int64_t axis = 1; // channel维度

        // 计算LpNormalization
        auto result = LpNormalization<float>::Compute(data, shape, axis, 2);

        // 验证输出形状与输入相同
        REQUIRE(result.second == shape);

        // 验证特定点
        // 批次0,位置(0,0)的两个通道: [1, 5]
        float norm_b0_h0_w0 = std::sqrt(1 * 1 + 5 * 5);
        REQUIRE(result.first[0] == Approx(1.0f / norm_b0_h0_w0));
        REQUIRE(result.first[4] == Approx(5.0f / norm_b0_h0_w0));
    }

    SECTION("4D归一化 (axis=2, p=2)")
    {
        // 创建一个4D输入: 2x2x2x2
        std::vector<float> data = {// batch 0
                                   1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                   // batch 1
                                   9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
        std::vector<int64_t> shape = {2, 2, 2, 2};
        int64_t axis = 2; // height维度

        // 计算LpNormalization
        auto result = LpNormalization<float>::Compute(data, shape, axis, 2);

        // 验证输出形状与输入相同
        REQUIRE(result.second == shape);

        // 验证特定点
        // 批次0,通道0,位置(0)和位置(1): [1, 3]
        float norm_b0_c0_w0 = std::sqrt(1 * 1 + 3 * 3);
        REQUIRE(result.first[0] == Approx(1.0f / norm_b0_c0_w0));
        REQUIRE(result.first[2] == Approx(3.0f / norm_b0_c0_w0));
    }

    SECTION("4D归一化 (axis=3, p=2)")
    {
        // 创建一个4D输入: 2x2x2x2
        std::vector<float> data = {// batch 0
                                   1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                   // batch 1
                                   9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
        std::vector<int64_t> shape = {2, 2, 2, 2};
        int64_t axis = 3; // width维度

        // 计算LpNormalization
        auto result = LpNormalization<float>::Compute(data, shape, axis, 2);

        // 验证输出形状与输入相同
        REQUIRE(result.second == shape);

        // 验证特定点
        // 批次0,通道0,行0: [1, 2]
        float norm_b0_c0_h0 = std::sqrt(1 * 1 + 2 * 2);
        REQUIRE(result.first[0] == Approx(1.0f / norm_b0_c0_h0));
        REQUIRE(result.first[1] == Approx(2.0f / norm_b0_c0_h0));
    }
}

TEST_CASE("5D LpNormalization操作 (1-48数据)", "[lp_normalization][5d]")
{
    SECTION("5D归一化 (axis=1, p=2)")
    {
        // 使用1-48的连续数字填充张量，模拟来自用户查询的5D测试数据
        std::vector<float> data;
        for (int i = 1; i <= 48; ++i)
        {
            data.push_back(static_cast<float>(i));
        }
        std::vector<int64_t> shape = {2, 3, 2, 2, 2};
        int64_t axis = 1; // 沿着通道维度归一化

        // 计算L2 LpNormalization
        auto result_l2 = LpNormalization<float>::Compute(data, shape, axis, 2);

        // 计算L1 LpNormalization
        auto result_l1 = LpNormalization<float>::Compute(data, shape, axis, 1);

        // 验证几个关键位置的值
        // 第一个切片: [1, 9, 17]
        float l2_norm_first = std::sqrt(1 * 1 + 9 * 9 + 17 * 17);
        float l1_norm_first = 1 + 9 + 17;

        REQUIRE(result_l2.first[0] == Approx(1.0f / l2_norm_first));
        REQUIRE(result_l2.first[8] == Approx(9.0f / l2_norm_first));
        REQUIRE(result_l2.first[16] == Approx(17.0f / l2_norm_first));

        REQUIRE(result_l1.first[0] == Approx(1.0f / l1_norm_first));
        REQUIRE(result_l1.first[8] == Approx(9.0f / l1_norm_first));
        REQUIRE(result_l1.first[16] == Approx(17.0f / l1_norm_first));
    }

    SECTION("5D归一化 (axis=2, p=2)")
    {
        // 使用1-48的连续数字填充张量
        std::vector<float> data;
        for (int i = 1; i <= 48; ++i)
        {
            data.push_back(static_cast<float>(i));
        }
        std::vector<int64_t> shape = {2, 3, 2, 2, 2};
        int64_t axis = 2; // 沿着第3个维度归一化

        // 计算L2 LpNormalization
        auto result = LpNormalization<float>::Compute(data, shape, axis, 2);

        // 验证输出形状与输入相同
        REQUIRE(result.second == shape);
    }
}

TEST_CASE("LpNormalization边界情况测试", "[lp_normalization][边界情况]")
{
    SECTION("零向量测试")
    {
        // 所有元素为0的输入
        std::vector<float> data = {0.0f, 0.0f, 0.0f, 0.0f};
        std::vector<int64_t> shape = {2, 2};
        int64_t axis = 1;

        // 计算LpNormalization
        auto result = LpNormalization<float>::Compute(data, shape, axis, 2);

        // 验证零向量应该保持为零
        for (auto& val : result.first)
        {
            REQUIRE(val == 0.0f);
        }
    }

    SECTION("单元素向量测试")
    {
        // 每个向量只有一个元素
        std::vector<float> data = {3.0f, 4.0f};
        std::vector<int64_t> shape = {2, 1};
        int64_t axis = 1;

        // 计算LpNormalization
        auto result = LpNormalization<float>::Compute(data, shape, axis, 2);

        // 单元素应该标准化为1.0
        REQUIRE(result.first[0] == 1.0f);
        REQUIRE(result.first[1] == 1.0f);
    }

    SECTION("负轴测试")
    {
        // 使用负轴索引
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<int64_t> shape = {2, 2};
        int64_t axis = -1; // 应该等同于1

        // 计算LpNormalization
        auto result = LpNormalization<float>::Compute(data, shape, axis, 2);

        // 验证结果与使用轴=1的结果相同
        float norm1 = std::sqrt(1 * 1 + 2 * 2);
        float norm2 = std::sqrt(3 * 3 + 4 * 4);

        REQUIRE(result.first[0] == Approx(1.0f / norm1));
        REQUIRE(result.first[1] == Approx(2.0f / norm1));
        REQUIRE(result.first[2] == Approx(3.0f / norm2));
        REQUIRE(result.first[3] == Approx(4.0f / norm2));
    }
}

TEST_CASE("LpNormalization错误处理测试", "[lp_normalization][错误处理]")
{
    SECTION("无效p值测试")
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<int64_t> shape = {2, 2};
        int64_t axis = 1;

        // p必须是1或2，使用其他值应该抛出异常
        REQUIRE_THROWS_AS(LpNormalization<float>::Compute(data, shape, axis, 3),
                          std::invalid_argument);
    }

    SECTION("超出范围的轴测试")
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<int64_t> shape = {2, 2};
        int64_t axis = 2; // 超出维度范围

        // 应该抛出异常
        REQUIRE_THROWS_AS(LpNormalization<float>::Compute(data, shape, axis, 2),
                          std::invalid_argument);
    }
}