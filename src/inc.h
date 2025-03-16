#pragma once

#include <vector>

#include <memory>





#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>


std::vector<std::vector<int64_t>> nonMaxSuppression(
    const std::vector<std::vector<std::vector<float>>>& boxes,
    const std::vector<std::vector<std::vector<float>>>& scores,
    int64_t max_output_boxes_per_class = 0,
    float iou_threshold = 0.0f,
    float score_threshold = 0.0f,
    int center_point_box = 0);

float calculateIOU(const std::vector<float>& box1,
    const std::vector<float>& box2,
    bool center_point_box);

template <typename T>
class Tensor
{
public:
    Tensor() = default;
    // ���캯��������������ά��
    Tensor(const std::vector<int64_t>& dims, const std::vector<T>& data) : dims(dims), data(data)
    {
        size_t total_size = 1;
        for (auto dim : dims)
        {
            if (dim <= 0)
                throw std::invalid_argument("Dimensions must be positive");
            total_size *= dim;
        }
        // data.resize(total_size);
    }
    ////
    /// ����������������е�Ԫ��?
    // T& operator()(const std::vector<int>& indices) {
    //     size_t flat_index = linearIndex(indices);
    //     return data[flat_index];
    // }
    //   const T& operator()(const std::vector<int>& indices) const {
    //     size_t flat_index = linearIndex(indices);
    //     return data[flat_index];
    // }

    // ��ȡά��
    const std::vector<int64_t>& getDims() const { return dims; }

    // �������?
    void setRandom()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
        {
            std::uniform_real_distribution<T> dis(-1.0, 1.0);
            for (auto& value : data)
            {
                value = dis(gen);
            }
        }
        else if constexpr (std::is_same_v<T, bool>)
        {
            std::uniform_int_distribution<int64_t> dis(0, 1);
            for (auto& value : data)
            {
                value = dis(gen) != 0;
            }
        }
        else if constexpr (std::is_integral_v<T>)
        {
            std::uniform_int_distribution<T> dis(-10, 10);
            for (auto& value : data)
            {
                value = dis(gen);
            }
        }
    }

    const std::vector<T>& getData() const { return data; }

    // ��ӡ����
    void print() const
    {
        std::cout << "Tensor shape: [";
        for (size_t i = 0; i < dims.size(); ++i)
        {
            std::cout << dims[i] << (i < dims.size() - 1 ? " " : "");
        }
        std::cout << "]\n";
        printRecursive(0, {});
        std::cout << std::endl;
    }

private:
    size_t linearIndex(const std::vector<int64_t>& indices) const
    {
        if (indices.size() != dims.size())
        {
            throw std::invalid_argument("Index dimension mismatch");
        }
        size_t index = 0;
        size_t stride = 1;

        for (int i = dims.size() - 1; i >= 0; --i)
        {
            if (indices[i] < 0 || indices[i] >= dims[i])
            {
                throw std::out_of_range("Index out of bounds");
            }
            index += indices[i] * stride;
            stride *= dims[i];
        }
        return index;
    }

    // �ݹ���?
    void printRecursive(int dim, std::vector<int64_t> indices) const
    {
        if (dim == dims.size())
        {
            std::cout << data[linearIndex(indices)];
            return;
        }
        std::cout << std::string(dim * 2, ' ') << "[";
        for (int i = 0; i < dims[dim]; ++i)
        {
            indices.push_back(i);
            printRecursive(dim + 1, indices);
            indices.pop_back();
            if (i < dims[dim] - 1)
            {
                std::cout << (dim == dims.size() - 1 ? " " : "\n" + std::string(dim * 2 + 2, ' '));
            }
        }
        std::cout << "]";
        if (dim == 0)
            std::cout << "\n";
    }
    std::vector<int64_t> dims; // ������ά��
    std::vector<T> data;   // ���������ݴ洢
};

// ������
template <typename T>
std::vector<Tensor<T>> SequenceConstruct(const Tensor<T>& first)
{
    std::vector<Tensor<T>> sequence;
    sequence.push_back(first);
    return sequence;
}

// ������
template <typename T, typename T2, typename... Ts>
std::vector<Tensor<T>> SequenceConstruct(const Tensor<T>& first, const Tensor<T2>& second,
                                         const Ts&... others)
{

    static_assert(std::is_same_v<T, T2>, "All input tensors must have the same data type");
    constexpr size_t num_tensors = 2 + sizeof...(Ts);
    if (num_tensors > 2147483647)
    {
        throw std::invalid_argument("Number of input tensors exceeds maximum limit of 2147483647");
    }
    auto sequence = SequenceConstruct(second, others...);
    sequence.insert(sequence.begin(), first); // �ڿ�ͷ�����һ������?
    return sequence;
}

// SequenceAt����ʵ��
template <typename T>
Tensor<T> SequenceAt(const std::vector<Tensor<T>>& input_sequence, int64_t position)
{
    size_t n = input_sequence.size();
    // ������
    if (position < 0)
    {
        position += n;
    }
    // ����Ƿ�Խ��?
    if (position < 0 || position >= n)
    {
        throw std::out_of_range("Position out of range.");
    }
    // ����ָ��λ�õ�����
    return input_sequence[position];
}

// SequenceInsert����ʵ��
template <typename T>
void SequenceInsert(std::vector<Tensor<T>>& input_sequence, Tensor<T> Insert_Tensor, int64_t position)
{
    size_t n = input_sequence.size();
    if (position < 0)
        position += n;
    if (position < 0 || position > n)
    {
        throw std::out_of_range("Position out of range");
    }

    input_sequence.insert(input_sequence.begin() + position, Insert_Tensor);
}
template <typename T>
void SequenceInsert(std::vector<Tensor<T>>& input_sequence, Tensor<T> Insert_Tensor)
{
    input_sequence.insert(input_sequence.end(), Insert_Tensor);
}

// SequenceErase����ʵ��
template <typename T>
void SequenceErase(std::vector<Tensor<T>>& input_sequence, int64_t position)
{
    size_t n = input_sequence.size();
    if (position < 0)
        position += n;
    if (position < 0 || position >= n)
    {
        throw std::out_of_range("Position out of range");
    }
    input_sequence.erase(input_sequence.begin() + position);
}
template <typename T>
void SequenceErase(std::vector<Tensor<T>>& input_sequence)
{
    input_sequence.erase(input_sequence.end() - 1);
}