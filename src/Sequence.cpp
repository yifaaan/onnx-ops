#include <iostream>
#include <vector>
#include <stdexcept>
#include <random>

template <typename T>
class Tensor {
public:
    Tensor() = default; 
    // 构造函数，接受张量的维度
    Tensor(const std::vector<int>& dims) : dims(dims) {
        size_t total_size = 1;
        for (int dim : dims) {
            if (dim <= 0) throw std::invalid_argument("Dimensions must be positive");
            total_size *= dim;
        }
        data.resize(total_size);
    }
    //// 根据坐标访问张量中的元素
    //T& operator()(const std::vector<int>& indices) {
    //    size_t flat_index = linearIndex(indices);
    //    return data[flat_index];
    //}
    //  const T& operator()(const std::vector<int>& indices) const {
    //    size_t flat_index = linearIndex(indices);
    //    return data[flat_index];
    //}

    // 获取维度
    const std::vector<int>& getDims() const {
        return dims;
    }
 
    // 填充张量
    void setRandom() {
        std::random_device rd;
        std::mt19937 gen(rd());
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            std::uniform_real_distribution<T> dis(-1.0, 1.0);
            for (auto& value : data) {
                value = dis(gen);
            }
        }
        else if constexpr (std::is_same_v<T, bool>) {
            std::uniform_int_distribution<int> dis(0, 1);
            for (auto& value : data) {
                value = dis(gen) != 0;
            }
        }
        else if constexpr (std::is_integral_v<T>) {
            std::uniform_int_distribution<T> dis(-10, 10);
            for (auto& value : data) {
                value = dis(gen);
            }
        }
    }

    // 打印张量
    void print() const {
        std::cout << "Tensor shape: [";
        for (size_t i = 0; i < dims.size(); ++i) {
            std::cout << dims[i] << (i < dims.size() - 1 ? " " : "");
        }
        std::cout << "]\n";
        printRecursive(0, {});
        std::cout << std::endl;
    }

private:
    
    size_t linearIndex(const std::vector<int>& indices) const {
        if (indices.size() != dims.size()) {
            throw std::invalid_argument("Index dimension mismatch");
        }
        size_t index = 0;
        size_t stride = 1;

        for (int i = dims.size() - 1; i >= 0; --i) {
            if (indices[i] < 0 || indices[i] >= dims[i]) {
                throw std::out_of_range("Index out of bounds");
            }
            index += indices[i] * stride;
            stride *= dims[i];
        }
        return index;
    }

    // 递归打印
    void printRecursive(int dim, std::vector<int> indices) const {
        if (dim == dims.size()) {
            std::cout << data[linearIndex(indices)];
            return;
        }
        std::cout << std::string(dim * 2, ' ') << "[";
        for (int i = 0; i < dims[dim]; ++i) {
            indices.push_back(i);
            printRecursive(dim + 1, indices);
            indices.pop_back();
            if (i < dims[dim] - 1) {
                std::cout << (dim == dims.size() - 1 ? " " : "\n" + std::string(dim * 2 + 2, ' '));
            }
        }
        std::cout << "]";
        if (dim == 0) std::cout << "\n";
    }
    std::vector<int> dims;  // 张量的维度
    std::vector<T> data;    // 张量的数据存储
};



// 单张量
template <typename T>
std::vector<Tensor<T>> SequenceConstruct(const Tensor<T>& first) {
    std::vector<Tensor<T>> sequence;
    sequence.push_back(first);
    return sequence;
}

// 多张量
template <typename T, typename T2, typename... Ts>
std::vector<Tensor<T>> SequenceConstruct(const Tensor<T>& first, const Tensor<T2>& second, const Ts&... others) {
   
    static_assert(std::is_same_v<T, T2>, "All input tensors must have the same data type");
    constexpr size_t num_tensors = 2 + sizeof...(Ts);
    if (num_tensors > 2147483647) {
        throw std::invalid_argument("Number of input tensors exceeds maximum limit of 2147483647");
    }
    auto sequence = SequenceConstruct(second, others...);
    sequence.insert(sequence.begin(), first); // 在开头插入第一个张量
    return sequence;
}

// SequenceAt算子实现
template <typename T>
Tensor<T> SequenceAt(const std::vector<Tensor<T>>& input_sequence, int position) {
    size_t n = input_sequence.size();
    //负索引
    if (position < 0) {
        position += n;
    }
    //检查是否越界
    if (position < 0 || position >= n) {
        throw std::out_of_range("Position out of range.");
    }
    // 返回指定位置的张量
    return input_sequence[position];
}


//SequenceInsert算子实现
template <typename T>
void SequenceInsert(std::vector<Tensor<T>>& input_sequence,Tensor<T> Insert_Tensor,int position) {
    size_t n = input_sequence.size();
    if (position < 0)  position += n;
    if (position < 0 || position > n) { throw std::out_of_range("Position out of range"); }
    
    input_sequence.insert(input_sequence.begin() + position, Insert_Tensor);
}
template <typename T>
void SequenceInsert(std::vector<Tensor<T>>& input_sequence, Tensor<T> Insert_Tensor) {
    input_sequence.insert(input_sequence.end(), Insert_Tensor);
    
}

//SequenceErase算子实现
template <typename T>
void SequenceErase(std::vector<Tensor<T>>& input_sequence, int position){
    size_t n = input_sequence.size();
    if (position < 0)  position += n;
    if (position < 0 || position >= n) { throw std::out_of_range("Position out of range"); }
    input_sequence.erase(input_sequence.begin() + position);
}
template <typename T>
void SequenceErase(std::vector<Tensor<T>>& input_sequence) {
    input_sequence.erase(input_sequence.end()-1);
 
}



int main() {
    ////测试SequenceAt算子
    //{
    //    // 测试2维张量
    //    std::cout << "Testing 2D Tensor:" << std::endl;
    //    Tensor<float> tensor1({ 2, 3 });
    //    tensor1.setRandom();
    //    Tensor<float> tensor2({ 3, 2 });
    //    tensor2.setRandom();
    //    std::vector<Tensor<float>> sequence2D = { tensor1, tensor2 };
    //    int position = -1;
    //    try {
    //        // 提取指定位置的张量
    //        auto result = SequenceAt(sequence2D, position);
    //        std::cout << "Extracted 2D Tensor (position " << position << "):\n";
    //        result.print();
    //    }
    //    catch (const std::exception& e) {
    //        std::cerr << "Error: " << e.what() << std::endl;
    //    }
    //    // 测试3维张量
    //    std::cout << "Testing 3D Tensor:" << std::endl;
    //    Tensor<float> tensor3({ 2, 3, 4 }); // 2x3x4 
    //    tensor3.setRandom();
    //    Tensor<float> tensor4({ 2, 2, 2 }); // 2x2x2 
    //    tensor4.setRandom();
    //    std::vector<Tensor<float>> sequence3D = { tensor3, tensor4 };
    //    position = 1;
    //    try {
    //        auto result3D = SequenceAt(sequence3D, position);
    //        std::cout << "Extracted 3D Tensor (position " << position << "):\n";
    //        result3D.print();
    //    }
    //    catch (const std::exception& e) {
    //        std::cerr << "Error: " << e.what() << std::endl;
    //    }
    //    // 测试4维张量
    //    std::cout << "Testing 4D Tensor:" << std::endl;
    //    Tensor<float> tensor5({ 2, 3, 4, 5 }); // 2x3x4x5 
    //    tensor5.setRandom();
    //    Tensor<float> tensor6({ 2, 2, 2, 2 }); // 2x2x2x2 
    //    tensor6.setRandom();
    //    std::vector<Tensor<float>> sequence4D = { tensor5, tensor6 };
    //    position = -1;
    //    try {
    //        auto result4D = SequenceAt(sequence4D, position);
    //        std::cout << "Extracted 4D Tensor (position " << position << "):\n";
    //        result4D.print();
    //    }
    //    catch (const std::exception& e) {
    //        std::cerr << "Error: " << e.what() << std::endl;
    //    }
    //    // 创建不同维度的张量
    //    Tensor<float> tensor7({ 2, 3 });     // 2D: 2x3
    //    tensor1.setRandom();
    //    Tensor<float> tensor8({ 3, 2, 1 });  // 3D: 3x2x1
    //    tensor2.setRandom();
    //    Tensor<float> tensor9({ 2 });        // 1D: 2
    //    tensor3.setRandom();
    //    // 创建张量序列
    //    std::vector<Tensor<float>> input_sequence = { tensor7, tensor8, tensor9 };
    //    // 测试提取不同位置的张量
    //    try {
    //        std::cout << "Extracting tensor at position 0 (2D):\n";
    //        auto result1 = SequenceAt(input_sequence, 0);
    //        result1.print();
    //        std::cout << "Extracting tensor at position 1 (3D):\n";
    //        auto result2 = SequenceAt(input_sequence, 1);
    //        result2.print();
    //        std::cout << "Extracting tensor at position -1 (1D):\n";
    //        auto result3 = SequenceAt(input_sequence, -1);
    //        result3.print();
    //    }
    //    catch (const std::exception& e) {
    //        std::cerr << "Error: " << e.what() << std::endl;
    //    }
    //}
    ////测试SequenceConstruct
    //{
    //    try {
    //        Tensor<float> t1({ 2, 3 }); // 2x3 张量
    //        t1.setRandom();
    //        Tensor<float> t2({ 1, 4 }); // 1x4 张量
    //        t2.setRandom();
    //        Tensor<float> t3({ 3 });    // 1D 张量
    //        t3.setRandom();
    //        // 调用算子构造序列
    //        auto sequence = SequenceConstruct(t1, t2, t3);
    //        // 打印结果
    //        std::cout << "Tensor Sequence contains " << sequence.size() << " tensors:\n";
    //        for (size_t i = 0; i < sequence.size(); ++i) {
    //            std::cout << "Tensor " << i + 1 << ":\n";
    //            sequence[i].print();
    //            std::cout << "\n";
    //        }
    //    }
    //    catch (const std::exception& e) {
    //        std::cerr << "Error: " << e.what() << std::endl;
    //    }
    //}
    
        //测试SequenceInsert算子
        Tensor<float> t1({ 2, 3 }); // 2x3 张量
        t1.setRandom();
        Tensor<float> t2({ 1, 4 }); // 1x4 张量
        t2.setRandom();
        Tensor<float> t3({ 3 });    // 1D 张量
        t3.setRandom();
        // 调用算子构造序列
        auto sequence = SequenceConstruct(t1, t2, t3);
        // 打印结果
        std::cout << "Tensor Sequence contains " << sequence.size() << " tensors:\n";
        for (size_t i = 0; i < sequence.size(); ++i) {
            std::cout << "Tensor " << i + 1 << ":\n";
            sequence[i].print();
            std::cout << "\n";
        }
        Tensor<float> t4({ 2 }); t4.setRandom();
        
        SequenceInsert(sequence, t4);
        SequenceInsert(sequence, t4,0);
        std::cout << "Tensor Sequence contains " << sequence.size() << " tensors:\n";
        for (size_t i = 0; i < sequence.size(); ++i) {
            std::cout << "Tensor " << i + 1 << ":\n";
            sequence[i].print();
            std::cout << "\n";
        }
        SequenceErase(sequence);//默认删去最后一个
        std::cout << "Tensor Sequence contains " << sequence.size() << " tensors:\n";
        for (size_t i = 0; i < sequence.size(); ++i) {
            std::cout << "Tensor " << i + 1 << ":\n";
            sequence[i].print();
            std::cout << "\n";
        }
        SequenceErase(sequence,0);
        std::cout << "Tensor Sequence contains " << sequence.size() << " tensors:\n";
        for (size_t i = 0; i < sequence.size(); ++i) {
            std::cout << "Tensor " << i + 1 << ":\n";
            sequence[i].print();
            std::cout << "\n";
        }
    return 0;
}