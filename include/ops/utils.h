#include <cstdint>
#include <stdexcept>
#include <vector>

/// 将n维坐标转换为1维偏移量（row-major顺序）
inline int64_t coords_to_offset(const std::vector<int64_t>& coords, const std::vector<int64_t>& shape) {
  int64_t offset = 0;
  int64_t stride = 1;
  for (int d = shape.size() - 1; d >= 0; --d) {
    if (coords[d] >= shape[d]) {
      throw std::out_of_range("索引超出范围");
    }
    offset += coords[d] * stride;
    // 更新步长
    stride *= shape[d];
  }
  return offset;
}

/// 将1维偏移量转换为n维坐标
inline std::vector<int64_t> offset_to_coords(int64_t offset, const std::vector<int64_t>& shape) {
  std::vector<int64_t> coords(shape.size());
  for (int d = shape.size() - 1; d >= 0; --d) {
    coords[d] = offset % shape[d];
    offset /= shape[d];
  }
  return coords;
}

/// 获得每个维度的步长
inline std::vector<int64_t> get_strides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size());
  strides.back() = 1;
  for (int64_t i = shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

/// 根据n维坐标获取值
template <typename T>
T get_value(const std::vector<int64_t>& coords, const std::vector<int64_t>& shape,std::vector<T>& data) {
  return data[coords_to_offset(coords, shape)];
}

// 根据n维坐标设置值
template <typename T>
void set_value(const std::vector<int64_t>& coords, T value, std::vector<T>& data, const std::vector<int64_t>& shape) {
  data[coords_to_offset(coords, shape)] = value;
}