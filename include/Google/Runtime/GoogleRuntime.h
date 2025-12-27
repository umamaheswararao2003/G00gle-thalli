#pragma once

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <cstdint>

namespace google {
namespace runtime {

// Forward declarations
class Tensor;

/// @brief Main runtime class for managing compiled kernels and execution
/// 
/// GoogleRuntime provides a centralized interface for:
/// - Registering compiled MLIR kernels
/// - Managing aligned memory allocation
/// - Executing registered kernels
/// 
/// This is a singleton class - use GoogleRuntime::instance() to access it.
class GoogleRuntime {
public:
    /// Get the singleton instance
    static GoogleRuntime& instance();
    
    /// Register a compiled kernel function
    /// @param name Unique name for the kernel
    /// @param func_ptr Pointer to the compiled function
    void registerKernel(const std::string& name, void* func_ptr);
    
    /// Execute a registered kernel
    /// @param name Name of the kernel to execute
    /// @param args Vector of argument pointers (inputs and outputs)
    void execute(const std::string& name, const std::vector<void*>& args);
    
    /// Allocate aligned memory
    /// @param size Size in bytes
    /// @param alignment Alignment requirement (default: 64 bytes for SIMD)
    /// @return Pointer to aligned memory
    void* allocateAligned(size_t size, size_t alignment = 64);
    
    /// Deallocate memory
    /// @param ptr Pointer to deallocate
    void deallocate(void* ptr);
    
    /// Get number of registered kernels
    size_t numKernels() const;
    
    /// Check if a kernel is registered
    bool hasKernel(const std::string& name) const;
    
private:
    GoogleRuntime();
    ~GoogleRuntime();
    
    // Delete copy and move constructors
    GoogleRuntime(const GoogleRuntime&) = delete;
    GoogleRuntime& operator=(const GoogleRuntime&) = delete;
    GoogleRuntime(GoogleRuntime&&) = delete;
    GoogleRuntime& operator=(GoogleRuntime&&) = delete;
    
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// @brief Tensor class for managing multi-dimensional arrays
///
/// Provides:
/// - Automatic memory management with shared_ptr
/// - Aligned memory allocation for SIMD optimization
/// - Shape and stride tracking
/// - Eager operations (element-wise ops, activations)
class Tensor {
public:
    /// Construct tensor with given shape
    /// @param shape Dimensions of the tensor
    explicit Tensor(const std::vector<int64_t>& shape);
    
    /// Construct tensor from existing data (copies data)
    /// @param data Source data
    /// @param shape Dimensions of the tensor
    Tensor(const float* data, const std::vector<int64_t>& shape);
    
    /// Get raw data pointer
    float* data() { return data_.get(); }
    const float* data() const { return data_.get(); }
    
    /// Get shape
    const std::vector<int64_t>& shape() const { return shape_; }
    
    /// Get strides
    const std::vector<int64_t>& strides() const { return strides_; }
    
    /// Get number of dimensions
    size_t ndim() const { return shape_.size(); }
    
    /// Get total number of elements
    int64_t size() const;
    
    /// Get number of rows (first dimension)
    int64_t rows() const { return shape_.empty() ? 0 : shape_[0]; }
    
    /// Get number of columns (second dimension)
    int64_t cols() const { return shape_.size() > 1 ? shape_[1] : 1; }
    
    /// Element access (for 2D tensors)
    float& operator()(int64_t i, int64_t j);
    const float& operator()(int64_t i, int64_t j) const;
    
    /// Eager operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;  // Element-wise
    Tensor operator/(float scalar) const;
    Tensor operator*(float scalar) const;
    
    /// Activation functions
    Tensor relu() const;
    Tensor sigmoid() const;
    Tensor tanh() const;
    
    /// Fill with value
    void fill(float value);
    
    /// Fill with random values
    void randn(float mean = 0.0f, float stddev = 1.0f);
    
private:
    std::shared_ptr<float[]> data_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    
    void computeStrides();
    void allocateMemory();
};

} // namespace runtime
} // namespace google
