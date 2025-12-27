#include "Google/Runtime/GoogleRuntime.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <sstream>

namespace google {
namespace runtime {

// ============================================================================
// GoogleRuntime Implementation
// ============================================================================

struct GoogleRuntime::Impl {
    std::map<std::string, void*> kernel_registry_;
    std::vector<void*> allocated_memory_;
};

GoogleRuntime::GoogleRuntime() : impl_(std::make_unique<Impl>()) {}

GoogleRuntime::~GoogleRuntime() {
    // Clean up allocated memory
    for (void* ptr : impl_->allocated_memory_) {
        free(ptr);
    }
}

GoogleRuntime& GoogleRuntime::instance() {
    static GoogleRuntime runtime;
    return runtime;
}

void GoogleRuntime::registerKernel(const std::string& name, void* func_ptr) {
    if (impl_->kernel_registry_.count(name)) {
        throw std::runtime_error("Kernel already registered: " + name);
    }
    impl_->kernel_registry_[name] = func_ptr;
}

void GoogleRuntime::execute(const std::string& name, 
                           const std::vector<void*>& args) {
    auto it = impl_->kernel_registry_.find(name);
    if (it == impl_->kernel_registry_.end()) {
        throw std::runtime_error("Kernel not found: " + name);
    }
    
    // For now, we assume a simple calling convention
    // In Phase 2, this will be more sophisticated
    void* func_ptr = it->second;
    
    // Call the function (this is a simplified version)
    // The actual calling convention depends on the compiled kernel signature
    using SimpleKernelFunc = void(*)(void**);
    auto kernel = reinterpret_cast<SimpleKernelFunc>(func_ptr);
    kernel(const_cast<void**>(args.data()));
}

void* GoogleRuntime::allocateAligned(size_t size, size_t alignment) {
#ifdef _WIN32
    void* ptr = _aligned_malloc(size, alignment);
#else
    void* ptr = aligned_alloc(alignment, size);
#endif
    if (!ptr) {
        throw std::bad_alloc();
    }
    impl_->allocated_memory_.push_back(ptr);
    return ptr;
}

void GoogleRuntime::deallocate(void* ptr) {
    auto it = std::find(impl_->allocated_memory_.begin(),
                       impl_->allocated_memory_.end(),
                       ptr);
    if (it != impl_->allocated_memory_.end()) {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        free(ptr);
#endif
        impl_->allocated_memory_.erase(it);
    }
}

size_t GoogleRuntime::numKernels() const {
    return impl_->kernel_registry_.size();
}

bool GoogleRuntime::hasKernel(const std::string& name) const {
    return impl_->kernel_registry_.count(name) > 0;
}

// ============================================================================
// Tensor Implementation
// ============================================================================

Tensor::Tensor(const std::vector<int64_t>& shape) : shape_(shape) {
    computeStrides();
    allocateMemory();
    fill(0.0f);  // Initialize to zero
}

Tensor::Tensor(const float* data, const std::vector<int64_t>& shape) 
    : shape_(shape) {
    computeStrides();
    allocateMemory();
    
    // Copy data
    int64_t total_size = size();
    std::memcpy(data_.get(), data, total_size * sizeof(float));
}

void Tensor::computeStrides() {
    strides_.resize(shape_.size());
    if (shape_.empty()) return;
    
    // Row-major order
    int64_t stride = 1;
    for (int i = shape_.size() - 1; i >= 0; --i) {
        strides_[i] = stride;
        stride *= shape_[i];
    }
}

void Tensor::allocateMemory() {
    int64_t total_size = size();
    if (total_size == 0) {
        throw std::runtime_error("Cannot allocate tensor with zero size");
    }
    
    // Allocate 64-byte aligned memory for SIMD optimization
#ifdef _WIN32
    float* raw_ptr = static_cast<float*>(_aligned_malloc(total_size * sizeof(float), 64));
#else
    float* raw_ptr = static_cast<float*>(aligned_alloc(64, total_size * sizeof(float)));
#endif
    
    if (!raw_ptr) {
        throw std::bad_alloc();
    }
    
    data_ = std::shared_ptr<float[]>(
        raw_ptr,
        [](float* p) {
#ifdef _WIN32
            _aligned_free(p);
#else
            free(p);
#endif
        }
    );
}

int64_t Tensor::size() const {
    if (shape_.empty()) return 0;
    return std::accumulate(shape_.begin(), shape_.end(), 
                          1LL, std::multiplies<int64_t>());
}

float& Tensor::operator()(int64_t i, int64_t j) {
    if (shape_.size() != 2) {
        throw std::runtime_error("Operator() only works for 2D tensors");
    }
    return data_[i * strides_[0] + j * strides_[1]];
}

const float& Tensor::operator()(int64_t i, int64_t j) const {
    if (shape_.size() != 2) {
        throw std::runtime_error("Operator() only works for 2D tensors");
    }
    return data_[i * strides_[0] + j * strides_[1]];
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch in addition");
    }
    
    Tensor result(shape_);
    int64_t total_size = size();
    for (int64_t i = 0; i < total_size; ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch in subtraction");
    }
    
    Tensor result(shape_);
    int64_t total_size = size();
    for (int64_t i = 0; i < total_size; ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch in multiplication");
    }
    
    Tensor result(shape_);
    int64_t total_size = size();
    for (int64_t i = 0; i < total_size; ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

Tensor Tensor::operator/(float scalar) const {
    Tensor result(shape_);
    int64_t total_size = size();
    for (int64_t i = 0; i < total_size; ++i) {
        result.data_[i] = data_[i] / scalar;
    }
    return result;
}

Tensor Tensor::operator*(float scalar) const {
    Tensor result(shape_);
    int64_t total_size = size();
    for (int64_t i = 0; i < total_size; ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

Tensor Tensor::relu() const {
    Tensor result(shape_);
    int64_t total_size = size();
    for (int64_t i = 0; i < total_size; ++i) {
        result.data_[i] = std::max(0.0f, data_[i]);
    }
    return result;
}

Tensor Tensor::sigmoid() const {
    Tensor result(shape_);
    int64_t total_size = size();
    for (int64_t i = 0; i < total_size; ++i) {
        result.data_[i] = 1.0f / (1.0f + std::exp(-data_[i]));
    }
    return result;
}

Tensor Tensor::tanh() const {
    Tensor result(shape_);
    int64_t total_size = size();
    for (int64_t i = 0; i < total_size; ++i) {
        result.data_[i] = std::tanh(data_[i]);
    }
    return result;
}

void Tensor::fill(float value) {
    int64_t total_size = size();
    for (int64_t i = 0; i < total_size; ++i) {
        data_[i] = value;
    }
}

void Tensor::randn(float mean, float stddev) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, stddev);
    
    int64_t total_size = size();
    for (int64_t i = 0; i < total_size; ++i) {
        data_[i] = dist(gen);
    }
}

} // namespace runtime
} // namespace google
