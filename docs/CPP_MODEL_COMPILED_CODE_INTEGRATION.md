# C++ Model and Compiled Code Integration

## Executive Summary

This document explains how **compiled MLIR code** integrates with **C++ models**, covering eager execution patterns, JIT compilation, runtime integration, and comparison with Python-based approaches. Unlike Python where decorators provide a clean boundary, C++ requires explicit design patterns for mixing eager and compiled execution.

**Key Insight**: In C++, the integration is more explicit but offers better control, zero-overhead abstractions, and direct memory management.

---

## Table of Contents

1. [C++ vs Python Integration](#cpp-vs-python-integration)
2. [Eager Execution in C++](#eager-execution-in-cpp)
3. [Compiled Code Integration Patterns](#compiled-code-integration-patterns)
4. [JIT Compilation in C++](#jit-compilation-in-cpp)
5. [Runtime Integration Mechanisms](#runtime-integration-mechanisms)
6. [Framework Comparison](#framework-comparison)
7. [Complete Implementation Examples](#complete-implementation-examples)
8. [Best Practices](#best-practices)

---

## C++ vs Python Integration

### Python Approach (Decorator-Based)

```python
class Model:
    @jax.jit  # Decorator marks compilation boundary
    def compiled_layer(self, x, w):
        return x @ w
    
    def forward(self, x):
        x = x / 255.0              # Eager
        x = self.compiled_layer(x) # Compiled
        return x
```

**Characteristics:**
- ✅ Clean syntax with decorators
- ✅ Automatic tracing and compilation
- ✅ Runtime overhead for tracing
- ❌ Less control over compilation

### C++ Approach (Explicit Design)

```cpp
class Model {
    // Compiled function (pre-compiled or JIT)
    CompiledFunction compiled_layer_;
    
    Tensor forward(const Tensor& x) {
        // Eager execution
        Tensor normalized = x / 255.0f;
        
        // Call compiled code explicitly
        Tensor result = compiled_layer_.execute({normalized});
        
        return result;
    }
};
```

**Characteristics:**
- ✅ Explicit control over compilation
- ✅ Zero runtime overhead
- ✅ Direct memory management
- ✅ Better performance
- ❌ More verbose
- ❌ Manual integration required

---

## Eager Execution in C++

### What is "Eager" in C++?

In C++, "eager execution" means operations execute immediately when called, as opposed to being deferred or compiled into a graph.

```cpp
// Eager execution - each operation executes immediately
Tensor eager_matmul(const Tensor& A, const Tensor& B) {
    // Allocate result
    Tensor C(A.rows(), B.cols());
    
    // Execute immediately (naive implementation)
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < B.cols(); ++j) {
            float sum = 0.0f;
            for (int k = 0; k < A.cols(); ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
    
    return C;
}

// Usage - executes immediately
Tensor A = Tensor::randn(1024, 512);
Tensor B = Tensor::randn(512, 256);
Tensor C = eager_matmul(A, B);  // Runs right now
```

### Eager Execution Frameworks in C++

#### 1. Eigen (Header-Only Library)

```cpp
#include <Eigen/Dense>

using namespace Eigen;

MatrixXf eager_computation(const MatrixXf& A, const MatrixXf& B) {
    // All operations execute immediately
    MatrixXf C = A * B;           // Matrix multiply
    C = C.array() + 1.0f;         // Element-wise add
    C = C.cwiseMax(0.0f);         // ReLU
    return C;
}
```

#### 2. ATen (PyTorch C++ API)

```cpp
#include <torch/torch.h>

torch::Tensor eager_forward(torch::Tensor x, torch::Tensor w) {
    // Eager execution - each op runs immediately
    auto h = torch::matmul(x, w);
    h = torch::relu(h);
    return h;
}
```

#### 3. Custom Tensor Library

```cpp
class Tensor {
    std::vector<float> data_;
    std::vector<int64_t> shape_;
    
public:
    // Eager operations
    Tensor operator*(const Tensor& other) const {
        // Execute immediately
        return matmul(*this, other);
    }
    
    Tensor relu() const {
        Tensor result(shape_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = std::max(0.0f, data_[i]);
        }
        return result;
    }
};
```

---

## Compiled Code Integration Patterns

### Pattern 1: Pre-Compiled Functions (AOT)

**Ahead-of-Time (AOT) compilation**: Compile MLIR to object files at build time, link with C++ code.

```cpp
// 1. MLIR code (compiled separately)
// File: matmul.mlir
/*
func.func @matmul_optimized(%A: memref<1024x512xf32>,
                            %B: memref<512x256xf32>,
                            %C: memref<1024x256xf32>) {
  linalg.matmul ins(%A, %B : memref<1024x512xf32>, memref<512x256xf32>)
                outs(%C : memref<1024x256xf32>)
  return
}
*/

// 2. Compile MLIR to object file
// $ google-opt matmul.mlir --google-extreme-pipeline -o matmul.ll
// $ clang -c matmul.ll -o matmul.o

// 3. Declare in C++ header
// File: compiled_kernels.h
extern "C" {
    void matmul_optimized(
        float* A_data, int64_t* A_shape, int64_t* A_strides,
        float* B_data, int64_t* B_shape, int64_t* B_strides,
        float* C_data, int64_t* C_shape, int64_t* C_strides
    );
}

// 4. Use in C++ model
class Model {
    Tensor weights_;
    
public:
    Tensor forward(const Tensor& input) {
        // Eager preprocessing
        Tensor normalized = input / 255.0f;
        
        // Call compiled function
        Tensor output(input.rows(), weights_.cols());
        
        int64_t A_shape[] = {input.rows(), input.cols()};
        int64_t A_strides[] = {input.cols(), 1};
        int64_t B_shape[] = {weights_.rows(), weights_.cols()};
        int64_t B_strides[] = {weights_.cols(), 1};
        int64_t C_shape[] = {output.rows(), output.cols()};
        int64_t C_strides[] = {output.cols(), 1};
        
        matmul_optimized(
            normalized.data(), A_shape, A_strides,
            weights_.data(), B_shape, B_strides,
            output.data(), C_shape, C_strides
        );
        
        // Eager postprocessing
        return output.relu();
    }
};

// 5. Link at build time
// $ g++ -o model model.cpp matmul.o -O3
```

**Pros:**
- ✅ Zero runtime compilation overhead
- ✅ Predictable performance
- ✅ Simple deployment (single binary)
- ✅ Best for production

**Cons:**
- ❌ No runtime flexibility
- ❌ Recompilation needed for changes
- ❌ Fixed input shapes

---

### Pattern 2: Dynamic Loading (Shared Libraries)

**Load compiled code at runtime from `.so` files.**

```cpp
#include <dlfcn.h>
#include <iostream>

class CompiledKernel {
    void* lib_handle_;
    using KernelFunc = void(*)(float*, int64_t*, int64_t*,
                               float*, int64_t*, int64_t*,
                               float*, int64_t*, int64_t*);
    KernelFunc kernel_func_;
    
public:
    CompiledKernel(const std::string& so_path, const std::string& func_name) {
        // Load shared library
        lib_handle_ = dlopen(so_path.c_str(), RTLD_LAZY);
        if (!lib_handle_) {
            throw std::runtime_error("Failed to load: " + std::string(dlerror()));
        }
        
        // Get function pointer
        kernel_func_ = reinterpret_cast<KernelFunc>(
            dlsym(lib_handle_, func_name.c_str())
        );
        if (!kernel_func_) {
            throw std::runtime_error("Failed to find function: " + func_name);
        }
    }
    
    ~CompiledKernel() {
        if (lib_handle_) dlclose(lib_handle_);
    }
    
    void execute(const Tensor& A, const Tensor& B, Tensor& C) {
        int64_t A_shape[] = {A.rows(), A.cols()};
        int64_t A_strides[] = {A.cols(), 1};
        int64_t B_shape[] = {B.rows(), B.cols()};
        int64_t B_strides[] = {B.cols(), 1};
        int64_t C_shape[] = {C.rows(), C.cols()};
        int64_t C_strides[] = {C.cols(), 1};
        
        kernel_func_(
            A.data(), A_shape, A_strides,
            B.data(), B_shape, B_strides,
            C.data(), C_shape, C_strides
        );
    }
};

// Usage
class Model {
    CompiledKernel matmul_kernel_;
    Tensor weights_;
    
public:
    Model() : matmul_kernel_("./matmul_optimized.so", "matmul_optimized") {}
    
    Tensor forward(const Tensor& input) {
        // Eager preprocessing
        Tensor normalized = input / 255.0f;
        
        // Call compiled kernel
        Tensor output(input.rows(), weights_.cols());
        matmul_kernel_.execute(normalized, weights_, output);
        
        // Eager postprocessing
        return output.relu();
    }
};
```

**Pros:**
- ✅ Runtime flexibility (swap kernels)
- ✅ No recompilation of main binary
- ✅ Easy to update kernels

**Cons:**
- ❌ Deployment complexity (multiple files)
- ❌ Dynamic loading overhead
- ❌ Version management

---

### Pattern 3: JIT Compilation (Runtime)

**Compile MLIR code at runtime using MLIR ExecutionEngine.**

```cpp
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

class JITCompiler {
    mlir::MLIRContext context_;
    std::unique_ptr<mlir::ExecutionEngine> engine_;
    
public:
    void compileFromMLIR(const std::string& mlir_code) {
        // Parse MLIR
        auto module = mlir::parseSourceString<mlir::ModuleOp>(
            mlir_code, &context_
        );
        
        // Create execution engine with optimization
        mlir::ExecutionEngineOptions options;
        options.transformer = mlir::makeOptimizingTransformer(
            /*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/nullptr
        );
        
        auto maybeEngine = mlir::ExecutionEngine::create(*module, options);
        if (!maybeEngine) {
            throw std::runtime_error("Failed to create execution engine");
        }
        
        engine_ = std::move(*maybeEngine);
    }
    
    template<typename... Args>
    void execute(const std::string& func_name, Args... args) {
        auto error = engine_->invokePacked(func_name, args...);
        if (error) {
            throw std::runtime_error("Execution failed");
        }
    }
};

// Usage
class Model {
    JITCompiler jit_;
    Tensor weights_;
    bool compiled_ = false;
    
public:
    Tensor forward(const Tensor& input) {
        // JIT compile on first call
        if (!compiled_) {
            std::string mlir_code = generateMLIR(input.shape(), weights_.shape());
            jit_.compileFromMLIR(mlir_code);
            compiled_ = true;
        }
        
        // Eager preprocessing
        Tensor normalized = input / 255.0f;
        
        // Execute JIT-compiled code
        Tensor output(input.rows(), weights_.cols());
        jit_.execute("matmul", 
                     normalized.data(), normalized.shape(),
                     weights_.data(), weights_.shape(),
                     output.data(), output.shape());
        
        // Eager postprocessing
        return output.relu();
    }
    
private:
    std::string generateMLIR(const std::vector<int64_t>& input_shape,
                            const std::vector<int64_t>& weight_shape) {
        // Generate MLIR code based on shapes
        std::ostringstream oss;
        oss << "func.func @matmul(%arg0: memref<" 
            << input_shape[0] << "x" << input_shape[1] << "xf32>, "
            << "%arg1: memref<" 
            << weight_shape[0] << "x" << weight_shape[1] << "xf32>, "
            << "%arg2: memref<" 
            << input_shape[0] << "x" << weight_shape[1] << "xf32>) {\n"
            << "  linalg.matmul ins(%arg0, %arg1 : memref<"
            << input_shape[0] << "x" << input_shape[1] << "xf32>, memref<"
            << weight_shape[0] << "x" << weight_shape[1] << "xf32>) "
            << "outs(%arg2 : memref<"
            << input_shape[0] << "x" << weight_shape[1] << "xf32>)\n"
            << "  return\n"
            << "}\n";
        return oss.str();
    }
};
```

**Pros:**
- ✅ Runtime flexibility
- ✅ Shape-specific optimization
- ✅ No separate compilation step

**Cons:**
- ❌ JIT compilation overhead
- ❌ Requires MLIR/LLVM at runtime
- ❌ Larger binary size

---

### Pattern 4: Hybrid Approach

**Combine AOT for common cases, JIT for dynamic cases.**

```cpp
class HybridModel {
    // AOT-compiled kernels for common shapes
    std::map<std::string, void*> aot_kernels_;
    
    // JIT compiler for dynamic shapes
    JITCompiler jit_compiler_;
    
    // Cache for JIT-compiled functions
    std::map<std::string, std::unique_ptr<CompiledFunction>> jit_cache_;
    
public:
    HybridModel() {
        // Register AOT kernels
        registerAOTKernel("matmul_1024x512", &matmul_1024x512_optimized);
        registerAOTKernel("matmul_512x256", &matmul_512x256_optimized);
    }
    
    Tensor forward(const Tensor& input) {
        // Eager preprocessing
        Tensor normalized = input / 255.0f;
        
        // Try AOT kernel first (fast path)
        std::string key = getKernelKey(normalized.shape(), weights_.shape());
        if (aot_kernels_.count(key)) {
            return executeAOT(aot_kernels_[key], normalized, weights_);
        }
        
        // Fallback to JIT (slow path, but flexible)
        if (!jit_cache_.count(key)) {
            jit_cache_[key] = compileJIT(normalized.shape(), weights_.shape());
        }
        return executeJIT(jit_cache_[key].get(), normalized, weights_);
    }
    
private:
    void registerAOTKernel(const std::string& name, void* func_ptr) {
        aot_kernels_[name] = func_ptr;
    }
    
    std::string getKernelKey(const std::vector<int64_t>& a_shape,
                            const std::vector<int64_t>& b_shape) {
        std::ostringstream oss;
        oss << "matmul_" << a_shape[0] << "x" << a_shape[1];
        return oss.str();
    }
};
```

**Pros:**
- ✅ Best of both worlds
- ✅ Fast for common cases
- ✅ Flexible for dynamic cases

**Cons:**
- ❌ Most complex implementation
- ❌ Larger binary size

---

## JIT Compilation in C++

### MLIR ExecutionEngine Integration

```cpp
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/TargetSelect.h"

class MLIRJITEngine {
    mlir::MLIRContext context_;
    
public:
    MLIRJITEngine() {
        // Initialize LLVM targets
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
    }
    
    std::unique_ptr<mlir::ExecutionEngine> compile(const std::string& mlir_code) {
        // Parse MLIR module
        auto module = mlir::parseSourceString<mlir::ModuleOp>(
            mlir_code, &context_
        );
        if (!module) {
            throw std::runtime_error("Failed to parse MLIR");
        }
        
        // Set up optimization pipeline
        mlir::ExecutionEngineOptions options;
        options.transformer = [](llvm::Module* m) {
            // Apply LLVM optimizations
            llvm::PassBuilder pb;
            llvm::ModuleAnalysisManager mam;
            llvm::ModulePassManager mpm;
            
            pb.registerModuleAnalyses(mam);
            mpm = pb.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
            mpm.run(*m, mam);
            
            return llvm::Error::success();
        };
        
        // Create execution engine
        auto maybeEngine = mlir::ExecutionEngine::create(*module, options);
        if (!maybeEngine) {
            throw std::runtime_error("Failed to create execution engine");
        }
        
        return std::move(*maybeEngine);
    }
};

// Usage example
class JITModel {
    MLIRJITEngine jit_engine_;
    std::unique_ptr<mlir::ExecutionEngine> compiled_matmul_;
    
public:
    void compileMatMul(int64_t M, int64_t K, int64_t N) {
        std::string mlir_code = R"(
            module {
              func.func @matmul(%arg0: memref<)" + std::to_string(M) + "x" + 
                std::to_string(K) + R"(xf32>,
                                %arg1: memref<)" + std::to_string(K) + "x" + 
                std::to_string(N) + R"(xf32>,
                                %arg2: memref<)" + std::to_string(M) + "x" + 
                std::to_string(N) + R"(xf32>) {
                linalg.matmul ins(%arg0, %arg1 : memref<)" + 
                  std::to_string(M) + "x" + std::to_string(K) + "xf32>, memref<" + 
                  std::to_string(K) + "x" + std::to_string(N) + R"(xf32>)
                              outs(%arg2 : memref<)" + std::to_string(M) + "x" + 
                  std::to_string(N) + R"(xf32>)
                return
              }
            }
        )";
        
        compiled_matmul_ = jit_engine_.compile(mlir_code);
    }
    
    void executeMatMul(float* A, float* B, float* C) {
        // Create memref descriptors
        struct MemRefDescriptor {
            float* allocated;
            float* aligned;
            int64_t offset;
            int64_t sizes[2];
            int64_t strides[2];
        };
        
        MemRefDescriptor A_desc = {A, A, 0, {M_, K_}, {K_, 1}};
        MemRefDescriptor B_desc = {B, B, 0, {K_, N_}, {N_, 1}};
        MemRefDescriptor C_desc = {C, C, 0, {M_, N_}, {N_, 1}};
        
        // Execute
        auto error = compiled_matmul_->invokePacked(
            "matmul", &A_desc, &B_desc, &C_desc
        );
        
        if (error) {
            throw std::runtime_error("Execution failed");
        }
    }
    
private:
    int64_t M_, K_, N_;
};
```

---

## Runtime Integration Mechanisms

### Memory Management

**Key Challenge**: Ensuring C++ tensors and compiled code share memory correctly.

```cpp
class Tensor {
    std::shared_ptr<float[]> data_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    
public:
    // Constructor
    Tensor(const std::vector<int64_t>& shape) : shape_(shape) {
        int64_t size = 1;
        for (auto dim : shape) size *= dim;
        
        // Allocate aligned memory (64-byte alignment for SIMD)
        data_ = std::shared_ptr<float[]>(
            static_cast<float*>(aligned_alloc(64, size * sizeof(float))),
            [](float* p) { free(p); }
        );
        
        // Compute strides (row-major)
        strides_.resize(shape.size());
        int64_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides_[i] = stride;
            stride *= shape[i];
        }
    }
    
    // Get raw pointer for compiled code
    float* data() { return data_.get(); }
    const float* data() const { return data_.get(); }
    
    // Get shape/strides for compiled code
    const int64_t* shape_ptr() const { return shape_.data(); }
    const int64_t* strides_ptr() const { return strides_.data(); }
    
    // Create memref descriptor for MLIR
    struct MemRefDescriptor {
        float* allocated;
        float* aligned;
        int64_t offset;
        int64_t sizes[2];
        int64_t strides[2];
    };
    
    MemRefDescriptor toMemRef() const {
        return {
            data_.get(),
            data_.get(),
            0,
            {shape_[0], shape_[1]},
            {strides_[0], strides_[1]}
        };
    }
};

// Usage with compiled code
void callCompiledKernel(const Tensor& A, const Tensor& B, Tensor& C) {
    // No data copying - just pass pointers!
    auto A_desc = A.toMemRef();
    auto B_desc = B.toMemRef();
    auto C_desc = C.toMemRef();
    
    compiled_kernel(&A_desc, &B_desc, &C_desc);
}
```

### Thread Safety

```cpp
class ThreadSafeModel {
    mutable std::mutex compilation_mutex_;
    std::atomic<bool> compiled_{false};
    std::unique_ptr<CompiledFunction> compiled_func_;
    
public:
    Tensor forward(const Tensor& input) {
        // Ensure compilation happens once
        if (!compiled_.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lock(compilation_mutex_);
            if (!compiled_.load(std::memory_order_relaxed)) {
                compiled_func_ = compileFunction();
                compiled_.store(true, std::memory_order_release);
            }
        }
        
        // Execute (thread-safe if compiled function is stateless)
        return compiled_func_->execute(input);
    }
};
```

---

## Framework Comparison

### TorchScript (PyTorch C++)

```cpp
#include <torch/script.h>

// Load pre-compiled TorchScript model
torch::jit::script::Module module = torch::jit::load("model.pt");

// Execute
std::vector<torch::jit::IValue> inputs;
inputs.push_back(torch::randn({1, 3, 224, 224}));
auto output = module.forward(inputs).toTensor();
```

**Architecture:**
```
Python Model → torch.jit.script → TorchScript IR → Serialized → C++ Load → Execute
```

**Pros:**
- ✅ Easy Python-to-C++ deployment
- ✅ Serialized format
- ✅ Good performance

**Cons:**
- ❌ Limited to PyTorch ecosystem
- ❌ Less control over compilation

---

### TensorRT (NVIDIA)

```cpp
#include <NvInfer.h>

// Build engine from ONNX
nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0);
nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);

// Parse ONNX model
parser->parseFromFile("model.onnx", ...);

// Build optimized engine
nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

// Execute
nvinfer1::IExecutionContext* context = engine->createExecutionContext();
context->executeV2(buffers);
```

**Architecture:**
```
ONNX Model → TensorRT Builder → Optimized Engine → Serialized → C++ Load → GPU Execute
```

**Pros:**
- ✅ Excellent GPU performance
- ✅ Aggressive optimizations
- ✅ Production-ready

**Cons:**
- ❌ NVIDIA GPUs only
- ❌ Complex API
- ❌ Limited flexibility

---

### MLIR-Based (Your Approach)

```cpp
// Compile MLIR to shared library
system("google-opt model.mlir --google-extreme-pipeline -o model.ll");
system("clang -shared model.ll -o model.so");

// Load and execute
CompiledKernel kernel("model.so", "forward");
Tensor output = kernel.execute(input);
```

**Architecture:**
```
MLIR Code → google-opt → LLVM IR → Clang → Shared Library → C++ Load → Execute
```

**Pros:**
- ✅ Full control over compilation
- ✅ Multi-backend (CPU, GPU, custom)
- ✅ Extensible

**Cons:**
- ❌ More manual work
- ❌ Need to build infrastructure

---

## Complete Implementation Examples

### Example 1: Simple AOT Integration

```cpp
// File: model.h
#pragma once
#include <vector>
#include <memory>

class Tensor {
    std::shared_ptr<float[]> data_;
    std::vector<int64_t> shape_;
public:
    Tensor(const std::vector<int64_t>& shape);
    float* data() { return data_.get(); }
    const std::vector<int64_t>& shape() const { return shape_; }
    int64_t size() const;
};

// Compiled kernel (from MLIR)
extern "C" void matmul_optimized(
    float* A, int64_t A_rows, int64_t A_cols,
    float* B, int64_t B_rows, int64_t B_cols,
    float* C, int64_t C_rows, int64_t C_cols
);

class SimpleModel {
    Tensor weights_;
public:
    SimpleModel(const std::vector<int64_t>& weight_shape);
    Tensor forward(const Tensor& input);
};
```

```cpp
// File: model.cpp
#include "model.h"
#include <algorithm>
#include <numeric>

Tensor::Tensor(const std::vector<int64_t>& shape) : shape_(shape) {
    int64_t size = std::accumulate(shape.begin(), shape.end(), 
                                   1LL, std::multiplies<int64_t>());
    data_ = std::shared_ptr<float[]>(new float[size]);
}

int64_t Tensor::size() const {
    return std::accumulate(shape_.begin(), shape_.end(), 
                          1LL, std::multiplies<int64_t>());
}

SimpleModel::SimpleModel(const std::vector<int64_t>& weight_shape)
    : weights_(weight_shape) {
    // Initialize weights (random or load from file)
    std::fill_n(weights_.data(), weights_.size(), 1.0f);
}

Tensor SimpleModel::forward(const Tensor& input) {
    // Eager preprocessing
    Tensor normalized(input.shape());
    for (int64_t i = 0; i < input.size(); ++i) {
        normalized.data()[i] = input.data()[i] / 255.0f;
    }
    
    // Call compiled kernel
    Tensor output({input.shape()[0], weights_.shape()[1]});
    matmul_optimized(
        normalized.data(), input.shape()[0], input.shape()[1],
        weights_.data(), weights_.shape()[0], weights_.shape()[1],
        output.data(), output.shape()[0], output.shape()[1]
    );
    
    // Eager postprocessing (ReLU)
    for (int64_t i = 0; i < output.size(); ++i) {
        output.data()[i] = std::max(0.0f, output.data()[i]);
    }
    
    return output;
}
```

```cpp
// File: main.cpp
#include "model.h"
#include <iostream>
#include <chrono>

int main() {
    // Create model
    SimpleModel model({512, 256});
    
    // Create input
    Tensor input({128, 512});
    std::fill_n(input.data(), input.size(), 1.0f);
    
    // Warm-up
    auto output = model.forward(input);
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        output = model.forward(input);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    double avg_time = std::chrono::duration<double, std::milli>(end - start).count() / 100.0;
    std::cout << "Average time: " << avg_time << " ms\n";
    
    return 0;
}
```

**Build:**
```bash
# Compile MLIR to object file
google-opt matmul.mlir --google-extreme-pipeline --convert-to-llvm -o matmul.ll
clang -c matmul.ll -o matmul.o

# Compile C++ code
g++ -c model.cpp -o model.o -O3 -std=c++17
g++ -c main.cpp -o main.o -O3 -std=c++17

# Link
g++ model.o main.o matmul.o -o model_app
```

---

### Example 2: Dynamic Loading with Caching

```cpp
// File: dynamic_model.h
#pragma once
#include <string>
#include <map>
#include <memory>
#include <dlfcn.h>

class DynamicKernel {
    void* lib_handle_;
    void* func_ptr_;
    
public:
    DynamicKernel(const std::string& lib_path, const std::string& func_name);
    ~DynamicKernel();
    
    void execute(float* A, int64_t A_rows, int64_t A_cols,
                 float* B, int64_t B_rows, int64_t B_cols,
                 float* C, int64_t C_rows, int64_t C_cols);
};

class DynamicModel {
    std::map<std::string, std::unique_ptr<DynamicKernel>> kernel_cache_;
    Tensor weights_;
    
public:
    DynamicModel(const std::vector<int64_t>& weight_shape);
    Tensor forward(const Tensor& input);
    
private:
    DynamicKernel* getKernel(const std::vector<int64_t>& input_shape);
    std::string getKernelPath(const std::vector<int64_t>& input_shape);
};
```

```cpp
// File: dynamic_model.cpp
#include "dynamic_model.h"
#include <stdexcept>
#include <sstream>

DynamicKernel::DynamicKernel(const std::string& lib_path, 
                             const std::string& func_name) {
    lib_handle_ = dlopen(lib_path.c_str(), RTLD_LAZY);
    if (!lib_handle_) {
        throw std::runtime_error("Failed to load library: " + 
                               std::string(dlerror()));
    }
    
    func_ptr_ = dlsym(lib_handle_, func_name.c_str());
    if (!func_ptr_) {
        throw std::runtime_error("Failed to find function: " + func_name);
    }
}

DynamicKernel::~DynamicKernel() {
    if (lib_handle_) {
        dlclose(lib_handle_);
    }
}

void DynamicKernel::execute(float* A, int64_t A_rows, int64_t A_cols,
                           float* B, int64_t B_rows, int64_t B_cols,
                           float* C, int64_t C_rows, int64_t C_cols) {
    using KernelFunc = void(*)(float*, int64_t, int64_t,
                              float*, int64_t, int64_t,
                              float*, int64_t, int64_t);
    auto kernel = reinterpret_cast<KernelFunc>(func_ptr_);
    kernel(A, A_rows, A_cols, B, B_rows, B_cols, C, C_rows, C_cols);
}

DynamicModel::DynamicModel(const std::vector<int64_t>& weight_shape)
    : weights_(weight_shape) {}

Tensor DynamicModel::forward(const Tensor& input) {
    // Get or compile kernel for this input shape
    auto* kernel = getKernel(input.shape());
    
    // Execute
    Tensor output({input.shape()[0], weights_.shape()[1]});
    kernel->execute(
        input.data(), input.shape()[0], input.shape()[1],
        weights_.data(), weights_.shape()[0], weights_.shape()[1],
        output.data(), output.shape()[0], output.shape()[1]
    );
    
    return output;
}

DynamicKernel* DynamicModel::getKernel(const std::vector<int64_t>& input_shape) {
    std::string key = std::to_string(input_shape[0]) + "x" + 
                     std::to_string(input_shape[1]);
    
    if (kernel_cache_.find(key) == kernel_cache_.end()) {
        std::string lib_path = getKernelPath(input_shape);
        kernel_cache_[key] = std::make_unique<DynamicKernel>(
            lib_path, "matmul_optimized"
        );
    }
    
    return kernel_cache_[key].get();
}

std::string DynamicModel::getKernelPath(const std::vector<int64_t>& input_shape) {
    std::ostringstream oss;
    oss << "./kernels/matmul_" << input_shape[0] << "x" << input_shape[1] << ".so";
    return oss.str();
}
```

---

## Best Practices

### 1. Memory Alignment

```cpp
// Always use aligned memory for SIMD
void* aligned_data = aligned_alloc(64, size * sizeof(float));

// Or use C++17 aligned new
float* data = new (std::align_val_t{64}) float[size];
```

### 2. Error Handling

```cpp
class CompiledKernel {
public:
    enum class Status { SUCCESS, ERROR_LOAD, ERROR_SYMBOL, ERROR_EXEC };
    
    Status execute(const Tensor& input, Tensor& output) {
        try {
            // Execute kernel
            kernel_func_(input.data(), output.data());
            return Status::SUCCESS;
        } catch (const std::exception& e) {
            std::cerr << "Execution error: " << e.what() << "\n";
            return Status::ERROR_EXEC;
        }
    }
};
```

### 3. Shape Validation

```cpp
void validateShapes(const Tensor& A, const Tensor& B) {
    if (A.shape()[1] != B.shape()[0]) {
        throw std::invalid_argument(
            "Incompatible shapes for matmul: " +
            shapeToString(A.shape()) + " and " + shapeToString(B.shape())
        );
    }
}
```

### 4. Performance Profiling

```cpp
class ProfiledKernel {
    std::chrono::duration<double, std::milli> total_time_{0};
    int64_t num_calls_{0};
    
public:
    void execute(const Tensor& input, Tensor& output) {
        auto start = std::chrono::high_resolution_clock::now();
        kernel_func_(input.data(), output.data());
        auto end = std::chrono::high_resolution_clock::now();
        
        total_time_ += std::chrono::duration<double, std::milli>(end - start);
        ++num_calls_;
    }
    
    double averageTime() const {
        return total_time_.count() / num_calls_;
    }
};
```

---

## Conclusion

### Key Takeaways

1. **C++ offers more control** than Python but requires explicit integration
2. **Three main patterns**: AOT (best performance), Dynamic Loading (flexibility), JIT (runtime optimization)
3. **Memory management is critical**: Use aligned memory, avoid copies
4. **Framework comparison**: TorchScript (easy), TensorRT (fast GPU), MLIR (flexible)

### Recommended Approach for Your Project

1. **Start with AOT** for production kernels
2. **Add dynamic loading** for flexibility
3. **Consider JIT** for research/experimentation
4. **Use hybrid approach** for best of both worlds

### Integration Checklist

- ✅ Define clear C ABI for compiled functions
- ✅ Use aligned memory for SIMD
- ✅ Implement shape validation
- ✅ Add error handling
- ✅ Profile performance
- ✅ Cache compiled kernels
- ✅ Thread-safe compilation

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-26  
**Author**: Google MLIR Compiler Team
