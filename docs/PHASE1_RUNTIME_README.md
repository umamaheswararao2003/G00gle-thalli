# Google MLIR Runtime - Phase 1

## Overview

Phase 1 implements an **Enhanced Embedded Runtime** that provides:
- Runtime wrapper class (`GoogleRuntime`)
- Tensor abstraction with automatic memory management
- Kernel registry for compiled MLIR functions
- Eager operations (element-wise ops, activations)

## Quick Start

### Build the Runtime

```powershell
# Windows (PowerShell)
.\scripts\build_runtime_phase1.ps1

# Or manually:
cmake -B build -DCMAKE_BUILD_TYPE=Release `
  -DLLVM_DIR="C:\path\to\llvm\build\lib\cmake\llvm" `
  -DMLIR_DIR="C:\path\to\llvm\build\lib\cmake\mlir"
  
cmake --build build --target GoogleRuntime
```

### Use the Runtime

```cpp
#include "Google/Runtime/GoogleRuntime.h"

using namespace google::runtime;

// Get runtime instance
auto& runtime = GoogleRuntime::instance();

// Register compiled kernel
extern "C" void my_kernel(float*, float*, float*);
runtime.registerKernel("my_kernel", reinterpret_cast<void*>(my_kernel));

// Create tensors
Tensor A({1024, 512});
Tensor B({512, 256});
Tensor C({1024, 256});

// Fill with data
A.fill(1.0f);
B.fill(1.0f);

// Execute kernel
std::vector<void*> args = {A.data(), B.data(), C.data()};
runtime.execute("my_kernel", args);

// Eager operations
Tensor normalized = A / 255.0f;
Tensor activated = normalized.relu();
```

## API Reference

### GoogleRuntime

**Singleton Access:**
```cpp
GoogleRuntime& runtime = GoogleRuntime::instance();
```

**Register Kernel:**
```cpp
void registerKernel(const std::string& name, void* func_ptr);
```

**Execute Kernel:**
```cpp
void execute(const std::string& name, const std::vector<void*>& args);
```

**Memory Management:**
```cpp
void* allocateAligned(size_t size, size_t alignment = 64);
void deallocate(void* ptr);
```

### Tensor

**Construction:**
```cpp
Tensor(const std::vector<int64_t>& shape);
Tensor(const float* data, const std::vector<int64_t>& shape);
```

**Data Access:**
```cpp
float* data();
const std::vector<int64_t>& shape() const;
int64_t size() const;
```

**Eager Operations:**
```cpp
Tensor operator+(const Tensor& other) const;  // Element-wise add
Tensor operator-(const Tensor& other) const;  // Element-wise subtract
Tensor operator*(const Tensor& other) const;  // Element-wise multiply
Tensor operator/(float scalar) const;         // Scalar division
Tensor operator*(float scalar) const;         // Scalar multiplication

Tensor relu() const;      // ReLU activation
Tensor sigmoid() const;   // Sigmoid activation
Tensor tanh() const;      // Tanh activation
```

**Utilities:**
```cpp
void fill(float value);                              // Fill with constant
void randn(float mean = 0.0f, float stddev = 1.0f); // Random normal
```

## Architecture

```
┌─────────────────────────────────────┐
│        User Code (Model)            │
└─────────────────────────────────────┘
                ↓
┌─────────────────────────────────────┐
│      GoogleRuntime (Singleton)      │
│  • Kernel Registry                  │
│  • Memory Management                │
│  • Execution Orchestration          │
└─────────────────────────────────────┘
                ↓
┌─────────────────────────────────────┐
│     Compiled MLIR Kernels           │
│  (Linked at build time)             │
└─────────────────────────────────────┘
```

## Example: Complete Model

```cpp
#include "Google/Runtime/GoogleRuntime.h"
#include <iostream>

using namespace google::runtime;

// Compiled kernels
extern "C" {
    void matmul_optimized(float*, float*, float*);
}

class SimpleModel {
    Tensor weights_;
    GoogleRuntime& runtime_;
    
public:
    SimpleModel() 
        : weights_({512, 256}),
          runtime_(GoogleRuntime::instance()) {
        
        // Register kernels
        runtime_.registerKernel("matmul", 
            reinterpret_cast<void*>(matmul_optimized));
        
        // Initialize weights
        weights_.randn(0.0f, 0.1f);
    }
    
    Tensor forward(const Tensor& input) {
        // Eager preprocessing
        Tensor normalized = input / 255.0f;
        
        // Compiled matmul
        Tensor hidden(input.rows(), weights_.cols());
        std::vector<void*> args = {
            normalized.data(), 
            weights_.data(), 
            hidden.data()
        };
        runtime_.execute("matmul", args);
        
        // Eager activation
        return hidden.relu();
    }
};

int main() {
    SimpleModel model;
    
    Tensor input({128, 512});
    input.randn();
    
    Tensor output = model.forward(input);
    
    std::cout << "Output shape: " 
              << output.rows() << "x" << output.cols() 
              << std::endl;
    
    return 0;
}
```

## Testing

Run the Phase 1 test:

```bash
# Build test (requires compiled matmul kernel)
cmake --build build --target test_runtime_phase1

# Run test
.\build\bin\test_runtime_phase1.exe
```

Expected output:
```
=== Google Runtime Test (Phase 1) ===

Registered kernels: 1
Has matmul_l3: yes

Configuration:
  Matrix size: 1024x1024
  Iterations: 5
  Total FLOPs per matmul: 2147483648

Testing correctness...
  ✓ Correctness: PASS

Testing eager operations...
  X / 2.0 = 1.00 (expected 1.0)
  relu(Y) = 1.00 (expected 1.0)
  ✓ Eager operations: PASS

Benchmarking performance...

Results:
  Average time: XX.XX ms
  Performance: XX.XX GFLOPS

=== Phase 1 Runtime Test Complete ===
```

## Performance

Phase 1 provides:
- **Zero-copy execution**: Tensors share memory with compiled code
- **Aligned memory**: 64-byte alignment for SIMD optimization
- **Minimal overhead**: Direct function calls to compiled kernels

## Next Steps

Phase 1 is complete! Next:
- **Phase 2**: Implement PJRT-style interface with device abstraction
- **Phase 3**: Add GPU backend (CUDA/ROCm)
- **Phase 4**: Production hardening (caching, profiling, multi-device)

## Files

- **Header**: `include/Google/Runtime/GoogleRuntime.h`
- **Implementation**: `lib/Google/Runtime/GoogleRuntime.cpp`
- **Test**: `test/test_runtime_phase1.cpp`
- **Build Script**: `scripts/build_runtime_phase1.ps1`
