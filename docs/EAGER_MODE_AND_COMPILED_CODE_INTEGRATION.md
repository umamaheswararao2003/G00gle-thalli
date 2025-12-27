# Eager Mode and Compiled Code Integration

## Executive Summary

This document explains how **compiled code** (from DL compilers) integrates with **eager execution** in ML frameworks, using decorators to mark compilation boundaries. We'll explore the mechanisms used by JAX, PyTorch, TensorFlow, and how this applies to MLIR-based compilers.

---

## Table of Contents

1. [The Problem: Mixing Eager and Compiled Code](#the-problem-mixing-eager-and-compiled-code)
2. [How Decorators Work](#how-decorators-work)
3. [Compilation and Integration Pipeline](#compilation-and-integration-pipeline)
4. [Detailed Mechanism Breakdown](#detailed-mechanism-breakdown)
5. [Framework Comparison](#framework-comparison)
6. [Implementation for Your MLIR Compiler](#implementation-for-your-mlir-compiler)

---

## The Problem: Mixing Eager and Compiled Code

### Eager Mode vs Compiled Mode

```python
# Eager mode - executes immediately, line by line
def eager_function(x, y):
    a = x + y          # Executes immediately
    b = a * 2          # Executes immediately
    return b           # Returns immediately

# Compiled mode - builds graph, optimizes, then executes
@jax.jit  # or @torch.compile, @tf.function
def compiled_function(x, y):
    a = x + y          # Traced into graph
    b = a * 2          # Traced into graph
    return b           # Graph compiled to native code
```

### The Challenge

How do you seamlessly call compiled code from eager code?

```python
# Model with mixed execution
class MyModel:
    def forward(self, x):
        # Eager preprocessing
        x = x.reshape(-1, 784)
        x = x / 255.0
        
        # COMPILED SECTION (decorator marks this)
        x = self.compiled_matmul(x, self.weights)
        
        # Eager postprocessing
        x = x.cpu().numpy()
        return x
    
    @jax.jit  # How does this integrate?
    def compiled_matmul(self, x, w):
        return x @ w
```

**Key Questions:**
1. How does the decorator intercept the function call?
2. How is the graph extracted and compiled?
3. How does compiled code connect back to eager execution?
4. How are tensors passed between eager and compiled code?

---

## How Decorators Work

### Python Decorator Mechanism

```python
# What a decorator does
@jax.jit
def my_function(x):
    return x * 2

# Is equivalent to:
def my_function(x):
    return x * 2
my_function = jax.jit(my_function)  # Wraps the function
```

### Decorator Implementation Pattern

```python
class JITDecorator:
    def __init__(self, func):
        self.original_func = func
        self.compiled_cache = {}  # Cache compiled versions
        
    def __call__(self, *args, **kwargs):
        # 1. Get input shapes/types (for cache key)
        cache_key = self._get_cache_key(args, kwargs)
        
        # 2. Check if already compiled
        if cache_key not in self.compiled_cache:
            # 3. Trace the function to build graph
            graph = self._trace_function(self.original_func, args, kwargs)
            
            # 4. Compile graph to native code
            compiled_code = self._compile_graph(graph)
            
            # 5. Cache it
            self.compiled_cache[cache_key] = compiled_code
        
        # 6. Execute compiled code
        return self._execute_compiled(
            self.compiled_cache[cache_key], 
            args, 
            kwargs
        )
```

---

## Compilation and Integration Pipeline

### Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    EAGER EXECUTION                              │
│  Python interpreter running user code                           │
└─────────────────────────────────────────────────────────────────┘
                            ↓
                    Function call with @jit
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: DECORATOR INTERCEPTS CALL                               │
│  • Decorator's __call__ method invoked                          │
│  • Extract arguments (tensors, shapes, dtypes)                  │
│  • Create cache key from input signatures                       │
└─────────────────────────────────────────────────────────────────┘
                            ↓
                    Check compilation cache
                            ↓
                    ┌───────┴───────┐
                    ↓               ↓
            Cache Hit          Cache Miss
                ↓                   ↓
        Skip to Step 5    ┌─────────────────────────────────────┐
                          │ STEP 2: TRACING                     │
                          │  • Replace tensors with tracers     │
                          │  • Execute function symbolically    │
                          │  • Record operations into graph     │
                          └─────────────────────────────────────┘
                                      ↓
                          ┌─────────────────────────────────────┐
                          │ STEP 3: GRAPH OPTIMIZATION          │
                          │  • Constant folding                 │
                          │  • Operator fusion                  │
                          │  • Dead code elimination            │
                          └─────────────────────────────────────┘
                                      ↓
                          ┌─────────────────────────────────────┐
                          │ STEP 4: COMPILATION                 │
                          │  • Lower to MLIR/XLA HLO            │
                          │  • Apply optimizations              │
                          │  • Generate LLVM IR                 │
                          │  • Compile to machine code          │
                          │  • Create callable wrapper          │
                          └─────────────────────────────────────┘
                                      ↓
                          ┌─────────────────────────────────────┐
                          │ STEP 5: CREATE EXECUTABLE           │
                          │  • Wrap compiled code               │
                          │  • Set up buffer descriptors        │
                          │  • Cache for future calls           │
                          └─────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: EXECUTE COMPILED CODE                                   │
│  • Convert Python tensors to C buffers                          │
│  • Call compiled function (FFI)                                 │
│  • Compiled code runs (native CPU/GPU)                          │
│  • Convert C buffers back to Python tensors                     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    RETURN TO EAGER EXECUTION                    │
│  • Python receives result tensors                               │
│  • Continue eager execution                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detailed Mechanism Breakdown

### Step 1: Decorator Intercepts Call

**What happens when you call a decorated function:**

```python
@jax.jit
def matmul(x, w):
    return x @ w

# When you call:
result = matmul(input_tensor, weight_tensor)

# The decorator intercepts this call
```

**Decorator Implementation:**

```python
class JIT:
    def __init__(self, func):
        self.func = func
        self.cache = {}
    
    def __call__(self, *args, **kwargs):
        # Extract metadata from arguments
        signature = self._get_signature(args, kwargs)
        
        # Check cache
        if signature not in self.cache:
            self.cache[signature] = self._compile(args, kwargs)
        
        # Execute
        return self.cache[signature].execute(args, kwargs)
    
    def _get_signature(self, args, kwargs):
        """Create cache key from input shapes/dtypes"""
        sig = []
        for arg in args:
            if isinstance(arg, Tensor):
                sig.append((arg.shape, arg.dtype))
        return tuple(sig)
```

---

### Step 2: Tracing (Graph Capture)

**How the graph is extracted:**

```python
# Original function
def matmul(x, w):
    return x @ w

# During tracing, inputs are replaced with "tracers"
class Tracer:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.operations = []  # Records operations
    
    def __matmul__(self, other):
        # Instead of computing, record the operation
        result = Tracer(
            shape=(self.shape[0], other.shape[1]),
            dtype=self.dtype
        )
        result.operations = self.operations + [
            ('matmul', self, other)
        ]
        return result
```

**Tracing Process:**

```python
def trace_function(func, example_inputs):
    # Replace inputs with tracers
    traced_inputs = [
        Tracer(inp.shape, inp.dtype) 
        for inp in example_inputs
    ]
    
    # Execute function with tracers
    traced_output = func(*traced_inputs)
    
    # Extract recorded operations
    graph = traced_output.operations
    
    return graph
```

**Example:**

```python
# Original code
def forward(x, w1, w2):
    h = x @ w1
    h = relu(h)
    y = h @ w2
    return y

# After tracing, we get a graph:
graph = [
    ('matmul', input_0, input_1, output_0),
    ('relu', output_0, output_1),
    ('matmul', output_1, input_2, output_2),
    ('return', output_2)
]
```

---

### Step 3: Graph Optimization

**Optimizations applied to the graph:**

```python
# Before optimization
graph = [
    ('constant', 2.0, c1),
    ('constant', 3.0, c2),
    ('add', c1, c2, c3),        # Can be folded
    ('mul', x, c3, y),
]

# After constant folding
graph = [
    ('constant', 5.0, c3),      # 2.0 + 3.0 = 5.0
    ('mul', x, c3, y),
]
```

**Fusion Example:**

```python
# Before fusion (2 kernels)
graph = [
    ('matmul', x, w, h),
    ('relu', h, y),
]

# After fusion (1 kernel)
graph = [
    ('fused_matmul_relu', x, w, y),
]
```

---

### Step 4: Compilation to Native Code

**MLIR Compilation Pipeline:**

```
Python Graph → MLIR High-Level → MLIR Optimized → LLVM IR → Machine Code
```

**Example MLIR Generation:**

```python
# Python graph
graph = [('matmul', x, w, y)]

# Generated MLIR
"""
func.func @matmul(%x: tensor<1024x512xf32>, 
                  %w: tensor<512x256xf32>) 
    -> tensor<1024x256xf32> {
  %y = linalg.matmul ins(%x, %w : tensor<1024x512xf32>, tensor<512x256xf32>)
                     outs(%init : tensor<1024x256xf32>)
                     -> tensor<1024x256xf32>
  return %y : tensor<1024x256xf32>
}
"""

# Compile MLIR → LLVM IR → Object code
compiled_obj = mlir_compile(mlir_code)
```

**Compilation Result:**

```cpp
// Compiled function signature (C ABI)
extern "C" void matmul_compiled(
    float* x_data, int64_t* x_shape, int64_t* x_strides,
    float* w_data, int64_t* w_shape, int64_t* w_strides,
    float* y_data, int64_t* y_shape, int64_t* y_strides
);
```

---

### Step 5: Creating Executable Wrapper

**Wrapper connects Python to compiled code:**

```python
class CompiledExecutable:
    def __init__(self, compiled_func_ptr, signature):
        self.func_ptr = compiled_func_ptr  # Pointer to compiled code
        self.signature = signature
    
    def execute(self, args, kwargs):
        # 1. Convert Python tensors to C arrays
        c_buffers = self._tensors_to_c_buffers(args)
        
        # 2. Call compiled function via FFI
        result_buffer = self._call_compiled(c_buffers)
        
        # 3. Convert C arrays back to Python tensors
        result_tensor = self._c_buffer_to_tensor(result_buffer)
        
        return result_tensor
    
    def _tensors_to_c_buffers(self, tensors):
        """Convert Python tensors to C-compatible buffers"""
        buffers = []
        for tensor in tensors:
            buffer = {
                'data': tensor.data_ptr(),  # Raw pointer
                'shape': tensor.shape,
                'strides': tensor.strides,
                'dtype': tensor.dtype
            }
            buffers.append(buffer)
        return buffers
    
    def _call_compiled(self, buffers):
        """Call compiled code via ctypes/cffi"""
        import ctypes
        
        # Load compiled function
        func = ctypes.CDLL(self.func_ptr)
        
        # Prepare arguments
        args = []
        for buf in buffers:
            args.extend([
                buf['data'],
                buf['shape'],
                buf['strides']
            ])
        
        # Call
        func.matmul_compiled(*args)
```

---

### Step 6: Execution and Data Transfer

**Memory Layout and Transfer:**

```
┌─────────────────────────────────────────────────────────┐
│              PYTHON/EAGER SIDE                          │
│                                                         │
│  Python Tensor Object                                  │
│  ┌──────────────────────────────────────┐             │
│  │ Metadata:                            │             │
│  │  - shape: (1024, 512)                │             │
│  │  - dtype: float32                    │             │
│  │  - strides: (512, 1)                 │             │
│  │                                      │             │
│  │ Data Pointer ───────────────────┐   │             │
│  └──────────────────────────────────│───┘             │
└────────────────────────────────────│────────────────────┘
                                     │
                                     ↓
┌────────────────────────────────────────────────────────┐
│              SHARED MEMORY                             │
│                                                        │
│  Raw Float Array (contiguous memory)                  │
│  [1.0, 2.0, 3.0, ..., 524288 elements]                │
│                                                        │
└────────────────────────────────────────────────────────┘
                                     ↑
                                     │
┌────────────────────────────────────│────────────────────┐
│              COMPILED CODE SIDE    │                    │
│                                    │                    │
│  C Function Parameters             │                    │
│  ┌──────────────────────────────────────┐             │
│  │ float* data ──────────────────────┘  │             │
│  │ int64_t* shape = {1024, 512}         │             │
│  │ int64_t* strides = {512, 1}          │             │
│  └──────────────────────────────────────┘             │
│                                                        │
│  Compiled Machine Code                                │
│  [0x48, 0x89, 0xf8, ...]  (x86-64 instructions)       │
└────────────────────────────────────────────────────────┘
```

**Key Insight**: **No data copying** happens if memory is already contiguous! Python tensor and compiled code share the same memory.

---

## Framework Comparison

### JAX (@jax.jit)

**Architecture:**

```
Python Code → JAX Tracer → XLA HLO → LLVM IR → Machine Code
                                         ↓
                                    PJRT Runtime
```

**Implementation:**

```python
@jax.jit
def matmul(x, w):
    return jnp.dot(x, w)

# What happens:
# 1. JAX traces the function
# 2. Converts to XLA HLO (High-Level Operations)
# 3. XLA compiles HLO to LLVM IR
# 4. LLVM generates machine code
# 5. PJRT runtime executes compiled code
```

**Integration Mechanism:**

```python
class JaxJIT:
    def __call__(self, *args):
        # 1. Trace to HLO
        hlo_module = self._trace_to_hlo(args)
        
        # 2. Compile via XLA
        compiled = xla_client.compile(hlo_module)
        
        # 3. Execute via PJRT
        pjrt_buffers = [arg._device_array for arg in args]
        result_buffers = compiled.execute(pjrt_buffers)
        
        # 4. Wrap in JAX array
        return jax.Array(result_buffers[0])
```

**Data Transfer:**

```python
# JAX uses device arrays (already on device)
x_jax = jnp.array([1, 2, 3])  # Creates DeviceArray

@jax.jit
def f(x):
    return x * 2

result = f(x_jax)  # No host-device transfer!
```

---

### PyTorch (@torch.compile)

**Architecture (PyTorch 2.0+):**

```
Python Code → TorchDynamo → FX Graph → TorchInductor → Triton/C++ → Machine Code
```

**Implementation:**

```python
@torch.compile
def matmul(x, w):
    return torch.matmul(x, w)

# What happens:
# 1. TorchDynamo captures bytecode
# 2. Converts to FX graph
# 3. TorchInductor generates Triton/C++ code
# 4. Compiles to CUDA kernels or CPU code
```

**Integration Mechanism:**

```python
class TorchCompile:
    def __call__(self, *args):
        # 1. Capture via TorchDynamo (bytecode analysis)
        graph = torch._dynamo.export(self.func, args)
        
        # 2. Optimize FX graph
        optimized_graph = optimize_fx_graph(graph)
        
        # 3. Generate code via TorchInductor
        compiled_fn = torch._inductor.compile(optimized_graph)
        
        # 4. Execute
        return compiled_fn(*args)
```

**Data Transfer:**

```python
# PyTorch tensors already have device info
x = torch.randn(1024, 512, device='cuda')

@torch.compile
def f(x):
    return x * 2

result = f(x)  # Stays on GPU, no transfer
```

---

### TensorFlow (@tf.function)

**Architecture:**

```
Python Code → AutoGraph → TF Graph → XLA/TFLite → Machine Code
```

**Implementation:**

```python
@tf.function
def matmul(x, w):
    return tf.matmul(x, w)

# What happens:
# 1. AutoGraph converts Python to TF graph
# 2. Graph optimization (Grappler)
# 3. Optional XLA compilation
# 4. Execute via TF runtime
```

**Integration Mechanism:**

```python
class TFFunction:
    def __call__(self, *args):
        # 1. Trace to TF graph
        with tf.Graph().as_default() as graph:
            traced_fn = self._trace(args)
        
        # 2. Optimize graph
        optimized_graph = tf.graph_util.optimize_graph(graph)
        
        # 3. Create session/function
        concrete_fn = tf.compat.v1.Session(graph=optimized_graph)
        
        # 4. Execute
        return concrete_fn.run(traced_fn, feed_dict={...})
```

---

### Comparison Table

| Framework | Tracing Method | IR | Compiler | Runtime | Integration |
|-----------|---------------|-----|----------|---------|-------------|
| **JAX** | Tracer objects | XLA HLO | XLA/LLVM | PJRT | DeviceArray wrapping |
| **PyTorch** | Bytecode capture | FX Graph | TorchInductor/Triton | ATen | Tensor metadata preserved |
| **TensorFlow** | AutoGraph | TF Graph | XLA/Grappler | TF Runtime | Eager tensor wrapping |
| **MLIR (Your project)** | Custom tracer | MLIR dialects | MLIR→LLVM | Custom runtime | **To be designed** |

---

## Implementation for Your MLIR Compiler

### Proposed Design

```python
# google_mlir/decorators.py

import ctypes
import numpy as np
from typing import Callable, Any

class GoogleJIT:
    """JIT decorator for Google MLIR compiler"""
    
    def __init__(self, func: Callable):
        self.func = func
        self.cache = {}
        self.mlir_compiler = GoogleMLIRCompiler()
    
    def __call__(self, *args, **kwargs):
        # 1. Get signature
        sig = self._get_signature(args, kwargs)
        
        # 2. Check cache
        if sig not in self.cache:
            # 3. Trace to MLIR
            mlir_module = self._trace_to_mlir(args, kwargs)
            
            # 4. Compile
            compiled_obj = self.mlir_compiler.compile(mlir_module)
            
            # 5. Create executable
            self.cache[sig] = GoogleExecutable(compiled_obj, sig)
        
        # 6. Execute
        return self.cache[sig].execute(args, kwargs)
    
    def _get_signature(self, args, kwargs):
        """Extract shapes and dtypes for cache key"""
        sig = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                sig.append((arg.shape, arg.dtype))
        return tuple(sig)
    
    def _trace_to_mlir(self, args, kwargs):
        """Trace function to MLIR"""
        # Create MLIR tracers
        tracers = [GoogleTracer(arg.shape, arg.dtype) for arg in args]
        
        # Execute function with tracers
        result = self.func(*tracers, **kwargs)
        
        # Generate MLIR from trace
        mlir_code = self._generate_mlir(result.trace)
        
        return mlir_code


class GoogleTracer:
    """Tracer object that records operations"""
    
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.trace = []
        self.id = id(self)
    
    def __matmul__(self, other):
        result = GoogleTracer(
            shape=(self.shape[0], other.shape[1]),
            dtype=self.dtype
        )
        result.trace = self.trace + other.trace + [
            ('google.matmul', self.id, other.id, result.id)
        ]
        return result
    
    def __add__(self, other):
        result = GoogleTracer(self.shape, self.dtype)
        result.trace = self.trace + other.trace + [
            ('google.add', self.id, other.id, result.id)
        ]
        return result


class GoogleExecutable:
    """Wrapper for compiled code"""
    
    def __init__(self, compiled_obj_path, signature):
        self.lib = ctypes.CDLL(compiled_obj_path)
        self.signature = signature
        self._setup_function()
    
    def _setup_function(self):
        """Set up ctypes function signature"""
        # Assuming compiled function has signature:
        # void compiled_func(float* in1, float* in2, float* out, 
        #                    int64_t* shapes, int64_t* strides)
        self.lib.compiled_func.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # input 1
            ctypes.POINTER(ctypes.c_float),  # input 2
            ctypes.POINTER(ctypes.c_float),  # output
            ctypes.POINTER(ctypes.c_int64),  # shapes
            ctypes.POINTER(ctypes.c_int64),  # strides
        ]
        self.lib.compiled_func.restype = None
    
    def execute(self, args, kwargs):
        """Execute compiled code"""
        # 1. Prepare input buffers
        input_ptrs = []
        for arg in args:
            ptr = arg.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            input_ptrs.append(ptr)
        
        # 2. Allocate output buffer
        output_shape = self._infer_output_shape(args)
        output = np.zeros(output_shape, dtype=np.float32)
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # 3. Prepare shape/stride arrays
        shapes = np.array([arg.shape for arg in args], dtype=np.int64)
        strides = np.array([arg.strides for arg in args], dtype=np.int64)
        
        # 4. Call compiled function
        self.lib.compiled_func(
            *input_ptrs,
            output_ptr,
            shapes.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            strides.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        )
        
        # 5. Return result
        return output


class GoogleMLIRCompiler:
    """MLIR compiler backend"""
    
    def compile(self, mlir_code: str) -> str:
        """Compile MLIR to object file"""
        import subprocess
        import tempfile
        
        # 1. Write MLIR to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
            f.write(mlir_code)
            mlir_file = f.name
        
        # 2. Compile MLIR to LLVM IR
        llvm_file = mlir_file.replace('.mlir', '.ll')
        subprocess.run([
            'google-opt', mlir_file,
            '--google-extreme-pipeline',
            '--convert-to-llvm',
            '-o', llvm_file
        ])
        
        # 3. Compile LLVM IR to object file
        obj_file = llvm_file.replace('.ll', '.so')
        subprocess.run([
            'clang', '-shared', '-fPIC',
            llvm_file, '-o', obj_file
        ])
        
        return obj_file
```

### Usage Example

```python
import numpy as np
from google_mlir.decorators import GoogleJIT

# Define model with decorator
class MyModel:
    def __init__(self):
        self.weights = np.random.randn(512, 256).astype(np.float32)
    
    def forward(self, x):
        # Eager preprocessing
        x = x.reshape(-1, 512)
        x = x / 255.0
        
        # COMPILED SECTION
        x = self.compiled_matmul(x, self.weights)
        
        # Eager postprocessing
        x = x.clip(0, 1)
        return x
    
    @GoogleJIT  # <-- Decorator marks compilation boundary
    def compiled_matmul(self, x, w):
        return x @ w  # Will be compiled to optimized MLIR


# Use the model
model = MyModel()
input_data = np.random.randn(128, 512).astype(np.float32)

# First call: traces, compiles, caches
output = model.forward(input_data)  # ~100ms (compilation overhead)

# Subsequent calls: uses cached compiled code
output = model.forward(input_data)  # ~1ms (fast!)
```

### Generated MLIR

```mlir
// From traced function
module {
  func.func @compiled_matmul(%arg0: tensor<128x512xf32>, 
                             %arg1: tensor<512x256xf32>) 
      -> tensor<128x256xf32> {
    %0 = google.matmul %arg0, %arg1 
      : tensor<128x512xf32>, tensor<512x256xf32> -> tensor<128x256xf32>
    return %0 : tensor<128x256xf32>
  }
}
```

---

## Key Takeaways

### How Integration Works

1. **Decorator intercepts** function calls
2. **Tracing** extracts computational graph
3. **Compilation** generates optimized native code
4. **Wrapper** connects Python to compiled code via FFI
5. **Shared memory** enables zero-copy data transfer
6. **Caching** avoids recompilation for same input shapes

### Critical Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Tracing** | Operator overloading | Pythonic, easy to implement |
| **Caching** | Shape + dtype signature | Balance between cache hits and memory |
| **Data transfer** | Zero-copy via pointers | Maximum performance |
| **ABI** | C calling convention | Standard, portable |
| **Compilation** | Lazy (on first call) | Avoid upfront cost |

### Comparison Summary

- **JAX**: Most sophisticated, uses PJRT, excellent for research
- **PyTorch**: Bytecode-level tracing, best eager-compiled integration
- **TensorFlow**: Graph-based, mature but complex
- **Your MLIR Compiler**: Opportunity to design clean, modern interface

---

## Conclusion

The integration of compiled code with eager execution is achieved through:

1. **Decorators** that intercept function calls
2. **Tracing** to capture computational graphs
3. **Compilation** to native code
4. **FFI wrappers** to bridge Python and compiled code
5. **Shared memory** for zero-copy data transfer

This design allows seamless mixing of eager and compiled code, giving users the flexibility of eager mode with the performance of compiled execution.
