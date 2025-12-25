# Google MLIR Dialect - Comprehensive Benchmark Documentation

## Table of Contents
1. [Overview](#overview)
2. [Benchmark Categories](#benchmark-categories)
3. [Memory-Bound Benchmarks](#memory-bound-benchmarks)
4. [Compute-Bound Benchmarks](#compute-bound-benchmarks)
5. [Compilation Benchmarks](#compilation-benchmarks)
6. [Build Evaluation Benchmarks](#build-evaluation-benchmarks)
7. [Command Reference](#command-reference)
8. [Interpreting Results](#interpreting-results)
9. [Troubleshooting](#troubleshooting)

---

## Overview

This benchmark suite measures performance across four dimensions:
- **Memory-bound operations**: Limited by memory bandwidth (360 GB/s on RTX 3060)
- **Compute-bound operations**: Limited by compute throughput (13 TFLOPS on RTX 3060)
- **Compilation performance**: Measures optimization effectiveness
- **Build evaluation**: Measures compilation time and efficiency

### Hardware Target: NVIDIA GeForce RTX 3060
- **CUDA Cores**: 3,584
- **Tensor Cores**: 112 (3rd gen)
- **Memory**: 12GB GDDR6
- **Memory Bandwidth**: 360 GB/s
- **Peak FP32**: 13 TFLOPS
- **L2 Cache**: 3MB
- **Shared Memory**: 48KB per SM

---

## Benchmark Categories

### 1. Memory-Bound Benchmarks
**Location**: `benchmarks/memory_bound/`  
**Count**: 5 test files  
**Purpose**: Test operations where performance is limited by memory bandwidth  
**Characteristic**: Low arithmetic intensity (<10 FLOPS/byte)  
**Target**: >320 GB/s memory bandwidth utilization (89% of peak)

### 2. Compute-Bound Benchmarks
**Location**: `benchmarks/compute_bound/`  
**Count**: 7 test files  
**Purpose**: Test operations where performance is limited by compute throughput  
**Characteristic**: High arithmetic intensity (>50 FLOPS/byte)  
**Target**: 7,000-13,000 GFLOPS for matmul operations

### 3. Compilation Benchmarks
**Location**: `benchmarks/compilation/`  
**Count**: 2 PowerShell scripts  
**Purpose**: Measure compilation performance and optimization effectiveness  
**Metrics**: GFLOPS, speedup ratios, optimization gains

### 4. Build Evaluation Benchmarks
**Location**: `benchmarks/build_evaluation/`  
**Count**: 1 PowerShell script  
**Purpose**: Measure end-to-end compilation time  
**Target**: <1000ms for typical workloads

---

## Memory-Bound Benchmarks

### Overview
Memory-bound operations are limited by how fast data can be moved between memory and compute units, not by computation speed.

**Arithmetic Intensity Formula**: FLOPS / Bytes Accessed  
**Memory-Bound Threshold**: <10 FLOPS/byte

### Test 1: Element-wise Operations
**File**: `bench_elementwise.mlir`  
**Size**: 1,524 bytes  
**Operations**: Add, Mul, ReLU, Sigmoid, Tanh  
**Tensor Size**: 4096×4096 (67 MB per tensor)  
**Arithmetic Intensity**: ~2 FLOPS/byte

**What it tests**:
- Element-wise addition: `C[i] = A[i] + B[i]`
- Element-wise multiplication: `C[i] = A[i] * B[i]`
- ReLU activation: `C[i] = max(0, A[i])`
- Sigmoid activation: `C[i] = 1 / (1 + exp(-A[i]))`
- Tanh activation: `C[i] = tanh(A[i])`

**Why it's memory-bound**:
- Each element is read once, computed once, written once
- Very few operations per byte of data
- Performance limited by memory bandwidth, not compute

**Command**:
```powershell
# Compile only
.\build\bin\google-opt.exe benchmarks\memory_bound\bench_elementwise.mlir --google-basic-pipeline

# With timing
Measure-Command { .\build\bin\google-opt.exe benchmarks\memory_bound\bench_elementwise.mlir --google-basic-pipeline | Out-Null }
```

**Expected Performance**:
- Memory bandwidth: >320 GB/s
- Compilation time: ~200-300ms
- Should be limited by memory, not compute

---

### Test 2: Transpose
**File**: `bench_transpose.mlir`  
**Size**: 1,372 bytes  
**Operations**: Matrix transpose  
**Tensor Sizes**: 1024×1024, 4096×4096, 2048×8192, 3D tensors  
**Arithmetic Intensity**: ~0 FLOPS/byte (pure memory operation)

**What it tests**:
- Square matrix transpose: `B[j,i] = A[i,j]`
- Rectangular matrix transpose
- 3D tensor transpose with different permutations

**Why it's memory-bound**:
- Zero computation, only memory reads and writes
- Tests memory coalescing (accessing contiguous memory)
- Non-coalesced access can reduce bandwidth by 10×

**Memory Access Pattern**:
```
Input:  A[0,0], A[0,1], A[0,2], ... (row-major, coalesced)
Output: B[0,0], B[1,0], B[2,0], ... (column-major, non-coalesced)
```

**Command**:
```powershell
.\build\bin\google-opt.exe benchmarks\memory_bound\bench_transpose.mlir --google-basic-pipeline
```

**Expected Performance**:
- Memory bandwidth: >300 GB/s (with proper coalescing)
- Without coalescing: ~30 GB/s (10× slower)
- Compilation time: ~250ms

---

### Test 3: Reshape
**File**: `bench_reshape.mlir`  
**Size**: 1,472 bytes  
**Operations**: Tensor reshape  
**Transformations**: 2D↔1D, 2D↔3D, 3D↔4D  
**Arithmetic Intensity**: ~0 FLOPS/byte

**What it tests**:
- 2D to 1D: `[4096, 4096] → [16777216]`
- 1D to 2D: `[16777216] → [4096, 4096]`
- 2D to 3D: `[1024, 1024] → [32, 32, 1024]`
- Complex reshapes

**Why it's memory-bound**:
- No computation, only memory layout changes
- Tests sequential memory access patterns
- May require memory copy if layout changes

**Command**:
```powershell
.\build\bin\google-opt.exe benchmarks\memory_bound\bench_reshape.mlir --google-basic-pipeline
```

**Expected Performance**:
- Memory bandwidth: >350 GB/s (sequential access is fastest)
- Compilation time: ~200ms
- May be optimized away if no actual copy needed

---

### Test 4: Concat
**File**: `bench_concat.mlir`  
**Size**: 1,938 bytes  
**Operations**: Tensor concatenation  
**Modes**: Axis 0, Axis 1, Multiple tensors, 3D tensors  
**Arithmetic Intensity**: ~0 FLOPS/byte

**What it tests**:
- Concatenate along rows (axis 0)
- Concatenate along columns (axis 1)
- Multiple tensor concatenation
- 3D tensor concatenation

**Why it's memory-bound**:
- Pure memory copy operation
- Tests memory write patterns
- Multiple source tensors → single destination

**Memory Pattern**:
```
Input:  [A: 2048×4096] + [B: 2048×4096]
Output: [C: 4096×4096] (A stacked on top of B)
```

**Command**:
```powershell
.\build\bin\google-opt.exe benchmarks\memory_bound\bench_concat.mlir --google-basic-pipeline
```

**Expected Performance**:
- Memory bandwidth: >300 GB/s
- Compilation time: ~300ms
- Write bandwidth may be lower than read bandwidth

---

### Test 5: Broadcast
**File**: `bench_broadcast.mlir`  
**Size**: 1,586 bytes  
**Operations**: Tensor broadcasting  
**Modes**: Vector→Matrix, Scalar→Matrix, 2D→3D, 1D→3D  
**Arithmetic Intensity**: ~0 FLOPS/byte

**What it tests**:
- Broadcast vector to matrix: `[4096] → [4096, 4096]`
- Broadcast scalar to matrix: `[1] → [4096, 4096]`
- Broadcast to higher dimensions
- Large broadcasts (8192×8192)

**Why it's memory-bound**:
- Memory write-heavy operation
- Single value written to many locations
- Tests write bandwidth and caching

**Memory Pattern**:
```
Input:  [1, 2, 3, 4] (4 values)
Output: [[1, 2, 3, 4],
         [1, 2, 3, 4],
         [1, 2, 3, 4],
         ...] (4096 rows)
```

**Command**:
```powershell
.\build\bin\google-opt.exe benchmarks\memory_bound\bench_broadcast.mlir --google-basic-pipeline
```

**Expected Performance**:
- Memory bandwidth: >320 GB/s
- Compilation time: ~250ms
- May benefit from caching small input

---

## Compute-Bound Benchmarks

### Overview
Compute-bound operations are limited by computational throughput, not memory bandwidth. They reuse data in cache, achieving high arithmetic intensity.

**Arithmetic Intensity Formula**: FLOPS / Bytes Accessed  
**Compute-Bound Threshold**: >50 FLOPS/byte

### MatMul Benchmarks (5 sizes)

All matmul benchmarks include **embedded L3 tiling transform**:
- **L3 tiling**: 256×256×256 (outermost loops for L3 cache)
- **L2 tiling**: 64×64×64 (middle loops for L2 cache)
- **L1 tiling**: 16×16×16 (innermost loops for L1 cache)

This creates a **9-level loop nest** for optimal cache utilization.

---

### Test 1: MatMul 256×256
**File**: `bench_matmul_256.mlir`  
**Size**: 1,674 bytes  
**Matrix Dimensions**: 256×256 × 256×256 → 256×256  
**FLOPs**: 2 × 256³ = 33,554,432 FLOPs  
**Memory**: 3 × 256² × 4 bytes = 768 KB  
**Arithmetic Intensity**: ~128 FLOPS/byte  
**Target**: 3,000 GFLOPS (23% of peak)

**Why this target**:
- Small matrix, fits entirely in L3 cache
- Limited by kernel launch overhead
- Good for testing tiling correctness

**Command**:
```powershell
# With extreme pipeline (includes L3 tiling)
.\build\bin\google-opt.exe benchmarks\compute_bound\bench_matmul_256.mlir --google-extreme-pipeline

# With basic pipeline (no tiling)
.\build\bin\google-opt.exe benchmarks\compute_bound\bench_matmul_256.mlir --google-basic-pipeline
```

**Expected Performance**:
- Extreme pipeline: ~3,000 GFLOPS
- Basic pipeline: ~500 GFLOPS (6× slower)
- Compilation time: ~150ms

---

### Test 2: MatMul 512×512
**File**: `bench_matmul_512.mlir`  
**Size**: 1,614 bytes  
**Matrix Dimensions**: 512×512 × 512×512 → 512×512  
**FLOPs**: 2 × 512³ = 268,435,456 FLOPs  
**Memory**: 3 × 512² × 4 bytes = 3 MB  
**Arithmetic Intensity**: ~170 FLOPS/byte  
**Target**: 5,000 GFLOPS (38% of peak)

**Why this target**:
- Matrix size exceeds L3 cache (3MB)
- Tiling becomes important
- Good balance between overhead and compute

**Command**:
```powershell
.\build\bin\google-opt.exe benchmarks\compute_bound\bench_matmul_512.mlir --google-extreme-pipeline
```

**Expected Performance**:
- Extreme pipeline: ~5,000 GFLOPS
- Basic pipeline: ~800 GFLOPS (6× slower)
- Compilation time: ~200ms

---

### Test 3: MatMul 1024×1024
**File**: `bench_matmul_1024.mlir`  
**Size**: 1,645 bytes  
**Matrix Dimensions**: 1024×1024 × 1024×1024 → 1024×1024  
**FLOPs**: 2 × 1024³ = 2,147,483,648 FLOPs  
**Memory**: 3 × 1024² × 4 bytes = 12 MB  
**Arithmetic Intensity**: ~213 FLOPS/byte  
**Target**: 7,000 GFLOPS (54% of peak)

**Why this target**:
- **Key benchmark** for performance validation
- Large enough to amortize overhead
- Tests L3 tiling effectiveness
- Commonly used size in ML workloads

**Command**:
```powershell
.\build\bin\google-opt.exe benchmarks\compute_bound\bench_matmul_1024.mlir --google-extreme-pipeline
```

**Expected Performance**:
- Extreme pipeline: ~7,000 GFLOPS
- Basic pipeline: ~1,200 GFLOPS (6× slower)
- Compilation time: ~300ms

**GFLOPS Calculation**:
```
FLOPs = 2 × M × N × K = 2 × 1024 × 1024 × 1024 = 2,147,483,648
Time = 0.307 seconds (example)
GFLOPS = 2,147,483,648 / (0.307 × 1e9) = 6,995 GFLOPS
```

---

### Test 4: MatMul 2048×2048
**File**: `bench_matmul_2048.mlir`  
**Size**: 1,646 bytes  
**Matrix Dimensions**: 2048×2048 × 2048×2048 → 2048×2048  
**FLOPs**: 2 × 2048³ = 17,179,869,184 FLOPs  
**Memory**: 3 × 2048² × 4 bytes = 48 MB  
**Arithmetic Intensity**: ~256 FLOPS/byte  
**Target**: 10,000 GFLOPS (77% of peak)

**Why this target**:
- Large matrix, high arithmetic intensity
- Tests multi-level tiling effectiveness
- Memory exceeds all cache levels
- Approaches peak performance

**Command**:
```powershell
.\build\bin\google-opt.exe benchmarks\compute_bound\bench_matmul_2048.mlir --google-extreme-pipeline
```

**Expected Performance**:
- Extreme pipeline: ~10,000 GFLOPS
- Basic pipeline: ~1,500 GFLOPS (7× slower)
- Compilation time: ~500ms

---

### Test 5: MatMul 4096×4096
**File**: `bench_matmul_4096.mlir`  
**Size**: 1,646 bytes  
**Matrix Dimensions**: 4096×4096 × 4096×4096 → 4096×4096  
**FLOPs**: 2 × 4096³ = 137,438,953,472 FLOPs  
**Memory**: 3 × 4096² × 4 bytes = 192 MB  
**Arithmetic Intensity**: ~341 FLOPS/byte  
**Target**: 12,000 GFLOPS (92% of peak)

**Why this target**:
- **Maximum performance test**
- Very high arithmetic intensity
- Tests sustained peak performance
- Memory pressure test (192 MB)

**Command**:
```powershell
.\build\bin\google-opt.exe benchmarks\compute_bound\bench_matmul_4096.mlir --google-extreme-pipeline
```

**Expected Performance**:
- Extreme pipeline: ~12,000 GFLOPS
- Basic pipeline: ~1,800 GFLOPS (7× slower)
- Compilation time: ~800ms

---

### Test 6: Softmax
**File**: `bench_softmax.mlir`  
**Size**: 1,036 bytes  
**Operations**: Softmax normalization  
**Tensor Sizes**: 1024×1024, 4096×4096, 3D (32×128×128)  
**Arithmetic Intensity**: ~20 FLOPS/byte  
**Target**: 4,000 GFLOPS

**What it computes**:
```
softmax(x_i) = exp(x_i) / Σ exp(x_j)
```

**Steps**:
1. Find max value (for numerical stability)
2. Compute exp(x - max)
3. Sum all exp values
4. Divide each by sum

**Why moderately compute-bound**:
- Multiple passes over data (max, exp, sum, divide)
- Exponential operation is compute-intensive
- Some data reuse in reduction

**Command**:
```powershell
.\build\bin\google-opt.exe benchmarks\compute_bound\bench_softmax.mlir --google-basic-pipeline
```

**Expected Performance**:
- ~4,000 GFLOPS
- Compilation time: ~250ms
- Memory bandwidth: ~200 GB/s

---

### Test 7: GELU
**File**: `bench_gelu.mlir`  
**Size**: 1,218 bytes  
**Operations**: Gaussian Error Linear Unit activation  
**Tensor Sizes**: 1024×1024, 4096×4096, 3D, 8192×8192  
**Arithmetic Intensity**: ~15 FLOPS/byte  
**Target**: 3,500 GFLOPS

**What it computes**:
```
GELU(x) = x * Φ(x)
where Φ(x) = 0.5 * (1 + erf(x / √2))
```

**Approximation**:
```
GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

**Why moderately compute-bound**:
- Multiple transcendental functions (tanh, cubic)
- More compute than simple element-wise ops
- Used in transformer models

**Command**:
```powershell
.\build\bin\google-opt.exe benchmarks\compute_bound\bench_gelu.mlir --google-basic-pipeline
```

**Expected Performance**:
- ~3,500 GFLOPS
- Compilation time: ~250ms
- Memory bandwidth: ~180 GB/s

---

## Compilation Benchmarks

### Script 1: Execution Efficiency
**File**: `bench_execution_efficiency.ps1`  
**Size**: 3,105 bytes  
**Purpose**: Measure GFLOPS for matmul operations  
**Tests**: All 5 matmul sizes (256 to 4096)

**What it measures**:
- Compilation time for each matmul size
- Calculates GFLOPS: `(2 × M × N × K) / (time_in_seconds × 1e9)`
- Compares against target GFLOPS
- Calculates efficiency percentage

**Output Metrics**:
- Average compilation time (ms)
- GFLOPS achieved
- Target GFLOPS
- Efficiency (%)

**Command**:
```powershell
cd benchmarks\compilation
.\bench_execution_efficiency.ps1

# With custom iterations
.\bench_execution_efficiency.ps1 -Iterations 20
```

**Output Example**:
```
Testing MatMul 1024x1024x1024... OK
  Time: 305.23ms
  GFLOPS: 7023.45 (Target: 7000)
  Efficiency: 100.3%
```

**CSV Output**:
```csv
Size,M,N,K,AvgTimeMs,GFLOPS,TargetGFLOPS,Efficiency
1024x1024x1024,1024,1024,1024,305.23,7023.45,7000,100.3
```

---

### Script 2: Optimization Gain
**File**: `bench_optimization_gain.ps1`  
**Size**: 3,360 bytes  
**Purpose**: Measure speedup across different pipelines  
**Test File**: `bench_matmul_1024.mlir`

**Pipelines Tested**:
1. `--google-basic-pipeline` (baseline)
2. `--google-optimized-pipeline`
3. `--google-extreme-l1` (L1 tiling only)
4. `--google-extreme-l2` (L1+L2 tiling)
5. `--google-extreme-l3` (L1+L2+L3 tiling)
6. `--google-extreme-l3-full` (L3 + LLVM lowering)
7. `--google-extreme-pipeline` (complete optimization)

**What it measures**:
- Compilation time for each pipeline
- Speedup vs baseline (basic pipeline)
- Success/failure for each pipeline

**Command**:
```powershell
cd benchmarks\compilation
.\bench_optimization_gain.ps1

# With custom iterations
.\bench_optimization_gain.ps1 -Iterations 20
```

**Output Example**:
```
Testing Basic... OK
  Time: 4523.12ms
  Speedup: 1.00x

Testing Extreme Pipeline... OK
  Time: 312.45ms
  Speedup: 14.48x

Maximum Speedup: 14.48x
Target (10x): ACHIEVED
```

**CSV Output**:
```csv
Pipeline,AvgTimeMs,Speedup,Success
Basic,4523.12,1.00,True
Extreme Pipeline,312.45,14.48,True
```

---

## Build Evaluation Benchmarks

### Script: Build Latency
**File**: `bench_latency.ps1`  
**Size**: 2,740 bytes  
**Purpose**: Measure end-to-end compilation time  
**Pipeline**: `--google-extreme-pipeline`

**Workloads Tested**:
1. Small (256×256): Target <200ms
2. Medium (1024×1024): Target <500ms
3. Large (4096×4096): Target <1000ms

**What it measures**:
- Average compilation time
- Minimum compilation time
- Maximum compilation time
- Whether within target

**Command**:
```powershell
cd benchmarks\build_evaluation
.\bench_latency.ps1

# With custom iterations
.\bench_latency.ps1 -Iterations 20
```

**Output Example**:
```
Testing Small (256x256)... OK
  Avg: 145.67ms (Target: 200ms)
  Min: 142.12ms | Max: 152.34ms

Testing Medium (1024x1024)... OK
  Avg: 405.23ms (Target: 500ms)
  Min: 398.45ms | Max: 415.67ms

Testing Large (4096x4096)... OK
  Avg: 823.45ms (Target: 1000ms)
  Min: 810.23ms | Max: 845.67ms

Average Latency: 458.12ms
All Within Target: YES
```

**CSV Output**:
```csv
Workload,AvgTimeMs,MinTimeMs,MaxTimeMs,TargetMs,WithinTarget
Small (256x256),145.67,142.12,152.34,200,True
Medium (1024x1024),405.23,398.45,415.67,500,True
Large (4096x4096),823.45,810.23,845.67,1000,True
```

---

## Command Reference

### Master Benchmark Runner

**Run all benchmarks**:
```powershell
cd c:\Users\Asus\Desktop\google\benchmarks
.\run_all_benchmarks.ps1
```

**Run specific category**:
```powershell
# Memory-bound only
.\run_all_benchmarks.ps1 -Category memory_bound

# Compute-bound only
.\run_all_benchmarks.ps1 -Category compute_bound

# Compilation benchmarks only
.\run_all_benchmarks.ps1 -Category compilation

# Build evaluation only
.\run_all_benchmarks.ps1 -Category build_evaluation
```

**Use different pipeline**:
```powershell
# Basic pipeline (fast, less optimized)
.\run_all_benchmarks.ps1 -Pipeline google-basic-pipeline

# Optimized pipeline (balanced)
.\run_all_benchmarks.ps1 -Pipeline google-optimized-pipeline

# Extreme pipeline (slow, highly optimized)
.\run_all_benchmarks.ps1 -Pipeline google-extreme-pipeline
```

**Change iterations**:
```powershell
# More iterations for better statistics
.\run_all_benchmarks.ps1 -Iterations 20

# Fewer iterations for quick test
.\run_all_benchmarks.ps1 -Iterations 3
```

**Combined options**:
```powershell
.\run_all_benchmarks.ps1 -Category compute_bound -Pipeline google-extreme-pipeline -Iterations 15
```

---

### Individual Test Files

**Compile single test**:
```powershell
# Basic compilation
.\build\bin\google-opt.exe benchmarks\memory_bound\bench_elementwise.mlir --google-basic-pipeline

# Extreme optimization
.\build\bin\google-opt.exe benchmarks\compute_bound\bench_matmul_1024.mlir --google-extreme-pipeline

# Save output
.\build\bin\google-opt.exe benchmarks\compute_bound\bench_matmul_1024.mlir --google-extreme-pipeline -o output\result.mlir
```

**Measure compilation time**:
```powershell
# Using Measure-Command
Measure-Command { .\build\bin\google-opt.exe benchmarks\compute_bound\bench_matmul_1024.mlir --google-extreme-pipeline | Out-Null }

# Multiple runs
1..10 | ForEach-Object { (Measure-Command { .\build\bin\google-opt.exe benchmarks\compute_bound\bench_matmul_1024.mlir --google-extreme-pipeline | Out-Null }).TotalMilliseconds }
```

**View generated IR**:
```powershell
# View output (will be long)
.\build\bin\google-opt.exe benchmarks\compute_bound\bench_matmul_1024.mlir --google-extreme-pipeline

# Save and view
.\build\bin\google-opt.exe benchmarks\compute_bound\bench_matmul_1024.mlir --google-extreme-pipeline -o output\matmul_1024_extreme.mlir
code output\matmul_1024_extreme.mlir
```

---

### Individual Benchmark Scripts

**Execution efficiency**:
```powershell
cd benchmarks\compilation
.\bench_execution_efficiency.ps1
.\bench_execution_efficiency.ps1 -Iterations 20
```

**Optimization gain**:
```powershell
cd benchmarks\compilation
.\bench_optimization_gain.ps1
.\bench_optimization_gain.ps1 -Iterations 15
```

**Build latency**:
```powershell
cd benchmarks\build_evaluation
.\bench_latency.ps1
.\bench_latency.ps1 -Iterations 10
```

---

## Interpreting Results

### Memory-Bound Benchmarks

**Good Performance Indicators**:
- Memory bandwidth >320 GB/s (89% of peak 360 GB/s)
- Compilation time <300ms
- Consistent performance across runs (low std dev)

**Poor Performance Indicators**:
- Memory bandwidth <200 GB/s (indicates inefficiency)
- High variance in timing (indicates instability)
- Compilation failures

**Common Issues**:
- **Low bandwidth on transpose**: Non-coalesced memory access
- **Slow broadcast**: Cache not being utilized
- **High compilation time**: Complex IR transformations

---

### Compute-Bound Benchmarks

**Good Performance Indicators**:
- GFLOPS within 10% of target
- Efficiency >80%
- Speedup >10x vs basic pipeline

**Poor Performance Indicators**:
- GFLOPS <50% of target
- No speedup vs basic pipeline
- Compilation failures

**Performance by Size**:
```
256×256:   3,000 GFLOPS (23% of peak) - Small overhead
512×512:   5,000 GFLOPS (38% of peak) - Good balance
1024×1024: 7,000 GFLOPS (54% of peak) - Key benchmark
2048×2048: 10,000 GFLOPS (77% of peak) - High performance
4096×4096: 12,000 GFLOPS (92% of peak) - Near peak
```

**GFLOPS Calculation Example**:
```
Matrix: 1024×1024 × 1024×1024
FLOPs = 2 × 1024 × 1024 × 1024 = 2,147,483,648
Time = 305.23ms = 0.30523 seconds
GFLOPS = 2,147,483,648 / (0.30523 × 1,000,000,000)
       = 7,036 GFLOPS
Efficiency = 7,036 / 7,000 = 100.5%
```

---

### Compilation Benchmarks

**Execution Efficiency Results**:
- **Excellent**: All tests >90% efficiency
- **Good**: Most tests >80% efficiency
- **Poor**: Tests <70% efficiency

**Optimization Gain Results**:
- **Excellent**: Speedup >15x
- **Good**: Speedup >10x
- **Poor**: Speedup <5x

**Expected Speedup by Pipeline**:
```
Basic:           1.0x (baseline)
Optimized:       2-3x
Extreme L1:      4-6x
Extreme L2:      7-10x
Extreme L3:      10-14x
Extreme L3 Full: 12-16x
Extreme Pipeline: 14-20x
```

---

### Build Evaluation Results

**Build Latency Targets**:
- Small (256×256): <200ms
- Medium (1024×1024): <500ms
- Large (4096×4096): <1000ms

**Interpretation**:
- **All within target**: Excellent compilation performance
- **1-2 over target**: Acceptable, may need optimization
- **Many over target**: Poor compilation performance

**Typical Latencies**:
```
256×256:   ~150ms (well within target)
1024×1024: ~400ms (within target)
4096×4096: ~850ms (within target)
```

---

## Troubleshooting

### Common Errors

#### Error: "google-opt.exe not found"
**Solution**:
```powershell
# Check if built
Test-Path .\build\bin\google-opt.exe

# If not, build the project
cmake --build build --target google-opt
```

#### Error: "could not find a nested named sequence"
**Cause**: Test file missing transform module  
**Solution**: Use files with embedded transforms (all compute_bound tests have them)

#### Error: "'affine.for' op operand cannot be used as a dimension id"
**Cause**: Matrix dimensions smaller than tile sizes  
**Solution**: Use larger matrices (≥1024) or reduce tile sizes

#### Error: "Out of memory"
**Cause**: Test too large for available memory  
**Solution**: Use smaller test or increase system memory

---

### Performance Issues

#### Low GFLOPS
**Possible Causes**:
- Not using extreme pipeline
- Matrix size too small
- System thermal throttling

**Solutions**:
```powershell
# Use extreme pipeline
.\build\bin\google-opt.exe test.mlir --google-extreme-pipeline

# Use larger matrices
# Use bench_matmul_2048.mlir or bench_matmul_4096.mlir

# Check system temperature
# Ensure adequate cooling
```

#### Low Memory Bandwidth
**Possible Causes**:
- Non-coalesced memory access
- Cache not being utilized
- Memory contention

**Solutions**:
- Check transpose implementation
- Verify sequential access patterns
- Close other applications

#### High Compilation Time
**Possible Causes**:
- Complex transformations
- Large matrix sizes
- Debug build

**Solutions**:
```powershell
# Use Release build
cmake -DCMAKE_BUILD_TYPE=Release -B build

# Use simpler pipeline for testing
.\build\bin\google-opt.exe test.mlir --google-basic-pipeline
```

---

### Verification

**Verify benchmark suite installation**:
```powershell
# Check all files exist
Get-ChildItem benchmarks -Recurse -File | Measure-Object
# Should show 17 files

# Check directory structure
Get-ChildItem benchmarks -Directory
# Should show: memory_bound, compute_bound, compilation, build_evaluation, results
```

**Quick smoke test**:
```powershell
# Test one file from each category
.\build\bin\google-opt.exe benchmarks\memory_bound\bench_elementwise.mlir --google-basic-pipeline
.\build\bin\google-opt.exe benchmarks\compute_bound\bench_matmul_256.mlir --google-extreme-pipeline
```

**Verify results directory**:
```powershell
# Results should be created after running benchmarks
Test-Path benchmarks\results
Get-ChildItem benchmarks\results
```

---

## Summary

### Benchmark Suite Contents

**Total Files**: 17
- 5 Memory-bound tests
- 7 Compute-bound tests
- 3 Compilation scripts
- 1 Build evaluation script
- 1 Master runner script

### Key Commands

```powershell
# Run everything
.\run_all_benchmarks.ps1

# Run specific category
.\run_all_benchmarks.ps1 -Category compute_bound

# Individual test
.\build\bin\google-opt.exe benchmarks\compute_bound\bench_matmul_1024.mlir --google-extreme-pipeline

# Individual script
.\benchmarks\compilation\bench_execution_efficiency.ps1
```

### Performance Targets

| Category | Metric | Target |
|----------|--------|--------|
| Memory-bound | Bandwidth | >320 GB/s |
| Compute-bound (MatMul 1024) | GFLOPS | 7,000 |
| Compute-bound (MatMul 4096) | GFLOPS | 12,000 |
| Optimization | Speedup | 10-20x |
| Compilation | Latency | <1000ms |

---

**For questions or issues, refer to the main README in `benchmarks/README.md`**
