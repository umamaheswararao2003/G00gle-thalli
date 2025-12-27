# Google MLIR Dialect Benchmark Suite

## Overview

Comprehensive benchmark suite for measuring performance of the Google MLIR dialect across multiple dimensions:
- **Memory-bound operations**: Limited by memory bandwidth
- **Compute-bound operations**: Limited by compute throughput
- **Compilation performance**: Build time and optimization effectiveness
- **Build evaluation**: End-to-end metrics

## Quick Start

```powershell
# Run all benchmarks
.\run_all_benchmarks.ps1

# Run specific category
.\run_all_benchmarks.ps1 -Category memory_bound
.\run_all_benchmarks.ps1 -Category compute_bound
.\run_all_benchmarks.ps1 -Category compilation
```

## Benchmark Categories

### 1. Memory-Bound Operations
**Location**: `memory_bound/`
**Characteristics**: Low arithmetic intensity (<10 FLOPS/byte)

Tests:
- Element-wise operations (add, mul, relu, sigmoid, tanh)
- Transpose
- Reshape
- Concat
- Broadcast

**Target**: >320 GB/s memory bandwidth utilization

### 2. Compute-Bound Operations
**Location**: `compute_bound/`
**Characteristics**: High arithmetic intensity (>50 FLOPS/byte)

Tests:
- MatMul (256x256, 512x512, 1024x1024, 2048x2048, 4096x4096)
- Softmax
- GELU

**Target**: 7,000-13,000 GFLOPS for matmul

### 3. Compilation Benchmarks
**Location**: `compilation/`

Metrics:
- Execution efficiency (GFLOPS)
- Throughput (ops/sec, GB/s)
- Memory efficiency (cache hit rates)
- Peak memory utilization
- Optimization gain (speedup ratios)
- Resource utilization (SM occupancy)
- Arithmetic intensity

### 4. Build Evaluation
**Location**: `build_evaluation/`

Metrics:
- Total latency (compilation time)
- Speedup (vs baseline)
- Peak memory (during build)

## Results

Results are saved to `results/` as CSV files:
- `memory_bound_results.csv`
- `compute_bound_results.csv`
- `compilation_results.csv`
- `build_evaluation_results.csv`

## Target Metrics (NVIDIA RTX 3060)

| Metric | Target | Notes |
|--------|--------|-------|
| MatMul 1024 GFLOPS | 7,000 | 54% of peak |
| MatMul 4096 GFLOPS | 12,000 | 92% of peak |
| Memory Bandwidth | >320 GB/s | 89% of peak |
| L3 Tiling Speedup | 10-20x | vs no tiling |
| Compilation Time | <1000ms | for typical workloads |
| Peak Memory | <12GB | Total VRAM |
| SM Occupancy | >75% | GPU utilization |

## Hardware Specifications

**NVIDIA GeForce RTX 3060**:
- CUDA Cores: 3,584
- Tensor Cores: 112 (3rd gen)
- Memory: 12GB GDDR6
- Memory Bandwidth: 360 GB/s
- L2 Cache: 3MB
- Shared Memory: 48KB per SM
- Compute Capability: 8.6
- Peak FP32: 13 TFLOPS

## Benchmark Methodology

1. **Warm-up**: Run 3 iterations to warm up caches
2. **Measurement**: Run 10 iterations and take median
3. **Metrics**: Calculate GFLOPS, bandwidth, speedup
4. **Validation**: Compare against reference implementation

## File Structure

```
benchmarks/
├── README.md                          # This file
├── run_all_benchmarks.ps1            # Master runner
├── memory_bound/                      # Memory-bound tests
│   ├── bench_elementwise.mlir
│   ├── bench_transpose.mlir
│   ├── bench_reshape.mlir
│   ├── bench_concat.mlir
│   └── bench_broadcast.mlir
├── compute_bound/                     # Compute-bound tests
│   ├── bench_matmul_256.mlir
│   ├── bench_matmul_512.mlir
│   ├── bench_matmul_1024.mlir
│   ├── bench_matmul_2048.mlir
│   ├── bench_matmul_4096.mlir
│   ├── bench_softmax.mlir
│   └── bench_gelu.mlir
├── compilation/                       # Compilation benchmarks
│   ├── bench_execution_efficiency.ps1
│   ├── bench_throughput.ps1
│   ├── bench_memory_efficiency.ps1
│   └── bench_optimization_gain.ps1
├── build_evaluation/                  # Build metrics
│   ├── bench_latency.ps1
│   ├── bench_speedup.ps1
│   └── bench_peak_memory.ps1
└── results/                           # Output results
    ├── memory_bound_results.csv
    ├── compute_bound_results.csv
    ├── compilation_results.csv
    └── build_evaluation_results.csv
```

## Contributing

When adding new benchmarks:
1. Follow the naming convention: `bench_<operation>.mlir`
2. Include embedded transform module for extreme pipeline
3. Add to appropriate category
4. Update this README
5. Run benchmarks and verify results
