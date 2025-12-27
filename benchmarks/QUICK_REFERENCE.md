# Benchmark Quick Reference Card

## 🚀 Quick Start
```powershell
cd c:\Users\Asus\Desktop\google\benchmarks
.\run_all_benchmarks.ps1
```

## 📁 File Structure
```
benchmarks/
├── README.md                    # Overview
├── BENCHMARK_GUIDE.md          # Detailed documentation (THIS IS COMPREHENSIVE!)
├── run_all_benchmarks.ps1      # Master runner
├── memory_bound/               # 5 tests (bandwidth-limited)
├── compute_bound/              # 7 tests (compute-limited)
├── compilation/                # 2 scripts (GFLOPS, speedup)
├── build_evaluation/           # 1 script (latency)
└── results/                    # CSV outputs
```

## 🎯 Benchmark Categories

### Memory-Bound (5 tests)
**Target**: >320 GB/s bandwidth
```powershell
.\run_all_benchmarks.ps1 -Category memory_bound
```
- `bench_elementwise.mlir` - Add, Mul, ReLU, Sigmoid, Tanh
- `bench_transpose.mlir` - Matrix transpose
- `bench_reshape.mlir` - Tensor reshape
- `bench_concat.mlir` - Concatenation
- `bench_broadcast.mlir` - Broadcasting

### Compute-Bound (7 tests)
**Target**: 7,000-13,000 GFLOPS
```powershell
.\run_all_benchmarks.ps1 -Category compute_bound
```
- `bench_matmul_256.mlir` - 3,000 GFLOPS target
- `bench_matmul_512.mlir` - 5,000 GFLOPS target
- `bench_matmul_1024.mlir` - 7,000 GFLOPS target ⭐
- `bench_matmul_2048.mlir` - 10,000 GFLOPS target
- `bench_matmul_4096.mlir` - 12,000 GFLOPS target
- `bench_softmax.mlir` - 4,000 GFLOPS target
- `bench_gelu.mlir` - 3,500 GFLOPS target

## 💻 Common Commands

### Run All Benchmarks
```powershell
.\run_all_benchmarks.ps1
```

### Run Specific Category
```powershell
.\run_all_benchmarks.ps1 -Category memory_bound
.\run_all_benchmarks.ps1 -Category compute_bound
```

### Use Different Pipeline
```powershell
.\run_all_benchmarks.ps1 -Pipeline google-basic-pipeline
.\run_all_benchmarks.ps1 -Pipeline google-extreme-pipeline
```

### Test Single File
```powershell
.\build\bin\google-opt.exe benchmarks\compute_bound\bench_matmul_1024.mlir --google-extreme-pipeline
```

### Measure Time
```powershell
Measure-Command { .\build\bin\google-opt.exe benchmarks\compute_bound\bench_matmul_1024.mlir --google-extreme-pipeline | Out-Null }
```

## 📊 Compilation Benchmarks

### Execution Efficiency (GFLOPS)
```powershell
cd benchmarks\compilation
.\bench_execution_efficiency.ps1
```
Measures: GFLOPS for all matmul sizes

### Optimization Gain (Speedup)
```powershell
cd benchmarks\compilation
.\bench_optimization_gain.ps1
```
Measures: Speedup across 7 pipelines
Target: 10-20x speedup

### Build Latency
```powershell
cd benchmarks\build_evaluation
.\bench_latency.ps1
```
Measures: Compilation time
Target: <1000ms

## 🎓 Understanding Results

### Memory-Bound
✅ Good: >320 GB/s bandwidth
⚠️ Poor: <200 GB/s bandwidth

### Compute-Bound (MatMul)
✅ Good: >90% of target GFLOPS
⚠️ Poor: <70% of target GFLOPS

### Optimization Gain
✅ Good: >10x speedup
⚠️ Poor: <5x speedup

## 📈 Expected Performance (RTX 3060)

| Test | Target | % of Peak |
|------|--------|-----------|
| MatMul 256 | 3,000 GFLOPS | 23% |
| MatMul 512 | 5,000 GFLOPS | 38% |
| MatMul 1024 | 7,000 GFLOPS | 54% |
| MatMul 2048 | 10,000 GFLOPS | 77% |
| MatMul 4096 | 12,000 GFLOPS | 92% |
| Memory Ops | 320 GB/s | 89% |

## 🔧 Troubleshooting

### google-opt.exe not found
```powershell
cmake --build build --target google-opt
```

### Transform module error
Use files with embedded transforms (all compute_bound tests have them)

### Low performance
- Use `--google-extreme-pipeline`
- Use larger matrices (≥1024)
- Check system cooling

## 📝 Output Files

Results saved to `benchmarks/results/`:
- `memory_bound_results_TIMESTAMP.csv`
- `compute_bound_results_TIMESTAMP.csv`
- `execution_efficiency_TIMESTAMP.csv`
- `optimization_gain_TIMESTAMP.csv`
- `build_latency_TIMESTAMP.csv`

## 📚 Full Documentation

**See `BENCHMARK_GUIDE.md` for complete details including**:
- Detailed explanation of each test
- What each benchmark measures
- Why operations are memory/compute-bound
- GFLOPS calculation formulas
- Arithmetic intensity explanations
- Troubleshooting guide
- Performance interpretation

## 🎯 Key Benchmarks

**Most Important Tests**:
1. `bench_matmul_1024.mlir` - Key performance indicator
2. `bench_optimization_gain.ps1` - Validates 10-20x speedup
3. `bench_execution_efficiency.ps1` - Validates GFLOPS targets

**Quick Validation**:
```powershell
# Test the key benchmark
.\build\bin\google-opt.exe benchmarks\compute_bound\bench_matmul_1024.mlir --google-extreme-pipeline

# Should complete in ~300-400ms
# Should achieve ~7,000 GFLOPS
```

---

**For detailed information, see `BENCHMARK_GUIDE.md` (comprehensive 50+ page guide)**
