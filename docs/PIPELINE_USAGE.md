# Google Dialect Optimization Pipelines

## Usage Guide

This document describes how to use optimization pipelines with the Google dialect.

---

## Quick Start

### Basic Pipeline (Fast Compilation)
```bash
google-opt input.mlir \
  --convert-google-to-linalg \
  --linalg-fuse-elementwise-ops \
  --one-shot-bufferize \
  --convert-linalg-to-loops \
  --convert-scf-to-cf \
  --convert-vector-to-llvm \
  --convert-func-to-llvm \
  --convert-arith-to-llvm \
  --finalize-memref-to-llvm \
  --reconcile-unrealized-casts \
  -o output.mlir
```

### Optimized Pipeline (Recommended)
```bash
google-opt input.mlir \
  --convert-google-to-linalg \
  --linalg-fuse-elementwise-ops \
  --linalg-generalize-named-ops \
  --one-shot-bufferize \
  --convert-linalg-to-affine-loops \
  --affine-loop-fusion \
  --affine-loop-coalescing \
  --lower-affine \
  --convert-scf-to-cf \
  --convert-vector-to-llvm \
  --convert-func-to-llvm \
  --convert-arith-to-llvm \
  --finalize-memref-to-llvm \
  --reconcile-unrealized-casts \
  -o output.mlir
```

### Extreme Pipeline (Maximum Performance)
```bash
google-opt input.mlir \
  --convert-google-to-linalg \
  --linalg-fuse-elementwise-ops \
  --linalg-generalize-named-ops \
  --one-shot-bufferize \
  --convert-linalg-to-affine-loops \
  --affine-loop-fusion \
  --affine-loop-coalescing \
  --lower-affine \
  --convert-scf-to-cf \
  --convert-vector-to-llvm \
  --convert-func-to-llvm \
  --convert-arith-to-llvm \
  --finalize-memref-to-llvm \
  --reconcile-unrealized-casts \
  -o output.mlir
```

---

## Pipeline Scripts

### Windows PowerShell Scripts

Create these scripts in `scripts/` directory:

**basic-pipeline.ps1**:
```powershell
param(
    [Parameter(Mandatory=$true)]
    [string]$InputFile,
    [string]$OutputFile = "output.mlir"
)

& google-opt $InputFile `
  --convert-google-to-linalg `
  --linalg-fuse-elementwise-ops `
  --one-shot-bufferize `
  --convert-linalg-to-loops `
  --convert-scf-to-cf `
  --convert-vector-to-llvm `
  --convert-func-to-llvm `
  --convert-arith-to-llvm `
  --finalize-memref-to-llvm `
  --reconcile-unrealized-casts `
  -o $OutputFile
```

**optimized-pipeline.ps1**:
```powershell
param(
    [Parameter(Mandatory=$true)]
    [string]$InputFile,
    [string]$OutputFile = "output.mlir"
)

& google-opt $InputFile `
  --convert-google-to-linalg `
  --linalg-fuse-elementwise-ops `
  --linalg-generalize-named-ops `
  --one-shot-bufferize `
  --convert-linalg-to-affine-loops `
  --affine-loop-fusion `
  --affine-loop-coalescing `
  --lower-affine `
  --convert-scf-to-cf `
  --convert-vector-to-llvm `
  --convert-func-to-llvm `
  --convert-arith-to-llvm `
  --finalize-memref-to-llvm `
  --reconcile-unrealized-casts `
  -o $OutputFile
```

---

## Example Usage

### Test MatMul + Bias + ReLU
```bash
# Using basic pipeline
.\scripts\basic-pipeline.ps1 -InputFile test\test_matmul_bias_relu.mlir -OutputFile output_basic.mlir

# Using optimized pipeline
.\scripts\optimized-pipeline.ps1 -InputFile test\test_matmul_bias_relu.mlir -OutputFile output_opt.mlir
```

### Test All 30 Operations
```bash
.\scripts\optimized-pipeline.ps1 -InputFile test\test_all_30_ops.mlir -OutputFile output_all.mlir
```

---

## Performance Comparison

Run benchmarks with different pipelines:

```powershell
# Benchmark script
$testFile = "test\test_matmul_bias_relu.mlir"

Write-Host "Basic Pipeline..."
Measure-Command { .\scripts\basic-pipeline.ps1 -InputFile $testFile }

Write-Host "Optimized Pipeline..."
Measure-Command { .\scripts\optimized-pipeline.ps1 -InputFile $testFile }
```

---

## Next Steps

1. **Test pipelines** with existing test files
2. **Measure performance** improvements
3. **Add more optimizations** as needed (tiling, vectorization)
4. **Create benchmarks** for different workloads
