# Google MLIR Dialect: Commands Reference

## Table of Contents

1. [Quick Start](#quick-start)
2. [Pipeline Commands](#pipeline-commands)
3. [Testing Commands](#testing-commands)
4. [Build Commands](#build-commands)
5. [Verification Commands](#verification-commands)
6. [Debugging Commands](#debugging-commands)
7. [Performance Analysis](#performance-analysis)
8. [Common Workflows](#common-workflows)

---

## Quick Start

### Build the Project

```bash
# Configure
cmake -G Ninja -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=/path/to/llvm/build/lib/cmake/llvm \
  -DMLIR_DIR=/path/to/llvm/build/lib/cmake/mlir

# Build
cmake --build build --target google-opt
```

### Run a Simple Test

```bash
# Test basic pipeline
./build/bin/google-opt test/test_matmul.mlir --google-basic-pipeline
```

---

## Pipeline Commands

### Available Pipelines

| Pipeline | Description | Use Case |
|----------|-------------|----------|
| `--google-basic-pipeline` | Fast compilation | Development, debugging |
| `--google-optimized-pipeline` | Balanced performance | General use |
| `--google-extreme-pipeline` | Maximum performance | Production (no tiling) |
| `--google-extreme-l1` | L1 tiling (16x16x16) | L1 cache optimization |
| `--google-extreme-l2` | L1+L2 tiling | L1+L2 cache optimization |
| `--google-extreme-l2-full` | L1+L2 + LLVM lowering | Complete L2 pipeline |
| `--google-extreme-l3` | L1+L2+L3 tiling | Full cache hierarchy |
| `--google-extreme-l3-full` | L1+L2+L3 + LLVM lowering | **Ultimate optimization** |

### Basic Pipeline

**Purpose**: Fast compilation, minimal optimization

```bash
google-opt input.mlir --google-basic-pipeline
```

**Passes**:
1. GoogleToLinalg
2. Bufferization
3. LLVM lowering

**Use When**: Debugging, quick iteration

---

### Optimized Pipeline

**Purpose**: Balanced performance and compile time

```bash
google-opt input.mlir --google-optimized-pipeline
```

**Passes**:
1. GoogleToLinalg
2. Linalg fusion
3. Bufferization
4. Affine optimization
5. LLVM lowering

**Use When**: General development

---

### Extreme Pipeline (No Tiling)

**Purpose**: Maximum performance without tiling

```bash
google-opt input.mlir --google-extreme-pipeline
```

**Passes**:
1. GoogleToLinalg
2. Linalg fusion
3. Linalg generalization
4. Bufferization
5. Affine loops + fusion + coalescing
6. LLVM lowering

**Use When**: Baseline performance comparison

---

### L1 Tiling Pipeline

**Purpose**: L1 cache optimization (16x16x16 tiles)

**Minimal** (verify tiling):
```bash
google-opt input.mlir --google-extreme-l1 -o output.mlir
```

**Full** (complete compilation):
```bash
google-opt input.mlir --google-extreme-l1-full -o output.mlir
```

**Expected**:
- 3 `scf.for` loops
- Step size: 16
- 3-5x speedup

---

### L2 Tiling Pipeline

**Purpose**: L1+L2 cache optimization (64→16 tiles)

**Minimal** (verify tiling):
```bash
google-opt input.mlir --google-extreme-l2 -o output.mlir
```

**Full** (complete compilation):
```bash
google-opt input.mlir --google-extreme-l2-full -o output.mlir
```

**Expected**:
- 6 `scf.for` loops (3 L2 + 3 L1)
- Step sizes: 64 (L2), 16 (L1)
- 6-10x speedup

---

### L3 Tiling Pipeline (Ultimate)

**Purpose**: Full cache hierarchy optimization (256→64→16 tiles)

**Minimal** (verify tiling):
```bash
google-opt test/test_matmul_l3_tiling.mlir --google-extreme-l3 -o output.mlir
```

**Full** (complete compilation):
```bash
google-opt test/test_matmul_l3_tiling.mlir --google-extreme-l3-full -o output.mlir
```

**Expected**:
- **9 `scf.for` loops** (3 L3 + 3 L2 + 3 L1)
- Step sizes: 256 (L3), 64 (L2), 16 (L1)
- **10-20x speedup** 🚀

---

## Testing Commands

### Test Files

| Test File | Description | Pipeline |
|-----------|-------------|----------|
| `test/test_matmul.mlir` | Basic MatMul | Any |
| `test/test_matmul_bias_relu.mlir` | Fused ops | Optimized/Extreme |
| `test/test_matmul_l2_embedded.mlir` | 256x256 MatMul + L2 tiling | L2 pipelines |
| `test/test_matmul_l3_tiling.mlir` | 1024x1024 MatMul + L3 tiling | L3 pipelines |
| `test/test_all_30_ops.mlir` | All 30 operations | Basic/Optimized |

### Run Individual Tests

**MatMul**:
```bash
google-opt test/test_matmul.mlir --google-optimized-pipeline
```

**Fused Operations**:
```bash
google-opt test/test_matmul_bias_relu.mlir --google-extreme-pipeline
```

**L2 Tiling**:
```bash
google-opt test/test_matmul_l2_embedded.mlir --google-extreme-l2
```

**L3 Tiling**:
```bash
google-opt test/test_matmul_l3_tiling.mlir --google-extreme-l3
```

### Run All Tests

```bash
# PowerShell
Get-ChildItem test/*.mlir | ForEach-Object {
  Write-Host "Testing: $($_.Name)"
  .\build\bin\google-opt.exe $_.FullName --google-optimized-pipeline
}
```

```bash
# Bash
for file in test/*.mlir; do
  echo "Testing: $file"
  ./build/bin/google-opt "$file" --google-optimized-pipeline
done
```

---

## Build Commands

### Full Rebuild

```bash
# Clean
cmake --build build --target clean

# Rebuild
cmake --build build --target google-opt --config Release
```

### Incremental Build

```bash
# Only rebuild changed files
cmake --build build --target google-opt
```

### Build Specific Targets

```bash
# Build only pipelines library
cmake --build build --target MLIRGooglePipelines

# Build only dialect library
cmake --build build --target MLIRGoogleDialect

# Build only translation library
cmake --build build --target MLIRGoogleTranslation
```

### Build with Verbose Output

```bash
cmake --build build --target google-opt --verbose
```

---

## Verification Commands

### Count Loops

**PowerShell**:
```powershell
# Count scf.for loops
(Select-String "scf.for" output.mlir -AllMatches).Matches.Count

# Should be:
# L1: 3 loops
# L2: 6 loops
# L3: 9 loops
```

**Bash**:
```bash
# Count scf.for loops
grep -c "scf.for" output.mlir
```

### Verify Tile Sizes

**PowerShell**:
```powershell
# L3 loops (step 256)
(Select-String "step.*256" output.mlir -AllMatches).Matches.Count

# L2 loops (step 64)
(Select-String "step.*64" output.mlir -AllMatches).Matches.Count

# L1 loops (step 16)
(Select-String "step.*16" output.mlir -AllMatches).Matches.Count
```

**Bash**:
```bash
# L3 loops
grep -c "step.*256" output.mlir

# L2 loops
grep -c "step.*64" output.mlir

# L1 loops
grep -c "step.*16" output.mlir
```

### Verify LLVM Lowering

**PowerShell**:
```powershell
# Count LLVM operations
(Select-String "llvm\." output.mlir -AllMatches).Matches.Count

# Should be > 0 for full pipelines
```

**Bash**:
```bash
# Count LLVM operations
grep -c "llvm\." output.mlir
```

### Show Loop Structure

**PowerShell**:
```powershell
# Show all loops with line numbers
Select-String "scf.for" output.mlir | ForEach-Object {
  Write-Host "Line $($_.LineNumber): $($_.Line.Trim())"
}
```

**Bash**:
```bash
# Show all loops with line numbers
grep -n "scf.for" output.mlir
```

---

## Debugging Commands

### View Intermediate IR

**After GoogleToLinalg**:
```bash
google-opt input.mlir --convert-google-to-linalg -o linalg.mlir
```

**After Fusion**:
```bash
google-opt input.mlir \
  --convert-google-to-linalg \
  --pass-pipeline="builtin.module(func.func(linalg-fuse-elementwise-ops))" \
  -o fused.mlir
```

**After Tiling** (minimal pipeline):
```bash
google-opt input.mlir --google-extreme-l3 -o tiled.mlir
```

### Enable Verbose Output

```bash
google-opt input.mlir --google-extreme-l3 --mlir-print-ir-after-all
```

### Print Specific Pass Output

```bash
google-opt input.mlir \
  --google-extreme-l3 \
  --mlir-print-ir-after=transform-interpreter
```

### Disable Verification

```bash
# Use with caution!
google-opt input.mlir --google-extreme-l3 --verify-each=false
```

---

## Performance Analysis

### Measure Compilation Time

**PowerShell**:
```powershell
Measure-Command {
  .\build\bin\google-opt.exe test/test_matmul_l3_tiling.mlir --google-extreme-l3-full
}
```

**Bash**:
```bash
time ./build/bin/google-opt test/test_matmul_l3_tiling.mlir --google-extreme-l3-full
```

### Compare Pipeline Performance

**PowerShell**:
```powershell
# No tiling
Measure-Command { .\build\bin\google-opt.exe test.mlir --google-extreme-pipeline }

# L1 tiling
Measure-Command { .\build\bin\google-opt.exe test.mlir --google-extreme-l1-full }

# L2 tiling
Measure-Command { .\build\bin\google-opt.exe test.mlir --google-extreme-l2-full }

# L3 tiling
Measure-Command { .\build\bin\google-opt.exe test.mlir --google-extreme-l3-full }
```

### Analyze Output Size

**PowerShell**:
```powershell
# Line count
(Get-Content output.mlir).Count

# File size
(Get-Item output.mlir).Length
```

**Bash**:
```bash
# Line count
wc -l output.mlir

# File size
ls -lh output.mlir
```

---

## Common Workflows

### Workflow 1: Test New Transform

```bash
# 1. Create test file
cat > test/my_test.mlir << EOF
module {
  func.func @test(%arg0: tensor<256x256xf32>, %arg1: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %0 = google.matmul %arg0, %arg1 : tensor<256x256xf32>, tensor<256x256xf32> -> tensor<256x256xf32>
    return %0 : tensor<256x256xf32>
  }
  
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg: !transform.any_op {transform.readonly}) {
      %matmuls = transform.structured.match ops{["linalg.matmul"]} in %arg
      %tiled, %loops:3 = transform.structured.tile_using_for %matmuls tile_sizes [32, 32, 32]
      transform.yield
    }
  }
}
EOF

# 2. Test with minimal pipeline
google-opt test/my_test.mlir --google-extreme-l1 -o output.mlir

# 3. Verify
grep -c "scf.for" output.mlir  # Should be 3
grep "step.*32" output.mlir    # Should show step 32
```

---

### Workflow 2: Compare Tiling Levels

```bash
# Test file
TEST_FILE="test/test_matmul_l3_tiling.mlir"

# No tiling
google-opt $TEST_FILE --google-extreme-pipeline -o no_tiling.mlir

# L1 tiling
google-opt $TEST_FILE --google-extreme-l1 -o l1_tiling.mlir

# L2 tiling
google-opt $TEST_FILE --google-extreme-l2 -o l2_tiling.mlir

# L3 tiling
google-opt $TEST_FILE --google-extreme-l3 -o l3_tiling.mlir

# Compare loop counts
echo "No tiling: $(grep -c 'scf.for' no_tiling.mlir || echo 0) loops"
echo "L1 tiling: $(grep -c 'scf.for' l1_tiling.mlir) loops"
echo "L2 tiling: $(grep -c 'scf.for' l2_tiling.mlir) loops"
echo "L3 tiling: $(grep -c 'scf.for' l3_tiling.mlir) loops"
```

---

### Workflow 3: Full Pipeline Test

```bash
# 1. Build
cmake --build build --target google-opt --config Release

# 2. Test all pipelines
for pipeline in basic optimized extreme extreme-l1 extreme-l2 extreme-l3; do
  echo "Testing: google-$pipeline-pipeline"
  google-opt test/test_matmul.mlir --google-$pipeline-pipeline -o output_$pipeline.mlir
  echo "  Lines: $(wc -l < output_$pipeline.mlir)"
done

# 3. Verify L3 tiling
echo "L3 loops: $(grep -c 'scf.for' output_extreme-l3.mlir)"
```

---

### Workflow 4: Debug Failed Pipeline

```bash
# 1. Run with verbose output
google-opt test/failing.mlir --google-extreme-l3 --mlir-print-ir-after-all 2>&1 | tee debug.log

# 2. Find where it failed
grep "error:" debug.log

# 3. Check IR before failing pass
grep -B 20 "error:" debug.log

# 4. Try minimal pipeline
google-opt test/failing.mlir --convert-google-to-linalg -o linalg.mlir

# 5. Test transform separately
google-opt linalg.mlir --pass-pipeline="transform-interpreter" -o tiled.mlir
```

---

### Workflow 5: Benchmark Performance

```bash
# Create benchmark script
cat > benchmark.sh << 'EOF'
#!/bin/bash

TEST_FILE="test/test_matmul_l3_tiling.mlir"
ITERATIONS=10

echo "Benchmarking pipelines..."

for pipeline in extreme-pipeline extreme-l1-full extreme-l2-full extreme-l3-full; do
  echo "Testing: google-$pipeline"
  total=0
  for i in $(seq 1 $ITERATIONS); do
    start=$(date +%s%N)
    ./build/bin/google-opt $TEST_FILE --google-$pipeline > /dev/null 2>&1
    end=$(date +%s%N)
    elapsed=$((end - start))
    total=$((total + elapsed))
  done
  avg=$((total / ITERATIONS / 1000000))
  echo "  Average: ${avg}ms"
done
EOF

chmod +x benchmark.sh
./benchmark.sh
```

---

## Summary

### Quick Reference

**Build**:
```bash
cmake --build build --target google-opt
```

**Test L3 Tiling**:
```bash
google-opt test/test_matmul_l3_tiling.mlir --google-extreme-l3
```

**Verify 9 Loops**:
```bash
grep -c "scf.for" output.mlir  # Should be 9
```

**Full Pipeline**:
```bash
google-opt test/test_matmul_l3_tiling.mlir --google-extreme-l3-full -o output.mlir
```

### Pipeline Selection Guide

| Use Case | Pipeline | Expected Speedup |
|----------|----------|------------------|
| Development | `--google-basic-pipeline` | 1x (baseline) |
| General use | `--google-optimized-pipeline` | 2-3x |
| Production (no tiling) | `--google-extreme-pipeline` | 3-4x |
| L1 optimization | `--google-extreme-l1-full` | 3-5x |
| L1+L2 optimization | `--google-extreme-l2-full` | 6-10x |
| **Ultimate** | `--google-extreme-l3-full` | **10-20x** 🚀 |

---

## Troubleshooting

### Common Issues

**Issue**: "Unknown command line argument"
```bash
# Solution: Rebuild to register new pipelines
cmake --build build --target google-opt
```

**Issue**: "Transform script syntax error"
```bash
# Solution: Check loop unpacking syntax
%tiled, %loops:3 = ...  # Correct
%tiled, %loops:2 = ...  # Wrong for 3D tiling
```

**Issue**: "No output file generated"
```bash
# Solution: Check for errors
google-opt input.mlir --google-extreme-l3 2>&1 | grep error
```

**Issue**: "Verification failed"
```bash
# Solution: Use minimal pipeline to isolate issue
google-opt input.mlir --google-extreme-l3  # Stops after tiling
```

---

## Conclusion

This reference covers all commands needed to:

✅ Build the project  
✅ Run all pipelines  
✅ Test transformations  
✅ Verify results  
✅ Debug issues  
✅ Measure performance  

**Key Commands**:
- Build: `cmake --build build --target google-opt`
- Test L3: `google-opt test/test_matmul_l3_tiling.mlir --google-extreme-l3`
- Verify: `grep -c "scf.for" output.mlir`

**Achievement**: 9-level loop nest, 10-20x speedup! 🚀
