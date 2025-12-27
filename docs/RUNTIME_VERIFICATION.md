# Runtime Engine Verification Guide

## Quick Verification (5 minutes)

This verifies the runtime engine works without requiring MLIR kernel compilation.

### Step 1: Build the Runtime Library

```powershell
cd c:\Users\Asus\Desktop\google

# Configure CMake (if not already done)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build GoogleRuntime library
cmake --build build --target GoogleRuntime
```

**Success Criteria:**
- ✅ No compilation errors
- ✅ `GoogleRuntime.lib` created in `build\lib\`

### Step 2: Build Quick Tests

```powershell
# Build test executable
cmake --build build --target test_quick_phase1
```

**Success Criteria:**
- ✅ No compilation errors
- ✅ `test_quick_phase1.exe` created in `build\bin\`

### Step 3: Run Quick Tests

```powershell
# Run the tests
.\build\bin\test_quick_phase1.exe
```

**Expected Output:**
```
=== Phase 1 Quick Tests ===

Testing Runtime singleton... ✓ PASS
Testing Kernel registration... ✓ PASS
Testing Kernel execution... ✓ PASS
Testing Memory alignment... ✓ PASS
Testing Tensor creation... ✓ PASS
Testing Tensor shape and strides... ✓ PASS
Testing Tensor fill... ✓ PASS
Testing Tensor scalar division... ✓ PASS
Testing Tensor scalar multiplication... ✓ PASS
Testing Tensor element-wise addition... ✓ PASS
Testing Tensor element-wise subtraction... ✓ PASS
Testing Tensor element-wise multiplication... ✓ PASS
Testing Tensor ReLU activation... ✓ PASS
Testing Tensor sigmoid activation... ✓ PASS
Testing Tensor tanh activation... ✓ PASS
Testing Tensor random initialization... ✓ PASS

=== Test Summary ===
Passed: 16
Failed: 0
Total:  16

✓ All tests passed!
```

**Success Criteria:**
- ✅ All 16 tests pass
- ✅ Exit code 0

## ✅ If All Tests Pass → Runtime Engine is Working!

If you see "All tests passed!", your runtime engine is **fully functional** and ready for use.

## What Each Test Validates

| Test | What It Proves |
|------|----------------|
| Runtime singleton | Runtime instance works correctly |
| Kernel registration | Can register compiled functions |
| Kernel execution | Can execute registered kernels |
| Memory alignment | Memory is 64-byte aligned (SIMD ready) |
| Tensor creation | Tensor objects can be created |
| Shape and strides | Shape/stride tracking works |
| Fill | Can initialize tensor data |
| Scalar division | Eager scalar ops work |
| Scalar multiplication | Eager scalar ops work |
| Element-wise add | Eager element-wise ops work |
| Element-wise subtract | Eager element-wise ops work |
| Element-wise multiply | Eager element-wise ops work |
| ReLU activation | Activation functions work |
| Sigmoid activation | Activation functions work |
| Tanh activation | Activation functions work |
| Random initialization | Random number generation works |

## Full Verification (With MLIR Kernel)

For complete verification including MLIR integration:

### Step 4: Compile MLIR Kernel (Optional)

```powershell
# Compile matmul kernel
.\build\bin\google-opt.exe test\test_matmul_l3_tiling.mlir `
  --google-extreme-l3-full | `
  .\build\bin\mlir-translate.exe --mlir-to-llvmir | `
  llc -filetype=obj -o test\matmul_l3.o
```

### Step 5: Build Integration Test

```powershell
cmake --build build --target test_runtime_phase1
```

### Step 6: Run Integration Test

```powershell
.\build\bin\test_runtime_phase1.exe
```

**Expected Output:**
```
=== Google Runtime Test (Phase 1) ===

Registered kernels: 1
Has matmul_l3: yes

Configuration:
  Matrix size: 1024x1024
  Iterations: 5

Testing correctness...
  ✓ Correctness: PASS

Testing eager operations...
  ✓ Eager operations: PASS

Benchmarking performance...
  Average time: XX.XX ms
  Performance: XX.XX GFLOPS

=== Phase 1 Runtime Test Complete ===
```

## Troubleshooting

### Build Fails

**Issue:** CMake configuration fails
**Solution:** 
```powershell
# Remove build directory and reconfigure
Remove-Item -Recurse -Force build
cmake -B build -DCMAKE_BUILD_TYPE=Release
```

### Tests Fail

**Issue:** Some tests fail
**Solution:** Check error messages - they indicate which component isn't working

### Can't Find Executable

**Issue:** `test_quick_phase1.exe` not found
**Solution:** Check `build\bin\` directory or build with verbose output:
```powershell
cmake --build build --target test_quick_phase1 --verbose
```

## Summary

**Minimum Verification (Runtime is Working):**
1. ✅ Build succeeds
2. ✅ All 16 quick tests pass

**Full Verification (Runtime + MLIR Integration):**
1. ✅ Build succeeds
2. ✅ All 16 quick tests pass
3. ✅ Integration test passes
4. ✅ Performance is acceptable (>10 GFLOPS for matmul)

If quick tests pass, **your runtime engine is working!** 🎉
