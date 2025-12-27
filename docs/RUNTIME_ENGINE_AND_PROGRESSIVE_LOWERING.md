# Runtime Engine and Progressive Lowering - Complete Guide

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [What is the Runtime Engine?](#what-is-the-runtime-engine)
3. [Purpose and Responsibilities](#purpose-and-responsibilities)
4. [Progressive Lowering Overview](#progressive-lowering-overview)
5. [Complete Example: Top to Bottom](#complete-example-top-to-bottom)
6. [Detailed Stage-by-Stage Breakdown](#detailed-stage-by-stage-breakdown)
7. [Memory Management](#memory-management)
8. [Optimization Opportunities](#optimization-opportunities)
9. [Performance Considerations](#performance-considerations)

---

## Executive Summary

The **runtime engine** in MLIR is the execution infrastructure that takes high-level operations and progressively lowers them through multiple intermediate representations (IRs) until they become executable machine code. This document explains the complete journey from high-level Google dialect operations to LLVM IR that can be executed on CPU or GPU.

**Key Concept**: MLIR doesn't have a single "runtime engine" component. Instead, it uses **progressive lowering** through multiple dialects, where each dialect provides a different level of abstraction. The "runtime" is the final executable code generated at the end of this lowering process.

---

## What is the Runtime Engine?

### Traditional Compiler View
```
Source Code → Compiler → Machine Code → Runtime Engine → Execution
```

### MLIR View (Progressive Lowering)
```
High-Level Dialect (Google)
    ↓ Lowering Pass
Mid-Level Dialect (Linalg)
    ↓ Lowering Pass
Loop Dialect (SCF/Affine)
    ↓ Lowering Pass
Low-Level Dialect (LLVM)
    ↓ Code Generation
Machine Code
    ↓
Execution (CPU/GPU)
```

### What "Runtime" Means in MLIR Context

The "runtime" refers to:
1. **Lowering Infrastructure**: Passes that transform IR from one dialect to another
2. **Memory Management**: Buffer allocation, deallocation, and access patterns
3. **Execution Model**: How operations map to hardware (CPU threads, GPU kernels, etc.)
4. **Support Libraries**: Math libraries, memory allocators, synchronization primitives

**Note**: MLIR itself doesn't execute code. It generates LLVM IR which is then compiled to machine code by LLVM's code generator.

---

## Purpose and Responsibilities

### 1. Progressive Abstraction Lowering

**Purpose**: Transform high-level operations into increasingly concrete representations

**Responsibilities**:
- Preserve semantics across transformations
- Enable optimizations at each level
- Maintain correctness guarantees

### 2. Optimization Opportunities

**Purpose**: Apply transformations that improve performance

**Responsibilities**:
- Loop tiling for cache locality
- Operation fusion to reduce memory traffic
- Parallelization for multi-core/GPU execution
- Vectorization for SIMD instructions

### 3. Memory Management

**Purpose**: Convert abstract tensors to concrete memory buffers

**Responsibilities**:
- Allocate memory buffers
- Manage buffer lifetimes
- Optimize memory access patterns
- Handle memory hierarchies (registers, cache, RAM, VRAM)

### 4. Hardware Mapping

**Purpose**: Map abstract operations to hardware capabilities

**Responsibilities**:
- CPU: Thread parallelism, SIMD vectorization
- GPU: Grid/block/thread hierarchy, shared memory
- Specialized hardware: Tensor cores, vector units

---

## Progressive Lowering Overview

### The 7-Stage Lowering Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Google Dialect (High-Level)                       │
│ - Domain-specific operations (matmul, relu, softmax)       │
│ - Tensor types (abstract, no memory layout)                │
│ - No execution semantics                                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
                GoogleToLinalg Pass
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Linalg Dialect (Structured Operations)            │
│ - Generic structured operations                            │
│ - Iteration spaces and indexing maps                       │
│ - Enables tiling and fusion                                │
└─────────────────────────────────────────────────────────────┘
                          ↓
                Transform Dialect (Tiling)
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Tiled Linalg (Cache-Optimized)                    │
│ - Multi-level loop nests (L3, L2, L1 tiling)               │
│ - Optimized for cache hierarchy                            │
│ - Ready for parallelization                                │
└─────────────────────────────────────────────────────────────┘
                          ↓
                Bufferization Pass
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 4: MemRef Dialect (Explicit Memory)                  │
│ - Concrete memory buffers (memref types)                   │
│ - Explicit loads and stores                                │
│ - Memory layout information                                │
└─────────────────────────────────────────────────────────────┘
                          ↓
                Linalg to Affine/SCF
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 5: Affine/SCF Dialect (Loops)                        │
│ - Explicit loop constructs (for, while, if)                │
│ - Affine analysis for optimization                         │
│ - Loop fusion, coalescing                                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
                Lower to ControlFlow
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 6: ControlFlow Dialect (Branches)                    │
│ - Basic blocks and branches                                │
│ - Conditional jumps                                        │
│ - Ready for LLVM lowering                                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
                Convert to LLVM
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 7: LLVM Dialect (Machine-Level)                      │
│ - LLVM IR operations                                       │
│ - Register allocation                                      │
│ - Machine code generation                                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
                LLVM Code Generator
                          ↓
                    Machine Code
```

---

## Complete Example: Top to Bottom

### Input: Simple Matrix Multiplication

Let's trace a simple 4×4 matrix multiplication through all lowering stages.

---

### Stage 1: Google Dialect (High-Level)

**Purpose**: Express intent in domain-specific terms

```mlir
module {
  func.func @matmul_example(%A: tensor<4x4xf32>, 
                            %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // High-level matmul operation
    %C = google.matmul %A, %B : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
    return %C : tensor<4x4xf32>
  }
}
```

**Characteristics**:
- ✅ Single operation: `google.matmul`
- ✅ Tensor types (abstract, no memory layout)
- ✅ No loops, no memory operations
- ✅ Declarative: "what" not "how"

**Command to generate**:
```bash
# This is your input file
cat > input.mlir << 'EOF'
module {
  func.func @matmul_example(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %C = google.matmul %A, %B : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
    return %C : tensor<4x4xf32>
  }
}
EOF
```

---

### Stage 2: Linalg Dialect (Structured Operations)

**Purpose**: Express computation as structured iteration

**Lowering Pass**: `GoogleToLinalg`

```mlir
module {
  func.func @matmul_example(%A: tensor<4x4xf32>, 
                            %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // Initialize output tensor with zeros
    %c0 = arith.constant 0.000000e+00 : f32
    %init = tensor.empty() : tensor<4x4xf32>
    %C_init = linalg.fill ins(%c0 : f32) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    
    // Linalg matmul: structured operation with iteration semantics
    %C = linalg.matmul 
      ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>) 
      outs(%C_init : tensor<4x4xf32>) -> tensor<4x4xf32>
    
    return %C : tensor<4x4xf32>
  }
}
```

**Characteristics**:
- ✅ `linalg.matmul`: Structured operation with implicit loops
- ✅ Iteration space: `(i, j, k)` where `i, j, k ∈ [0, 4)`
- ✅ Indexing maps: `A[i,k], B[k,j], C[i,j]`
- ✅ Reduction: `C[i,j] += A[i,k] * B[k,j]`
- ✅ Still uses tensors (no memory yet)

**Semantic Meaning**:
```python
# Equivalent Python
for i in range(4):
    for j in range(4):
        for k in range(4):
            C[i,j] += A[i,k] * B[k,j]
```

**Command to generate**:
```bash
google-opt input.mlir --google-to-linalg-lowering
```

---

### Stage 3: Tiled Linalg (Cache-Optimized)

**Purpose**: Apply multi-level tiling for cache hierarchy

**Lowering Pass**: Transform Dialect (embedded in input)

**Input with Transform Module**:
```mlir
module {
  func.func @matmul_example(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %c0 = arith.constant 0.000000e+00 : f32
    %init = tensor.empty() : tensor<4x4xf32>
    %C_init = linalg.fill ins(%c0 : f32) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    %C = linalg.matmul ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>) 
                       outs(%C_init : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %C : tensor<4x4xf32>
  }
  
  // Transform module for tiling
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg: !transform.any_op {transform.readonly}) {
      %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg 
        : (!transform.any_op) -> !transform.any_op
      
      // Tile into 2×2×2 blocks (for this small 4×4 example)
      %tiled, %loops:3 = transform.structured.tile_using_for %matmul 
        tile_sizes [2, 2, 2] 
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      
      transform.yield
    }
  }
}
```

**Output after Tiling**:
```mlir
module {
  func.func @matmul_example(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %c0 = arith.constant 0.000000e+00 : f32
    %c0_idx = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    
    %init = tensor.empty() : tensor<4x4xf32>
    %C_init = linalg.fill ins(%c0 : f32) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    
    // Tiled loops: 3-level loop nest
    %result = scf.for %i = %c0_idx to %c4 step %c2 iter_args(%C_i = %C_init) -> (tensor<4x4xf32>) {
      %result_j = scf.for %j = %c0_idx to %c4 step %c2 iter_args(%C_j = %C_i) -> (tensor<4x4xf32>) {
        %result_k = scf.for %k = %c0_idx to %c4 step %c2 iter_args(%C_k = %C_j) -> (tensor<4x4xf32>) {
          
          // Extract 2×2 tiles
          %A_tile = tensor.extract_slice %A[%i, %k] [2, 2] [1, 1] 
            : tensor<4x4xf32> to tensor<2x2xf32>
          %B_tile = tensor.extract_slice %B[%k, %j] [2, 2] [1, 1] 
            : tensor<4x4xf32> to tensor<2x2xf32>
          %C_tile = tensor.extract_slice %C_k[%i, %j] [2, 2] [1, 1] 
            : tensor<4x4xf32> to tensor<2x2xf32>
          
          // Compute on tile
          %C_tile_result = linalg.matmul 
            ins(%A_tile, %B_tile : tensor<2x2xf32>, tensor<2x2xf32>) 
            outs(%C_tile : tensor<2x2xf32>) -> tensor<2x2xf32>
          
          // Insert result back
          %C_updated = tensor.insert_slice %C_tile_result into %C_k[%i, %j] [2, 2] [1, 1] 
            : tensor<2x2xf32> into tensor<4x4xf32>
          
          scf.yield %C_updated : tensor<4x4xf32>
        }
        scf.yield %result_k : tensor<4x4xf32>
      }
      scf.yield %result_j : tensor<4x4xf32>
    }
    
    return %result : tensor<4x4xf32>
  }
}
```

**Characteristics**:
- ✅ 3-level loop nest (i, j, k)
- ✅ Tile size: 2×2×2
- ✅ Processes 2×2 sub-matrices at a time
- ✅ Better cache locality
- ✅ Still uses tensors (functional style)

**Semantic Meaning**:
```python
# Tiled matmul
for i in range(0, 4, 2):  # Step by 2
    for j in range(0, 4, 2):
        for k in range(0, 4, 2):
            # Process 2×2 tiles
            C[i:i+2, j:j+2] += A[i:i+2, k:k+2] @ B[k:k+2, j:j+2]
```

**Command to generate**:
```bash
google-opt input_with_transform.mlir --google-extreme-l1
```

---

### Stage 4: MemRef Dialect (Explicit Memory)

**Purpose**: Convert tensors to explicit memory buffers

**Lowering Pass**: Bufferization

```mlir
module {
  func.func @matmul_example(%A: memref<4x4xf32>, 
                            %B: memref<4x4xf32>, 
                            %C: memref<4x4xf32>) {
    %c0 = arith.constant 0.000000e+00 : f32
    %c0_idx = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    
    // Initialize C to zero
    scf.for %i = %c0_idx to %c4 step %c2 {
      scf.for %j = %c0_idx to %c4 step %c2 {
        memref.store %c0, %C[%i, %j] : memref<4x4xf32>
      }
    }
    
    // Tiled matmul with explicit memory operations
    scf.for %i = %c0_idx to %c4 step %c2 {
      scf.for %j = %c0_idx to %c4 step %c2 {
        scf.for %k = %c0_idx to %c4 step %c2 {
          
          // Allocate temporary buffers for tiles
          %A_tile = memref.alloc() : memref<2x2xf32>
          %B_tile = memref.alloc() : memref<2x2xf32>
          %C_tile = memref.alloc() : memref<2x2xf32>
          
          // Copy tiles from global memory
          scf.for %ii = %c0_idx to %c2 step %c2 {
            scf.for %kk = %c0_idx to %c2 step %c2 {
              %i_idx = arith.addi %i, %ii : index
              %k_idx = arith.addi %k, %kk : index
              %val = memref.load %A[%i_idx, %k_idx] : memref<4x4xf32>
              memref.store %val, %A_tile[%ii, %kk] : memref<2x2xf32>
            }
          }
          
          // Similar for B_tile and C_tile...
          
          // Compute on tiles (will be lowered to loops)
          linalg.matmul ins(%A_tile, %B_tile : memref<2x2xf32>, memref<2x2xf32>) 
                        outs(%C_tile : memref<2x2xf32>)
          
          // Copy result back
          scf.for %ii = %c0_idx to %c2 step %c2 {
            scf.for %jj = %c0_idx to %c2 step %c2 {
              %i_idx = arith.addi %i, %ii : index
              %j_idx = arith.addi %j, %jj : index
              %val = memref.load %C_tile[%ii, %jj] : memref<2x2xf32>
              %old = memref.load %C[%i_idx, %j_idx] : memref<4x4xf32>
              %new = arith.addf %old, %val : f32
              memref.store %new, %C[%i_idx, %j_idx] : memref<4x4xf32>
            }
          }
          
          // Deallocate temporary buffers
          memref.dealloc %A_tile : memref<2x2xf32>
          memref.dealloc %B_tile : memref<2x2xf32>
          memref.dealloc %C_tile : memref<2x2xf32>
        }
      }
    }
    
    return
  }
}
```

**Characteristics**:
- ✅ `memref` types instead of `tensor`
- ✅ Explicit `memref.load` and `memref.store`
- ✅ Explicit `memref.alloc` and `memref.dealloc`
- ✅ In-place updates (no functional semantics)
- ✅ Memory layout information

**Key Changes**:
- `tensor<4x4xf32>` → `memref<4x4xf32>`
- Implicit operations → Explicit load/store
- Functional style → Imperative style

**Command to generate**:
```bash
google-opt input.mlir --google-extreme-l1 \
  --one-shot-bufferize
```

---

### Stage 5: Affine/SCF Dialect (Explicit Loops)

**Purpose**: Lower structured operations to explicit loop constructs

**Lowering Pass**: `ConvertLinalgToAffineLoops`

```mlir
module {
  func.func @matmul_example(%A: memref<4x4xf32>, 
                            %B: memref<4x4xf32>, 
                            %C: memref<4x4xf32>) {
    %c0 = arith.constant 0.000000e+00 : f32
    
    // Fully explicit nested loops
    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 4 {
        affine.for %k = 0 to 4 {
          // Load values
          %a = affine.load %A[%i, %k] : memref<4x4xf32>
          %b = affine.load %B[%k, %j] : memref<4x4xf32>
          %c = affine.load %C[%i, %j] : memref<4x4xf32>
          
          // Compute
          %prod = arith.mulf %a, %b : f32
          %sum = arith.addf %c, %prod : f32
          
          // Store result
          affine.store %sum, %C[%i, %j] : memref<4x4xf32>
        }
      }
    }
    
    return
  }
}
```

**Characteristics**:
- ✅ `affine.for` loops with static bounds
- ✅ `affine.load` and `affine.store` with affine indexing
- ✅ Arithmetic operations (`arith.mulf`, `arith.addf`)
- ✅ Enables affine analysis and optimization

**Affine Analysis Enables**:
- Loop fusion
- Loop interchange
- Loop tiling (if not already done)
- Dependence analysis
- Parallelization opportunities

**Command to generate**:
```bash
google-opt input.mlir --google-extreme-l1 \
  --one-shot-bufferize \
  --convert-linalg-to-affine-loops
```

---

### Stage 6: SCF/ControlFlow Dialect (Lowered Loops)

**Purpose**: Convert affine loops to standard control flow

**Lowering Pass**: `LowerAffine`, `SCFToControlFlow`

```mlir
module {
  func.func @matmul_example(%A: memref<4x4xf32>, 
                            %B: memref<4x4xf32>, 
                            %C: memref<4x4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0_f32 = arith.constant 0.000000e+00 : f32
    
    // SCF loops (structured control flow)
    scf.for %i = %c0 to %c4 step %c1 {
      scf.for %j = %c0 to %c4 step %c1 {
        scf.for %k = %c0 to %c4 step %c1 {
          // Load
          %a = memref.load %A[%i, %k] : memref<4x4xf32>
          %b = memref.load %B[%k, %j] : memref<4x4xf32>
          %c = memref.load %C[%i, %j] : memref<4x4xf32>
          
          // Compute
          %prod = arith.mulf %a, %b : f32
          %sum = arith.addf %c, %prod : f32
          
          // Store
          memref.store %sum, %C[%i, %j] : memref<4x4xf32>
        }
      }
    }
    
    return
  }
}
```

**After SCFToControlFlow**:
```mlir
module {
  func.func @matmul_example(%A: memref<4x4xf32>, 
                            %B: memref<4x4xf32>, 
                            %C: memref<4x4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    
    // Control flow with basic blocks
    cf.br ^bb_i_init
    
  ^bb_i_init:
    %i_init = arith.constant 0 : index
    cf.br ^bb_i_cond(%i_init : index)
    
  ^bb_i_cond(%i: index):
    %i_cmp = arith.cmpi slt, %i, %c4 : index
    cf.cond_br %i_cmp, ^bb_i_body(%i : index), ^bb_exit
    
  ^bb_i_body(%i_body: index):
    // Similar structure for j and k loops...
    // Load, compute, store operations
    %i_next = arith.addi %i_body, %c1 : index
    cf.br ^bb_i_cond(%i_next : index)
    
  ^bb_exit:
    return
  }
}
```

**Characteristics**:
- ✅ Basic blocks (`^bb_name`)
- ✅ Conditional branches (`cf.cond_br`)
- ✅ Unconditional branches (`cf.br`)
- ✅ No structured loops (just jumps)
- ✅ Ready for LLVM lowering

**Command to generate**:
```bash
google-opt input.mlir --google-extreme-pipeline
# (includes all previous passes plus SCFToControlFlow)
```

---

### Stage 7: LLVM Dialect (Machine-Level IR)

**Purpose**: Convert to LLVM IR for final code generation

**Lowering Pass**: Multiple LLVM conversion passes

```mlir
module {
  llvm.func @matmul_example(%A: !llvm.ptr, %B: !llvm.ptr, %C: !llvm.ptr) {
    %c0 = llvm.mlir.constant(0 : index) : i64
    %c1 = llvm.mlir.constant(1 : index) : i64
    %c4 = llvm.mlir.constant(4 : index) : i64
    
    llvm.br ^bb_i_init
    
  ^bb_i_init:
    llvm.br ^bb_i_cond(%c0 : i64)
    
  ^bb_i_cond(%i: i64):
    %i_cmp = llvm.icmp "slt" %i, %c4 : i64
    llvm.cond_br %i_cmp, ^bb_i_body(%i : i64), ^bb_exit
    
  ^bb_i_body(%i_body: i64):
    llvm.br ^bb_j_init(%i_body : i64)
    
  ^bb_j_init(%i_j: i64):
    llvm.br ^bb_j_cond(%i_j, %c0 : i64, i64)
    
  ^bb_j_cond(%i_jc: i64, %j: i64):
    %j_cmp = llvm.icmp "slt" %j, %c4 : i64
    llvm.cond_br %j_cmp, ^bb_j_body(%i_jc, %j : i64, i64), ^bb_i_inc(%i_jc : i64)
    
  ^bb_j_body(%i_jb: i64, %j_body: i64):
    llvm.br ^bb_k_init(%i_jb, %j_body : i64, i64)
    
  ^bb_k_init(%i_k: i64, %j_k: i64):
    llvm.br ^bb_k_cond(%i_k, %j_k, %c0 : i64, i64, i64)
    
  ^bb_k_cond(%i_kc: i64, %j_kc: i64, %k: i64):
    %k_cmp = llvm.icmp "slt" %k, %c4 : i64
    llvm.cond_br %k_cmp, ^bb_k_body(%i_kc, %j_kc, %k : i64, i64, i64), 
                         ^bb_j_inc(%i_kc, %j_kc : i64, i64)
    
  ^bb_k_body(%i_kb: i64, %j_kb: i64, %k_body: i64):
    // Calculate array indices
    %c4_stride = llvm.mlir.constant(4 : index) : i64
    
    // A[i, k] index calculation
    %i_offset = llvm.mul %i_kb, %c4_stride : i64
    %a_idx = llvm.add %i_offset, %k_body : i64
    %a_ptr = llvm.getelementptr %A[%a_idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %a_val = llvm.load %a_ptr : !llvm.ptr -> f32
    
    // B[k, j] index calculation
    %k_offset = llvm.mul %k_body, %c4_stride : i64
    %b_idx = llvm.add %k_offset, %j_kb : i64
    %b_ptr = llvm.getelementptr %B[%b_idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %b_val = llvm.load %b_ptr : !llvm.ptr -> f32
    
    // C[i, j] index calculation
    %c_idx = llvm.add %i_offset, %j_kb : i64
    %c_ptr = llvm.getelementptr %C[%c_idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %c_val = llvm.load %c_ptr : !llvm.ptr -> f32
    
    // Compute: c = c + a * b
    %prod = llvm.fmul %a_val, %b_val : f32
    %sum = llvm.fadd %c_val, %prod : f32
    
    // Store result
    llvm.store %sum, %c_ptr : f32, !llvm.ptr
    
    // Increment k
    %k_next = llvm.add %k_body, %c1 : i64
    llvm.br ^bb_k_cond(%i_kb, %j_kb, %k_next : i64, i64, i64)
    
  ^bb_j_inc(%i_ji: i64, %j_inc: i64):
    %j_next = llvm.add %j_inc, %c1 : i64
    llvm.br ^bb_j_cond(%i_ji, %j_next : i64, i64)
    
  ^bb_i_inc(%i_inc: i64):
    %i_next = llvm.add %i_inc, %c1 : i64
    llvm.br ^bb_i_cond(%i_next : i64)
    
  ^bb_exit:
    llvm.return
  }
}
```

**Characteristics**:
- ✅ LLVM operations (`llvm.load`, `llvm.store`, `llvm.fadd`, etc.)
- ✅ Pointer arithmetic (`llvm.getelementptr`)
- ✅ Integer comparisons (`llvm.icmp`)
- ✅ Floating-point operations (`llvm.fmul`, `llvm.fadd`)
- ✅ Control flow (`llvm.br`, `llvm.cond_br`)
- ✅ Ready for LLVM code generation

**Command to generate**:
```bash
google-opt input.mlir --google-extreme-pipeline
# Output is LLVM dialect IR
```

---

### Final Stage: Machine Code

**LLVM IR (Text Format)**:
```llvm
define void @matmul_example(float* %A, float* %B, float* %C) {
entry:
  br label %bb_i_init

bb_i_init:
  br label %bb_i_cond

bb_i_cond:
  %i = phi i64 [ 0, %bb_i_init ], [ %i_next, %bb_i_inc ]
  %i_cmp = icmp slt i64 %i, 4
  br i1 %i_cmp, label %bb_i_body, label %bb_exit

bb_i_body:
  br label %bb_j_init

bb_j_init:
  br label %bb_j_cond

bb_j_cond:
  %j = phi i64 [ 0, %bb_j_init ], [ %j_next, %bb_j_inc ]
  %j_cmp = icmp slt i64 %j, 4
  br i1 %j_cmp, label %bb_j_body, label %bb_i_inc

bb_j_body:
  br label %bb_k_init

bb_k_init:
  br label %bb_k_cond

bb_k_cond:
  %k = phi i64 [ 0, %bb_k_init ], [ %k_next, %bb_k_body ]
  %k_cmp = icmp slt i64 %k, 4
  br i1 %k_cmp, label %bb_k_body, label %bb_j_inc

bb_k_body:
  ; Calculate A[i,k] address
  %i_offset = mul i64 %i, 4
  %a_idx = add i64 %i_offset, %k
  %a_ptr = getelementptr float, float* %A, i64 %a_idx
  %a_val = load float, float* %a_ptr
  
  ; Calculate B[k,j] address
  %k_offset = mul i64 %k, 4
  %b_idx = add i64 %k_offset, %j
  %b_ptr = getelementptr float, float* %B, i64 %b_idx
  %b_val = load float, float* %b_ptr
  
  ; Calculate C[i,j] address
  %c_idx = add i64 %i_offset, %j
  %c_ptr = getelementptr float, float* %C, i64 %c_idx
  %c_val = load float, float* %c_ptr
  
  ; Compute
  %prod = fmul float %a_val, %b_val
  %sum = fadd float %c_val, %prod
  
  ; Store
  store float %sum, float* %c_ptr
  
  %k_next = add i64 %k, 1
  br label %bb_k_cond

bb_j_inc:
  %j_next = add i64 %j, 1
  br label %bb_j_cond

bb_i_inc:
  %i_next = add i64 %i, 1
  br label %bb_i_cond

bb_exit:
  ret void
}
```

**x86-64 Assembly (Simplified)**:
```asm
matmul_example:
    push    rbp
    mov     rbp, rsp
    xor     r8, r8              ; i = 0
.L_i_loop:
    cmp     r8, 4
    jge     .L_exit
    xor     r9, r9              ; j = 0
.L_j_loop:
    cmp     r9, 4
    jge     .L_i_inc
    xor     r10, r10            ; k = 0
.L_k_loop:
    cmp     r10, 4
    jge     .L_j_inc
    
    ; Load A[i,k]
    mov     rax, r8
    shl     rax, 2              ; i * 4
    add     rax, r10            ; + k
    movss   xmm0, [rdi + rax*4] ; A[i,k]
    
    ; Load B[k,j]
    mov     rax, r10
    shl     rax, 2              ; k * 4
    add     rax, r9             ; + j
    movss   xmm1, [rsi + rax*4] ; B[k,j]
    
    ; Load C[i,j]
    mov     rax, r8
    shl     rax, 2              ; i * 4
    add     rax, r9             ; + j
    movss   xmm2, [rdx + rax*4] ; C[i,j]
    
    ; Compute: C[i,j] += A[i,k] * B[k,j]
    mulss   xmm0, xmm1          ; a * b
    addss   xmm2, xmm0          ; c + (a * b)
    
    ; Store C[i,j]
    movss   [rdx + rax*4], xmm2
    
    inc     r10                 ; k++
    jmp     .L_k_loop
.L_j_inc:
    inc     r9                  ; j++
    jmp     .L_j_loop
.L_i_inc:
    inc     r8                  ; i++
    jmp     .L_i_loop
.L_exit:
    pop     rbp
    ret
```

**Command to generate machine code**:
```bash
# Generate LLVM IR
google-opt input.mlir --google-extreme-pipeline | \
  mlir-translate --mlir-to-llvmir > output.ll

# Compile to assembly
llc output.ll -o output.s

# Compile to object file
llc output.ll -filetype=obj -o output.o

# Link to executable
clang output.o -o matmul_executable
```

---

## Summary of Progressive Lowering

### Complete Transformation Chain

```
google.matmul                          (1 operation, high-level)
    ↓
linalg.matmul                          (1 operation, structured)
    ↓
3 nested scf.for loops                 (3 loops, tiled)
    ↓
memref.load/store operations           (explicit memory)
    ↓
3 nested affine.for loops              (optimizable loops)
    ↓
9 basic blocks with cf.br              (control flow)
    ↓
llvm.load/store/fadd/fmul             (LLVM operations)
    ↓
LLVM IR (SSA form)                     (machine-independent)
    ↓
x86-64 assembly                        (machine code)
    ↓
Executable binary                      (runs on CPU)
```

### Lines of Code Growth

| Stage | Dialect | Lines of IR | Complexity |
|-------|---------|-------------|------------|
| 1 | Google | ~10 | Very simple |
| 2 | Linalg | ~15 | Simple |
| 3 | Tiled Linalg | ~50 | Moderate |
| 4 | MemRef | ~80 | Complex |
| 5 | Affine | ~25 | Moderate |
| 6 | ControlFlow | ~100 | Very complex |
| 7 | LLVM | ~150 | Extremely complex |

### Abstraction Level

```
High Abstraction (Easy to write, hard to optimize)
    ↓
google.matmul
    ↓
linalg.matmul
    ↓
Tiled loops
    ↓
Explicit memory
    ↓
Affine loops
    ↓
Control flow
    ↓
LLVM IR
    ↓
Low Abstraction (Hard to write, easy to optimize)
```

---

## Complete Command Sequence

### Generate All Stages

```bash
# Stage 1: Input (Google dialect)
cat > input.mlir << 'EOF'
module {
  func.func @matmul_example(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %C = google.matmul %A, %B : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
    return %C : tensor<4x4xf32>
  }
}
EOF

# Stage 2: Lower to Linalg
google-opt input.mlir --google-to-linalg-lowering > stage2_linalg.mlir

# Stage 3: Apply tiling (requires embedded transform)
# (Use input_with_transform.mlir)
google-opt input_with_transform.mlir --google-extreme-l1 > stage3_tiled.mlir

# Stage 4: Bufferization
google-opt stage3_tiled.mlir --one-shot-bufferize > stage4_memref.mlir

# Stage 5: Lower to affine
google-opt stage4_memref.mlir --convert-linalg-to-affine-loops > stage5_affine.mlir

# Stage 6: Lower to control flow
google-opt stage5_affine.mlir --lower-affine --scf-to-cf > stage6_cf.mlir

# Stage 7: Lower to LLVM
google-opt stage6_cf.mlir \
  --convert-func-to-llvm \
  --convert-arith-to-llvm \
  --finalize-memref-to-llvm \
  --reconcile-unrealized-casts > stage7_llvm.mlir

# Final: Generate LLVM IR
mlir-translate --mlir-to-llvmir stage7_llvm.mlir > output.ll

# Compile to executable
llc output.ll -filetype=obj -o output.o
clang output.o -o matmul_executable

# Run
./matmul_executable
```

---

## Conclusion

The "runtime engine" in MLIR is not a single component but a **progressive lowering infrastructure** that transforms high-level operations through multiple intermediate representations until reaching executable machine code.

**Key Takeaways**:
1. Each dialect provides a different abstraction level
2. Lowering preserves semantics while enabling optimizations
3. Memory management transitions from implicit (tensors) to explicit (memrefs)
4. Control flow evolves from structured (loops) to unstructured (branches)
5. Final LLVM IR is optimized and compiled to machine code

This architecture enables:
- ✅ High-level expressiveness (Google dialect)
- ✅ Powerful optimizations (Linalg, Affine)
- ✅ Efficient execution (LLVM code generation)
- ✅ Hardware portability (CPU, GPU, TPU)
