# MLIR Core Concepts: Deep Dive

## Table of Contents

1. [Introduction](#introduction)
2. [SSA Form (Static Single Assignment)](#ssa-form)
3. [Operations, Regions, and Blocks](#operations-regions-and-blocks)
4. [Type System: Tensor vs MemRef](#type-system-tensor-vs-memref)
5. [Attributes vs Operands](#attributes-vs-operands)
6. [Polymorphism: Traits and Interfaces](#polymorphism-traits-and-interfaces)
7. [Dominance and Control Flow](#dominance-and-control-flow)
8. [Dialects and Extensibility](#dialects-and-extensibility)
9. [Common Patterns and Idioms](#common-patterns-and-idioms)
10. [Practical Examples](#practical-examples)

---

## Introduction

MLIR (Multi-Level Intermediate Representation) is a revolutionary compiler infrastructure that many developers use daily without fully understanding its core concepts. This guide demystifies the fundamental principles that make MLIR powerful and flexible.

**Why This Matters**: Understanding these concepts helps you:
- Write better transformations
- Debug IR issues faster
- Design cleaner dialects
- Avoid common pitfalls

---

## SSA Form (Static Single Assignment)

### What is SSA?

**SSA** means every value is **defined exactly once** and **used multiple times**.

### Traditional vs SSA

**Traditional (Mutable Variables)**:
```c
int x = 5;      // x defined
x = x + 1;      // x redefined
x = x * 2;      // x redefined again
return x;
```

**SSA Form**:
```mlir
%0 = arith.constant 5 : i32        // %0 defined once
%1 = arith.addi %0, %c1 : i32      // %1 defined once
%2 = arith.muli %1, %c2 : i32      // %2 defined once
return %2 : i32
```

**Key Rules**:
1. Each SSA value has **exactly one definition**
2. Values are **immutable** after definition
3. Uses must be **dominated** by definitions

### Why SSA?

**Benefits**:
- ✅ **Clear data dependencies**: No ambiguity about where values come from
- ✅ **Simpler optimizations**: No need to track variable mutations
- ✅ **Easier analysis**: Def-use chains are explicit
- ✅ **Better parallelization**: Independent values can be computed in parallel

### SSA Values in MLIR

**Every operation result is an SSA value**:

```mlir
// Operation produces SSA values
%result = arith.addi %a, %b : i32
//^^^^^^ SSA value (defined once)
//                   ^^  ^^ SSA values (used here)
```

**Types of SSA Values**:
1. **Operation Results**: `%0 = some.op ...`
2. **Block Arguments**: `^bb0(%arg0: i32):`
3. **Function Arguments**: `func.func @foo(%arg0: tensor<256xf32>)`

### Block Arguments (MLIR's PHI Nodes)

**Traditional PHI Nodes** (LLVM):
```llvm
bb3:
  %x = phi i32 [ %a, %bb1 ], [ %b, %bb2 ]
```

**MLIR Block Arguments**:
```mlir
^bb3(%x: i32):  // Block takes argument
  // Use %x here
  
^bb1:
  cf.br ^bb3(%a : i32)  // Pass %a to bb3
  
^bb2:
  cf.br ^bb3(%b : i32)  // Pass %b to bb3
```

**Why Better?**:
- ✅ More explicit: Arguments are part of block signature
- ✅ Clearer semantics: Parallel copy is obvious
- ✅ Easier transformations: No special PHI handling

---

## Operations, Regions, and Blocks

### The Hierarchy

```
Operation
  ├─ Operands (inputs)
  ├─ Results (outputs)
  ├─ Attributes (compile-time data)
  └─ Regions (nested IR)
      └─ Blocks (sequences of operations)
          └─ Operations (recursive!)
```

### Operations

**Definition**: The fundamental unit of MLIR. Everything is an operation.

**Anatomy**:
```mlir
%result = dialect.operation_name %operand1, %operand2 {attribute = value} : type
//^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^   ^^^^
//result   operation name          operands            attributes           type
```

**Example**:
```mlir
%0 = arith.addi %a, %b : i32
//   ^^^^^^^^^^  Operation name
//              ^^  ^^  Operands
//                      ^^^  Result type
```

**Operations Can Contain Regions**:
```mlir
scf.for %i = %c0 to %c10 step %c1 {
  // This is a REGION containing operations
  %x = arith.addi %i, %c1 : index
  scf.yield
}
```

### Regions

**Definition**: Ordered lists of blocks. Provide scoping and control flow.

**Two Types**:

**1. SSACFG Regions** (Control Flow Graph):
- Traditional control flow
- SSA dominance rules apply
- Example: Function bodies

```mlir
func.func @example(%arg0: i32) -> i32 {
  // SSACFG region
  %0 = arith.addi %arg0, %c1 : i32
  return %0 : i32
}
```

**2. Graph Regions** (Dataflow):
- Relaxed dominance
- Can have cycles
- Example: Hardware netlists

```mlir
hw.module @example(%in: i32) -> (out: i32) {
  // Graph region - different semantics
  %0 = comb.add %in, %1 : i32
  %1 = comb.mul %0, %c2 : i32  // Can reference %0 before it's "defined"
  hw.output %0 : i32
}
```

### Blocks

**Definition**: Linear sequences of operations with a single entry point.

**Structure**:
```mlir
^block_name(%arg1: type1, %arg2: type2):
  operation1
  operation2
  ...
  terminator_operation
```

**Example**:
```mlir
^bb0(%arg0: i32):
  %0 = arith.cmpi slt, %arg0, %c10 : i32
  cf.cond_br %0, ^bb1, ^bb2
  
^bb1:
  %1 = arith.addi %arg0, %c1 : i32
  cf.br ^bb3(%1 : i32)
  
^bb2:
  %2 = arith.muli %arg0, %c2 : i32
  cf.br ^bb3(%2 : i32)
  
^bb3(%result: i32):
  return %result : i32
```

**Key Points**:
- Blocks must end with a **terminator** (br, return, yield, etc.)
- Blocks can take **arguments** (like function parameters)
- Arguments replace PHI nodes

---

## Type System: Tensor vs MemRef

### The Fundamental Difference

| Aspect | `tensor` | `memref` |
|--------|----------|----------|
| **Abstraction** | High-level, mathematical | Low-level, physical memory |
| **Mutability** | Immutable (functional) | Mutable (imperative) |
| **Memory** | Abstract, no layout | Explicit layout, strides |
| **Aliasing** | No aliasing | Can alias |
| **Pointers** | No pointers | Has pointers |
| **Use Case** | High-level optimization | Low-level code generation |

### Tensor Type

**Definition**: Immutable, abstract N-dimensional data.

**Syntax**:
```mlir
tensor<256x256xf32>          // 2D tensor, static shape
tensor<?x?xf32>              // 2D tensor, dynamic shape
tensor<*xf32>                // Unranked tensor
```

**Characteristics**:
- **Immutable**: Operations create new tensors
- **No memory layout**: Compiler decides
- **Functional**: Pure, no side effects

**Example**:
```mlir
func.func @tensor_example(%arg0: tensor<256x256xf32>, %arg1: tensor<256x256xf32>) 
    -> tensor<256x256xf32> {
  // Operations on tensors are FUNCTIONAL
  %0 = linalg.matmul 
    ins(%arg0, %arg1 : tensor<256x256xf32>, tensor<256x256xf32>) 
    outs(%init : tensor<256x256xf32>) 
    -> tensor<256x256xf32>
  // %0 is a NEW tensor, %arg0 and %arg1 unchanged
  return %0 : tensor<256x256xf32>
}
```

**Why Tensors?**:
- ✅ **High-level**: Express intent, not implementation
- ✅ **Optimizable**: Compiler has freedom
- ✅ **Portable**: Works across different backends
- ✅ **Composable**: Easy to fuse operations

### MemRef Type

**Definition**: Mutable reference to a memory buffer.

**Syntax**:
```mlir
memref<256x256xf32>                           // Simple memref
memref<256x256xf32, affine_map<(d0,d1) -> (d0*256+d1)>>  // With layout
memref<256x256xf32, 1>                        // In address space 1
memref<?x?xf32>                               // Dynamic dimensions
```

**Characteristics**:
- **Mutable**: Can read/write in place
- **Explicit layout**: Strides, offsets specified
- **Memory management**: Alloc/dealloc explicit

**Example**:
```mlir
func.func @memref_example() {
  // Allocate memory
  %mem = memref.alloc() : memref<256x256xf32>
  
  // Write to memory (MUTATION!)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1.0 : f32
  memref.store %c1, %mem[%c0, %c0] : memref<256x256xf32>
  
  // Read from memory
  %val = memref.load %mem[%c0, %c0] : memref<256x256xf32>
  
  // Deallocate
  memref.dealloc %mem : memref<256x256xf32>
  return
}
```

**Why MemRefs?**:
- ✅ **Low-level**: Close to hardware
- ✅ **Efficient**: In-place updates
- ✅ **Explicit**: No hidden copies
- ✅ **Controllable**: Precise memory layout

### Bufferization: Tensor → MemRef

**The Bridge**:

```mlir
// High-level (Tensor)
%result = linalg.matmul 
  ins(%A, %B : tensor<256x256xf32>, tensor<256x256xf32>) 
  outs(%C : tensor<256x256xf32>) 
  -> tensor<256x256xf32>

// After Bufferization (MemRef)
%A_mem = bufferization.to_memref %A : memref<256x256xf32>
%B_mem = bufferization.to_memref %B : memref<256x256xf32>
%C_mem = memref.alloc() : memref<256x256xf32>
linalg.matmul 
  ins(%A_mem, %B_mem : memref<256x256xf32>, memref<256x256xf32>) 
  outs(%C_mem : memref<256x256xf32>)
%result = bufferization.to_tensor %C_mem : tensor<256x256xf32>
```

**When to Use What?**:
- **Tensor**: High-level transformations (fusion, tiling)
- **MemRef**: Low-level optimizations (memory layout, vectorization)

---

## Attributes vs Operands

### The Fundamental Difference

| Aspect | Operands | Attributes |
|--------|----------|------------|
| **When Known** | Runtime | Compile-time |
| **Type** | SSA values | Constant data |
| **Can Change** | Yes (different values) | No (fixed at compile time) |
| **Examples** | Function arguments, results | Tile sizes, array shapes |

### Operands

**Definition**: Runtime SSA values passed to operations.

**Example**:
```mlir
%result = arith.addi %a, %b : i32
//                    ^^  ^^ Operands (runtime values)
```

**Characteristics**:
- **Runtime values**: Not known at compile time
- **SSA values**: Defined by other operations
- **Can vary**: Different values in different executions

### Attributes

**Definition**: Compile-time constant data attached to operations.

**Example**:
```mlir
%result = arith.constant 42 : i32
//                       ^^ Attribute (compile-time constant)

%tiled, %loops:3 = transform.structured.tile_using_for %op 
  tile_sizes [16, 16, 16]
//           ^^^^^^^^^^^^ Attribute (compile-time array)
```

**Common Attribute Types**:

**1. Integer Attributes**:
```mlir
%c = arith.constant 42 : i32
//                  ^^ IntegerAttr
```

**2. Array Attributes**:
```mlir
tile_sizes [16, 16, 16]
//         ^^^^^^^^^^^^ ArrayAttr of IntegerAttrs
```

**3. String Attributes**:
```mlir
func.func @foo() attributes {sym_visibility = "private"}
//                                            ^^^^^^^^^ StringAttr
```

**4. Type Attributes**:
```mlir
%0 = llvm.mlir.constant(0 : i32) : i32
//                      ^^^^^^^ TypeAttr
```

**5. Dictionary Attributes**:
```mlir
{attribute1 = value1, attribute2 = value2}
```

**Why Attributes?**:
- ✅ **Optimization**: Compiler can reason about constants
- ✅ **Verification**: Check constraints at compile time
- ✅ **Metadata**: Attach information to operations

---

## Polymorphism: Traits and Interfaces

### The Problem

How do you write **generic transformations** that work on **any operation** without knowing its specific type?

**Example**: How to check if an operation is commutative?

### Solution 1: Traits

**Definition**: Static properties attached to operations.

**How They Work**:
- Traits are **mixins** (base classes)
- Provide **static methods** and **properties**
- No virtual dispatch

**Common Traits**:

**1. `Pure` (formerly `NoSideEffect`)**:
```cpp
def MyOp : Op<MyDialect, "my_op", [Pure]> {
  // This operation has no side effects
}
```

**Usage**:
```cpp
if (op->hasTrait<OpTrait::Pure>()) {
  // Safe to eliminate if result unused
}
```

**2. `Commutative`**:
```cpp
def AddOp : Op<Arith_Dialect, "addi", [Commutative]> {
  // a + b == b + a
}
```

**3. `Terminator`**:
```cpp
def ReturnOp : Op<Func_Dialect, "return", [Terminator]> {
  // This operation terminates a block
}
```

**4. `SameOperandsAndResultType`**:
```cpp
def AddOp : Op<Arith_Dialect, "addi", [SameOperandsAndResultType]> {
  // All operands and result have same type
}
```

**Checking Traits**:
```cpp
Operation *op = ...;
if (op->hasTrait<OpTrait::Commutative>()) {
  // Can reorder operands
}
```

### Solution 2: Interfaces

**Definition**: Polymorphic APIs for operations.

**How They Work**:
- **Concept-based polymorphism**: Like virtual methods
- **Model**: Implements interface for specific operation
- **Concept**: Defines interface methods

**Common Interfaces**:

**1. `InferTypeOpInterface`**:
```cpp
def MyOp : Op<MyDialect, "my_op", [InferTypeOpInterface]> {
  let extraClassDeclaration = [{
    static LogicalResult inferReturnTypes(
        MLIRContext *context,
        Optional<Location> location,
        ValueRange operands,
        DictionaryAttr attributes,
        RegionRange regions,
        SmallVectorImpl<Type> &inferredReturnTypes) {
      // Infer result type from operands
      inferredReturnTypes.push_back(operands[0].getType());
      return success();
    }
  }];
}
```

**2. `LoopLikeOpInterface`**:
```mlir
// scf.for implements LoopLikeOpInterface
scf.for %i = %c0 to %c10 step %c1 {
  // ...
}
```

**Usage**:
```cpp
if (auto loopOp = dyn_cast<LoopLikeOpInterface>(op)) {
  // Can call interface methods
  loopOp.moveOutOfLoop(invariantOp);
}
```

**3. `TilingInterface`**:
```cpp
// linalg.matmul implements TilingInterface
auto tilingInterface = dyn_cast<TilingInterface>(matmulOp);
if (tilingInterface) {
  // Can tile this operation
  auto tiledOps = tilingInterface.getTiledImplementation(...);
}
```

### Traits vs Interfaces

| Aspect | Traits | Interfaces |
|--------|--------|------------|
| **Polymorphism** | Static (compile-time) | Dynamic (runtime) |
| **Dispatch** | No virtual methods | Virtual methods |
| **Use Case** | Simple properties | Complex behavior |
| **Performance** | Faster (no virtual calls) | Slower (virtual calls) |
| **Flexibility** | Less flexible | More flexible |

**When to Use**:
- **Traits**: Simple boolean properties (Pure, Commutative)
- **Interfaces**: Complex behavior (tiling, type inference)

---

## Dominance and Control Flow

### What is Dominance?

**Definition**: Block A **dominates** block B if **all paths** from entry to B go through A.

**Why It Matters**: SSA requires values to be defined before use. Dominance ensures this.

### Dominance in MLIR

**1. Block Dominance** (Traditional):
```mlir
^entry:
  cf.br ^bb1
  
^bb1:  // entry dominates bb1
  cf.br ^bb2
  
^bb2:  // entry and bb1 dominate bb2
  return
```

**2. Hierarchical Dominance** (MLIR-specific):

```mlir
func.func @example() {
  // Outer region
  %0 = arith.constant 42 : i32
  
  scf.for %i = %c0 to %c10 step %c1 {
    // Inner region
    // %0 is visible here (outer dominates inner)
    %1 = arith.addi %0, %i : i32
    
    scf.yield
  }
  
  // %1 is NOT visible here (inner doesn't dominate outer)
  // %x = arith.addi %1, %c1 : i32  // ERROR!
  
  return
}
```

**Rules**:
1. **Parent dominates child**: Values from outer regions visible in inner regions
2. **Child doesn't dominate parent**: Values from inner regions NOT visible outside
3. **Siblings don't dominate**: Values from one region not visible in sibling regions

### Dominance Violations

**Common Error**:
```mlir
func.func @bad_example(%cond: i1) -> i32 {
  cf.cond_br %cond, ^bb1, ^bb2
  
^bb1:
  %0 = arith.constant 1 : i32
  cf.br ^bb3
  
^bb2:
  cf.br ^bb3
  
^bb3:
  return %0 : i32  // ERROR! %0 not defined in all paths
}
```

**Fix with Block Arguments**:
```mlir
func.func @good_example(%cond: i1) -> i32 {
  cf.cond_br %cond, ^bb1, ^bb2
  
^bb1:
  %0 = arith.constant 1 : i32
  cf.br ^bb3(%0 : i32)
  
^bb2:
  %1 = arith.constant 2 : i32
  cf.br ^bb3(%1 : i32)
  
^bb3(%arg: i32):  // Block argument!
  return %arg : i32  // OK!
}
```

---

## Dialects and Extensibility

### What are Dialects?

**Definition**: Namespaces for operations, types, and attributes.

**Examples**:
- `arith`: Arithmetic operations
- `func`: Functions
- `scf`: Structured control flow
- `linalg`: Linear algebra
- `tensor`: Tensor operations
- `memref`: Memory operations
- `llvm`: LLVM dialect

### Why Dialects?

**Modularity**:
```mlir
// Mix dialects freely
func.func @example(%arg0: tensor<256xf32>) -> tensor<256xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.fill ins(%c0 : f32) outs(%arg0 : tensor<256xf32>) -> tensor<256xf32>
  return %0 : tensor<256xf32>
}
//     ^^^^^        ^^^^^^^^^^^                ^^^^^^
//     func         arith                      linalg
```

**Extensibility**:
```cpp
// Define your own dialect
class MyDialect : public Dialect {
  static constexpr StringLiteral getDialectNamespace() {
    return "my";
  }
};

// Define operations
def MyOp : Op<MyDialect, "my_op"> {
  let summary = "My custom operation";
}
```

**Progressive Lowering**:
```
High-Level Dialect (google)
    ↓ (GoogleToLinalg)
Mid-Level Dialect (linalg)
    ↓ (Bufferization)
Low-Level Dialect (memref)
    ↓ (LLVM Lowering)
LLVM Dialect
    ↓ (Code Generation)
Machine Code
```

---

## Common Patterns and Idioms

### Pattern 1: Iteration with `iter_args`

**Problem**: How to accumulate values in loops (SSA forbids mutation)?

**Solution**: Use `iter_args`:

```mlir
// Accumulate sum in loop
%sum = scf.for %i = %c0 to %c10 step %c1 
    iter_args(%acc = %c0_i32) -> (i32) {
  //        ^^^^^^^^^^^^^^^^ Loop-carried value
  %next = arith.addi %acc, %i : i32
  scf.yield %next : i32  // Pass to next iteration
}
// %sum contains final accumulated value
```

**How It Works**:
- `iter_args` are like function parameters for each iteration
- `scf.yield` passes values to next iteration
- Final `yield` becomes loop result

### Pattern 2: Tensor Updates

**Problem**: Tensors are immutable, how to "update" them?

**Solution**: Use `tensor.insert_slice`:

```mlir
// "Update" tensor (actually creates new tensor)
%updated = tensor.insert_slice %tile into %tensor[%i, %j] [16, 16] [1, 1]
  : tensor<16x16xf32> into tensor<256x256xf32>
// %updated is a NEW tensor, %tensor unchanged
```

### Pattern 3: Type Conversion

**Problem**: Convert between tensor and memref?

**Solution**: Use bufferization operations:

```mlir
// Tensor → MemRef
%mem = bufferization.to_memref %tensor : memref<256x256xf32>

// MemRef → Tensor
%tensor = bufferization.to_tensor %mem : tensor<256x256xf32>
```

### Pattern 4: Conditional Execution

**Problem**: How to express if-then-else in SSA?

**Solution**: Use `scf.if` with results:

```mlir
%result = scf.if %cond -> (i32) {
  %true_val = arith.constant 1 : i32
  scf.yield %true_val : i32
} else {
  %false_val = arith.constant 2 : i32
  scf.yield %false_val : i32
}
// %result is 1 if %cond true, 2 otherwise
```

---

## Practical Examples

### Example 1: Understanding Tiling IR

**Before Tiling**:
```mlir
%0 = linalg.matmul 
  ins(%A, %B : tensor<256x256xf32>, tensor<256x256xf32>) 
  outs(%C : tensor<256x256xf32>) 
  -> tensor<256x256xf32>
```

**After Tiling** (with annotations):
```mlir
// Outer loop (SSA value %0)
%0 = scf.for %i = %c0 to %c256 step %c16 
    iter_args(%arg = %C) -> (tensor<256x256xf32>) {
  //        ^^^^^^^^^^^ Loop-carried tensor
  
  // Middle loop (SSA value %1)
  %1 = scf.for %j = %c0 to %c256 step %c16 
      iter_args(%arg2 = %arg) -> (tensor<256x256xf32>) {
    
    // Inner loop (SSA value %2)
    %2 = scf.for %k = %c0 to %c256 step %c16 
        iter_args(%arg3 = %arg2) -> (tensor<256x256xf32>) {
      
      // Extract tiles (immutable!)
      %A_tile = tensor.extract_slice %A[%i, %k] [16, 16] [1, 1]
      %B_tile = tensor.extract_slice %B[%k, %j] [16, 16] [1, 1]
      %C_tile = tensor.extract_slice %arg3[%i, %j] [16, 16] [1, 1]
      
      // Compute on tiles
      %result_tile = linalg.matmul 
        ins(%A_tile, %B_tile) outs(%C_tile) 
        -> tensor<16x16xf32>
      
      // Insert result (creates new tensor!)
      %updated = tensor.insert_slice %result_tile into %arg3[%i, %j] [16, 16] [1, 1]
      
      // Yield to next iteration
      scf.yield %updated : tensor<256x256xf32>
    }
    scf.yield %2 : tensor<256x256xf32>
  }
  scf.yield %1 : tensor<256x256xf32>
}
// %0 is final result
```

**Key Concepts Used**:
- ✅ SSA: Each `%0`, `%1`, `%2` defined once
- ✅ `iter_args`: Carry tensor through iterations
- ✅ Immutability: `insert_slice` creates new tensor
- ✅ Dominance: Inner values visible in their scope

### Example 2: Understanding Bufferization

**High-Level (Tensor)**:
```mlir
func.func @matmul_tensor(%A: tensor<256x256xf32>, %B: tensor<256x256xf32>) 
    -> tensor<256x256xf32> {
  %C = tensor.empty() : tensor<256x256xf32>
  %result = linalg.matmul ins(%A, %B) outs(%C) -> tensor<256x256xf32>
  return %result : tensor<256x256xf32>
}
```

**After Bufferization (MemRef)**:
```mlir
func.func @matmul_memref(%A: memref<256x256xf32>, %B: memref<256x256xf32>) 
    -> memref<256x256xf32> {
  // Allocate output buffer
  %C = memref.alloc() : memref<256x256xf32>
  
  // In-place computation (mutation!)
  linalg.matmul ins(%A, %B : memref<256x256xf32>, memref<256x256xf32>) 
                outs(%C : memref<256x256xf32>)
  
  return %C : memref<256x256xf32>
}
```

**Key Changes**:
- `tensor` → `memref`
- `tensor.empty()` → `memref.alloc()`
- Functional → Imperative
- Immutable → Mutable

---

## Conclusion

### Key Takeaways

**SSA Form**:
- ✅ Every value defined once
- ✅ Block arguments replace PHI nodes
- ✅ Immutability is fundamental

**Operations, Regions, Blocks**:
- ✅ Hierarchical structure
- ✅ Regions provide scoping
- ✅ Blocks are linear sequences

**Tensor vs MemRef**:
- ✅ Tensor: High-level, immutable, abstract
- ✅ MemRef: Low-level, mutable, concrete
- ✅ Bufferization bridges the gap

**Attributes vs Operands**:
- ✅ Attributes: Compile-time constants
- ✅ Operands: Runtime SSA values

**Polymorphism**:
- ✅ Traits: Static properties
- ✅ Interfaces: Dynamic behavior

**Dominance**:
- ✅ Hierarchical scoping
- ✅ Values visible in nested regions
- ✅ Block arguments for control flow merges

### Why These Concepts Matter

Understanding these fundamentals helps you:
1. **Write better transformations**: Know what's legal
2. **Debug faster**: Understand error messages
3. **Design cleaner dialects**: Follow MLIR conventions
4. **Optimize effectively**: Leverage MLIR's power

**MLIR is powerful because it gets these fundamentals right!** 🚀
