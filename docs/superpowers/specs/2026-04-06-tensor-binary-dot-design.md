# Tensor Binary Ops & Dot Product — Design Spec

**Date:** 2026-04-06
**Branch:** LoopTensorizebyClaude
**Status:** Approved

---

## 1. Background

The TPlan lowering pass currently supports N-D tensor contraction via the
`llvm.tensor.contract.<Ra>d.<Rb>d.<type>` intrinsic family (Ra, Rb ∈ 1..4).
These intrinsics are declared dynamically via `getOrInsertFunction()` rather
than being registered in `Intrinsics.td`.

Two gaps remain:

| Gap | Description |
|-----|-------------|
| **RankC=0** | 1D×1D dot product produces a scalar output (no output dims). Current code rejects `RankC < 1` and falls back to scalar. |
| **Element-wise / broadcast ops** | add, sub, mul, div, and, or, xor across N-D tensors with broadcasting are not covered by the contraction family. |

Additionally, all existing tensor intrinsics should be registered in
`Intrinsics.td` to make them first-class LLVM intrinsics.

---

## 2. Design Decisions

### 2.1 No New LLVM Types

We extend the existing intrinsic family rather than introducing a new
`TensorType` to the LLVM type system. The pointer + stride convention used by
`llvm.tensor.contract.*` is preserved throughout.

### 2.2 Generalized Tensor-Tensor Model

A vector is a rank-1 tensor; a matrix is a rank-2 tensor. All operation
combinations (vector-vector, vector-matrix, matrix-tensor, tensor-tensor) are
expressed as N-D × M-D intrinsics. This covers:

1. vector × matrix (1D × 2D)
2. tensor × tensor (N-D × M-D)
3. vector × tensor (1D × N-D)
4. matrix × tensor (2D × N-D)

### 2.3 Two Intrinsic Families

| Family | Reduction dim | Use case |
|--------|--------------|----------|
| `llvm.tensor.contract.<Ra>d.<Rb>d.<Rc>d.<type>` | Yes (K) | dot product, GEMV, GEMM, batched GEMM, general contraction |
| `llvm.tensor.binary.<op>.<Ra>d.<Rb>d.<Rc>d.<type>` | No | element-wise add/sub/mul/div and boolean and/or/xor with optional broadcasting |

> **Why `<Rc>d` in the name?** The same `(Ra, Rb)` pair can produce different RankC values depending
> on how many dimensions are shared between the two operands. For example, `contract.3d.3d` with
> batched GEMM (shared batch dim) has RankC=3, while a `contract.3d.3d` with no shared output dims
> has RankC=4 — giving different parameter counts. Including `<Rc>d` makes the signature
> unambiguous and is required for correct `Intrinsics.td` registration.
> **This is a breaking rename from the existing dynamic declarations** — existing test CHECK lines
> (e.g., `contract.2d.2d.f32` → `contract.2d.2d.2d.f32`) must be updated.

---

## 3. `llvm.tensor.contract` — RankC=0 Fix

### 3.1 Why RankC=0 Happens

For the dot product `acc += A[k] * B[k]`:

```
A.DimSet     = {k}
B.DimSet     = {k}
ContractDim  = k
OutputDimSet = (A.DimSet ∪ B.DimSet) − {ContractDim}
             = {k} − {k}
             = ∅   →  RankC = 0
```

The only dimension is contracted away, leaving a scalar accumulator.

### 3.2 Intrinsic Signature

`llvm.tensor.contract.1d.1d.0d.<type>` (RankC=0):

```
void(
  ptr   C,             ; scalar accumulator pointer (no stride slots)
  ptr   A,
  i64   A_stride,
  ptr   B,
  i64   B_stride,
  i64   K              ; trip count of the contraction dimension
)
```

This is the general formula `3 + 4·RankC + 3` with `RankC=0` → 6 parameters.
No special-casing of the formula is required.

### 3.3 `Intrinsics.td` Entry (per element type)

```tablegen
def int_tensor_contract_1d_1d_0d_f32 : Intrinsic<
  [],
  [llvm_ptr_ty,                  // C
   llvm_ptr_ty, llvm_i64_ty,     // A, A_stride
   llvm_ptr_ty, llvm_i64_ty,     // B, B_stride
   llvm_i64_ty],                 // K
  [IntrArgMemOnly, IntrWillReturn]>;
```

### 3.4 `emitContraction()` Change

Remove the `RankC < 1` rejection guard. When `RankC == 0`, skip the
output-dim stride loop (nothing to iterate) and emit the 6-parameter call.

---

## 4. `llvm.tensor.binary` — New Element-wise / Broadcast Family

### 4.1 Supported Operations

| Op string | Semantics |
|-----------|-----------|
| `add` | element-wise addition (float or integer) |
| `sub` | element-wise subtraction |
| `mul` | element-wise multiplication |
| `div` | element-wise division |
| `and` | bitwise / boolean AND |
| `or`  | bitwise / boolean OR |
| `xor` | bitwise / boolean XOR |

### 4.2 Intrinsic Name

```
llvm.tensor.binary.<op>.<Ra>d.<Rb>d.<Rc>d.<type>
```

Examples:
```
llvm.tensor.binary.add.2d.2d.2d.f32    ; 2D + 2D element-wise add (RankC=2)
llvm.tensor.binary.add.1d.2d.2d.f32    ; 1D broadcast-add into 2D (RankC=2)
llvm.tensor.binary.mul.1d.1d.1d.f32    ; 1D element-wise multiply (RankC=1)
llvm.tensor.binary.and.1d.1d.1d.i1     ; 1D boolean AND (RankC=1)
llvm.tensor.binary.mul.3d.3d.3d.f32    ; 3D element-wise multiply (RankC=3)
```

### 4.3 OutputDimSet Computation

Unlike contraction, there is no reduction dimension:

```
OutputDimSet = A.DimSet ∪ B.DimSet
RankC        = OutputDimSet.count()
```

### 4.4 Broadcast Convention

Stride = 0 on any output dimension not present in an operand's DimSet.
This is identical to the existing convention in `llvm.tensor.contract.*`.

Example — `binary.add.1d.2d.f32` where A is a 1D row vector broadcast-added
to a 2D matrix B:

```
A.DimSet     = {j}         ; A spans only the column dimension
B.DimSet     = {i, j}
OutputDimSet = {i, j}      ; RankC = 2

A_strides[i] = 0           ; A does not span row dim → broadcast
A_strides[j] = <stride>
B_strides[i] = <stride>
B_strides[j] = <stride>
```

### 4.5 Intrinsic Signature

```
void(
  ptr   C,   i64×RankC C_strides,
  ptr   A,   i64×RankC A_strides,   ; 0 if A ∌ that dim
  ptr   B,   i64×RankC B_strides,   ; 0 if B ∌ that dim
  i64×RankC  output_dims
)
```

Parameter count: `3 + 4·RankC` (3 fewer than contract — no K or contract strides).

### 4.6 `Intrinsics.td` Registration Strategy

All combinations Ra ∈ {1,2,3,4} × Rb ∈ {1,2,3,4} × op × type are enumerated
using a TableGen multiclass. Maximum entries: 4 × 4 × 7 ops × 8 types = 896,
but in practice only the combinations actually emitted by TPlan are needed.
We register all combinations up front for completeness.

```tablegen
multiclass TensorBinaryIntrinsic<string Op, int Ra, int Rb, int Rc,
                                  LLVMType ElemTy, string TypeSuffix> {
  def _#Op#_#Ra#d_#Rb#d_#TypeSuffix : Intrinsic<
    [],
    !listconcat(
      [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),   // C, C_strides
      [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),   // A, A_strides
      [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),   // B, B_strides
      !listsplat(llvm_i64_ty, Rc)                   // output_dims
    ),
    [IntrArgMemOnly, IntrWillReturn]>;
}
```

---

## 5. TPlan Lowering Changes

### 5.1 Stage 2: Pattern Matching

| Current | After |
|---------|-------|
| `Contraction` — rejects RankC=0 | `Contraction` — allows RankC=0 (dot product) |
| `ElementWise` + `BroadcastBinary` — two separate classes | `BinaryOp` — unified, records op kind (add/sub/mul/div/and/or/xor) |

### 5.2 Stage 3: IR Emission

`TPWidenRecipe::execute()` dispatch:

```
Contraction → emitContraction()    (modified: RankC=0 allowed)
BinaryOp    → emitBinaryOp()       (new)
```

### 5.3 New: `getTensorBinaryFn()`

Helper analogous to `getTensorContractFn()`:

```cpp
static FunctionCallee getTensorBinaryFn(Module &M, StringRef Op,
                                         unsigned RankA, unsigned RankB,
                                         unsigned RankC, Type *ElemTy) {
  std::string Name = ("llvm.tensor.binary." + Op + "." +
                      Twine(RankA) + "d." +
                      Twine(RankB) + "d." +
                      Twine(RankC) + "d." +
                      getTypeSuffix(ElemTy)).str();
  // Parameter count: 3 + 4·RankC
  // Build FunctionType and return via M.getOrInsertFunction()
}
```

### 5.4 New: `emitBinaryOp()`

1. Extract RankA, RankB from DimSets
2. Validate Ra, Rb ∈ [1, 4]
3. Compute `OutputDimSet = A.DimSet ∪ B.DimSet`, `RankC = count()`
4. Validate RankC ∈ [1, 4] (at least one operand must have ≥1 dim)
5. Locate C pointer (same recipe-user-walk + IR fallback as `emitContraction`)
6. For each output dim D in `OutputDimSet` order:
   - `C_strides[D]` = dense stride
   - `A_strides[D]` = `A.DimSet.test(D) ? stride : 0`
   - `B_strides[D]` = `B.DimSet.test(D) ? stride : 0`
   - `output_dims[D]` = `Plan.getPFForDim(D)`
7. Call `getTensorBinaryFn(op, RankA, RankB, RankC, ElemTy)` and emit

---

## 6. Test Plan

New test files in `llvm/test/Transforms/LoopTensorize/basic/`:

| File | Case |
|------|------|
| `tensor-contract-dot.ll` | 1D×1D dot product → `contract.1d.1d.0d.f32` (RankC=0, 6 params) |
| `tensor-binary-add-1d.ll` | 1D+1D add → `binary.add.1d.1d.1d.f32` |
| `tensor-binary-add-broadcast.ll` | 1D+2D broadcast add → `binary.add.1d.2d.2d.f32` (A_stride=0 on row dim) |
| `tensor-binary-sub-2d.ll` | 2D−2D sub → `binary.sub.2d.2d.2d.f32` |
| `tensor-binary-div-2d.ll` | 2D/2D div → `binary.div.2d.2d.2d.f32` |
| `tensor-binary-and-1d.ll` | 1D AND 1D → `binary.and.1d.1d.1d.i1` |
| `tensor-binary-mul-3d.ll` | 3D×3D element-wise mul → `binary.mul.3d.3d.3d.f32` |

**Note:** Existing contract test CHECK lines must be updated:
- `contract.1d.2d.f32` → `contract.1d.2d.1d.f32`
- `contract.2d.2d.f32` → `contract.2d.2d.2d.f32`
- `contract.3d.3d.f32` → `contract.3d.3d.3d.f32`

Each test checks:
1. Correct intrinsic name is emitted
2. Stride=0 broadcast convention on broadcast dims
3. For dot product: no stride params in call (RankC=0)
4. Signature matches the `Intrinsics.td` declaration

---

## 7. Non-Goals

- Runtime implementation of the emitted intrinsics (backend codegen is out of scope)
- Rank > 4 per operand
- Multiple contraction dimensions
- Ternary or higher-arity tensor ops
- In-place / aliasing operand support
