# Tensor Stride & Shape Generalization Design

**Date:** 2026-03-26
**Branch:** LoopTensorizebyClaude
**Status:** Approved

---

## 1. Problem Statement

`getTPValueShape()` is currently called only inside `emitContraction()` for 2D matrix multiply. All other `TensorOpKind` lowering paths (`ElementWise`, `BroadcastBinary`, `OuterProduct`, `PlainReduction`) ignore shape entirely and fall back to scalar instruction cloning. Additionally:

- The existing approach flattens tensors to bare vectors, implicitly assuming dense (no-stride) memory layout.
- Real LLM training workloads have dynamic batch and sequence dimensions, so strides are not always compile-time constants.
- `@llvm.matrix.multiply` requires constant `M`, `K`, `N` — unusable when any dimension is dynamic.

**Goals:**
1. Use `getTPValueShape()` (and a new companion `getTPValueStrides()`) for **all** `TensorOpKind`s.
2. Support **N-dimensional** tensors, not just 2D.
3. Handle **dynamic strides and dimensions** correctly for LLM workloads.

---

## 2. Memory Stride Model

### 2.1 Why flat-vector lowering is insufficient

A 2D matrix `A[M][N]` stored with row stride `S > N` (alignment padding) has this layout:

```
Row 0:  [a00 .. a0(N-1) | PAD .. PAD]   ← S elements total
Row 1:  [a10 .. a1(N-1) | PAD .. PAD]
```

A naive `load <M*N x float>` includes the padding — incorrect data.
`@llvm.matrix.row.major.load` (row-major convention, matching C arrays) accepts a runtime `i64` stride and handles this correctly.

### 2.2 DimIdx convention

TPlan uses **innermost-first** dim indexing (matching `DimPFMap` in `TPlan.h`):

- **dim 0** = innermost loop (fastest-varying index, unit stride for packed data)
- **dim D** = Dth loop from the innermost

All stride and shape computations in this spec follow this convention.

### 2.3 Two-level stride storage

Stride information lives at two granularities:

#### `TPlan` — dim-level dense stride (shared default)

`TPlan` stores the **dense stride** for each loop dimension: the stride a tensor *would* have if its memory were packed (no padding). Under the innermost-first convention this is:

```
denseStride(dim D) = product of PFs of all dims with index < D
```

Example for dims `{0,1,2}` with PFs `[1024, 512, 256]`
(dim 0 = innermost/j, dim 1 = middle/k, dim 2 = outermost/i):

| Dim | Role | PF | Dense stride |
|-----|------|----|-------------|
| 0   | j (innermost) | 1024 | 1 |
| 1   | k (middle)    | 512  | 1024 |
| 2   | i (outermost) | 256  | 512 × 1024 = 524288 |

New API:
```cpp
// In TPlan:
DenseMap<unsigned, TPValue *> DimDenseStride;
TPValue *getDenseStrideForDim(unsigned D) const;
// Returns compile-time constant when all inner-dim PFs are constant;
// returns a symbolic TPValue* when any inner-dim PF is dynamic.
```

**Population:** `TPlanWidener_widen()` computes and stores `DimDenseStride` after all DimSets are propagated, as a product of inner-dim PFs.

#### `TPSingleDefRecipe` — per-tensor stride override (load/store recipes only)

Each load or store recipe's actual memory layout may differ from the dense default (e.g., `lda > N` due to padding, or a runtime dynamic stride from a dynamic batch dimension). The recipe stores overrides only for dims where the actual stride differs from the dense default:

```cpp
// In TPSingleDefRecipe:
DenseMap<unsigned, TPValue *> MemStrides;
// No entry for dim D → inherit Plan.getDenseStrideForDim(D)
// TPValue* entry for dim D → override (static constant or dynamic SSA value)

// Query: returns override if present, else TPlan dense default
TPValue *getMemStride(unsigned Dim, const TPlan &Plan) const;
```

`MemStrides` is populated during `TPlanWidener_widen()` by inspecting the SCEV expression of each GEP's index coefficient for each loop dimension. If the coefficient differs from the corresponding `DimDenseStride`, it is stored as an override.

#### Concrete GEMM example

```c
// Outermost→innermost: i(dim2), k(dim1), j(dim0)
// A[256][512] lda=640, B[512][1024] ldb=1024, C[256][1024] ldc=1280
for (int i = 0; i < 256; i++)      // dim 2 (outermost)
  for (int k = 0; k < 512; k++)    // dim 1
    for (int j = 0; j < 1024; j++) // dim 0 (innermost)
      C[i*ldc + j] += A[i*lda + k] * B[k*ldb + j]
```

PFs: dim0=1024, dim1=512, dim2=256.
Dense strides: dim0=1, dim1=1024, dim2=512×1024=524288.

| Recipe  | Dim | Dense default | Actual stride | Stored in recipe? |
|---------|-----|--------------|:-------------:|:-----------------:|
| A load  | 2   | 524288       | lda×1024=655360 | yes |
| A load  | 0   | 1            | 1             | no |
| B load  | 1   | 1024         | ldb=1024      | no (matches) |
| B load  | 0   | 1            | 1             | no |
| C store | 2   | 524288       | ldc×1024=1310720 | yes |
| C store | 0   | 1            | 1             | no |

---

## 3. New Intrinsic Family

Two new target-independent intrinsics replace the flat-vector `@llvm.matrix.multiply` as the canonical output of TPlan lowering. All dimensions and strides are `i64` SSA values — fully dynamic.

### 3.1 `@llvm.tensor.matmul`

```llvm
; All dimensions and strides are i64 SSA values
declare void @llvm.tensor.matmul.f32(
    ptr %C, i64 %rows_C, i64 %cols_C, i64 %ldc,
    ptr %A, i64 %rows_A, i64 %cols_A, i64 %lda,
    ptr %B, i64 %rows_B, i64 %cols_B, i64 %ldb)
; Semantics: C[rows_C x cols_C] += A[rows_A x cols_A] * B[rows_B x cols_B]
; where cols_A == rows_B (the contraction dimension K).
; All tensors are row-major with their respective leading dimensions.
```

### 3.2 `@llvm.tensor.elementwise.<op>.<rank>d`

Rank is encoded in the intrinsic name so the number of stride/dim arguments is statically known to the backend. Strides precede dimensions in the argument list.

2D `fadd`:
```llvm
declare void @llvm.tensor.elementwise.fadd.2d.f32(
    ptr %C, i64 %strideC0, i64 %strideC1,
    ptr %A, i64 %strideA0, i64 %strideA1,
    ptr %B, i64 %strideB0, i64 %strideB1,
    i64 %D0, i64 %D1)
```

3D `fadd`:
```llvm
declare void @llvm.tensor.elementwise.fadd.3d.f32(
    ptr %C, i64 %strideC0, i64 %strideC1, i64 %strideC2,
    ptr %A, i64 %strideA0, i64 %strideA1, i64 %strideA2,
    ptr %B, i64 %strideB0, i64 %strideB1, i64 %strideB2,
    i64 %D0, i64 %D1, i64 %D2)
```

---

## 4. Backend Lowering Decision Trees

### 4.1 `@llvm.tensor.matmul`

`@llvm.tensor.matmul` is a **target-independent semantic anchor**. The backend selects an implementation based on what the target supports and whether dims/strides are constant:

```
@llvm.tensor.matmul(rows_C, cols_C, ldc, rows_A, cols_A, lda, rows_B, cols_B, ldb)
        │
        ├─ all dims + strides constant
        │       └─→ @llvm.matrix.row.major.load (×2)
        │           + @llvm.matrix.multiply (existing path)
        │           + @llvm.matrix.row.major.store
        │
        ├─ dims constant, strides dynamic
        │       └─→ @llvm.matrix.row.major.load with runtime stride
        │           + @llvm.matrix.multiply
        │           + @llvm.matrix.row.major.store with runtime stride
        │
        └─ any dim dynamic
                ├─ ARM SME available    → @llvm.aarch64.sme.mopa (scalable tiles)
                ├─ Custom NPU available → target-specific dynamic matmul intrinsic
                └─ generic fallback     → @cblas_sgemm or tiled scalar loop
```

### 4.2 `@llvm.tensor.elementwise.*`

```
@llvm.tensor.elementwise.fadd.<rank>d(...)
        │
        ├─ all dims + strides constant
        │       └─→ flat <N x float> vector fadd (N = product of all dims)
        │
        ├─ dims constant, strides dynamic
        │       └─→ strided gather + vector fadd + strided scatter
        │
        └─ any dim dynamic
                ├─ target supports scalable vectors (SVE/RVV)
                │       └─→ scalable vector fadd with vsetvli
                └─ generic fallback → scalar loop
```

### 4.3 LLM workload classification

| Scenario | Dims static? | Strides static? | Lowering path |
|----------|:-----------:|:---------------:|---------------|
| Weight matrices (fixed hidden dim) | yes | yes | `@llvm.matrix.multiply` |
| Fixed hidden dim, padded alloc | yes | no | stride-aware load + `@llvm.matrix.multiply` |
| Dynamic batch / seq_len | no | no | SME / NPU / BLAS / loop |
| Attention head slicing | no | no | SME / NPU / BLAS / loop |

---

## 5. `getTPValueShape` and `getTPValueStrides` — Generalization

### 5.1 Existing `getTPValueShape` (unchanged)

```cpp
// TPRecipeMatcher.h — unchanged
SmallVector<unsigned> getTPValueShape(const TPSingleDefRecipe &V,
                                      const TPlan &Plan);
// Returns PFForDim for each set bit in V.DimSet, in dim order (innermost first).
// Already N-dimensional by implementation.
```

### 5.2 New `getTPValueStrides`

```cpp
// TPRecipeMatcher.h — new
SmallVector<TPValue *> getTPValueStrides(const TPSingleDefRecipe &V,
                                          const TPlan &Plan);
// For each dim D in V.DimSet (innermost first):
//   returns V.getMemStride(D, Plan)
//   → V.MemStrides[D] if set (recipe-level override)
//   → Plan.getDenseStrideForDim(D) otherwise (dense default)
//
// Only meaningful for load/store recipes (TPWidenLoadSC, TPWidenStoreSC).
// Calling on an arithmetic recipe returns dense defaults, which may be
// incorrect; callers must use the LHS/RHS load recipes for contraction.
```

### 5.3 Resolving `TPValue*` to IR `Value*` at call sites

`getTPValueStrides` returns `TPValue *` — the plan-level representation. At lowering time, the caller resolves each stride to an IR `Value *` via `State.getValue()`:

```cpp
SmallVector<TPValue *> StrideVals = getTPValueStrides(*LoadDR, State.Plan);
SmallVector<Value *>   Strides;
for (TPValue *SV : StrideVals)
  Strides.push_back(State.getValue(SV));  // TPValue* accepted directly
```

`State.getValue()` accepts any `TPValue *` — recipe values, constants, and symbolics alike. Constants (e.g., from `getDenseStrideForDim`) are registered in the `TPTransformState` value map during plan construction and resolve to `ConstantInt` IR values. No cast to `TPSingleDefRecipe` is needed or correct here.

### 5.4 Call sites expand to all TensorOpKinds (load/store operands)

For all non-scalar `TensorOpKind`s, strides are fetched from the **load/store operand recipes**, not from the arithmetic update recipe:

```cpp
// For ElementWise, BroadcastBinary, OuterProduct, PlainReduction:
auto *LoadDR = dyn_cast<TPSingleDefRecipe>(getOperand(0)); // the load recipe
SmallVector<unsigned>  Shape   = getTPValueShape(*LoadDR, State.Plan);
SmallVector<TPValue *> Strides = getTPValueStrides(*LoadDR, State.Plan);

// For Contraction: already uses LHSDR / RHSDR (the mul operands' load recipes)
// — pattern unchanged from existing emitContraction().
```

---

## 6. IR Examples

### 6.1 ElementWise 2D add — mixed static dims, dynamic stride on A

```c
// C[i][j] = A[i][j] + B[i][j]
// dim0=j (innermost, PF=512), dim1=i (outermost, PF=256)
// A: lda=640 (padded) → stride for dim1 = 640*elemSize; B, C: dense (stride=512)
```

```llvm
; Emitted by TPlan lowering — strides are i64 SSA values
; Argument order matches intrinsic signature: strideX0 (dim0/innermost) first.
call void @llvm.tensor.elementwise.fadd.2d.f32(
    ptr %C, i64 1, i64 512,      ; C: strideC0=1 (dim0/j), strideC1=512 (dim1/i)
    ptr %A, i64 1, i64 %lda,     ; A: strideA0=1 (dim0/j), strideA1=lda (dim1/i, override)
    ptr %B, i64 1, i64 512,      ; B: strideB0=1 (dim0/j), strideB1=512 (dim1/i)
    i64 512, i64 256)             ; D0=512 (j), D1=256 (i)
; Backend: dims constant, stride of A dim1 is dynamic
;   → strided gather on A + vector fadd + scatter to C
```

### 6.2 Contraction — dynamic batch dimension

```c
// C[M][N] += A[M][K] * B[K][N], where M = batch * seq_len (runtime)
```

```llvm
%M = mul i64 %batch, %seq_len
call void @llvm.tensor.matmul.f32(
    ptr %C, i64 %M, i64 1024, i64 %ldc,
    ptr %A, i64 %M, i64 4096, i64 %lda,
    ptr %B, i64 4096, i64 1024, i64 1024)
; Backend: any dim dynamic → ARM SME / BLAS / tiled loop
```

### 6.3 Contraction — all static (existing path preserved)

```llvm
; Backend recognizes all-constant dims + strides → @llvm.matrix.multiply
%A_flat = call <131072 x float> @llvm.matrix.row.major.load(
              ptr %A, i64 512, i1 false, i32 256, i32 512)
%B_flat = call <524288 x float> @llvm.matrix.row.major.load(
              ptr %B, i64 1024, i1 false, i32 512, i32 1024)
%result = call <262144 x float> @llvm.matrix.multiply(
              <131072 x float> %A_flat,
              <524288 x float> %B_flat,
              i32 256, i32 512, i32 1024)
call void @llvm.matrix.row.major.store(
              <262144 x float> %result, ptr %C, i64 1024,
              i1 false, i32 256, i32 1024)
```

---

## 7. Files Affected

| File | Change |
|------|--------|
| `llvm/include/llvm/Transforms/Vectorize/TPlan.h` | Add `DimDenseStride` map and `getDenseStrideForDim()` to `TPlan`; add `MemStrides` map and `getMemStride()` to `TPSingleDefRecipe` |
| `llvm/include/llvm/Transforms/Vectorize/TPRecipeMatcher.h` | Declare `getTPValueStrides()` |
| `llvm/lib/Transforms/Vectorize/TPlanWidener.cpp` | After DimSet propagation: compute and store `DimDenseStride` in `TPlan`; populate `TPSingleDefRecipe::MemStrides` from SCEV-derived GEP index coefficients |
| `llvm/lib/Transforms/Vectorize/TPRecipeMatcher.cpp` | Implement `getTPValueStrides()` |
| `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` | Expand `getTPValueShape` + `getTPValueStrides` call sites to all `TensorOpKind`s; emit new intrinsics; resolve `TPValue*` strides via `State.getValue()` |
| `llvm/include/llvm/IR/Intrinsics.td` | Declare `@llvm.tensor.matmul` and `@llvm.tensor.elementwise.*` intrinsic families |
