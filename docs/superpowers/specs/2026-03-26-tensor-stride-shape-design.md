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

A naive `load <M*N x float>` includes the padding — incorrect data. Stride-aware load intrinsics (`@llvm.matrix.column.major.load`) accept a runtime `i64` stride and handle this correctly.

### 2.2 Two-level stride storage

Stride information lives at two granularities:

#### `TPlan` — dim-level dense stride (shared default)

`TPlan` stores the **dense stride** for each loop dimension: the stride a tensor *would* have if its memory were packed (no padding). This is derived from the PF (tile size) of inner dimensions:

```
denseStride(dim D) = product of PFs of all dims with index > D
```

Example for dims `{0,1,2}` with PFs `[256, 512, 1024]`:

| Dim | PF | Dense stride |
|-----|----|-------------|
| 0   | 256 | 512 × 1024 = 524288 |
| 1   | 512 | 1024 |
| 2   | 1024 | 1 |

New API:
```cpp
// In TPlan:
DenseMap<unsigned, TPValue *> DimDenseStride;
TPValue *getDenseStrideForDim(unsigned D) const;
```

#### `TPSingleDefRecipe` — per-tensor stride override

Each tensor's actual memory layout may differ from the dense default (e.g., `lda > N` due to padding, or a runtime dynamic stride from a dynamic batch dimension). The recipe stores overrides only when the actual stride differs:

```cpp
// In TPSingleDefRecipe:
SmallDenseMap<unsigned, TPValue *> MemStrides;
// nullptr entry → inherit TPlan dense default
// TPValue* entry → override (static constant or dynamic SSA value)

TPValue *getMemStride(unsigned Dim, const TPlan &Plan) const;
// Returns MemStrides[Dim] if set, else Plan.getDenseStrideForDim(Dim)
```

#### Concrete GEMM example

```c
// A[256][512] lda=640, B[512][1024] ldb=1024, C[256][1024] ldc=1280
for (int i = 0; i < 256; i++)     // dim 0
  for (int k = 0; k < 512; k++)   // dim 1 (reduction)
    for (int j = 0; j < 1024; j++) // dim 2
      C[i*ldc + j] += A[i*lda + k] * B[k*ldb + j]
```

| Recipe | Dim | TPlan dense default | Actual stride | Stored in recipe? |
|--------|-----|--------------------|--------------:|:-----------------:|
| A load | 0   | 512                | lda = 640     | yes               |
| A load | 1   | 1                  | 1             | no                |
| B load | 1   | 1                  | 1             | no                |
| B load | 2   | 1                  | ldb = 1024    | no (matches)      |
| C store| 0   | 512                | ldc = 1280    | yes               |
| C store| 2   | 1                  | 1             | no                |

---

## 3. New Intrinsic Family

Two new target-independent intrinsics replace the flat-vector `@llvm.matrix.multiply` as the canonical output of TPlan lowering.

### 3.1 `@llvm.tensor.matmul`

All dimensions and strides are `i64` SSA values — fully dynamic:

```llvm
declare void @llvm.tensor.matmul.f32(
    ptr %C, i64 %M, i64 %N, i64 %ldc,
    ptr %A, i64 %M, i64 %K, i64 %lda,
    ptr %B, i64 %K, i64 %N, i64 %ldb)
```

### 3.2 `@llvm.tensor.elementwise.<op>.<rank>d`

Rank is encoded in the intrinsic name so the number of stride/dim arguments is statically known to the backend. Example for 2D `fadd`:

```llvm
declare void @llvm.tensor.elementwise.fadd.2d.f32(
    ptr %C, i64 %strideC0, i64 %strideC1,
    ptr %A, i64 %strideA0, i64 %strideA1,
    ptr %B, i64 %strideB0, i64 %strideB1,
    i64 %D0, i64 %D1)
```

For 3D:

```llvm
declare void @llvm.tensor.elementwise.fadd.3d.f32(
    ptr %C, i64 %strideC0, i64 %strideC1, i64 %strideC2,
    ptr %A, i64 %strideA0, i64 %strideA1, i64 %strideA2,
    ptr %B, i64 %strideB0, i64 %strideB1, i64 %strideB2,
    i64 %D0, i64 %D1, i64 %D2)
```

---

## 4. Backend Lowering Decision Tree

`@llvm.tensor.matmul` is a **target-independent semantic anchor**. The backend selects an implementation based on what the target supports and whether dims/strides are constant at compile time:

```
@llvm.tensor.matmul(M, K, N, lda, ldb, ldc)
        │
        ├─ all dims + strides constant
        │       └─→ @llvm.matrix.multiply  (existing path, unchanged)
        │
        ├─ dims constant, strides dynamic
        │       └─→ @llvm.matrix.column.major.load
        │           + @llvm.matrix.multiply
        │
        └─ any dim dynamic
                ├─ ARM SME available    → @llvm.aarch64.sme.mopa
                ├─ Custom NPU available → target-specific dynamic intrinsic
                └─ generic fallback     → @cblas_sgemm or tiled scalar loop
```

### 4.1 LLM workload classification

| Scenario | Dims static? | Strides static? | Lowering path |
|----------|:-----------:|:---------------:|---------------|
| Weight matrices (fixed hidden dim) | yes | yes | `@llvm.matrix.multiply` |
| Fixed hidden, padded alloc | yes | no | stride-aware load + `@llvm.matrix.multiply` |
| Dynamic batch / seq_len | no | no | SME / NPU / BLAS / loop |
| Attention head slicing | no | no | SME / NPU / BLAS / loop |

---

## 5. `getTPValueShape` and `getTPValueStrides` — Generalization

### 5.1 Existing `getTPValueShape` (unchanged signature)

```cpp
// TPRecipeMatcher.h — unchanged
SmallVector<unsigned> getTPValueShape(const TPSingleDefRecipe &V,
                                      const TPlan &Plan);
// Returns PFForDim for each set bit in V.DimSet, in dim order.
// Already N-dimensional by implementation.
```

### 5.2 New companion `getTPValueStrides`

```cpp
// TPRecipeMatcher.h — new
SmallVector<TPValue *> getTPValueStrides(const TPSingleDefRecipe &V,
                                          const TPlan &Plan);
// For each dim D in V.DimSet (in order):
//   returns V.getMemStride(D, Plan)
//   → V.MemStrides[D] if set (recipe-level override)
//   → Plan.getDenseStrideForDim(D) otherwise (dense default)
```

### 5.3 Call sites expand to all TensorOpKinds

Currently only `emitContraction` calls `getTPValueShape`. After this change, every non-scalar `execute()` path calls both:

```cpp
// In TPWidenRecipe::execute(), for every non-Scalar TensorOpKind:
SmallVector<unsigned>  Shape   = getTPValueShape(*DR, State.Plan);
SmallVector<TPValue *> Strides = getTPValueStrides(*DR, State.Plan);
// Shape and Strides are passed to the appropriate new intrinsic.
```

---

## 6. IR Examples

### 6.1 ElementWise 2D add — dynamic stride

```c
// C[i][j] = A[i][j] + B[i][j], shape [256][512], lda=640 (padded)
```

```llvm
; Emitted by TPlan lowering:
call void @llvm.tensor.elementwise.fadd.2d.f32(
    ptr %C, i64 512,  i64 1,     ; C strides: dense default
    ptr %A, i64 %lda, i64 1,     ; A stride dim0 = runtime lda (override)
    ptr %B, i64 512,  i64 1,     ; B strides: dense default
    i64 256, i64 512)
```

### 6.2 Contraction — dynamic batch dimension

```c
// C[batch*seq][N] += A[batch*seq][K] * B[K][N], batch is runtime
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
; Backend recognizes all-constant dims → @llvm.matrix.multiply as before
%A_flat = call <131072 x float> @llvm.matrix.column.major.load(
              ptr %A, i64 512, i1 false, i32 256, i32 512)
%B_flat = call <524288 x float> @llvm.matrix.column.major.load(
              ptr %B, i64 1024, i1 false, i32 512, i32 1024)
%result = call <262144 x float> @llvm.matrix.multiply(
              <131072 x float> %A_flat,
              <524288 x float> %B_flat,
              i32 256, i32 512, i32 1024)
```

---

## 7. Files Affected

| File | Change |
|------|--------|
| `llvm/include/llvm/Transforms/Vectorize/TPlan.h` | Add `DimDenseStride` map and `getDenseStrideForDim()` to `TPlan`; add `MemStrides` map and `getMemStride()` to `TPSingleDefRecipe` |
| `llvm/include/llvm/Transforms/Vectorize/TPRecipeMatcher.h` | Declare `getTPValueStrides()` |
| `llvm/lib/Transforms/Vectorize/TPRecipeMatcher.cpp` | Implement `getTPValueStrides()`; populate `DimDenseStride` in `TPplan` during widening |
| `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` | Expand `getTPValueShape` + `getTPValueStrides` call sites to all `TensorOpKind`s; emit new intrinsics |
| `llvm/lib/Transforms/Vectorize/TPlanWidener.cpp` | Populate `TPSingleDefRecipe::MemStrides` from SCEV-derived strides during widening |
| `llvm/include/llvm/IR/Intrinsics.td` | Declare `@llvm.tensor.matmul` and `@llvm.tensor.elementwise.*` intrinsic families |
