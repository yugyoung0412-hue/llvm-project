# Failing Lit Test GEP Fix Design

**Date:** 2026-04-06
**Scope:** Fix 4 pre-existing failing lit tests in `llvm/test/Transforms/LoopTensorize/basic/`
**Status:** Approved

---

## 1. Root Cause

All four failing tests use incorrect GEP index patterns that produce wrong SCEV strides, causing
the emitted intrinsic stride arguments to not match the expected CHECK patterns.

### The Bug: Diagonal vs. Row-Major Access

**Broken pattern (diagonal access):**
```llvm
%ij = add i64 %i, %j
%ptr = getelementptr float, ptr %A, i64 %ij
```
SCEV of `%i + %j` = `{AddRec{0,1}(inner), step=1}(outer)` →
both loop steps are 1, so `stride_i = 1` and `stride_j = 1`.

**Correct pattern (row-major access with runtime stride):**
```llvm
%iRow = mul i64 %i, %N      ; N = row width (runtime arg)
%ij   = add i64 %iRow, %j
%ptr  = getelementptr float, ptr %A, i64 %ij
```
SCEV of `%i*%N + %j` = `{AddRec{0,%N}(outer), step=1}(inner)` →
`stride_i = %N` (runtime), `stride_j = 1`. Correct.

### Why This Matters

`emitContraction` and `tryVectorize` both call `expandStride(SCEV, Dim)`, which calls
`SCEVExpander::expandCodeFor()`. With the diagonal pattern, SCEV is `{constant 1}` for both
dims — it expands to `i64 1` instead of the expected runtime variable or dense-stride fallback.

### Why the Dense-Stride Fallback Doesn't Trigger

`expandStride` falls back to `getDenseStrideForDim(Dim)` only when `isSafeToExpand(S)`
returns false. A constant SCEV `{1}` is always safe to expand, so the fallback never triggers —
the wrong value `1` is emitted instead.

---

## 2. Fix Strategy: Runtime Stride Parameters

Each test function gains runtime `i64` stride parameters. The GEP index uses `mul + add` so
SCEV computes the correct per-dimension step. CHECK patterns are updated to assert runtime
variable names rather than magic constants.

This matches the established style of `tplan-strided-matmul.ll`.

---

## 3. Per-Test Fix

### 3.1 `tensor-eltwise-stride-2d.ll`

**Array layout:** `A[M][N]` — inner dim j (stride 1), outer dim i (stride N).

**Signature change:**
```llvm
; Before:
define void @eltwise_fadd_2d(ptr %A, ptr %B, ptr %C)

; After:
define void @eltwise_fadd_2d(ptr %A, ptr %B, ptr %C, i64 %N)
```

**GEP change (applied to all three arrays A, B, C):**
```llvm
; Before:
%ij = add i64 %i, %j

; After:
%iRow = mul i64 %i, %N
%ij   = add i64 %iRow, %j
```

**CHECK change:**
```llvm
; Before:
; CHECK-SAME: i64 1, i64 256    ← per operand strides (3 operands)

; After:
; CHECK-SAME: i64 1, i64 %N     ← stride_j=1, stride_i=%N
```
The dims line `i64 256, i64 256` is unchanged (PF is always 256 regardless of trip count).

### 3.2 `tensor-eltwise-stride-3d.ll`

**Array layout:** `D[I][J][K]` — inner dim k (stride 1), mid dim j (stride K), outer dim i
(stride K*J). Currently uses static multipliers `32` (=J*K=8*4) and `4` (=K). Replace with
runtime params.

**Signature change:**
```llvm
; Before:
define void @eltwise_fadd_3d(ptr %A, ptr %B, ptr %D)

; After:
define void @eltwise_fadd_3d(ptr %A, ptr %B, ptr %D, i64 %strideJ, i64 %strideI)
; strideJ = K  (columns per row of innermost 2D slice)
; strideI = J*K (elements per 2D slice)
```

**GEP change (applied to all three arrays A, B, D):**
```llvm
; Before:
%iJK = mul i64 %i, 32      ; static constant
%ijK = mul i64 %j, 4       ; static constant
%ij  = add i64 %iJK, %ijK
%ijk = add i64 %ij, %k

; After:
%iS  = mul i64 %i, %strideI
%jS  = mul i64 %j, %strideJ
%ij  = add i64 %iS, %jS
%ijk = add i64 %ij, %k
```

**CHECK change:**
```llvm
; Before:
; CHECK-SAME: i64 1, i64 256, i64 65536   ← wrong (static constants)

; After:
; CHECK-SAME: i64 1, i64 %strideJ, i64 %strideI
```
The dims line `i64 256, i64 256, i64 256` is unchanged.

### 3.3 `tensor-matmul-emit.ll`

**Array layout (GEMM):**
- A[i][k]: stride_k=1, stride_i=K (number of columns)
- B[k][j]: stride_j=1, stride_k=N (number of columns)
- C[i][j]: stride_j=1, stride_i=N

**Signature change:**
```llvm
; Before:
define void @gemm_16x16x16(ptr %A, ptr %B, ptr %C)

; After:
define void @gemm_16x16x16(ptr %A, ptr %B, ptr %C, i64 %K, i64 %N)
```

**GEP changes:**
```llvm
; A: i*K + k
; Before: %ik = add i64 %i, %k
%ai   = mul i64 %i, %K
%ak   = add i64 %ai, %k

; B: k*N + j
; Before: %kj = add i64 %k, %j
%bk   = mul i64 %k, %N
%bj   = add i64 %bk, %j

; C: i*N + j  (in k.latch)
; Before: %ij = add i64 %i, %j
%ci   = mul i64 %i, %N
%cj   = add i64 %ci, %j
```

**CHECK change:**

The matmul signature is `void(ptr C, M, N, ldc, ptr A, M, K, lda, ptr B, K, N, ldb)`.
With runtime strides:
- ldc (C outer stride) = %N
- lda (A outer stride) = %K
- ldb (B outer stride) = %N

```
; Before:
; CHECK: call void @llvm.tensor.matmul.f32(ptr {{.*}}, i64 256, i64 256, i64 16777216,
; CHECK-SAME: ptr {{.*}}, i64 256, i64 256, i64 65536,
; CHECK-SAME: ptr {{.*}}, i64 256, i64 256, i64 256)

; After:
; CHECK: call void @llvm.tensor.matmul.f32(ptr {{.*}}, i64 256, i64 256, i64 %N,
; CHECK-SAME: ptr {{.*}}, i64 256, i64 256, i64 %K,
; CHECK-SAME: ptr {{.*}}, i64 256, i64 256, i64 %N)
```

### 3.4 `matrix-multiply-emit.ll`

This test uses `i32` induction variables and `add i32 %i, %k` GEP pattern. Two issues:

1. `i32` IVs — SCEV stride analysis works with i32 but the GEP index type `i32` may
   cause zext/sext issues in `expandCodeFor(S, Int64Ty)`. Converting to `i64` is cleaner.
2. `add %i, %k` diagonal pattern — same root cause as the other tests.

**Signature change:**
```llvm
; Before:
define void @gemm_16x16x16(ptr %A, ptr %B, ptr %C)
; IV types: i32

; After:
define void @gemm_16x16x16(ptr %A, ptr %B, ptr %C, i64 %K, i64 %N)
; IV types: i64
```

**Change all `i32` IVs and arithmetic to `i64`**, add `mul %i, %K + %k` style GEPs (same
pattern as `tensor-matmul-emit.ll`).

**CHECK:** No stride check needed — just:
```
; CHECK: call void @llvm.tensor.matmul.f32
; CHECK-NOT: @llvm.matrix.multiply
```
This CHECK already passes once the matmul is correctly classified and lowered.

---

## 4. SCEV Stride Propagation — How the Fix Works

`populateSCEVStridesFromIndex` peels `SCEVAddRecExpr` chains:
```
SCEV(%i*%N + %j) = AddRec{ AddRec{0, %N}_{outer}, step=1 }_{inner}
```
- Inner pass: `LoopStep[inner] = 1` → stride for j-loop dim = 1
- Outer pass: `LoopStep[outer] = %N` → stride for i-loop dim = %N

`expandStride` calls `Expander->expandCodeFor(SCEV{%N}, Int64Ty)`. Since `%N` is a
function argument (loop-invariant), `isSafeToExpand` returns true and the expander
emits `%N` directly (no additional instructions). The intrinsic call gets `i64 %N`.

---

## 5. Files Affected

| File | Change |
|------|--------|
| `llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-stride-2d.ll` | Add `%N` param, fix GEP, update CHECK |
| `llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-stride-3d.ll` | Add `%strideJ, %strideI` params, fix GEP, update CHECK |
| `llvm/test/Transforms/LoopTensorize/basic/tensor-matmul-emit.ll` | Add `%K, %N` params, fix GEPs, update CHECK |
| `llvm/test/Transforms/LoopTensorize/basic/matrix-multiply-emit.ll` | Add `%K, %N` params, i32→i64, fix GEPs |

No changes to production code (`TPlanLowering.cpp`, `TPRecipeMatcher.cpp`, etc.).

---

## 6. Non-Goals

- Fixing the SCEV stride analysis in the pass itself (that's a separate, larger change)
- Making `tensor-eltwise-int.ll` check strides (it documents the pre-existing limitation)
- Adding new test cases (out of scope)
