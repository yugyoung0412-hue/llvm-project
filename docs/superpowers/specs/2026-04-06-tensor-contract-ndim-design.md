# N-Dim Tensor Contraction Design

**Date:** 2026-04-06
**Scope:** Generalize `emitContraction()` from 2D-only to rank 1–4 per operand;
introduce `llvm.tensor.contract.<Ra>d.<Rb>d.<type>` intrinsic family.
**Status:** Approved

---

## 1. Background

`emitContraction()` in `TPlanLowering.cpp` hard-rejects any operand whose
DimSet rank is not exactly 2:

```cpp
if (LHSDR->DimSet.count() != 2 || RHSDR->DimSet.count() != 2)
  return nullptr;   // scalar fallback
```

Additionally, the shape-extraction code assumes rank-2 operands:

```cpp
uint64_t M = LHSShape[1 - LHSPos];   // only valid for rank-2
uint64_t N = RHSShape[1 - RHSPos];
```

And the current intrinsic `llvm.tensor.matmul.<type>` has a BLAS-style
fixed signature (M, N, K, lda, ldb, ldc) that cannot express rank > 2.

### Cases that currently fall back to scalar

| Pattern | DimSet sizes | Needed for |
|---------|-------------|-----------|
| GEMV `C[j] += A[k] * B[k][j]` | 1D × 2D | FC layer with unbatched input |
| Batched GEMM `C[b,i,j] += A[b,i,k] * B[b,k,j]` | 3D × 3D | Transformer attention batch |
| Multi-head attention `C[b,h,s,sk] += Q[b,h,s,d] * K[b,h,d,sk]` | 4D × 4D | MHA |

---

## 2. New Intrinsic Family: `llvm.tensor.contract`

### 2.1 Naming

```
llvm.tensor.contract.<Ra>d.<Rb>d.<type>
```

- `<Ra>` = rank of operand A (DimSet.count()), 1–4
- `<Rb>` = rank of operand B (DimSet.count()), 1–4
- `<type>` = element type suffix from `getTypeSuffix()` (e.g. `f32`, `f64`)

Examples: `contract.1d.2d.f32`, `contract.2d.2d.f32`, `contract.3d.3d.f32`,
`contract.4d.4d.f32`.

**`llvm.tensor.matmul.<type>` is kept unchanged** for the OuterProduct
lowering path. Only the Contraction path switches to `tensor.contract`.

### 2.2 Output DimSet

```
OutputDimSet = (A.DimSet | B.DimSet) − {ContractDim}
RankC        = OutputDimSet.count()          // 1 ≤ RankC ≤ Ra + Rb − 2
```

### 2.3 Signature

```
void(
  ptr   C,
  i64×RankC  C_strides,          // C stride per output dim (output order)

  ptr   A,
  i64×RankC  A_strides,          // A stride per output dim; 0 if A ∌ that dim
  i64        A_contract_stride,  // A stride along ContractDim

  ptr   B,
  i64×RankC  B_strides,          // B stride per output dim; 0 if B ∌ that dim
  i64        B_contract_stride,  // B stride along ContractDim

  i64        K,                  // PF for ContractDim
  i64×RankC  output_dims         // PF per output dim
)
```

Total parameters = 3 ptrs + 4·RankC i64s + 3 i64s (A_cstride, B_cstride, K).

### 2.4 Output-Order Stride Convention

Strides for A and B are listed in **output dim order** (DimSet bits of
OutputDimSet iterated via `find_first()` to `find_last()`).

- If A **spans** output dim D → put `A.getMemStride(D)` (expanded to i64)
- If A **does not span** output dim D → put `i64 0` (**broadcast**,
  consistent with the BroadcastBinary stride-0 convention)

This makes the runtime loop trivial:

```python
for each output_dim i:
    C_ptr  += out_idx[i] * C_strides[i]
    A_ptr  += out_idx[i] * A_strides[i]   # 0 → A doesn't move along this dim
    B_ptr  += out_idx[i] * B_strides[i]   # 0 → B doesn't move along this dim
for k in [0, K):
    acc += A_ptr[k * A_contract_stride] * B_ptr[k * B_contract_stride]
```

### 2.5 Shared Free Dims (ggml compatibility)

When A and B both span a free dim (e.g. batch dim `b` in batched GEMM),
both `A_strides[b]` and `B_strides[b]` are non-zero. The runtime advances
both pointers along `b`. This naturally matches ggml's batch-broadcast model:
if one operand has `ne[2]=1`, its batch stride would be 0 (broadcast), while
the other has a real stride — identical to our stride-0 convention.

---

## 3. Concrete Examples

### 3.1 GEMV — `contract.1d.2d.f32`

Loop: `j` outer (dim 1), `k` inner + reduction (dim 0).

```
A:  DimSet={0}           RankA=1
B:  DimSet={0,1}         RankB=2
OutputDimSet = {1}       RankC=1
```

Call (output dim order = [j]):

```
@llvm.tensor.contract.1d.2d.f32(
  ptr C,   i64 1,           // C_strides[j]=1
  ptr A,   i64 0,  i64 1,  // A_strides[j]=0,  A_contract_stride=1
  ptr B,   i64 1,  i64 %N, // B_strides[j]=1,  B_contract_stride=%N
  i64 256,                  // K (PF for dim 0)
  i64 256                   // output_dims[j] (PF for dim 1)
)
```

### 3.2 GEMM — `contract.2d.2d.f32`  (replaces `tensor.matmul`)

Loop: `i` outer (dim 2), `j` middle (dim 1), `k` inner + reduction (dim 0).

```
A:  DimSet={0,2}         RankA=2
B:  DimSet={0,1}         RankB=2
OutputDimSet = {1,2}     RankC=2
```

Call (output dim order = [j, i]):

```
@llvm.tensor.contract.2d.2d.f32(
  ptr C,   i64 1,   i64 %N,           // C_strides[j]=1, C_strides[i]=%N
  ptr A,   i64 0,   i64 %K,  i64 1,  // A_strides[j]=0, A_strides[i]=%K, A_cs=1
  ptr B,   i64 1,   i64 0,   i64 %N, // B_strides[j]=1, B_strides[i]=0,  B_cs=%N
  i64 256,                             // K
  i64 256, i64 256                     // output_dims[j,i]
)
```

### 3.3 Batched GEMM — `contract.3d.3d.f32`

Loop: `b` outer (dim 3), `i` (dim 2), `j` (dim 1), `k` inner + reduction (dim 0).

```
A:  DimSet={0,2,3}       RankA=3
B:  DimSet={0,1,3}       RankB=3
OutputDimSet = {1,2,3}   RankC=3
Shared free dim: 3 (b)
```

Call (output dim order = [j, i, b]):

```
@llvm.tensor.contract.3d.3d.f32(
  ptr C, i64 1, i64 %N, i64 %IN,           // C_strides[j,i,b]
  ptr A, i64 0, i64 %K, i64 %IK,  i64 1,  // A_strides[j,i,b]=0/%K/%IK, A_cs=1
  ptr B, i64 1, i64 0,  i64 %KN,  i64 %N, // B_strides[j,i,b]=1/0/%KN,  B_cs=%N
  i64 256,                                  // K
  i64 256, i64 256, i64 256                 // output_dims[j,i,b]
)
```

Note `A_strides[j]=0` (A∌j) and `B_strides[i]=0` (B∌i).

---

## 4. Implementation Changes

### 4.1 New helper: `getTensorContractFn()`

```cpp
static FunctionCallee getTensorContractFn(Module &M, unsigned RankA,
                                           unsigned RankB, unsigned RankC,
                                           Type *ElemTy);
```

Builds the intrinsic name from `RankA`, `RankB`, `getTypeSuffix(ElemTy)`.
Constructs a `FunctionType` with `3 + 4·RankC + 3` parameters (see §2.3).

### 4.2 Rewrite `emitContraction()`

Replace the existing body with:

1. Check `RankA, RankB ∈ [1, 4]` (replaces `count() != 2` guard)
2. Compute `OutputDimSet`, `RankC`; check `RankC ∈ [1, 4]`
3. Find C pointer (same fallback chain as before)
4. For each dim D in OutputDimSet (find_first order):
   - `C_strides[D]` = `expandStride(CStoreRecipe->getMemStride(D), D)` or dense default
   - `A_strides[D]` = `A.DimSet.test(D) ? expandStride(A.getMemStride(D), D) : i64(0)`
   - `B_strides[D]` = `B.DimSet.test(D) ? expandStride(B.getMemStride(D), D) : i64(0)`
   - `output_dims[D]` = `i64(Plan.getPFForDim(D))`
5. `A_contract_stride` = `expandStride(A.getMemStride(ContractDim), ContractDim)`
6. `B_contract_stride` = `expandStride(B.getMemStride(ContractDim), ContractDim)`
7. `K` = `i64(Plan.getPFForDim(ContractDim))`
8. Call `getTensorContractFn(*Mod, RankA, RankB, RankC, ElemTy)` with args assembled above

### 4.3 Element-type restriction lifted

The old code restricts to `float` and `double`. `getTypeSuffix()` already
handles all supported types. The new code uses `getTypeSuffix()` and falls
back to scalar on empty return.

---

## 5. Files Affected

| File | Change |
|------|--------|
| `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` | Add `getTensorContractFn()`; rewrite `emitContraction()` |
| `llvm/test/Transforms/LoopTensorize/basic/tplan-strided-matmul.ll` | Update CHECK: `matmul` → `contract.2d.2d` |
| `llvm/test/Transforms/LoopTensorize/basic/tensor-matmul-emit.ll` | Update CHECK: `matmul` → `contract.2d.2d` |
| `llvm/test/Transforms/LoopTensorize/basic/matrix-multiply-emit.ll` | Update CHECK: `matmul` → `contract.2d.2d` |
| `llvm/test/Transforms/LoopTensorize/basic/tensor-contract-gemv.ll` | New: 1D×2D GEMV test |
| `llvm/test/Transforms/LoopTensorize/basic/tensor-contract-batched.ll` | New: 3D×3D batched GEMM test |

---

## 6. Non-Goals

- `llvm.tensor.matmul` removal — kept for OuterProduct path
- Rank > 4 support (combinatorial explosion; not needed for known AI workloads)
- Multiple contraction dimensions (not classifiable by current `TPRecipePatternMatcher`)
- Runtime implementation of the new intrinsics (backend codegen is out of scope)
- 4D×4D multi-head attention test (deferred; requires 5-level loop nest in lit test)
