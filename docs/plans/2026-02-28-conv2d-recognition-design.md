# CONV2D Recognition + Pre-lowering im2col Design

**Date:** 2026-02-28
**Status:** Approved
**Author:** (your name)

---

## 1. Motivation

The `LoopTensorize` pass currently recognizes GEMM patterns and emits
`llvm.matrix.multiply` or AMX intrinsics. `PatternKind::Conv2D` is
declared in the enum but has no implementation. This design adds:

- Pattern recognition for 2D convolution loop nests
- A pre-lowering lowering-strategy decision (im2col vs. direct tiling)
- Code generation for both paths

---

## 2. Goals & Non-Goals

**Goals:**
- Add `isConv2D()` to `TensorPatternClassifier`
- Detect the sliding-window access signature via SCEV coefficient equality
- Choose im2col → GEMM when the col_matrix fits in L2 cache; otherwise
  fall through to direct tiling + vectorize
- Full lit test coverage for recognition, lowering selection, and
  dynamic-shape fallback

**Non-Goals:**
- ARM SME / x86 AMX native conv2d intrinsics (no such LLVM intrinsics exist)
- Runtime shape profiling or ML-based cost models
- CONV2D performance benchmarks (deferred to a follow-on PR)

---

## 3. Pattern Recognition

### 3.1 Detection Criteria

`isConv2D(LoopNestInfo)` returns true when all of the following hold:

1. `Depth >= 4`
2. `IsPerfectNest && IsAffine`
3. Exactly 3 distinct base pointers (`input`, `kernel`, `output`)
4. Exactly 2 reads + 1 write
5. **Sliding-window coefficient equality**: at least one read access has
   two distinct `SCEVAddRecExpr` terms sharing the same constant
   multiplicative coefficient in the linearized pointer SCEV

### 3.2 Why Coefficient Equality

For `input[n][oh+kh][ow+kw][c]` in row-major layout the pointer SCEV is:

```
base + n*(H*W*C) + oh*(W*C) + kh*(W*C) + ow*C + kw*C + c
```

- `{oh}` and `{kh}` share coefficient `W*C`
- `{ow}` and `{kw}` share coefficient `C`

In GEMM `A[i][k]` all IV coefficients are distinct. The duplicate-
coefficient test distinguishes Conv2D from GEMM, elementwise ops, and
generic reductions without false positives. It also handles dilated
convolutions and non-unit strides because it checks equality, not value.

### 3.3 Implementation: `collectMulTerms`

```cpp
// Walk a top-level SCEVAddExpr; for each SCEVMulExpr(constant, AddRec)
// term, group AddRec nodes by their constant factor.
static void collectMulTerms(const SCEV *S,
    DenseMap<APInt, SmallVector<const SCEVAddRecExpr*>, APIntKeyInfo> &Out,
    ScalarEvolution &SE);
```

`isConv2D` iterates `MA.IndexExprs[0]` (the full pointer SCEV stored by
`LoopNestAnalyzer`) for each read access and calls `collectMulTerms`.
If any coefficient maps to two or more distinct `SCEVAddRecExpr` nodes,
the sliding-window signature is confirmed.

---

## 4. Lowering Decision

Runs in `LoopTensorize.cpp` immediately after `classifyPattern()` returns
`PatternKind::Conv2D`, before the beam search:

```
col_matrix_bytes = OH * OW * KH * KW * C_in * sizeof(ElemTy)
L2_size          = TTI.getCacheSize(/*Level=*/2)
use_im2col       = col_matrix_bytes <= L2_size
```

Trip counts come from `LoopNestInfo.IVs[i].TripCount` (extracted by
`LoopNestAnalyzer` via SCEV). If any trip count is `SCEVCouldNotCompute`
(dynamic shape), `use_im2col = false` unconditionally — no buffer
allocation for unknown sizes.

---

## 5. Lowering Paths

### 5.1 Path A — im2col → GEMM

When `use_im2col = true`:

1. **Buffer allocation**: insert an `alloca` for the col_matrix in the
   function's entry block.
   `col_matrix: [OH * OW, KH * KW * C_in]` flat array.

2. **im2col loop nest**: emit new loops (N outer, then OH×OW×KH×KW×C_in
   inner) that copy `input[n][oh+kh][ow+kw][c]` into
   `col[n][oh*OW + ow][kh*KW*C_in + kw*C_in + c]`.

3. **Rewrite original nest**: replace the input read pointer with a read
   from `col_matrix`. The loop nest now has the 3-pointer, 2-read-1-write
   structure of a GEMM.

4. **Delegate to existing GEMM path**: run `classifyPattern()` again on
   the rewritten `LoopNestInfo`; it returns `PatternKind::GEMM`. The
   normal beam search + `applyPlan()` handles the rest (including
   `llvm.matrix.multiply` emission or AMX if available).

### 5.2 Path B — Direct Tiling + Vectorize

When `use_im2col = false` (large kernel or dynamic shape):

1. Keep `PatternHint.Kind = PatternKind::Conv2D`, `PreferredOpIdx = -1`.
2. Run the normal beam search. `LoopTile` + `LoopPermute` + `Vectorize`
   are the terminal path (same as existing GEMM → `InnerLoopVectorizer`
   fallback).
3. In `applyPlan()`, add:
   ```cpp
   else if (Hint.Kind == PatternKind::Conv2D && !use_im2col) {
     // Invoke InnerLoopVectorizer on the tiled, permuted nest.
     // Same code path as the existing GEMM vectorize fallback.
   }
   ```

---

## 6. Files Changed

| File | Change |
|---|---|
| `llvm/lib/Transforms/Vectorize/TensorPatternClassifier.cpp` | Add `collectMulTerms()`, `isConv2D()`, call site in `classifyPattern()` |
| `llvm/include/llvm/Transforms/Vectorize/TensorPatternClassifier.h` | (no change — `PatternKind::Conv2D` already declared) |
| `llvm/lib/Transforms/Vectorize/LoopTensorize.cpp` | Add col_matrix size estimation + lowering decision before beam search |
| `llvm/lib/Transforms/Vectorize/TensorCodeGen.cpp` | Add `applyPlan()` branch for `Conv2D` direct-tile path |
| `llvm/unittests/Transforms/LoopTensorize/PatternClassifierTest.cpp` | New Conv2D unit test cases |
| `llvm/test/Transforms/LoopTensorize/basic/conv2d-*.ll` | New lit tests (see §7) |

---

## 7. Testing

### 7.1 Unit Tests

New cases in `PatternClassifierTest.cpp`:

| Test | Expected result |
|---|---|
| `isConv2DSmallKernel` | 5-deep nest, 3×3 kernel → `PatternKind::Conv2D` |
| `isConv2DDilated` | Dilated strides → `PatternKind::Conv2D` |
| `notConv2D_GEMM` | 3-deep GEMM nest → `PatternKind::GEMM`, not Conv2D |
| `notConv2D_Elementwise` | Depth=1 → `PatternKind::Elementwise` |
| `collectMulTermsFindsEqualCoeff` | Two AddRec terms with coeff=W*C detected |

### 7.2 Lit Tests

`llvm/test/Transforms/LoopTensorize/basic/`:

| File | What it verifies |
|---|---|
| `conv2d-recognition.ll` | 5-deep nest classified and transformed; no scalar loop remains |
| `conv2d-no-match-gemm.ll` | 3-deep GEMM not misclassified as Conv2D |
| `conv2d-no-match-non-affine.ll` | Indirect index → pass is a no-op |
| `conv2d-im2col-small.ll` | Small kernel (3×3×32) → col_matrix alloca + GEMM emission |
| `conv2d-direct-tile-large.ll` | Large kernel (7×7×512) → no alloca, vectorized loops |
| `conv2d-dynamic-shape.ll` | SCEVCouldNotCompute on OH/OW → direct tile fallback, no alloca |

---

## 8. Open Questions

1. **Col_matrix lifetime**: the `alloca` is in the function entry block.
   For large batch sizes (N >> 1) a per-batch allocation might be
   preferable. Deferred until benchmarks show it matters.
2. **Stride/padding**: strided convolutions add `stride_h * oh` to the
   index; the coefficient equality check still works but the im2col loop
   needs an additional bounds check for padding. Not handled in this PR.
3. **Interaction with inlining**: if the convolution loop is inside an
   inlined function, the im2col alloca stacks up per callsite. Investigate
   in a follow-on.
