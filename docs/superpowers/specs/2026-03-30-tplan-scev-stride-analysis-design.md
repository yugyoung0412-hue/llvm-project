# TPlan SCEV-Based Dynamic Stride Analysis

**Date:** 2026-03-30
**Branch:** LoopTensorizebyClaude
**Status:** Design approved, pending implementation

---

## Problem

`getTPValueStrides()` always returns dense defaults. `TPSingleDefRecipe::MemStrides`
(`DenseMap<unsigned, uint64_t>`) is never populated. As a result, stride arguments
passed to tensor intrinsics (`llvm.tensor.matmul`, `llvm.tensor.elementwise`) are
always computed as products of partition factors — correct only for contiguous/packed
tensors. Non-dense layouts (submatrices, padded rows, transposed views) silently
produce wrong leading-dimension arguments.

---

## Goals

1. Populate `MemStrides` with SCEV-derived strides during pattern matching.
2. Support dynamic (runtime) strides via `Value*` — not just compile-time `uint64_t`.
3. Change tensor intrinsic stride args from `ConstantInt*` to runtime `Value*`.
4. Fall back silently to dense defaults when SCEV analysis cannot determine a stride.

---

## Pipeline

```
Widening → Pattern Matching (+ SCEV stride analysis) → Lowering (execute())
                     ↑                                        ↑
           Populate MemStrides                    SCEVExpander → Value*
           with const SCEV* per dim              passed to intrinsics
```

No new passes. SCEV analysis is woven into the existing pattern matcher.
The dense fallback path survives as a SCEV constant — same code path as a
dynamic stride.

---

## Section 1: Data Model Changes

### `TPSingleDefRecipe::MemStrides` (TPlan.h)

```cpp
// Before:
DenseMap<unsigned, uint64_t> MemStrides;

// After:
DenseMap<unsigned, const SCEV *> MemStrides;
```

### `TPSingleDefRecipe::getMemStride()` (TPlan.h)

Gains `ScalarEvolution &SE` to construct the dense fallback as a SCEV constant:

```cpp
// Before:
uint64_t getMemStride(unsigned Dim, const TPlan &Plan) const;

// After:
const SCEV *getMemStride(unsigned Dim, const TPlan &Plan,
                          ScalarEvolution &SE) const;

// Implementation:
inline const SCEV *TPSingleDefRecipe::getMemStride(
    unsigned Dim, const TPlan &Plan, ScalarEvolution &SE) const {
  auto It = MemStrides.find(Dim);
  if (It != MemStrides.end())
    return It->second;
  return SE.getConstant(APInt(64, Plan.getDenseStrideForDim(Dim)));
}
```

### `getTPValueStrides()` (TPRecipeMatcher.h / .cpp)

```cpp
// Before:
SmallVector<uint64_t> getTPValueStrides(const TPSingleDefRecipe &V,
                                         const TPlan &Plan);
// After:
SmallVector<const SCEV *> getTPValueStrides(const TPSingleDefRecipe &V,
                                             const TPlan &Plan,
                                             ScalarEvolution &SE);
```

Implementation: same loop over `V.DimSet`, calls updated `getMemStride()`.

### `TPWidenStoreRecipe` extension (TPlan.h)

Store recipes currently have no `MemStrides` or `DimSet`. Add both to enable
strided-C support in matmul and elementwise lowering (resolves existing TODO):

```cpp
SmallBitVector DimSet;                        // copied from stored-value operand's DimSet
DenseMap<unsigned, const SCEV *> MemStrides;  // populated same as load recipes
const SCEV *getMemStride(unsigned Dim, const TPlan &Plan,
                          ScalarEvolution &SE) const;
```

`getDenseStrideForDim()` on `TPlan` remains `uint64_t`-based — it is only used
to construct SCEV constants at the boundary and is not otherwise changed.

---

## Section 2: SCEV Analysis in Pattern Matcher

### Signature change

```cpp
// Before:
void TPRecipePatternMatcher_match(const TPlan &Plan, RecipeClassMap &Out);

// After:
void TPRecipePatternMatcher_match(const TPlan &Plan, RecipeClassMap &Out,
                                   ScalarEvolution &SE, LoopInfo &LI);
```

Both `SE` and `LI` are already available at the `TPlanLowering_lower()` call site.

### Step 1 — Build `DimToLoop` mapping

Before classifying recipes, scan IV recipes in the TPlan header block:

```cpp
DenseMap<unsigned, Loop *> DimToLoop;
for (auto &R : *Plan.getHeader()) {
  if (auto *IV = dyn_cast<TPWidenInductionRecipe>(&R)) {
    auto *Phi = cast<PHINode>(IV->getInstruction());
    Loop *L = LI.getLoopFor(Phi->getParent());
    DimToLoop[IV->getDimIdx()] = L;
  }
}
```

### Step 2 — `populateSCEVStrides` helper

Static helper in TPRecipeMatcher.cpp. Walks the nested `SCEVAddRecExpr` chain
and maps each loop's step to a TPlan dimension:

```cpp
static void populateSCEVStrides(TPSingleDefRecipe &R, Value *Ptr,
                                 const DenseMap<unsigned, Loop *> &DimToLoop,
                                 ScalarEvolution &SE) {
  const SCEV *S = SE.getSCEV(Ptr);
  DenseMap<const Loop *, const SCEV *> LoopStep;
  while (const auto *AR = dyn_cast<SCEVAddRecExpr>(S)) {
    LoopStep[AR->getLoop()] = AR->getStepRecurrence(SE);
    S = AR->getStart();
  }
  for (int D = R.DimSet.find_first(); D >= 0; D = R.DimSet.find_next(D)) {
    auto DIt = DimToLoop.find(static_cast<unsigned>(D));
    if (DIt == DimToLoop.end()) continue;
    auto SIt = LoopStep.find(DIt->second);
    if (SIt != LoopStep.end())
      R.MemStrides[D] = SIt->second;
    // Absent → getMemStride() falls back to dense SCEV constant.
  }
}
```

An overload handles `TPWidenStoreRecipe` the same way (store recipes are not
`TPSingleDefRecipe`, so a separate overload is needed):

```cpp
static void populateSCEVStrides(TPWidenStoreRecipe &SR, Value *Ptr,
                                 const DenseMap<unsigned, Loop *> &DimToLoop,
                                 ScalarEvolution &SE) {
  // Same body as above, operating on SR.MemStrides and SR.DimSet.
}
```

### Step 3 — Call sites in the matcher

After recipe classification, for each `TPWidenLoadRecipe`:
```cpp
Value *Ptr = cast<LoadInst>(LR->getInstruction())->getPointerOperand();
populateSCEVStrides(*LR, Ptr, DimToLoop, SE);
```

For each `TPWidenStoreRecipe`:
```cpp
Value *Ptr = cast<StoreInst>(SR->getInstruction())->getPointerOperand();
// Copy DimSet from stored-value operand, then populate strides.
if (auto *ValDR = dyn_cast<TPSingleDefRecipe>(SR->getOperand(1)))
  SR->DimSet = ValDR->DimSet;
populateSCEVStrides(*SR, Ptr, DimToLoop, SE);  // overload for store recipe
```

---

## Section 3: Lowering — Materialization & Intrinsic Changes

### `TPTransformState` gains `SE` and `SCEVExpander`

```cpp
ScalarEvolution *SE = nullptr;     // set before execute() loop
SCEVExpander *Expander = nullptr;  // set before execute() loop
```

Constructed once at the start of `TPlanLowering_lower()`:

```cpp
SCEVExpander Expander(SE, F.getParent()->getDataLayout(), "tplan.stride");
State.SE = &SE;
State.Expander = &Expander;
```

`State.SE` is used by `getTPValueStrides()` calls inside `execute()` methods.

### Call sites updated

Every call site that was:
```cpp
SmallVector<uint64_t> Strides = getTPValueStrides(*DR, State.Plan);
Args.push_back(B.getInt64(Strides[i]));
```

becomes:
```cpp
SmallVector<const SCEV *> Strides = getTPValueStrides(*DR, State.Plan, SE);
Value *SV = State.Expander->expandCodeFor(
    Strides[i], B.getInt64Ty(), &*B.GetInsertPoint());
Args.push_back(SV);
```

For constant SCEV (dense default), `SCEVExpander` emits a `ConstantInt` — zero
runtime overhead. For non-trivial SCEV, it emits minimal IR before the call.

The hardcoded `LDC` in the matmul path also migrates to this pattern, reading
from the store recipe's `MemStrides` via `getMemStride()`.

### Intrinsic signatures

`llvm.tensor.matmul` and `llvm.tensor.elementwise` stride args were always `i64`.
They remain `i64` — but callers now pass runtime `Value*` instead of `ConstantInt*`.
No tablegen or `.td` changes needed (declared programmatically via
`getOrInsertFunction`). The only change is at call sites.

---

## Section 4: Error Handling & Fallback

### SCEV analysis fails (no AddRec)

If `SE.getSCEV(Ptr)` does not yield an `SCEVAddRecExpr` for a dimension
(e.g., pointer is a function argument, select, or non-affine expression),
`MemStrides[D]` is left unset. `getMemStride()` returns the dense SCEV constant.
Silent, correct, conservative.

### `SCEVExpander` cannot expand

Guard before expansion:

```cpp
if (!Expander.isSafeToExpand(Strides[i]))
  SV = B.getInt64(Plan.getDenseStrideForDim(D));
else
  SV = Expander->expandCodeFor(Strides[i], B.getInt64Ty(), &*B.GetInsertPoint());
```

Falls back to dense constant. Correct in all cases.

### Invariant

`getDenseStrideForDim()` on `TPlan` is never removed or changed. It remains the
authoritative source for the dense default, wrapped into SCEV at the boundary.
No mixed-type confusion: `uint64_t` lives only inside `getDenseStrideForDim()`.

---

## Files Changed

| File | Change |
|------|--------|
| `llvm/include/llvm/Transforms/Vectorize/TPlan.h` | `MemStrides` type, `getMemStride()` signature, `TPWidenStoreRecipe` extension |
| `llvm/include/llvm/Transforms/Vectorize/TPRecipeMatcher.h` | `getTPValueStrides()` signature, `TPRecipePatternMatcher_match()` signature |
| `llvm/lib/Transforms/Vectorize/TPRecipeMatcher.cpp` | `populateSCEVStrides()` helper, `getTPValueStrides()` impl, matcher body |
| `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` | `SCEVExpander` in state, all stride call sites, `LDC` migration |
