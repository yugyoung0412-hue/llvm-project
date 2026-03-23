# PF DimSet System for TPlan — Design Spec

**Date:** 2026-03-23
**Branch:** TPlan (LoopTensorizebyClaude)
**Status:** Draft

---

## 1. Motivation

The existing `TPlanWidener_widen()` propagates a single scalar `ParallelFactor`
(PF) per `TPValue` using BFS with `max()`. This loses information about *which*
loop dimensions a value spans. As a result, the widener cannot distinguish:

- `%b` loaded inside loops `i,k` → shape `[256, 512]`
- `%c` loaded inside loops `k,j` → shape `[512, 1024]`
- `%mul = fmul %b, %c`          → should become a matmul `[256×512] × [512×1024]`

The new PF DimSet system replaces the scalar PF on `TPValue` with a
`SmallBitVector DimSet` that tracks which loop dimension indices a value spans.
The shape of any value is then:

```
shape = [PFMap[d] for d in DimSet]   // in ascending dim-index order
```

Beyond matmul, the system generalises to element-wise ops, broadcast ops, outer
products, and plain reductions — all driven by the same DimSet analysis.

---

## 2. Design Goals

1. **Generality** — handle fmul, fdiv, fsub, outer products, broadcasts uniformly.
2. **Separation of concerns** — shape analysis, pattern detection, and IR emission
   are three distinct stages with no cross-contamination.
3. **Minimal recipe mutation** — no recipe reclassification during widening.
4. **Dumb execute()** — lowering dispatches on a pre-computed classification;
   no analysis inside `execute()`.
5. **Additive extensibility** — new op patterns require adding one rule to the
   matcher; nothing else changes.

---

## 3. Pipeline

```
TPlanBuilder
  LoopNestInfo → scalar TPlan
  (all DimSets empty, PF=1)
        │
TPlanWidener  [extended]
  Phase 1 (existing): scalar PF BFS from header PHIs
  Phase 2 (new):      DimSet BFS — union propagation
        │
TPRecipePatternMatcher  [new pass]
  classifies every recipe by DimSet patterns
  produces RecipeClassMap
        │
TPlanLowering  [extended]
  execute() dispatches on RecipeClassMap
  no pattern detection inside execute()
```

**Invariants:**
| Stage | Owns |
|---|---|
| `TPlanWidener` | Shape analysis only |
| `TPRecipePatternMatcher` | All pattern detection |
| `execute()` | Pure IR emission |

---

## 4. Data Structures

### 4.1 TPValue — DimSet field

```cpp
// Added to TPValue (TPRecipe.h):
SmallBitVector DimSet;   // set of loop dim indices this value spans
```

`DimSet` is added to `TPValue`, which is the base class for all value-defining
recipes via `TPSingleDefRecipe`. Recipes that define no values
(`TPWidenStoreRecipe`, `TPBranchOnCountRecipe`, etc.) inherit only from
`TPRecipeBase`, not `TPValue`, and therefore carry no `DimSet` — this is
correct; the BFS skips recipes with `getNumDefinedValues() == 0` silently.

The scalar `ParallelFactor` field is retained for backward compatibility with
existing code paths. `DimSet` is the authoritative shape source going forward.

**Shape helper — free function** (not a method on `TPValue`, since `TPValue`
has no back-pointer to its plan):

```cpp
// TPRecipeMatcher.h
SmallVector<unsigned> getTPValueShape(const TPValue &V, const TPlan &Plan);
// returns { Plan.getPF(d) for d in V.DimSet } in ascending dim-index order
// returns {} if DimSet is empty (scalar value)
```

### 4.2 TensorOpKind

```cpp
// TPRecipeMatcher.h
enum class TensorOpKind {
  Scalar,           // DimSet empty on all operands — scalar op
  ElementWise,      // binary op, operand DimSets equal
  BroadcastBinary,  // binary op, one DimSet is strict subset of the other
  OuterProduct,     // binary op, operand DimSets are disjoint
  Contraction,      // reduction(mul-like), shared dim is a reduction dim
  PlainReduction,   // reduction, no fuseable mul-like producer
  // Note: unary ops, casts, cmp, select fall to Scalar (conservative).
  // Future TensorOpKind values (e.g. UnaryElementWise) will cover these.
};
```

### 4.3 RecipeClassification

```cpp
struct RecipeClassification {
  TensorOpKind  Kind           = TensorOpKind::Scalar;
  int           ContractDim    = -1;         // set for Contraction only
  TPRecipeBase *FusedMulRecipe = nullptr;    // set for Contraction only
  // Stores the resolved mul-like producer so execute() never re-derives it.
  // The Matcher walks through any intervening cast/select nodes to find the
  // first mul-like ancestor and stores the pointer here.
};
```

Storing `FusedMulRecipe` in `RecipeClassification` (rather than re-deriving it
with a naked `cast<>` in `execute()`) ensures execute() never performs analysis
and is safe even if intervening cast or select recipes sit between the fmul and
the reduction.

### 4.4 RecipeClassMap

```cpp
using RecipeClassMap = DenseMap<const TPRecipeBase *, RecipeClassification>;
```

Produced once by `TPRecipePatternMatcher_match()`, passed into
`TPTransformState` before lowering begins.

---

## 5. TPlan — getReductionDims()

```cpp
// Added to TPlan (TPlan.h):
SmallBitVector ReductionDims;   // set of dim indices that are reduction dims

const SmallBitVector &getReductionDims() const { return ReductionDims; }
```

**Population:** `LoopNestInfo` is extended with a `SmallBitVector ReductionDims`
field populated by `analyzeLoopNest()`, where `ScalarEvolution` and
`DependenceAnalysis` are already available. A dim `d` is a reduction dim if its
induction variable does not appear in the `IndexExprs` of any *output* (store)
`MemAccess` in `LoopNestInfo::Accesses`. `TPlanBuilder_build()` then copies
`LoopNestInfo::ReductionDims` directly into `TPlan::ReductionDims` without
additional analysis.

Specifically, in `analyzeLoopNest()`, for each dimension `d`:
- Collect all store `MemAccess` entries (write accesses).
- For each store access, check whether `IVs[d].IndVar` appears in any of its
  `IndexExprs` via `SE.dominates()` / containment in the SCEV expression tree.
- If `d` appears in no store index expression → `ReductionDims.set(d)`.

For GEMM: `k` (dim 1) appears in load accesses for B and C but not in the store
access for A → `ReductionDims = {1}`. `i` and `j` appear in the store → parallel.

`getReductionDims()` may return an empty bitset (all dims are parallel — no
contractions will be classified).

---

## 6. DimSet Propagation (TPlanWidener — Phase 2)

**Seeds:** each `TPHeaderPHIRecipe` for loop dimension `d` gets `DimSet = {d}`.
Reduction accumulator PHIs get `DimSet = {}` (they carry scalar accumulated
values, not tensors).

**Rule:** for every user recipe `R` of value `V`, for every value `DV` defined
by `R` (i.e. `getNumDefinedValues() > 0`):

```
DV.DimSet |= V.DimSet
```

Recipes with `getNumDefinedValues() == 0` (stores, branches) are silently
skipped in the BFS — they consume values but define none.

This is union-always — no contraction at widening time. Contraction is a
lowering-phase instruction-selection decision, not a shape-analysis decision.

**Example (GEMM, PFMap={0→256, 1→512, 2→1024}, ReductionDims={1}):**

```
%i.DimSet={0}   %k.DimSet={1}   %j.DimSet={2}
%b (load [i,k]) → DimSet={0,1}  → shape [256, 512]
%c (load [k,j]) → DimSet={1,2}  → shape [512, 1024]
%mul = fmul %b,%c → DimSet={0,1,2} → logical shape (never materialised)
%sum = reduction  → DimSet={0,2}   → shape [256, 1024]
```

---

## 7. Pattern Matcher (TPRecipePatternMatcher)

**File:** `llvm/lib/Transforms/Vectorize/TPRecipeMatcher.cpp`
**Header:** `llvm/include/llvm/Transforms/Vectorize/TPRecipeMatcher.h`
**Entry point:** `void TPRecipePatternMatcher_match(const TPlan &, RecipeClassMap &)`

### Classification rules (in priority order)

```
isReduction(R):
  // Walk through intervening cast/unary/replicate recipes to find mul-like ancestor.
  // skipIntermediateRecipes() is null-safe: returns null when given null.
  // This handles block-argument reductions (e.g. loop-carried values, function args).
  producer = skipIntermediateRecipes(R.getOperand(0).getDefiningRecipe())
  if (producer != nullptr && isMulLike(producer)):
    shared = producer.op0.DimSet & producer.op1.DimSet
    if shared & Plan.getReductionDims() ≠ ∅:
      contractDim = (shared & Plan.getReductionDims()).find_first()
      → Contraction(ContractDim=contractDim, FusedMulRecipe=producer)
  → PlainReduction(ContractDim=-1, FusedMulRecipe=nullptr)
    Note: PlainReduction::execute() reduces along the first reduction dim
    present in the input's DimSet that also appears in Plan.getReductionDims().
    If none found, reduces along the last dim (innermost) as a fallback.

isBinaryOp(R) [i.e. TPWidenBinaryOpRecipe with 2 operands]:
  D0 = R.op0.DimSet,  D1 = R.op1.DimSet
  D0 == D1                  → ElementWise
  D0 ⊂ D1 or D1 ⊂ D0      → BroadcastBinary
  (D0 & D1).none()          → OuterProduct
  partial overlap            → INVALID: assert/unreachable in debug builds;
                               emit diagnostic and bail in release builds.
                               A partially-overlapping binary op cannot be
                               correctly lowered as ElementWise (mismatched
                               shapes) and is not a valid TPlan program.

otherwise (unary, cast, cmp, select, load, store, PHI, branch):
  → Scalar
  Known limitation: unary ops and casts that produce shaped results are
  conservatively classified as Scalar. A future UnaryElementWise kind will
  handle these.
```

**skipIntermediateRecipes helper:** walks the def-use chain past any recipe
whose kind is in the set {`WidenCast`, `WidenUnary`, `Replicate`} until it
reaches a recipe not in that set or a block argument. Returns null when given
a null input (null-safe). This covers all recipes that may appear between a
fmul and a reduction without changing the mathematical value (type casts,
negations, scalar broadcasts).

**Coverage examples:**

| Expression | D0 | D1 | Kind |
|---|---|---|---|
| `fdiv %a{i,j}, %b{i,j}` | {0,2} | {0,2} | ElementWise |
| `fdiv %a{i,j}, %b{j}` | {0,2} | {2} | BroadcastBinary |
| `fmul %a{i}, %b{j}` | {0} | {2} | OuterProduct |
| `reduction(fmul(%b{i,k},%c{k,j}))` | — | — | Contraction(dim=1) |
| `reduction(fmul(%b{i,j},%c{i,j}))` | no reduction dim shared | — | PlainReduction |

**Extension:** adding a new op pattern requires adding one `classifyXxx()`
helper and one branch in the matcher loop. No changes to `execute()`.

---

## 8. Lowering / execute() Dispatch

### 8.1 TPTransformState — ClassMap attachment

`TPlan.h` cannot directly include `TPRecipeMatcher.h` (which needs `TPlan.h`
for `TPlan` and `TPValue`) without a circular dependency. This is resolved by
placing `TensorOpKind`, `RecipeClassification`, and `RecipeClassMap` in a
separate **`TPlanTypes.h`** header that has no dependency on `TPlan.h` or
`TPRecipe.h`. Both `TPlan.h` and `TPRecipeMatcher.h` include `TPlanTypes.h`.

```cpp
// New header: llvm/include/llvm/Transforms/Vectorize/TPlanTypes.h
// No includes from TPlan.h or TPRecipe.h — only LLVM ADT headers.
enum class TensorOpKind { ... };
struct RecipeClassification { ... };
using RecipeClassMap = DenseMap<const TPRecipeBase *, RecipeClassification>;
// Note: getTPValueShape() is declared in TPRecipeMatcher.h (needs TPValue/TPlan)
```

```cpp
// Added to TPTransformState (TPlan.h, which now includes TPlanTypes.h):
const RecipeClassMap *ClassMap = nullptr;

TensorOpKind   getKind(const TPRecipeBase *R) const;
int            getContractDim(const TPRecipeBase *R) const;
TPRecipeBase  *getFusedMulRecipe(const TPRecipeBase *R) const;
```

`TPlanLowering_lower()` retains its existing signature:

```cpp
bool TPlanLowering_lower(TPlan &Plan, Function &F,
                          LoopInfo &LI, ScalarEvolution &SE,
                          DominatorTree &DT);
```

The `RecipeClassMap` is constructed inside `TPlanLowering_lower()` (or its
internal `TPlanLoweringImpl::lower()`) and attached to the internally-constructed
`TPTransformState` before recipes begin executing. No caller-visible signature
change is required.

### 8.2 TPWidenBinaryOpRecipe::execute() — pure dispatcher

```
ElementWise     → CreateBinOp(opcode, LHS, RHS)
BroadcastBinary → emitBroadcastBinaryOp(opcode, LHS, RHS, D0, D1)
OuterProduct    → emitOuterProduct(LHS, RHS)
Scalar          → CreateBinOp(opcode, LHS, RHS)   // scalar fallback
Contraction     → no-op  (fmul deferred to its reduction consumer)
```

### 8.3 TPReductionRecipe::execute() — B1-α fusion

Fusion semantics: **B1-α** — reduction recipe uses the pre-resolved
`FusedMulRecipe` pointer from `RecipeClassification` (stored by the Matcher).
Load outputs (`%b`, `%c`) are guaranteed in ValueMap because load recipes
always precede arithmetic recipes in program order, enforced by def-use
constraints within a `TPBasicBlock`.

```
Contraction:
  MulRecipe = State.getFusedMulRecipe(this)   // pre-resolved, no cast<> needed
  LHS = State.getValue(MulRecipe.getOperand(0))   // %b [256×512]
  RHS = State.getValue(MulRecipe.getOperand(1))   // %c [512×1024]
  LHSShape = getTPValueShape(*MulRecipe.getOperand(0), Plan)  // [256, 512]
  RHSShape = getTPValueShape(*MulRecipe.getOperand(1), Plan)  // [512, 1024]

  // Guard: only 2D matrix operands are supported. Use hard error (not debug-only
  // assert) so release builds fail cleanly rather than computing a wrong index.
  if (MulRecipe.getOperand(0).DimSet.count() != 2 ||
      MulRecipe.getOperand(1).DimSet.count() != 2):
    emit diagnostic "Contraction requires 2D operands" and return

  // ContractDim is a loop-dim index, not a shape-vector position.
  // Find its position within the sorted DimSet of each operand.
  contractDim = State.getContractDim(this)
  LHSPos = position of contractDim in sorted(MulRecipe.getOperand(0).DimSet)
  RHSPos = position of contractDim in sorted(MulRecipe.getOperand(1).DimSet)
  // Sanity: LHSShape[LHSPos] == RHSShape[RHSPos] (both equal PFMap[contractDim])
  ContractSize = LHSShape[LHSPos]
  M = LHSShape[1 - LHSPos]     // the non-contracted LHS dimension
  N = RHSShape[1 - RHSPos]     // the non-contracted RHS dimension
  emit @llvm.matrix.multiply(LHS, RHS, M, ContractSize, N)

PlainReduction:
  Input = State.getValue(getOperand(0))
  ReduceAxis = first dim in Input.DimSet ∩ Plan.getReductionDims()
               (fallback: last dim of Input.DimSet if intersection empty)
  emit reduce_intrinsic(Input, axis=ReduceAxis)
```

### 8.4 Full execute() trace (GEMM)

```
load(%b).execute()       → State[%b] = IR value [256×512]
load(%c).execute()       → State[%c] = IR value [512×1024]
fmul_recipe.execute()    → Kind=Contraction → no-op
reduction.execute()      → Kind=Contraction
                           MulRecipe = pre-resolved fmul recipe
                           LHS = State[%b],  RHS = State[%c]
                           emit @llvm.matrix.multiply(%b, %c, 256, 512, 1024)
                           State[%sum] = [256×1024]
store.execute()          → store [256×1024] → A_ptr
```

---

## 9. Updated Call Sequence in TPlanLowering_lower()

```cpp
// Inside TPlanLoweringImpl::lower() (no signature change to public API):

// 1. Widen DimSets (Phase 2 added to existing widener)
TPlanWidener_widen(Plan);

// 2. Classify recipes — build RecipeClassMap
RecipeClassMap CM;
TPRecipePatternMatcher_match(Plan, CM);

// 3. Attach to the internally-constructed TPTransformState
State.ClassMap = &CM;

// 4. Execute recipes in program order
Plan.getEntry()->execute(State);
Plan.getVectorBody()->execute(State);
Plan.getMiddleBlock()->execute(State);
```

---

## 10. Files Changed / Added

| File | Change |
|---|---|
| `llvm/include/llvm/Transforms/Vectorize/TPlanTypes.h` | **New** — `TensorOpKind`, `RecipeClassification`, `RecipeClassMap`; no circular deps |
| `llvm/include/llvm/Transforms/Vectorize/TPRecipe.h` | Add `DimSet` to `TPValue`; include `TPlanTypes.h` |
| `llvm/include/llvm/Transforms/Vectorize/TPlan.h` | Add `ReductionDims` + `getReductionDims()` to `TPlan`; add `ClassMap` + accessors to `TPTransformState`; include `TPlanTypes.h` |
| `llvm/include/llvm/Transforms/Vectorize/TPRecipeMatcher.h` | **New** — declare `TPRecipePatternMatcher_match()` and `skipIntermediateRecipes()`; include `TPlanTypes.h` |
| `llvm/include/llvm/Transforms/Vectorize/LoopNestAnalyzer.h` | Add `SmallBitVector ReductionDims` to `LoopNestInfo` |
| `llvm/lib/Transforms/Vectorize/LoopNestAnalyzer.cpp` | Populate `LoopNestInfo::ReductionDims` in `analyzeLoopNest()` via SCEV containment check |
| `llvm/lib/Transforms/Vectorize/TPlanBuilder.cpp` | Copy `LoopNestInfo::ReductionDims` into `TPlan::ReductionDims` |
| `llvm/lib/Transforms/Vectorize/TPlanWidener.cpp` | Add Phase 2 DimSet BFS after existing scalar BFS |
| `llvm/lib/Transforms/Vectorize/TPRecipeMatcher.cpp` | **New** — implement `TPRecipePatternMatcher_match()`, `skipIntermediateRecipes()`, `getTPValueShape()` |
| `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` | Call matcher inside `TPlanLoweringImpl::lower()`; attach `ClassMap` to state; update `execute()` dispatch in `TPWidenBinaryOpRecipe` and `TPReductionRecipe` |
| `llvm/lib/Transforms/Vectorize/CMakeLists.txt` | Add `TPRecipeMatcher.cpp` |
| `llvm/unittests/Transforms/LoopTensorize/TPlanTest.cpp` | Unit tests (see Section 11) |
| `llvm/test/Transforms/LoopTensorize/` | Lit tests (see Section 11) |

---

## 11. Testing Plan

### Unit tests (TPlanTest.cpp)

**DimSet propagation:**
- DimSet seeds correctly from `TPHeaderPHIRecipe` (one bit per dim)
- Reduction accumulator PHI gets empty DimSet
- DimSet propagates correctly through load → fmul → reduction chain (GEMM)
- Recipe with `getNumDefinedValues() == 0` (store, branch) does not crash BFS
- Value with `PF > 1` but empty `DimSet` is backward-compatible (scalar PF field unaffected)

**Matcher classification:**
- GEMM pattern → `Contraction`, correct `ContractDim`, non-null `FusedMulRecipe`
- `fdiv` same dims → `ElementWise`
- `fdiv` subset dims → `BroadcastBinary`
- `fmul` disjoint dims → `OuterProduct`
- Reduction with no mul producer → `PlainReduction`
- `getReductionDims()` empty (all dims parallel) → no `Contraction` fires
- Reduction with intervening `TPWidenCastRecipe` between fmul and reduction → `Contraction` (`skipIntermediateRecipes` works)
- Reduction with intervening `TPWidenUnaryOpRecipe` between fmul and reduction → `Contraction` (`skipIntermediateRecipes` walks unary)
- Partial-overlap binary op (D0={0,1}, D1={1,2}, non-reduction) → assert/diagnostic fired, not silently lowered
- Higher-rank operand (DimSet has 3+ bits) in Contraction path → diagnostic emitted and execute() returns early (only 2D matrix ops supported)

### Lit tests

- `gemm-4x4.ll` — 3-level nest (i,k,j), verifies `@llvm.matrix.multiply` emitted
- `eltwise-div.ll` — 2-level nest (i,j), same dims, verifies vector `fdiv` emitted
- `broadcast-div.ll` — 2-level nest, one operand with subset dims, verifies broadcast
- `outer-product.ll` — 2-level nest, disjoint dims, verifies outer product intrinsic
- `plain-reduction.ll` — reduction with no mul producer, verifies reduce intrinsic + correct axis
