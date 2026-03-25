# TPlan + TPRecipe Design

**Date:** 2026-03-04
**Status:** Approved
**Author:** (your name)

---

## 1. Motivation

`LoopTensorize` currently uses `SearchState` (wrapping `LoopNestInfo`) as the central IR
for both pattern classification and the beam-search transform engine. This has two
limitations:

1. Classification and the cost model operate on raw SCEV/loop data rather than a
   structured recipe IR, making it hard to add new patterns or lowering passes.
2. The vectorization factor is a single scalar; there is no way to assign independent
   parallel factors (PFs) to each loop induction variable.

This design introduces **TPlan** — a VPlan-inspired IR for nested-loop tensorization —
and **TPRecipe** — typed recipe nodes that live inside a TPlan. TPlan becomes the
central structure used by classification, the search engine, and code emission.

---

## 2. Goals & Non-Goals

**Goals:**
- Define `TPRecipeBase`, `TPInductionRecipe`, `TPMemRecipe`, `TPComputeRecipe`.
- Define `TPlan` with a flat ordered recipe list and a per-dim PF vector.
- Update `classifyPattern()` to accept a `TPlan` and inspect recipes.
- Add `searchTPlan()` — a per-dimension beam search that assigns PFs via the cost model.
- Add `applyTPlan()` for code emission driven by recipe kinds and PFs.
- Wire the new path in `LoopTensorize.cpp`: `buildInitial → classifyPattern(TPlan) →
  searchTPlan → applyTPlan`.
- Unit and lit test coverage for construction, classification, search, and emission.

**Non-Goals:**
- Migrating the existing GEMM `SearchState` / `applyPlan()` path (deferred).
- `TPBasicBlock` / multi-block TPlan CFG (deferred; flat list suffices now).
- Elementwise and Reduction emission (stubs only; return `false`).
- ARM SME or x86 AMX native multi-dim intrinsics.

---

## 3. Recipe Class Hierarchy

```
TPRecipeBase          (ilist_node — owned by TPlan's iplist)
  ├── TPInductionRecipe   one per loop IV; carries PF for that dimension
  ├── TPMemRecipe         tiled load or store; knows which dims it spans
  └── TPComputeRecipe     compute op: Elementwise | Reduction | MatMul | Conv
```

### 3.1 `TPRecipeBase`

```cpp
class TPRecipeBase : public ilist_node<TPRecipeBase> {
public:
  enum class RecipeKind { Induction, Mem, Compute };
  RecipeKind getKind() const;
  TPlan *getPlan() const;           // back-pointer, set on insertion
  virtual void print(raw_ostream &) const = 0;
  virtual ~TPRecipeBase() = default;
};
```

### 3.2 `TPInductionRecipe`

Wraps `InductionDesc` from `LoopNestInfo`. `PF` is the parallel factor for this
dimension; set to 1 by `buildInitial()` and updated by `withPFs()`.

```cpp
class TPInductionRecipe : public TPRecipeBase {
  InductionDesc   Desc;
  uint32_t        PF = 1;
  unsigned        DimIndex;   // index into TPlan::PFs[]
};
```

### 3.3 `TPMemRecipe`

Wraps `MemAccess`. `AccessPFs` is resolved lazily from the plan's PF vector (not
stored redundantly).

```cpp
class TPMemRecipe : public TPRecipeBase {
  MemAccess  MA;
  bool       IsWrite;
  // PFs for each accessed dimension resolved via getPlan()->getPF(d)
};
```

### 3.4 `TPComputeRecipe`

```cpp
class TPComputeRecipe : public TPRecipeBase {
public:
  enum class ComputeKind { Elementwise, Reduction, MatMul, Conv };
  ComputeKind  Kind;
  PatternKind  Pattern = PatternKind::Generic;  // set by classifyPattern()
  bool         UseIm2Col = false;               // set by searchTPlan() for Conv
};
```

---

## 4. TPlan Structure

```cpp
class TPlan {
  LoopNestInfo            NestInfo;   // copy from analyzeLoopNest()
  SmallVector<uint32_t>   PFs;        // PFs[i] = parallel factor for IVs[i]
  iplist<TPRecipeBase>    Recipes;    // induction → mem reads → compute → mem writes
  float                   Cost = std::numeric_limits<float>::infinity();

public:
  static TPlan buildInitial(const LoopNestInfo &Info);
  TPlan withPFs(ArrayRef<uint32_t> NewPFs) const;   // cheap copy + PF update
  uint32_t getPF(unsigned Dim) const;
  void setCost(float C);
  float getCost() const;
  const LoopNestInfo &getNestInfo() const;
  iterator_range<iplist<TPRecipeBase>::iterator> recipes();
};
```

**`buildInitial()`** constructs:
- One `TPInductionRecipe` per `LoopNestInfo::IVs[i]` (PF=1).
- One `TPMemRecipe` per `LoopNestInfo::Accesses[j]`.
- One `TPComputeRecipe` with `Kind` inferred from access count and depth
  (2R+1W + depth≥3 → MatMul; 2R+1W + depth≥4 → Conv; otherwise Elementwise).

**`withPFs()`** returns a new `TPlan` with the recipe list shallow-copied and
`PFs` replaced. Recipes query PFs from the plan pointer, so no per-recipe mutation
is needed.

**No block structure in this PR.** A flat `iplist` suffices. `TPBasicBlock` can be
added in a follow-on when remainder / predication control flow is needed.

---

## 5. Pipeline Integration

### 5.1 New flow in `LoopTensorize.cpp`

```
analyzeLoopNest()
  → TPlan::buildInitial()
  → classifyPattern(TPlan)          // inspects recipes, annotates TPComputeRecipe
  → searchTPlan(TPlan, Ops, Params) // per-dim beam search, sets UseIm2Col if Conv2D
  → applyTPlan()
```

The existing `classifyPattern(LoopNestInfo)` / `runBeamSearch(SearchState)` /
`applyPlan()` path is **kept unchanged** for the legacy GEMM codegen path during
this PR. Migration is a follow-on.

### 5.2 `classifyPattern(const TPlan &Plan)`

New overload in `TensorPatternClassifier.h`. Inspects recipes:

- Counts `TPInductionRecipe`s → `Depth`.
- Counts `TPMemRecipe`s by `IsWrite` → determines read/write signature.
- Inspects `TPComputeRecipe::Kind` and `TPMemRecipe` SCEV index exprs for the
  sliding-window coefficient test (Conv2D).
- Writes `Pattern` and annotates `TPComputeRecipe` in-place; also returns a
  `PatternHint` for backward compatibility.

Old `classifyPattern(const LoopNestInfo &)` is kept as a deprecated overload.

---

## 6. Search Engine — `searchTPlan()`

### 6.1 Signature (in `TensorTransformSpace.h`)

```cpp
TPlan searchTPlan(TPlan Initial,
                  ArrayRef<TensorOpDesc> SupportedOps,
                  const TensorCostModelParams &Params,
                  unsigned BeamWidth = 8);
```

### 6.2 PF candidate generation

For dimension `d` with trip count `T`:
- If `T` is a `SCEVConstant`: candidates = powers of 2 from 1 to `min(T, 512)`.
- If `T` is `SCEVCouldNotCompute`: candidates = `{1, 4, 8, 16}`.

### 6.3 Per-dimension beam search

```
beam = { Initial }
for d = 0 .. Depth-1 (outer → inner):
    next_beam = {}
    for each plan in beam:
        for each pf in candidates[d]:
            plan' = plan.withPFs(pf at dim d)
            plan'.setCost(costTPlan(plan', SupportedOps, Params))
            next_beam.push(plan')
    beam = top-BeamWidth plans by cost from next_beam
return lowest-cost plan in beam
```

Complexity: `O(Depth × BeamWidth × max_candidates)` — avoids exponential blowup
of full combinatorial search.

### 6.4 `costTPlan()` (in `TensorCostModel.cpp`)

- **MatMul**: `cost = (PF[i] × PF[j] × PF[k] × 2) / PeakTensorFLOPS` (roofline).
- **Conv2D + im2col**: im2col copy cost + delegated MatMul cost.
- **Conv2D direct-tile**: vectorization cost scaled by PFs.
- Fallback to `PeakVectorFLOPS` when no tensor ops are available.

`UseIm2Col` annotation: after selecting the best plan, `searchTPlan()` reads the
`TPComputeRecipe::Pattern == Conv2D` flag, computes `col_matrix_bytes` from PFs, and
sets `UseIm2Col` on the recipe.

---

## 7. Code Emission — `applyTPlan()`

```cpp
// In TensorCodeGen.h / TensorCodeGen.cpp
bool llvm::applyTPlan(const TPlan &Plan, Function &F,
                      LoopInfo &LI, ScalarEvolution &SE, DominatorTree &DT);
```

Walks the recipe list in order:

| Recipe | Emission |
|---|---|
| `TPInductionRecipe` | No IR; PFs read by downstream recipes |
| `TPMemRecipe` (read) | `MatrixBuilder::CreateColumnMajorLoad` sized by PFs of spanned dims |
| `TPComputeRecipe` MatMul | `emitMatrixMultiply(PF[i], PF[k], PF[j], ...)` |
| `TPComputeRecipe` Conv + im2col | alloca col_matrix, emit copy loops, re-emit as MatMul |
| `TPComputeRecipe` Conv direct | InnerLoopVectorizer delegation (stub, returns false) |
| `TPComputeRecipe` Elementwise/Reduction | Stub; returns false (deferred) |
| `TPMemRecipe` (write) | `MatrixBuilder::CreateColumnMajorStore` sized by PFs |

After emission: `EliminateUnreachableBlocks(F)` removes the dead original loop.

Both `applyTPlan()` and `applyPlan()` coexist in `TensorCodeGen.cpp`.
`LoopTensorize.cpp` dispatches to `applyTPlan()` for the new path.

---

## 8. Files Changed

| File | Change |
|---|---|
| `llvm/include/llvm/Transforms/Vectorize/TPlan.h` | **New** — full class hierarchy |
| `llvm/lib/Transforms/Vectorize/TPlan.cpp` | **New** — `buildInitial`, `withPFs`, recipe constructors, `print` |
| `llvm/lib/Transforms/Vectorize/CMakeLists.txt` | Add `TPlan.cpp` |
| `TensorPatternClassifier.h/.cpp` | Add `classifyPattern(const TPlan &)` overload |
| `TensorTransformSpace.h/.cpp` | Add `searchTPlan()` |
| `TensorCostModel.h/.cpp` | Add `costTPlan()` |
| `TensorCodeGen.h/.cpp` | Add `applyTPlan()` |
| `LoopTensorize.cpp` | Wire new path; keep legacy path |

---

## 9. Testing

### 9.1 Unit Tests (`llvm/unittests/Transforms/LoopTensorize/`)

| Test | What it checks |
|---|---|
| `TPlanBuildInitial` | Correct recipe count and PF=1 on all dims |
| `TPlanWithPFs` | `withPFs()` returns new plan; original unchanged |
| `ClassifyPatternFromTPlan_GEMM` | 3-deep nest → `PatternKind::GEMM` via recipe inspection |
| `ClassifyPatternFromTPlan_Conv2D` | 5-deep sliding-window nest → `PatternKind::Conv2D` |
| `SearchTPlan_PicksBestPF` | Cost model selects PF={256,128,64} over PF={1,1,1} |

### 9.2 Lit Tests (`llvm/test/Transforms/LoopTensorize/basic/`)

| File | What it checks |
|---|---|
| `tplan-gemm-pf.ll` | GEMM with PF-assigned dims emits `llvm.matrix.multiply` at correct sizes |
| `tplan-pf-fallback-dynamic.ll` | Dynamic trip count → conservative PF candidates, no crash |
| `tplan-classify-from-recipes.ll` | Classification on TPlan recipes; correct `PatternKind` in debug output |

---

## 10. Open Questions

1. **PF legality vs. trip count**: if `PF[d] > TripCount[d]`, emission is undefined.
   For now, `searchTPlan()` clamps candidates to `min(pf, TripCount)`. Remainder
   handling is deferred.
2. **Recipe ownership**: `TPlan` owns its recipes via `iplist`. `withPFs()` does a
   deep copy. If copy cost matters for large nests, a copy-on-write scheme can be
   added later.
3. **Interaction with `runBeamSearch`**: until GEMM is migrated, both search paths
   coexist. A follow-on PR should unify them and delete `SearchState`.
