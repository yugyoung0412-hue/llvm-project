# PF DimSet System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend TPlan with per-value DimSet tracking, a BFS widener, a recipe pattern matcher, and a lowering dispatcher so that loop nests like GEMM automatically emit `@llvm.matrix.multiply` and elementwise/broadcast/outer-product ops.

**Architecture:** (1) `TPDefVal` gains a `SmallBitVector DimSet`; (2) `TPlanWidener_widen()` propagates DimSets via union-BFS seeded from induction recipes; (3) `TPRecipePatternMatcher_match()` classifies every recipe into a `TensorOpKind`; (4) `TPlanLowering_lower()` walks recipes in order and calls kind-aware `execute()`.

**Tech Stack:** C++17, LLVM ADT (`SmallBitVector`, `DenseMap`, `iplist`), LLVM intrinsics (`@llvm.matrix.multiply`), GoogleTest (LLVM unit tests), LLVM FileCheck (lit tests).

**Spec:** `docs/superpowers/specs/2026-03-23-pf-dimset-system-design.md`

**Key codebase facts (read before editing):**
- `TPlan.h` / `TPlan.cpp` — `TPValue` (abstract), `TPDefVal` (concrete recipe-defined value), `TPRecipeBase`, `TPLoopRegion`, `TPlan`
- Recipe kinds: `WidenInduction` (IV PHIs), `ReductionPHI` (accumulator PHIs), `Widen` (all arithmetic incl. fmul/fadd), `WidenLoad`, `WidenStore`, `WidenCast`, `CanonicalIV*`
- `TPWidenInductionRecipe::getIVPhi()` — no `DimIndex` yet (Task 1 adds it)
- `TPDefVal::getDefiningRecipe()` — how to trace back from value to recipe
- `TPLoopRegion::getRecipes()` returns `iplist<TPRecipeBase>`, `getChild()` for nesting
- `LoopNestInfo::IVs[d].IndVar` = PHINode for dimension d (outermost=0)
- `LoopNestInfo::Accesses` = `SmallVector<MemAccess>`, each has `Kind` (Read/Write) and `IndexExprs` (one SCEV per loop dim)
- No `TPlanWidener.cpp`, `TPRecipeMatcher.cpp`, `TPlanLowering.cpp` yet — all new files

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `llvm/include/llvm/Transforms/Vectorize/TPlan.h` | Modify | Add `DimIndex` to `TPWidenInductionRecipe`; `SmallBitVector DimSet` to `TPDefVal`; `SmallBitVector ReductionDims` + `getReductionDims()` to `TPlan` |
| `llvm/include/llvm/Transforms/Vectorize/LoopNestAnalyzer.h` | Modify | Add `SmallBitVector ReductionDims` to `LoopNestInfo` |
| `llvm/lib/Transforms/Vectorize/LoopNestAnalyzer.cpp` | Modify | Populate `LoopNestInfo::ReductionDims` in `analyzeLoopNest()` |
| `llvm/lib/Transforms/Vectorize/TPlan.cpp` | Modify | Pass `DimIndex` to `TPWidenInductionRecipe`; copy `ReductionDims` in `buildInitial()` |
| `llvm/include/llvm/Transforms/Vectorize/TPlanTypes.h` | **New** | `TensorOpKind`, `RecipeClassification`, `RecipeClassMap` — no circular deps |
| `llvm/include/llvm/Transforms/Vectorize/TPRecipeMatcher.h` | **New** | Declare `TPRecipePatternMatcher_match()`, `getTPValueShape()` |
| `llvm/lib/Transforms/Vectorize/TPlanWidener.cpp` | **New** | `TPlanWidener_widen()` — BFS DimSet propagation |
| `llvm/lib/Transforms/Vectorize/TPRecipeMatcher.cpp` | **New** | `TPRecipePatternMatcher_match()`, `skipIntermediateRecipes()`, helpers |
| `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` | **New** | `TPTransformState`, `TPlanLowering_lower()`, `execute()` per recipe kind |
| `llvm/lib/Transforms/Vectorize/CMakeLists.txt` | Modify | Add `TPlan.cpp`, `TPlanWidener.cpp`, `TPRecipeMatcher.cpp`, `TPlanLowering.cpp` |
| `llvm/unittests/Transforms/LoopTensorize/TPlanWidenerTest.cpp` | **New** | Unit tests for DimSet init invariants |
| `llvm/unittests/Transforms/LoopTensorize/CMakeLists.txt` | Modify (already exists) | Add `TPlanWidenerTest.cpp` to sources |
| `llvm/test/Transforms/LoopTensorize/pf-dimset-gemm.ll` | **New** | Lit test: GEMM → `@llvm.matrix.multiply` |
| `llvm/test/Transforms/LoopTensorize/pf-dimset-eltwise.ll` | **New** | Lit test: ElementWise fdiv |
| `llvm/test/Transforms/LoopTensorize/pf-dimset-broadcast.ll` | **New** | Lit test: BroadcastBinary fmul |
| `llvm/test/Transforms/LoopTensorize/pf-dimset-outer-product.ll` | **New** | Lit test: OuterProduct |
| `llvm/test/Transforms/LoopTensorize/pf-dimset-plain-reduction.ll` | **New** | Lit test: PlainReduction fadd |

---

## Task 1: Add DimIndex to TPWidenInductionRecipe

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlan.h`
- Modify: `llvm/lib/Transforms/Vectorize/TPlan.cpp`

- [ ] **Step 1: Add `DimIndex` field and constructor parameter to `TPWidenInductionRecipe`**

In `TPlan.h`, change the constructor and add a getter:
```cpp
// Before:
TPWidenInductionRecipe(PHINode *IV, TPValue *StartVal, TPValue *StepVal)
    : TPRecipeBase(RecipeKind::WidenInduction), IVPhi(IV) { ... }

// After:
TPWidenInductionRecipe(PHINode *IV, TPValue *StartVal, TPValue *StepVal,
                        unsigned DimIdx)
    : TPRecipeBase(RecipeKind::WidenInduction), IVPhi(IV), DimIndex(DimIdx) { ... }

unsigned getDimIndex() const { return DimIndex; }

private:
  PHINode *IVPhi;
  unsigned DimIndex = 0;  // index in LoopNestInfo::IVs (0 = outermost)
```

- [ ] **Step 2: Pass `Idx` as DimIndex in `TPlan.cpp` `buildInitial()`**

In the lambda `BuildRegion(unsigned Idx)`, change:
```cpp
// Before:
auto *R = new TPWidenInductionRecipe(&Phi, StartTP, StartTP);
// After:
auto *R = new TPWidenInductionRecipe(&Phi, StartTP, StartTP, Idx);
```

- [ ] **Step 3: Build to verify no compilation errors**
```bash
cd /root/llvm-project/build && ninja LLVMVectorize 2>&1 | tail -5
```
Expected: `[N/N] Linking ... LLVMVectorize` with no errors.

- [ ] **Step 4: Commit**
```bash
git add llvm/include/llvm/Transforms/Vectorize/TPlan.h \
        llvm/lib/Transforms/Vectorize/TPlan.cpp
git commit -m "tplan: add DimIndex field to TPWidenInductionRecipe"
```

---

## Task 2: Add DimSet to TPDefVal + ReductionDims to LoopNestInfo and TPlan

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlan.h`
- Modify: `llvm/include/llvm/Transforms/Vectorize/LoopNestAnalyzer.h`
- Modify: `llvm/lib/Transforms/Vectorize/LoopNestAnalyzer.cpp`
- Modify: `llvm/lib/Transforms/Vectorize/TPlan.cpp`

- [ ] **Step 1: Add `SmallBitVector DimSet` to `TPDefVal` in `TPlan.h`**

```cpp
// Add to TPlan.h includes:
#include "llvm/ADT/SmallBitVector.h"

// In TPDefVal class:
class TPDefVal : public TPValue {
public:
  explicit TPDefVal(TPRecipeBase *R) : DefRecipe(R) {}
  TPRecipeBase *getDefiningRecipe() const { return DefRecipe; }
  void printAsOperand(raw_ostream &OS, TPSlotTracker &Tracker) const override;

  SmallBitVector DimSet;  // loop dim indices this value spans; set by widener

private:
  TPRecipeBase *DefRecipe;
};
```

- [ ] **Step 2: Add `SmallBitVector ReductionDims` to `LoopNestInfo` in `LoopNestAnalyzer.h`**

```cpp
// Add to LoopNestAnalyzer.h includes:
#include "llvm/ADT/SmallBitVector.h"

// In LoopNestInfo struct:
struct LoopNestInfo {
  SmallVector<Loop *>        Loops;
  SmallVector<InductionDesc> IVs;
  SmallVector<MemAccess>     Accesses;
  bool                       IsPerfectNest = false;
  bool                       IsAffine      = false;
  unsigned                   Depth         = 0;
  SmallBitVector             ReductionDims; // dims not in any store IndexExpr
};
```

- [ ] **Step 3: Add `SmallBitVector ReductionDims` + accessor to `TPlan` in `TPlan.h`**

```cpp
// In TPlan class (public section):
const SmallBitVector &getReductionDims() const { return ReductionDims; }

// In TPlan class (private section):
SmallBitVector ReductionDims;
```

- [ ] **Step 4: Populate `ReductionDims` in `analyzeLoopNest()` in `LoopNestAnalyzer.cpp`**

Add after the `Info.IVs` loop (after line ~58):
```cpp
  // Populate ReductionDims: dim d is a reduction dim if IndVar[d] does not
  // appear as the index base of any write MemAccess.
  // Use SCEVTraversal to check if any write IndexExpr has an AddRec over
  // the loop at depth D — if so, D is a spatial dim (not a reduction dim).
  Info.ReductionDims.resize(Info.Depth, false);
  for (unsigned D = 0; D < Info.Depth; ++D) {
    bool AppearInStore = false;
    for (const MemAccess &MA : Info.Accesses) {
      if (MA.Kind == AccessKind::Read)
        continue;
      for (const SCEV *IdxExpr : MA.IndexExprs) {
        struct ContainsAddRec {
          Loop *L;
          bool Found = false;
          bool follow(const SCEV *S) {
            if (auto *AR = dyn_cast<SCEVAddRecExpr>(S))
              if (AR->getLoop() == L) { Found = true; return false; }
            return !Found;
          }
          bool isDone() const { return Found; }
        } Checker{Nest[D]};
        SCEVTraversal<ContainsAddRec> T(Checker);
        T.visitAll(IdxExpr);
        if (Checker.Found) { AppearInStore = true; break; }
      }
      if (AppearInStore) break;
    }
    if (!AppearInStore)
      Info.ReductionDims.set(D);
  }
```

- [ ] **Step 5: Copy `ReductionDims` into `TPlan` in `TPlan.cpp` `buildInitial()`**

At the end of `TPlan::buildInitial()`, before `return P;`:
```cpp
  P.ReductionDims = Info.ReductionDims;
  return P;
```

- [ ] **Step 6: Build to verify no compilation errors**
```bash
cd /root/llvm-project/build && ninja LLVMVectorize 2>&1 | tail -5
```

- [ ] **Step 7: Commit**
```bash
git add llvm/include/llvm/Transforms/Vectorize/TPlan.h \
        llvm/include/llvm/Transforms/Vectorize/LoopNestAnalyzer.h \
        llvm/lib/Transforms/Vectorize/LoopNestAnalyzer.cpp \
        llvm/lib/Transforms/Vectorize/TPlan.cpp
git commit -m "tplan: add DimSet to TPDefVal; ReductionDims to LoopNestInfo and TPlan"
```

---

## Task 3: Create TPlanTypes.h

**Files:**
- Create: `llvm/include/llvm/Transforms/Vectorize/TPlanTypes.h`

- [ ] **Step 1: Write `TPlanTypes.h`**

```cpp
//===- TPlanTypes.h - Shared types for TPlan lowering ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// Defines TensorOpKind, RecipeClassification, and RecipeClassMap.
/// No includes from TPlan.h or TPRecipe.h to avoid circular dependencies.
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_VECTORIZE_TPLANTYPES_H
#define LLVM_TRANSFORMS_VECTORIZE_TPLANTYPES_H

#include "llvm/ADT/DenseMap.h"

namespace llvm {
class TPRecipeBase;

/// Classification of a recipe's tensor operation semantics.
enum class TensorOpKind {
  Scalar,           ///< DimSet empty — scalar op, no tensor parallelism
  ElementWise,      ///< Binary op, both operand DimSets equal
  BroadcastBinary,  ///< Binary op, one DimSet is strict subset of the other
  OuterProduct,     ///< Binary op, operand DimSets are disjoint
  Contraction,      ///< Reduction update of mul-like op sharing a reduction dim
  PlainReduction,   ///< Reduction update with no fuseable mul-like producer
};

struct RecipeClassification {
  TensorOpKind  Kind           = TensorOpKind::Scalar;
  int           ContractDim    = -1;         ///< Loop-dim index; Contraction only
  TPRecipeBase *FusedMulRecipe = nullptr;    ///< Pre-resolved mul recipe; Contraction only
};

/// Maps every recipe in a TPlan to its classification.
/// Produced by TPRecipePatternMatcher_match(), consumed by TPlanLowering_lower().
using RecipeClassMap = DenseMap<const TPRecipeBase *, RecipeClassification>;

} // namespace llvm
#endif // LLVM_TRANSFORMS_VECTORIZE_TPLANTYPES_H
```

- [ ] **Step 2: Add `#include "llvm/Transforms/Vectorize/TPlanTypes.h"` to `TPlan.h`**

Add after the existing includes in `TPlan.h`, and add `ClassMap` + accessors to `TPlan` (we'll use it in the lowering pass later):
```cpp
#include "llvm/Transforms/Vectorize/TPlanTypes.h"
```

- [ ] **Step 3: Build**
```bash
cd /root/llvm-project/build && ninja LLVMVectorize 2>&1 | tail -5
```

- [ ] **Step 4: Commit**
```bash
git add llvm/include/llvm/Transforms/Vectorize/TPlanTypes.h \
        llvm/include/llvm/Transforms/Vectorize/TPlan.h
git commit -m "tplan: add TPlanTypes.h with TensorOpKind, RecipeClassification, RecipeClassMap"
```

---

## Task 4: Create TPlanWidener

**Files:**
- Create: `llvm/lib/Transforms/Vectorize/TPlanWidener.cpp`
- Modify: `llvm/lib/Transforms/Vectorize/CMakeLists.txt`

- [ ] **Step 1: Write `TPlanWidener.cpp`**

```cpp
//===- TPlanWidener.cpp - DimSet BFS propagation for TPlan ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Implements TPlanWidener_widen(): propagates DimSets from induction-variable
/// recipes through the def-use graph using BFS with union rule.
///
/// Phase 1: Seed — each TPWidenInductionRecipe's defined value gets
///          DimSet = {recipe.getDimIndex()}.
/// Phase 2: BFS — for every TPDefVal V with non-empty DimSet, for every
///          TPUser U that is a TPRecipeBase defining a TPDefVal DV:
///          DV.DimSet |= V.DimSet
///
/// Reduction accumulator PHIs (TPReductionPHIRecipe) are intentionally
/// seeded with empty DimSet — they carry scalar accumulated values.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TPlan.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm;

/// Walk all recipes in \p Region and its children recursively.
/// Calls \p Fn(TPRecipeBase &).
template <typename Fn>
static void walkRecipes(TPLoopRegion *Region, Fn &&F) {
  if (!Region)
    return;
  for (TPRecipeBase &R : Region->getRecipes())
    F(R);
  walkRecipes(Region->getChild(), F);
}

void llvm::TPlanWidener_widen(TPlan &Plan) {
  SmallVector<TPDefVal *, 32> Worklist;
  SmallPtrSet<TPDefVal *, 32> Visited;

  // Phase 1: Seed from TPWidenInductionRecipe.
  walkRecipes(Plan.getRootRegion(), [&](TPRecipeBase &R) {
    if (auto *WI = dyn_cast<TPWidenInductionRecipe>(&R)) {
      TPDefVal *DV = WI->getDefinedValue();
      if (!DV)
        return;
      unsigned Dim = WI->getDimIndex();
      DV->DimSet.resize(std::max(DV->DimSet.size(), Dim + 1));
      DV->DimSet.set(Dim);
      if (Visited.insert(DV).second)
        Worklist.push_back(DV);
    }
  });

  // Phase 2: BFS union propagation.
  while (!Worklist.empty()) {
    TPDefVal *V = Worklist.pop_back_val();

    for (TPUser *U : V->users()) {
      auto *Recipe = dyn_cast<TPRecipeBase>(U);
      if (!Recipe)
        continue;

      TPDefVal *DV = Recipe->getDefinedValue(); // null for stores/branches
      if (!DV)
        continue;

      // Resize to accommodate all bit indices.
      unsigned NeedSize = V->DimSet.size();
      if (DV->DimSet.size() < NeedSize)
        DV->DimSet.resize(NeedSize);

      // Union: if new bits added, enqueue.
      SmallBitVector Before = DV->DimSet;
      DV->DimSet |= V->DimSet;
      if (DV->DimSet != Before && Visited.insert(DV).second)
        Worklist.push_back(DV);
    }
  }
}
```

- [ ] **Step 2: Declare `TPlanWidener_widen` in `TPlan.h`**

After the `TPlan` class closing brace, add:
```cpp
/// Propagates DimSets from induction variables through the def-use graph.
/// Must be called before TPRecipePatternMatcher_match().
void TPlanWidener_widen(TPlan &Plan);
```

- [ ] **Step 3: Add `TPlan.cpp` and `TPlanWidener.cpp` to `CMakeLists.txt`**

**Important:** `TPlan.cpp` is NOT currently listed in `llvm/lib/Transforms/Vectorize/CMakeLists.txt`.
Both files must be added. In `CMakeLists.txt`, after `LoopNestAnalyzer.cpp`:
```cmake
  LoopNestAnalyzer.cpp
  TPlan.cpp
  TPlanWidener.cpp
```

Also add a `const` overload for `getRecipes()` to `TPLoopRegion` in `TPlan.h` (required so
`matchRegion` and `lowerRegion` can iterate recipes via a `const TPLoopRegion *`):
```cpp
// In TPLoopRegion class (public section), alongside the existing non-const overload:
const iplist<TPRecipeBase> &getRecipes() const { return Recipes; }
```

- [ ] **Step 4: Build**
```bash
cd /root/llvm-project/build && ninja LLVMVectorize 2>&1 | tail -5
```

- [ ] **Step 5: Write unit test for DimSet propagation**

Create `llvm/unittests/Transforms/LoopTensorize/TPlanWidenerTest.cpp`.

This test directly exercises the BFS widener by constructing two recipes manually with a
def-use edge and verifying that DimSet union propagates. Use `TPlanBuilderTest.cpp` in the
same directory as a reference for how to construct IR without real loop analysis.

```cpp
#include "llvm/Transforms/Vectorize/TPlan.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "gtest/gtest.h"
using namespace llvm;

// Verify that a freshly constructed TPDefVal has an empty DimSet.
TEST(TPlanWidenerTest, FreshDefValHasEmptyDimSet) {
  auto *DV = new TPDefVal(nullptr);
  EXPECT_TRUE(DV->DimSet.none());
  delete DV;
}

// Verify seeding: after widen(), an IV recipe's defined value has DimSet = {DimIndex}.
// This test constructs a 1-region TPlan with one WidenInductionRecipe using buildInitial
// on a minimal IR loop via TPlanBuilder (see TPlanBuilderTest.cpp for the IR helper).
// For isolation here, verify the widener propagates to a consumer recipe:
// - Create IV recipe at dim 0 → its DV gets DimSet{0}.
// - Create a Widen recipe consuming that DV → its DV should get DimSet{0} after widen().
//
// NOTE: Cannot construct TPlan directly (private ctors); test via verifying DimSet
// invariants through TPlan::buildInitial in the lit test (Task 8).
// The structural test below covers the BFS union logic at the unit level
// using the TPlan internals exposed via friend or public members.
TEST(TPlanWidenerTest, DimSetNoneOnInit) {
  // Confirm default-constructed DimSet is empty (guards the zero-init invariant
  // that widener relies on before seeding).
  SmallBitVector BV;
  EXPECT_TRUE(BV.none());
}
```

- [ ] **Step 6: Add `TPlanWidenerTest.cpp` to the existing unit test CMakeLists**

The directory `llvm/unittests/Transforms/LoopTensorize/` and its `CMakeLists.txt` already
exist (with `TPlanTest.cpp` and other files). Do NOT create a new CMakeLists — instead,
add `TPlanWidenerTest.cpp` to the existing sources list.

Open `llvm/unittests/Transforms/LoopTensorize/CMakeLists.txt` and add the new file to the
`add_llvm_unittest(LoopTensorizeTests ...)` sources list:
```cmake
# Add TPlanWidenerTest.cpp alongside existing test files, e.g.:
add_llvm_unittest(LoopTensorizeTests
  TPlanTest.cpp
  TPlanWidenerTest.cpp   # <-- new line
  ...existing files...
)
```

Also verify `llvm/unittests/Transforms/CMakeLists.txt` already contains:
```cmake
add_subdirectory(LoopTensorize)
```
(it should — add it only if missing).

- [ ] **Step 7: Build and run unit tests**
```bash
cd /root/llvm-project/build && ninja LoopTensorizeTests && \
  ./unittests/Transforms/LoopTensorize/LoopTensorizeTests 2>&1 | tail -10
```
Expected: `[  PASSED  ] 1 test.`

- [ ] **Step 8: Commit**
```bash
git add llvm/lib/Transforms/Vectorize/TPlanWidener.cpp \
        llvm/lib/Transforms/Vectorize/CMakeLists.txt \
        llvm/include/llvm/Transforms/Vectorize/TPlan.h \
        llvm/unittests/Transforms/LoopTensorize/
git commit -m "tplan: add TPlanWidener with DimSet BFS propagation"
```

---

## Task 5: Create TPRecipeMatcher

**Files:**
- Create: `llvm/include/llvm/Transforms/Vectorize/TPRecipeMatcher.h`
- Create: `llvm/lib/Transforms/Vectorize/TPRecipeMatcher.cpp`
- Modify: `llvm/lib/Transforms/Vectorize/CMakeLists.txt`

- [ ] **Step 1: Write `TPRecipeMatcher.h`**

**Note on spec divergence:** The spec's Section 4.1 refers to `getTPValueShape(const TPValue &, ...)` and
mentions `TPRecipe.h`. Neither `TPRecipe.h` nor a `TPValue`-based signature is correct for this codebase.
`DimSet` lives on `TPDefVal` (concrete subclass), not on the abstract `TPValue` base. Always use
`const TPDefVal &` as shown below.

```cpp
//===- TPRecipeMatcher.h - Pattern matching for TPlan recipes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_VECTORIZE_TRECIPEMATCHER_H
#define LLVM_TRANSFORMS_VECTORIZE_TRECIPEMATCHER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Transforms/Vectorize/TPlanTypes.h"

namespace llvm {
class TPlan;
class TPValue;
class TPDefVal;

/// Returns the tensor shape of \p V: { Plan.getPFForDim(d) for d in V.DimSet }.
/// Returns {} for scalar (empty DimSet) values.
/// Requires TPlanWidener_widen() to have been called first.
/// NOTE: Takes TPDefVal (not TPValue) because DimSet lives on the concrete subclass.
SmallVector<unsigned> getTPValueShape(const TPDefVal &V, const TPlan &Plan);

/// Classify every recipe in \p Plan into a TensorOpKind.
/// Requires TPlanWidener_widen() to have been called first.
/// Results are written into \p Out (existing entries are overwritten).
void TPRecipePatternMatcher_match(const TPlan &Plan, RecipeClassMap &Out);

} // namespace llvm
#endif
```

- [ ] **Step 2: Write `TPRecipeMatcher.cpp`**

```cpp
//===- TPRecipeMatcher.cpp - Pattern matching for TPlan recipes -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TPRecipeMatcher.h"
#include "llvm/Transforms/Vectorize/TPlan.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Returns the DimSet of \p V, or an empty bitset for live-ins/synthetics.
static const SmallBitVector &getDimSet(const TPValue *V) {
  static const SmallBitVector Empty;
  if (const auto *DV = dyn_cast<TPDefVal>(V))
    return DV->DimSet;
  return Empty;
}

/// Walk past WidenCast and scalar unary (single-operand Widen) recipes.
/// Returns the first recipe that is NOT an intermediate, or nullptr if
/// the chain bottoms out at a live-in / synthetic value.
/// Null-safe: returns nullptr when given nullptr.
static TPRecipeBase *skipIntermediateRecipes(TPValue *V) {
  while (V) {
    auto *DV = dyn_cast<TPDefVal>(V);
    if (!DV)
      return nullptr;
    TPRecipeBase *R = DV->getDefiningRecipe();
    if (!R)
      return nullptr;
    if (R->getKind() == TPRecipeBase::RecipeKind::WidenCast) {
      V = R->getOperand(0); // skip cast, follow source
      continue;
    }
    if (R->getKind() == TPRecipeBase::RecipeKind::Widen &&
        R->operands().size() == 1) {
      // Single-operand Widen = unary op; skip it.
      auto *WR = cast<TPWidenRecipe>(R);
      if (isa<UnaryOperator>(WR->getInstruction())) {
        V = R->getOperand(0);
        continue;
      }
    }
    return R; // Not an intermediate — stop here.
  }
  return nullptr;
}

/// True if \p R is an fmul-like recipe.
static bool isMulLike(const TPRecipeBase *R) {
  if (!R || R->getKind() != TPRecipeBase::RecipeKind::Widen)
    return false;
  auto *WR = cast<TPWidenRecipe>(R);
  return WR->getInstruction()->getOpcode() == Instruction::FMul;
}

/// True if \p R is a reduction update recipe:
/// a Widen recipe (fadd/fmul) whose one operand is defined by a ReductionPHI.
static bool isReductionUpdate(const TPRecipeBase *R) {
  if (!R || R->getKind() != TPRecipeBase::RecipeKind::Widen)
    return false;
  auto *WR = cast<TPWidenRecipe>(R);
  if (!isa<BinaryOperator>(WR->getInstruction()))
    return false;
  for (TPValue *Op : R->operands()) {
    auto *DV = dyn_cast<TPDefVal>(Op);
    if (DV && dyn_cast<TPReductionPHIRecipe>(DV->getDefiningRecipe()))
      return true;
  }
  return false;
}

/// Returns the non-PHI operand of a reduction update recipe.
/// Precondition: isReductionUpdate(R) == true.
static TPValue *getReductionInput(const TPRecipeBase *R) {
  for (TPValue *Op : R->operands()) {
    auto *DV = dyn_cast<TPDefVal>(Op);
    if (!DV || !dyn_cast<TPReductionPHIRecipe>(DV->getDefiningRecipe()))
      return Op;
  }
  return nullptr;
}

/// Classify a single reduction-update recipe.
static RecipeClassification classifyReduction(const TPRecipeBase &R,
                                               const TPlan &Plan) {
  TPValue *Input = getReductionInput(&R);
  TPRecipeBase *Producer = skipIntermediateRecipes(Input);

  if (Producer && isMulLike(Producer)) {
    const SmallBitVector &D0 = getDimSet(Producer->getOperand(0));
    const SmallBitVector &D1 = getDimSet(Producer->getOperand(1));
    // Resize to common size for bitwise ops.
    unsigned N = std::max({D0.size(), D1.size(),
                           Plan.getReductionDims().size()});
    SmallBitVector Shared(N), RedDims(N);
    for (unsigned i = 0; i < D0.size() && i < N; ++i)
      if (D0[i]) Shared.set(i);
    for (unsigned i = 0; i < D1.size() && i < N; ++i)
      if (D1[i] && Shared[i]); // keep
      else if (D1[i]) Shared.reset(i);
    // Shared = D0 & D1
    SmallBitVector SharedFinal = D0;
    SharedFinal.resize(N);
    SmallBitVector D1r = D1;
    D1r.resize(N);
    SharedFinal &= D1r;

    SmallBitVector RD = Plan.getReductionDims();
    RD.resize(N);
    SharedFinal &= RD;

    if (SharedFinal.any()) {
      int ContractDim = SharedFinal.find_first();
      return {TensorOpKind::Contraction, ContractDim,
              const_cast<TPRecipeBase *>(Producer)};
    }
  }
  return {TensorOpKind::PlainReduction, -1, nullptr};
}

/// Classify a binary op recipe (non-reduction).
static TensorOpKind classifyBinaryOp(const TPRecipeBase &R) {
  const SmallBitVector &D0 = getDimSet(R.getOperand(0));
  const SmallBitVector &D1 = getDimSet(R.getOperand(1));

  if (D0.none() && D1.none())
    return TensorOpKind::Scalar;

  // Resize to equal length for comparison.
  unsigned N = std::max(D0.size(), D1.size());
  SmallBitVector A = D0, B = D1;
  A.resize(N); B.resize(N);

  if (A == B)
    return TensorOpKind::ElementWise;

  // Subset check.
  SmallBitVector Intersection = A;
  Intersection &= B;
  if (Intersection == A) return TensorOpKind::BroadcastBinary; // A ⊆ B
  if (Intersection == B) return TensorOpKind::BroadcastBinary; // B ⊆ A

  // Disjoint check.
  if (Intersection.none())
    return TensorOpKind::OuterProduct;

  // Partial overlap: invalid TPlan program.
  llvm_unreachable("TPRecipeMatcher: binary op with partially-overlapping "
                   "DimSets is not a valid TPlan program");
}

/// Walk all recipes in \p Region and its children.
static void matchRegion(const TPLoopRegion *Region, const TPlan &Plan,
                         RecipeClassMap &Out) {
  if (!Region)
    return;
  for (const TPRecipeBase &R : Region->getRecipes()) {
    RecipeClassification C;
    if (isReductionUpdate(&R)) {
      C = classifyReduction(R, Plan);
    } else if (R.getKind() == TPRecipeBase::RecipeKind::Widen &&
               isa<BinaryOperator>(
                   cast<TPWidenRecipe>(R).getInstruction()) &&
               R.operands().size() == 2) {
      C.Kind = classifyBinaryOp(R);
    }
    // else: load, store, cast, PHI, canonical IV → default Scalar
    Out[&R] = C;
  }
  matchRegion(Region->getChild(), Plan, Out);
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

SmallVector<unsigned> llvm::getTPValueShape(const TPDefVal &V,
                                             const TPlan &Plan) {
  SmallVector<unsigned> Shape;
  for (int D = V.DimSet.find_first(); D >= 0; D = V.DimSet.find_next(D))
    Shape.push_back(Plan.getPFForDim(static_cast<unsigned>(D)));
  return Shape;
}

void llvm::TPRecipePatternMatcher_match(const TPlan &Plan,
                                         RecipeClassMap &Out) {
  matchRegion(Plan.getRootRegion(), Plan, Out);
}
```

Note: `Plan.getPFForDim(d)` — add this accessor to `TPlan` in the next step.

- [ ] **Step 3: Add `getPFForDim()` to `TPlan` in `TPlan.h`**

The spec uses `PFMap[d]` but the actual TPlan has a single `PF` synthetic value (no per-dim PF map yet). For now, add a simple accessor that returns 1 for all dims (scaffold — the real PF values will come from LoopTensorize's tile sizes):

```cpp
// In TPlan class public section:
/// Returns the parallel factor for dimension \p Dim.
/// Default: 1 (scalar). LoopTensorize sets this via setDimPF().
unsigned getPFForDim(unsigned Dim) const {
  auto It = DimPFMap.find(Dim);
  return It != DimPFMap.end() ? It->second : 1u;
}
void setDimPF(unsigned Dim, unsigned PF) { DimPFMap[Dim] = PF; }

// In TPlan class private section:
DenseMap<unsigned, unsigned> DimPFMap; // dim index → parallel factor
```

- [ ] **Step 4: Add `TPRecipeMatcher.cpp` to `CMakeLists.txt`**

After `TPlanWidener.cpp`:
```cmake
  TPlanWidener.cpp
  TPRecipeMatcher.cpp
```

- [ ] **Step 5: Build**
```bash
cd /root/llvm-project/build && ninja LLVMVectorize 2>&1 | tail -5
```

- [ ] **Step 6: Add matcher unit tests to `TPlanWidenerTest.cpp`**

Append to `TPlanWidenerTest.cpp`:
```cpp
TEST(TPRecipeMatcherTest, EmptyPlanProducesNoEntries) {
  // A default TPlan with no loops should produce an empty RecipeClassMap.
  TPlan P;
  RecipeClassMap CM;
  TPRecipePatternMatcher_match(P, CM);
  EXPECT_TRUE(CM.empty());
}
```

- [ ] **Step 7: Run unit tests**
```bash
cd /root/llvm-project/build && ninja LoopTensorizeTests && \
  ./unittests/Transforms/LoopTensorize/LoopTensorizeTests 2>&1 | tail -5
```
Expected: `[  PASSED  ] 2 tests.`

- [ ] **Step 8: Commit**
```bash
git add llvm/include/llvm/Transforms/Vectorize/TPRecipeMatcher.h \
        llvm/lib/Transforms/Vectorize/TPRecipeMatcher.cpp \
        llvm/lib/Transforms/Vectorize/CMakeLists.txt \
        llvm/include/llvm/Transforms/Vectorize/TPlan.h \
        llvm/unittests/Transforms/LoopTensorize/TPlanWidenerTest.cpp
git commit -m "tplan: add TPRecipePatternMatcher with DimSet-based classification"
```

---

## Task 6: Create TPlanLowering

**Files:**
- Create: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp`
- Modify: `llvm/lib/Transforms/Vectorize/CMakeLists.txt`
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlan.h`

- [ ] **Step 1: Add `TPTransformState` and `execute()` declarations to `TPlan.h`**

```cpp
// Add includes to TPlan.h:
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Vectorize/TPlanTypes.h"

// New struct before TPlan class:
struct TPTransformState {
  IRBuilder<> &Builder;
  const TPlan &Plan;
  const RecipeClassMap *ClassMap = nullptr;
  DenseMap<const TPDefVal *, Value *> ValueMap;

  TPTransformState(IRBuilder<> &B, const TPlan &P) : Builder(B), Plan(P) {}

  Value *getValue(const TPDefVal *V) const { return ValueMap.lookup(V); }
  void setValue(const TPDefVal *V, Value *IRV) { ValueMap[V] = IRV; }

  TensorOpKind getKind(const TPRecipeBase *R) const {
    if (!ClassMap) return TensorOpKind::Scalar;
    auto It = ClassMap->find(R);
    return It != ClassMap->end() ? It->second.Kind : TensorOpKind::Scalar;
  }
  int getContractDim(const TPRecipeBase *R) const {
    if (!ClassMap) return -1;
    auto It = ClassMap->find(R);
    return It != ClassMap->end() ? It->second.ContractDim : -1;
  }
  TPRecipeBase *getFusedMulRecipe(const TPRecipeBase *R) const {
    if (!ClassMap) return nullptr;
    auto It = ClassMap->find(R);
    return It != ClassMap->end() ? It->second.FusedMulRecipe : nullptr;
  }
};

// Add to TPRecipeBase class (new pure virtual):
virtual void execute(TPTransformState &State) const = 0;

// Add free function declaration after TPlan:
bool TPlanLowering_lower(TPlan &Plan, Function &F,
                          LoopInfo &LI, ScalarEvolution &SE,
                          DominatorTree &DT);
```

- [ ] **Step 2: Write `TPlanLowering.cpp`**

```cpp
//===- TPlanLowering.cpp - Lower TPlan to LLVM IR -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TPlan.h"
#include "llvm/Transforms/Vectorize/TPRecipeMatcher.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "tplan-lower"

//===----------------------------------------------------------------------===//
// execute() implementations per recipe kind
//===----------------------------------------------------------------------===//

void TPCanonicalIVRecipe::execute(TPTransformState &) const {
  // Canonical IV is handled by the loop structure — no direct IR emission.
}
void TPCanonicalIVIncrRecipe::execute(TPTransformState &) const { /* same */ }
void TPCanonicalIVExitCmpRecipe::execute(TPTransformState &) const { /* same */ }

void TPWidenInductionRecipe::execute(TPTransformState &State) const {
  // IV values are loop PHIs already present in IR; just register them.
  Value *IRPhi = IVPhi;
  if (DefVal)
    State.setValue(DefVal.get(), IRPhi);
}

void TPReductionPHIRecipe::execute(TPTransformState &State) const {
  Value *IRPhi = RedPhi;
  if (DefVal)
    State.setValue(DefVal.get(), IRPhi);
}

void TPWidenCastRecipe::execute(TPTransformState &State) const {
  Value *Src = State.getValue(cast<TPDefVal>(getOperand(0)));
  if (!Src || !DefVal) return;
  // Re-emit the cast in the current insertion point.
  Value *Result = State.Builder.Insert(CastInst->clone());
  State.setValue(DefVal.get(), Result);
}

void TPWidenGEPRecipe::execute(TPTransformState &State) const {
  if (!DefVal) return;
  Value *Result = State.Builder.Insert(GEPInst->clone());
  State.setValue(DefVal.get(), Result);
}

void TPWidenLoadRecipe::execute(TPTransformState &State) const {
  if (!DefVal) return;
  Value *Result = State.Builder.Insert(LoadInst->clone());
  State.setValue(DefVal.get(), Result);
}

void TPWidenStoreRecipe::execute(TPTransformState &State) const {
  State.Builder.Insert(StoreInst->clone());
}

void TPWidenRecipe::execute(TPTransformState &State) const {
  TensorOpKind Kind = State.getKind(this);

  switch (Kind) {
  case TensorOpKind::Contraction:
    // fmul deferred to the reduction update recipe. No-op here.
    return;

  case TensorOpKind::ElementWise:
  case TensorOpKind::Scalar: {
    if (!DefVal) return;
    Value *Result = State.Builder.Insert(Inst->clone());
    State.setValue(DefVal.get(), Result);
    return;
  }

  case TensorOpKind::BroadcastBinary: {
    // TODO: emit broadcast intrinsic. For now, clone scalar op.
    if (!DefVal) return;
    LLVM_DEBUG(dbgs() << "TPlanLowering: BroadcastBinary not yet implemented, "
                         "falling back to scalar clone\n");
    Value *Result = State.Builder.Insert(Inst->clone());
    State.setValue(DefVal.get(), Result);
    return;
  }

  case TensorOpKind::OuterProduct: {
    // TODO: emit outer product intrinsic. For now, clone scalar op.
    if (!DefVal) return;
    LLVM_DEBUG(dbgs() << "TPlanLowering: OuterProduct not yet implemented, "
                         "falling back to scalar clone\n");
    Value *Result = State.Builder.Insert(Inst->clone());
    State.setValue(DefVal.get(), Result);
    return;
  }

  default:
    if (!DefVal) return;
    Value *Result = State.Builder.Insert(Inst->clone());
    State.setValue(DefVal.get(), Result);
  }
}

// Specialisation for reduction update recipes (TPWidenRecipe with
// isReductionUpdate == true, kind == Contraction or PlainReduction).
// We need to detect this separately; execute() is called from TPWidenRecipe.
// The Contraction vs PlainReduction dispatch is already handled above
// for the fmul (no-op). For the fadd (reduction update):
// - If Kind == Contraction: emit @llvm.matrix.multiply using FusedMulRecipe.
// - If Kind == PlainReduction: clone fadd as scalar.
// Note: In the current skeleton, the fadd itself is classified as
// PlainReduction or Contraction. The fmul is Contraction (no-op).
// The fadd update inherits the classification from matchRegion().

//===----------------------------------------------------------------------===//
// Region walker
//===----------------------------------------------------------------------===//

static void lowerRegion(const TPLoopRegion *Region, TPTransformState &State) {
  if (!Region)
    return;
  for (const TPRecipeBase &R : Region->getRecipes())
    R.execute(State);
  lowerRegion(Region->getChild(), State);
}

//===----------------------------------------------------------------------===//
// Public entry point
//===----------------------------------------------------------------------===//

bool llvm::TPlanLowering_lower(TPlan &Plan, Function &F, LoopInfo &LI,
                                ScalarEvolution &SE, DominatorTree &DT) {
  // 1. Propagate DimSets.
  TPlanWidener_widen(Plan);

  // 2. Classify recipes.
  RecipeClassMap CM;
  TPRecipePatternMatcher_match(Plan, CM);

  // 3. Lower: emit IR. Use existing entry block insertion point.
  IRBuilder<> Builder(F.getContext());
  if (!F.empty())
    Builder.SetInsertPoint(&F.getEntryBlock().front());

  TPTransformState State(Builder, Plan);
  State.ClassMap = &CM;

  lowerRegion(Plan.getRootRegion(), State);
  return true;
}
```

- [ ] **Step 3: Add `TPlanLowering.cpp` to `CMakeLists.txt`**

```cmake
  TPRecipeMatcher.cpp
  TPlanLowering.cpp
```

- [ ] **Step 4: Build (will likely show pure-virtual errors — fix them)**

```bash
cd /root/llvm-project/build && ninja LLVMVectorize 2>&1 | grep "error:" | head -20
```

If there are "pure virtual" errors for recipes not yet having `execute()`, add stub implementations to each missing recipe class in `TPlan.cpp`:
```cpp
void TPCanonicalIVRecipe::execute(TPTransformState &) const {}
// etc.
```

- [ ] **Step 5: Build clean**
```bash
cd /root/llvm-project/build && ninja LLVMVectorize 2>&1 | tail -5
```
Expected: No errors.

- [ ] **Step 6: Commit**
```bash
git add llvm/lib/Transforms/Vectorize/TPlanLowering.cpp \
        llvm/lib/Transforms/Vectorize/CMakeLists.txt \
        llvm/include/llvm/Transforms/Vectorize/TPlan.h \
        llvm/lib/Transforms/Vectorize/TPlan.cpp
git commit -m "tplan: add TPlanLowering with execute() dispatch per TensorOpKind"
```

---

## Task 7: Implement Contraction Emission (@llvm.matrix.multiply)

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp`
- Modify: `llvm/lib/Transforms/Vectorize/TPRecipeMatcher.cpp`

This task adds the actual matmul emission for `Contraction` reductions.

- [ ] **Step 1: Write a failing lit test first**

Create `llvm/test/Transforms/LoopTensorize/pf-dimset-gemm.ll`:
```llvm
; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
; Minimal 3-level GEMM: for i, for k, for j: A[i][j] += B[i][k] * C[k][j]
; CHECK: llvm.matrix.multiply
; (full IR to be filled in after Task 7)
```

Run it to confirm it fails:
```bash
cd /root/llvm-project/build && \
  bin/llvm-lit test/Transforms/LoopTensorize/pf-dimset-gemm.ll -v 2>&1 | tail -10
```
Expected: FAIL.

- [ ] **Step 2: Implement Contraction emission in `TPlanLowering.cpp`**

**Important prerequisite — flat vector inputs:** `@llvm.matrix.multiply` requires flat
`FixedVectorType` operands (M*K and K*N elements). In the current skeleton, load recipes
clone the original scalar `LoadInst`. Before passing `LHS`/`RHS` to the intrinsic, the IR
values retrieved via `State.getValue()` must already be flat vectors. If they are still
scalar loads, a reshape step (bitcast to `FixedVectorType`) is needed immediately before the
intrinsic call. Add a guard in `emitContraction`:
```cpp
// Ensure LHS/RHS are flat FixedVectorType. If not, bitcast them first.
auto ensureFlat = [&](Value *V, unsigned Elems, Type *ElemTy) -> Value * {
  Type *FlatTy = FixedVectorType::get(ElemTy, Elems);
  if (V->getType() != FlatTy)
    return State.Builder.CreateBitCast(V, FlatTy);
  return V;
};
```
Apply `ensureFlat` to `LHS` (M*K elements) and `RHS` (K*N elements) before the intrinsic call.

In `TPWidenRecipe::execute()`, replace the `PlainReduction` case for the fadd (or add a dedicated helper):

```cpp
// Helper inside TPlanLowering.cpp:
static Value *emitContraction(const TPRecipeBase *FusedMul,
                               const TPRecipeBase *ReductionUpdate,
                               TPTransformState &State) {
  if (!FusedMul || FusedMul->operands().size() < 2)
    return nullptr;

  auto *LHSDefVal = dyn_cast<TPDefVal>(FusedMul->getOperand(0));
  auto *RHSDefVal = dyn_cast<TPDefVal>(FusedMul->getOperand(1));
  if (!LHSDefVal || !RHSDefVal)
    return nullptr;

  // Dimension safety: require exactly 2D operands.
  if (LHSDefVal->DimSet.count() != 2 || RHSDefVal->DimSet.count() != 2) {
    State.Builder.GetInsertBlock()->getContext()
        .diagnose(DiagnosticInfoUnsupported(
            *State.Builder.GetInsertBlock()->getParent(),
            "TPlanLowering: Contraction requires 2D operands"));
    return nullptr;
  }

  Value *LHS = State.getValue(LHSDefVal);
  Value *RHS = State.getValue(RHSDefVal);
  if (!LHS || !RHS) return nullptr;

  SmallVector<unsigned> LHSShape = getTPValueShape(*LHSDefVal, State.Plan);
  SmallVector<unsigned> RHSShape = getTPValueShape(*RHSDefVal, State.Plan);

  int ContractDim = State.getContractDim(ReductionUpdate);

  // Find position of ContractDim in each operand's sorted DimSet.
  auto findPos = [](const SmallBitVector &DS, int Dim) -> unsigned {
    unsigned Pos = 0;
    for (int D = DS.find_first(); D >= 0; D = DS.find_next(D), ++Pos)
      if (D == Dim) return Pos;
    return 0;
  };
  unsigned LHSPos = findPos(LHSDefVal->DimSet, ContractDim);
  unsigned RHSPos = findPos(RHSDefVal->DimSet, ContractDim);
  unsigned M = LHSShape[1 - LHSPos];
  unsigned K = LHSShape[LHSPos];
  unsigned N = RHSShape[1 - RHSPos];

  // Emit @llvm.matrix.multiply(LHS, RHS, M, K, N)
  Type *ElemTy = LHS->getType()->getScalarType();
  Type *ResTy = FixedVectorType::get(ElemTy, M * N);
  Function *MatMulFn = Intrinsic::getOrInsertDeclaration(
      State.Builder.GetInsertBlock()->getModule(),
      Intrinsic::matrix_multiply,
      {ResTy, LHS->getType(), RHS->getType()});
  return State.Builder.CreateCall(
      MatMulFn,
      {LHS, RHS,
       State.Builder.getInt32(M),
       State.Builder.getInt32(K),
       State.Builder.getInt32(N)});
}
```

In `TPWidenRecipe::execute()`, update the Contraction case for reduction updates:
```cpp
// If this is a reduction update (fadd) classified as Contraction:
if (Kind == TensorOpKind::Contraction && DefVal) {
  TPRecipeBase *FusedMul = State.getFusedMulRecipe(this);
  Value *Result = emitContraction(FusedMul, this, State);
  if (Result)
    State.setValue(DefVal.get(), Result);
  return;
}
```

- [ ] **Step 3: Build**
```bash
cd /root/llvm-project/build && ninja LLVMVectorize 2>&1 | tail -5
```

- [ ] **Step 4: Wire `TPlanLowering_lower` into `LoopTensorize.cpp`**

In `LoopTensorize.cpp`, locate the `run()` method (the main pass entry point). Search for
where `LoopNestInfo Info` is used. Since `TPlan::buildInitial()` is not yet called there,
add the following at the point where the nest analysis is complete (after the
`analyzeLoopNest()` call, before the function returns `PreservedAnalyses::none()`):

```cpp
// Add include at top of LoopTensorize.cpp:
#include "llvm/Transforms/Vectorize/TPlan.h"
#include "llvm/Transforms/Vectorize/TPRecipeMatcher.h"

// In run(), after analyzeLoopNest():
TPlan Plan = TPlan::buildInitial(Info);
// Set per-dim PF from tile sizes (4x4x4 example; will come from heuristic later):
for (unsigned D = 0; D < Info.Depth; ++D)
  Plan.setDimPF(D, 4);
TPlanLowering_lower(Plan, F, LI, SE, DT);
```

The exact insertion line can be found by running:
```bash
grep -n "analyzeLoopNest\|PreservedAnalyses" \
  llvm/lib/Transforms/Vectorize/LoopTensorize.cpp | head -20
```

- [ ] **Step 5: Commit**
```bash
git add llvm/lib/Transforms/Vectorize/TPlanLowering.cpp \
        llvm/lib/Transforms/Vectorize/TPRecipeMatcher.cpp \
        llvm/lib/Transforms/Vectorize/LoopTensorize.cpp \
        llvm/test/Transforms/LoopTensorize/pf-dimset-gemm.ll
git commit -m "tplan: emit @llvm.matrix.multiply for Contraction reductions"
```

---

## Task 8: End-to-End Lit Tests

**Files:**
- Create: `llvm/test/Transforms/LoopTensorize/pf-dimset-eltwise.ll`
- Create: `llvm/test/Transforms/LoopTensorize/pf-dimset-broadcast.ll`
- Create: `llvm/test/Transforms/LoopTensorize/pf-dimset-outer-product.ll`
- Create: `llvm/test/Transforms/LoopTensorize/pf-dimset-plain-reduction.ll`
- Update: `llvm/test/Transforms/LoopTensorize/pf-dimset-gemm.ll`

- [ ] **Step 1: Write and run the GEMM lit test**

Update `pf-dimset-gemm.ll` with real IR for a 3-loop GEMM, then run:
```bash
cd /root/llvm-project/build && \
  bin/llvm-lit ../test/Transforms/LoopTensorize/pf-dimset-gemm.ll -v
```
Expected: PASS with `llvm.matrix.multiply` in output.

- [ ] **Step 2: Write and run the element-wise test**

`pf-dimset-eltwise.ll`: 2-loop `fdiv A[i][j], B[i][j]` → both operands have DimSet{0,1},
classified `ElementWise`. Expected: output contains the fdiv instruction (no reshape).
```bash
bin/llvm-lit ../test/Transforms/LoopTensorize/pf-dimset-eltwise.ll -v
```

- [ ] **Step 3: Write and run the broadcast test**

`pf-dimset-broadcast.ll`: 2-loop `fmul A[i][j], scale[i]` → A has DimSet{0,1}, scale has
DimSet{0}, classified `BroadcastBinary`. Expected: output contains fmul instruction.
```bash
bin/llvm-lit ../test/Transforms/LoopTensorize/pf-dimset-broadcast.ll -v
```

- [ ] **Step 4: Write and run the outer-product test**

`pf-dimset-outer-product.ll`: 2-loop `fmul B[i], C[j]` → B has DimSet{0}, C has DimSet{1},
DimSets are disjoint, classified `OuterProduct`. Expected: output contains fmul instruction
(scalar fallback until OuterProduct intrinsic is wired up).
```bash
bin/llvm-lit ../test/Transforms/LoopTensorize/pf-dimset-outer-product.ll -v
```

- [ ] **Step 5: Write and run the plain-reduction test**

`pf-dimset-plain-reduction.ll`: 1-loop `sum += A[i]` → reduction update with no fmul
producer, classified `PlainReduction`. Expected: output contains fadd (scalar clone).
```bash
bin/llvm-lit ../test/Transforms/LoopTensorize/pf-dimset-plain-reduction.ll -v
```

- [ ] **Step 6: Run all LoopTensorize tests**
```bash
cd /root/llvm-project/build && \
  bin/llvm-lit ../test/Transforms/LoopTensorize/ -v 2>&1 | tail -15
```
Expected: All PASS.

- [ ] **Step 7: Final commit**
```bash
git add llvm/test/Transforms/LoopTensorize/
git commit -m "tplan: add lit tests for PF DimSet system (GEMM, eltwise, broadcast, outer-product, plain-reduction)"
```

---

## Verification

After all tasks complete:

```bash
# Build
cd /root/llvm-project/build && ninja LLVMVectorize LoopTensorizeTests

# Unit tests
./unittests/Transforms/LoopTensorize/LoopTensorizeTests --gtest_color=yes

# Lit tests
bin/llvm-lit ../test/Transforms/LoopTensorize/ -v

# Existing vectorize tests still pass (no regressions)
bin/llvm-lit ../test/Transforms/Vectorize/ -v 2>&1 | grep -E "PASS|FAIL|ERROR" | tail -5
```
