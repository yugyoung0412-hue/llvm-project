# TPlan Multi-Region Refactor Design Spec

**Date:** 2026-03-25
**Branch:** LoopTensorizebyClaude
**Goal:** Extend `TPRegionBlock` and `TPlan` with the named structural fields described in `TPlanSummary.txt` (`Middle`, `Scalar`, `Inner`, `Loop2HeaderTPB`, `Loop2LatchTPB`, `Preheader`, `Regions`, `LoopIdx2TPRB`), update `buildInitial()` to populate them, and replace the generic `constructionOrder` traversal in `TPRegionBlock::print()`/`execute()` with an explicit Inner-aware traversal while retaining the original as `printFlat()`/`executeFlat()`.

---

## 1. Motivation

The 2026-03-24 refactor introduced the `TPBlockBase`/`TPBasicBlock`/`TPRegionBlock` hierarchy, wiring all structural blocks together via successor/predecessor links. However the `TPRegionBlock` class stores only `Entry` and `Exiting` — the remaining structural blocks (`Middle`, `Scalar`) and the nested region pointer (`Inner`) are only reachable by following successor edges. Similarly `TPlan` has no first-class list of regions and no Loop→region lookup.

This refactor adds those named fields so that passes can navigate the region hierarchy directly without traversing the successor graph.

---

## 2. Index Convention

Throughout TPlan the **innermost loop is index 0, outermost is N-1**.

> **Note:** `TPlanSummary.txt` Section 3 describes `Regions` as "outer-most first" (`Regions[0]` = outermost). This spec intentionally diverges: `Regions[0]` = innermost, `Regions[N-1]` = outermost. This matches the natural recursion-unwind order of `BuildRegion` and the `DimIdx` convention below.

| `AllLoops[Idx]` | `DimIdx = Depth-1-Idx` | `Level` (block naming) | `Regions[…]` |
|---|---|---|---|
| 0 (outermost) | N-1 | N-1 | pushed last → `Regions[N-1]` |
| 1 (middle) | N-2 | N-2 | — |
| N-1 (innermost) | 0 | 0 | pushed first → `Regions[0]` |

`DimIdx` is used for `TPWidenInductionRecipe::getDimIndex()`, `DimPFs` access, and `DimSet` bit positions in `TPlanWidener`. Block names (`tensor.latch0`, `tensor.loop0`, etc.) use `Level = Depth-1-Idx` which equals `DimIdx`, so naming stays consistent.

### 2.1 `ReductionDims` remapping

`P.ReductionDims` is currently set as:
```cpp
P.ReductionDims = Info.ReductionDims;
```
`Info.ReductionDims` uses the old outermost=0 convention (bit 0 = outermost). After the DimIdx reversal, `TPlanWidener` seeds `DimSet` bits using `getDimIndex()` (innermost=0). `TPRecipeMatcher` compares those bits against `ReductionDims`. To keep them consistent, `ReductionDims` must be remapped at assignment time:

```cpp
// Remap ReductionDims: old bit i (outermost=0) → new bit Depth-1-i (innermost=0).
// IMPORTANT: always test Info.ReductionDims (the source), not P.ReductionDims,
// to avoid reading back previously written destination bits.
P.ReductionDims.resize(P.Depth);
for (unsigned I = 0; I < P.Depth; ++I)
  if (Info.ReductionDims.test(I))
    P.ReductionDims.set(P.Depth - 1 - I);
```

This is a behavioral change and **must be validated** by running the full LoopTensorize lit suite, particularly `pf-dimset-gemm.ll` (Contraction classification) and `pf-dimset-plain-reduction.ll`.

---

## 3. Data Model Changes

### 3.1 `TPRegionBlock` additions

```cpp
// New private fields:
TPBlockBase   *Middle = nullptr;    // epilogue-check block; null for innermost
TPBlockBase   *Scalar = nullptr;    // scalar preheader; null for innermost
TPRegionBlock *Inner  = nullptr;    // next-inner nested region; null if leaf
DenseMap<Loop *, TPBlockBase *> Loop2HeaderTPB;
DenseMap<Loop *, TPBlockBase *> Loop2LatchTPB;

// New public accessors:
TPBlockBase   *getMiddle() const         { return Middle; }
void           setMiddle(TPBlockBase *B) { Middle = B; }

TPBlockBase   *getScalar() const         { return Scalar; }
void           setScalar(TPBlockBase *B) { Scalar = B; }

TPRegionBlock *getInner() const               { return Inner; }
void           setInner(TPRegionBlock *R)     { Inner = R; }

// Callers that need BasicBlock* access should cast<TPIRBasicBlock>(getHeaderForLoop(L)).
TPBlockBase   *getHeaderForLoop(Loop *L) const     { return Loop2HeaderTPB.lookup(L); }
void           setHeaderForLoop(Loop *L, TPBlockBase *B) { Loop2HeaderTPB[L] = B; }

TPBlockBase   *getLatchForLoop(Loop *L) const      { return Loop2LatchTPB.lookup(L); }
void           setLatchForLoop(Loop *L, TPBlockBase *B)  { Loop2LatchTPB[L] = B; }
```

`HeaderBB` is always a `TPIRBasicBlock*`; `LatchBB` is always a `TPBasicBlock*`. Both are stored as `TPBlockBase*` in the maps. Callers that need `getIRBasicBlock()` should `cast<TPIRBasicBlock>(getHeaderForLoop(L))`.

### 3.2 `TPlan` additions

```cpp
// New private fields:
TPBasicBlock *Preheader = nullptr;               // reserved for SCEV expansions; empty in Commit 1
SmallVector<TPRegionBlock *, 4> Regions;         // Regions[0]=innermost, Regions[N-1]=outermost
MapVector<Loop *, TPRegionBlock *> LoopIdx2TPRB; // insertion-order for deterministic iteration

// New public accessors:
TPBasicBlock              *getPreheader() const           { return Preheader; }
void                       setPreheader(TPBasicBlock *B)  { Preheader = B; }

ArrayRef<TPRegionBlock *>  getRegions() const             { return Regions; }

TPRegionBlock             *getRegionForLoop(Loop *L) const {
  auto It = LoopIdx2TPRB.find(L);
  return It != LoopIdx2TPRB.end() ? It->second : nullptr;
}
```

`Entry` remains `TPBlockBase*` pointing to `OuterPH` (the outer preheader block). `TPlanSummary.txt` describes `Entry` as `TPBasicBlock*`; the existing field stores a `TPBasicBlock` at runtime but is typed `TPBlockBase*` for generality. No type change is made in this refactor.

`Preheader` will be wired as a predecessor of `Entry` in a future SCEV-expansion commit. In Commit 1 it is created but not connected to the CFG.

`MapVector` is chosen over `DenseMap` to provide deterministic insertion-order iteration for print and analysis passes.

---

## 4. `buildInitial()` Changes

### 4.1 DimIdx reversal

At the top of `BuildRegion(Idx)`, compute:

```cpp
unsigned DimIdx = P.Depth - 1 - Idx;
```

Replace all uses of `Idx` as a dimension index:

| Old | New |
|---|---|
| `new TPWidenIntOrFpInductionRecipe(..., Idx)` | `new TPWidenIntOrFpInductionRecipe(..., DimIdx)` |
| `new TPWidenPointerInductionRecipe(..., Idx)` | `new TPWidenPointerInductionRecipe(..., DimIdx)` |
| `P.DimPFs[Idx].get()` (both innermost and non-innermost branches) | `P.DimPFs[DimIdx].get()` |

Block naming (`Level = P.Depth - 1 - Idx`) is unchanged — it already equals `DimIdx`.

> **Semantic note:** `DimPFs` is created with names `"PF[0]"…"PF[N-1]"` (by the creation loop `D = 0..Depth-1`). After this refactor `DimPFs[0]` (named `"PF[0]"`) is used by the innermost loop. The `tplan-print.ll` CHECK lines verify the label text (`PF[0]`, `PF[1]`, `PF[2]`) only, not which loop level each corresponds to, so no CHECK update is needed. However the semantic mapping inverts: `PF[0]` was previously the outermost PF, afterwards it is the innermost. Any comments or documentation equating `PF[0]` with "outermost loop" must be updated.

`ReductionDims` remapping is applied once after `BuildRegion` completes (see Section 2.1).

### 4.2 `BuildRegion` additions

The additions are applied inside the `BuildRegion` lambda at two separate points: immediately after header/latch creation (both branches), and before `return Region` (each branch separately).

**Immediately after `HeaderBB` and `LatchBB` are created (both innermost and non-innermost):**
```cpp
// (done after Region is created and setEntry/setExiting called)
Region->setHeaderForLoop(L, HeaderBB);
Region->setLatchForLoop(L, LatchBB);
```

**Non-innermost branch — before `return Region`:**
```cpp
Region->setMiddle(MiddleBB);
Region->setScalar(ScalarPH);
Region->setInner(Child);  // Child = result of BuildRegion(Idx+1), computed before Region

// Register — innermost pushed first (recursion unwinds inner before outer)
P.Regions.push_back(Region);
P.LoopIdx2TPRB[L] = Region;
return Region;
```

**Innermost branch — before `return Region`:**
```cpp
// Middle and Scalar stay null for innermost.
// Inner stays null (leaf region).

P.Regions.push_back(Region);
P.LoopIdx2TPRB[L] = Region;
return Region;
```

Because `BuildRegion(Idx+1)` is called (and pushes its region) before `BuildRegion(Idx)` pushes its own region, the push order is innermost-first: `Regions[0]` = innermost, `Regions[N-1]` = outermost.

After all `BuildRegion` calls complete, the `Inner` chain invariant holds: `Regions[N-1].getInner() == Regions[N-2]`, ..., `Regions[1].getInner() == Regions[0]`, `Regions[0].getInner() == nullptr`.

### 4.3 Outermost region `Middle`/`Scalar` population

The outermost region's `MiddleBB` and `ScalarPH` are created in the **top-level wiring block** (lines 905, 908 of `buildInitial`), outside `BuildRegion`. After `BuildRegion(0)` returns `Outer`, add:

```cpp
Outer->setMiddle(MiddleBB);
Outer->setScalar(ScalarPH);
```

This ensures `getRegions().back().getMiddle()` and `getRegions().back().getScalar()` return non-null for the outermost region, consistent with all non-innermost regions.

### 4.4 Preheader creation

After the top-level wiring block (after `P.setEntry(OuterPH)`):

```cpp
P.Preheader = P.createTPBasicBlock("tensor.preheader");
// Not connected to the CFG in Commit 1.
```

### 4.5 `ReductionDims` remapping

Replace `P.ReductionDims = Info.ReductionDims` with the remapping loop from Section 2.1.

---

## 5. Traversal Helpers

Two free functions in `TPlan.cpp`:

| Function | Scope | Behavior |
|---|---|---|
| `constructionOrder(TPBlockBase *Start)` | Declared in `TPlan.h` (existing) | DFS pre-order following all successors freely — crosses region boundaries. |
| `intraRegionOrder(TPBlockBase *Start)` | File-local `static` in `TPlan.cpp` | DFS pre-order; treats `TPRegionBlock` nodes as **opaque leaf nodes** — adds them to the result list but does not descend into them. Used to traverse a single region's own blocks. |

`intraRegionOrder` is kept `static` in `TPlan.cpp` (not declared in `TPlan.h`) because it is only needed by `print()` and `execute()` internally.

---

## 6. `TPRegionBlock` Print/Execute

### 6.1 Inner-aware (new primary — Commit 2)

**`TPRegionBlock::print()`** replaces the `constructionOrder(Entry)` loop with `intraRegionOrder(Entry)` and dispatches to `getInner()->print()` when the inner region node is encountered:

```
TPRegionBlock::print(OS, Indent, Tracker):
  OS << Indent << "<x1> " << getName() << ": {\n"
  std::string InnerIndentStr = (Indent + "  ").str()
  for B in intraRegionOrder(Entry):
    if B == getInner():
      getInner()->print(OS, InnerIndentStr, Tracker)
    else:
      B->print(OS, InnerIndentStr, Tracker)
  OS << Indent << "}\n"
  printBlockSuccessors(OS, Indent, this)
  OS << "\n"
```

**`TPRegionBlock::execute()`** same pattern — `intraRegionOrder` + explicit `getInner()->execute(State)` when `B == getInner()`.

Printing order is **outermost-to-innermost**: the outer region's `print()` recursively calls `Inner->print()`, which calls its own `Inner->print()`, etc.:

```
tensor.ph2:
<x1> tensor loop2: {
  ir-bb<outer>:
  tensor.latch2:
  tensor.ph1:
  <x1> tensor loop1: {
    ir-bb<middle>:
    tensor.latch1:
    tensor.ph0:
    <x1> tensor loop0: {
      ir-bb<inner>:
      tensor.latch0:
      tensor.body.0:
    }
  }
  middle.block1:
  scalar.ph1:
}
middle.block2:
scalar.ph2:
```

### 6.2 Flat alternative (kept — Commit 2)

The existing `constructionOrder(Entry)` logic is moved into two new methods on `TPRegionBlock`:

```cpp
void printFlat(raw_ostream &OS, const Twine &Indent, TPSlotTracker &Tracker) const;
void executeFlat(TPTransformState &State);
```

These retain the original behavior exactly (including the trailing `OS << "\n"`). No external callers use them initially; they exist for debugging and regression testing.

---

## 7. Delivery Plan

### Commit 1 — Data model + `buildInitial()` + DimIdx reversal

| File | Change |
|---|---|
| `llvm/include/llvm/Transforms/Vectorize/TPlan.h` | Add `Middle`/`Scalar`/`Inner`/`Loop2*` to `TPRegionBlock` with accessors; add `Preheader`/`Regions`/`LoopIdx2TPRB` + accessors to `TPlan` |
| `llvm/lib/Transforms/Vectorize/TPlan.cpp` | Update `buildInitial()`: DimIdx reversal, `ReductionDims` remapping, populate new fields, create `Preheader` |

Gate: `ninja -C build LLVMVectorize` + all LoopTensorize lit tests pass. Pay particular attention to `pf-dimset-gemm.ll` and `pf-dimset-plain-reduction.ll` — the `ReductionDims` remapping is a behavioral change; verify Contraction/PlainReduction classification is preserved. No `tplan-print.ll` CHECK changes expected (print output format is unchanged in Commit 1).

### Commit 2 — Inner-aware traversal + `printFlat`/`executeFlat`

| File | Change |
|---|---|
| `llvm/include/llvm/Transforms/Vectorize/TPlan.h` | Declare `printFlat()`/`executeFlat()` on `TPRegionBlock` |
| `llvm/lib/Transforms/Vectorize/TPlan.cpp` | Add `static intraRegionOrder()`; update `TPRegionBlock::print()` and `execute()` to Inner-aware; add `printFlat()`/`executeFlat()` with original logic |
| `llvm/test/Transforms/LoopTensorize/basic/tplan-print.ll` | Verify CHECK lines pass (output format is unchanged) |

Gate: full LoopTensorize lit suite passes. Note: no existing lit test directly exercises the multi-region `execute()` dispatch path (lowering tests exist but cover single-level loops). The Inner-aware `execute()` is validated structurally by confirming it compiles and produces identical output to `executeFlat()` on existing test inputs; full multi-level lowering test coverage is deferred to a future commit.

---

## 8. What Stays Unchanged

- All `TPRecipeBase` subclasses, recipe `print()`/`execute()` implementations
- `TPValue`, `TPUser`, `TPDef`, `TPRecipeValue`, `TPSlotTracker`
- `TPBasicBlock::print()`/`execute()`, `TPIRBasicBlock::print()`/`execute()`
- `TPlan::print()` — top-level `constructionOrder(Entry)` traversal is unchanged; Inner-aware behavior flows through virtual dispatch into `TPRegionBlock::print()`
- `TPlanWidener.cpp` — DimSet seeding uses `getDimIndex()` which changes values, but the widener logic itself (BFS union) is unchanged. The `ReductionDims` remapping in `buildInitial()` ensures the relative dim relationships (shared vs. disjoint) are preserved for the Contraction classifier
- `TPlanLowering.cpp` — calls `constructionOrder(Plan.getEntry())` at the top level (not on a region), so it is unaffected by the `TPRegionBlock::execute()` change. The Inner-aware behavior is fully encapsulated inside `TPRegionBlock::execute()`
- `TPRecipeMatcher.cpp` — unchanged
