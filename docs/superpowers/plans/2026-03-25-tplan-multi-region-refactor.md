# TPlan Multi-Region Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add named structural fields (`Middle`, `Scalar`, `Inner`, `Loop2HeaderTPB`, `Loop2LatchTPB`, `Preheader`, `Regions`, `LoopIdx2TPRB`) to `TPRegionBlock` and `TPlan`, update `buildInitial()` to populate them with an innermost=0 DimIdx convention, and replace `constructionOrder` in `TPRegionBlock::print()`/`execute()` with explicit Inner-aware traversal while keeping the original as `printFlat()`/`executeFlat()`.

**Architecture:** Two commits. Commit 1 is purely additive: new fields and accessors in the header, wiring in `buildInitial()`, DimIdx reversal, and `ReductionDims` remapping — no print output change. Commit 2 adds `intraRegionOrder()` and swaps `TPRegionBlock::print()`/`execute()` to the Inner-aware traversal, preserving the original as flat alternatives.

**Tech Stack:** C++17, LLVM ADT (`DenseMap`, `MapVector`, `SmallVector`, `SmallBitVector`), LLVM lit (`llvm-lit`), ninja build system.

---

## File Map

| File | Role in this plan |
|---|---|
| `llvm/include/llvm/Transforms/Vectorize/TPlan.h` | Class declarations: new fields + accessors on `TPRegionBlock` (lines 436–471) and `TPlan` (lines 1274–1290); new method declarations for `printFlat`/`executeFlat` (Commit 2) |
| `llvm/lib/Transforms/Vectorize/TPlan.cpp` | `buildInitial()` wiring (lines 660–927); `TPRegionBlock::print()` (lines 531–544); `TPRegionBlock::execute()` (lines 564–571); new `static intraRegionOrder()` (Commit 2); new `printFlat()`/`executeFlat()` (Commit 2) |
| `llvm/test/Transforms/LoopTensorize/basic/tplan-print.ll` | Verify CHECK lines unchanged after Commit 2 |

---

## Task 1: Commit 1 — Data model + `buildInitial()` + DimIdx reversal

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlan.h:436-471` (TPRegionBlock private fields)
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlan.h:1274-1290` (TPlan private fields)
- Modify: `llvm/lib/Transforms/Vectorize/TPlan.cpp:660-927` (buildInitial)

**Build command:** `ninja -C build LLVMVectorize 2>&1 | tail -20`
**Test command:** `build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/`

---

- [ ] **Step 1: Add new private fields to `TPRegionBlock` in `TPlan.h`**

  In `TPlan.h`, find the `TPRegionBlock` private section (lines 467–471):
  ```cpp
  private:
    TPBlockBase *Entry = nullptr;
    TPBlockBase *Exiting = nullptr;
    bool IsReplicator = false;
  ```

  Replace with:
  ```cpp
  private:
    TPBlockBase *Entry    = nullptr;
    TPBlockBase *Exiting  = nullptr;
    TPBlockBase *Middle   = nullptr;  ///< Epilogue-check block; null for innermost.
    TPBlockBase *Scalar   = nullptr;  ///< Scalar preheader; null for innermost.
    TPRegionBlock *Inner  = nullptr;  ///< Next-inner nested region; null if leaf.
    DenseMap<Loop *, TPBlockBase *> Loop2HeaderTPB;
    DenseMap<Loop *, TPBlockBase *> Loop2LatchTPB;
    bool IsReplicator = false;
  ```

- [ ] **Step 2: Add new public accessors to `TPRegionBlock` in `TPlan.h`**

  In `TPlan.h`, find the existing `TPRegionBlock` public accessor block (after `isReplicator()`):
  ```cpp
    bool isReplicator() const { return IsReplicator; }

    void execute(TPTransformState &State) override;
  ```

  Insert the new accessors between `isReplicator()` and `execute()`:
  ```cpp
    bool isReplicator() const { return IsReplicator; }

    TPBlockBase   *getMiddle() const          { return Middle; }
    void           setMiddle(TPBlockBase *B)  { Middle = B; }

    TPBlockBase   *getScalar() const          { return Scalar; }
    void           setScalar(TPBlockBase *B)  { Scalar = B; }

    TPRegionBlock *getInner() const               { return Inner; }
    void           setInner(TPRegionBlock *R)     { Inner = R; }

    /// Returns the header block for loop \p L. Cast to TPIRBasicBlock* to access
    /// the underlying BasicBlock*.
    TPBlockBase *getHeaderForLoop(Loop *L) const { return Loop2HeaderTPB.lookup(L); }
    void setHeaderForLoop(Loop *L, TPBlockBase *B) { Loop2HeaderTPB[L] = B; }

    TPBlockBase *getLatchForLoop(Loop *L) const { return Loop2LatchTPB.lookup(L); }
    void setLatchForLoop(Loop *L, TPBlockBase *B) { Loop2LatchTPB[L] = B; }

    void execute(TPTransformState &State) override;
  ```

- [ ] **Step 3: Add new private fields to `TPlan` in `TPlan.h`**

  `TPlan.h` does not currently include `MapVector.h`. Add the include alongside the other ADT includes at the top of the file (after the existing `#include "llvm/ADT/ilist_node.h"` line):
  ```cpp
  #include "llvm/ADT/MapVector.h"
  ```

  In `TPlan.h`, find the `TPlan` private section (lines 1274–1290):
  ```cpp
  private:
    SmallVector<std::unique_ptr<TPSymbolicValue>> DimPFs;
    std::string FuncName;
    unsigned Depth = 0;
    SmallBitVector ReductionDims;
    DenseMap<unsigned, unsigned> DimPFMap;
    SmallVector<std::unique_ptr<TPIRValue>> LiveIns;
    TPBlockBase *Entry = nullptr;
    SmallVector<TPBlockBase *> CreatedBlocks;
    mutable TPSlotTracker Tracker;
    DenseMap<Value *, TPValue *> ValueMap;
  ```

  Add the three new fields after `Entry`:
  ```cpp
  private:
    SmallVector<std::unique_ptr<TPSymbolicValue>> DimPFs;
    std::string FuncName;
    unsigned Depth = 0;
    SmallBitVector ReductionDims;
    DenseMap<unsigned, unsigned> DimPFMap;
    SmallVector<std::unique_ptr<TPIRValue>> LiveIns;
    TPBlockBase *Entry = nullptr;
    TPBasicBlock *Preheader = nullptr;               ///< Reserved for SCEV expansions.
    SmallVector<TPRegionBlock *, 4> Regions;         ///< [0]=innermost, [N-1]=outermost.
    MapVector<Loop *, TPRegionBlock *> LoopIdx2TPRB; ///< Insertion-order for deterministic iteration.
    SmallVector<TPBlockBase *> CreatedBlocks;
    mutable TPSlotTracker Tracker;
    DenseMap<Value *, TPValue *> ValueMap;
  ```

- [ ] **Step 4: Add new public accessors to `TPlan` in `TPlan.h`**

  In `TPlan.h`, find the `TPlan` public section where `getEntry()`/`setEntry()` are declared (around line 1252):
  ```cpp
    /// Entry block (outermost preheader, a TPBasicBlock).
    TPBlockBase *getEntry() const { return Entry; }
    /// Set the plan's outermost entry block.
    void setEntry(TPBlockBase *B) { Entry = B; }
  ```

  Add the new accessors immediately after `setEntry`:
  ```cpp
    /// Entry block (outermost preheader, a TPBasicBlock).
    TPBlockBase *getEntry() const { return Entry; }
    /// Set the plan's outermost entry block.
    void setEntry(TPBlockBase *B) { Entry = B; }

    /// Preheader block reserved for SCEV expansions (empty in initial plan).
    TPBasicBlock *getPreheader() const           { return Preheader; }
    void          setPreheader(TPBasicBlock *B)  { Preheader = B; }

    /// All TPRegionBlocks: Regions[0]=innermost, Regions[N-1]=outermost.
    ArrayRef<TPRegionBlock *> getRegions() const { return Regions; }

    /// Returns the TPRegionBlock that owns loop \p L, or nullptr.
    TPRegionBlock *getRegionForLoop(Loop *L) const {
      auto It = LoopIdx2TPRB.find(L);
      return It != LoopIdx2TPRB.end() ? It->second : nullptr;
    }
  ```

- [ ] **Step 5: Verify the header compiles**

  ```bash
  ninja -C build LLVMVectorize 2>&1 | tail -20
  ```
  Expected: compiles cleanly (any `Loop` forward-declaration issue: add `class Loop;` to the forward-decl block at the top of `TPlan.h` if not already present — check with `grep -n "^class Loop" llvm/include/llvm/Transforms/Vectorize/TPlan.h`).

- [ ] **Step 6: Add DimIdx variable to `BuildRegion` lambda in `TPlan.cpp`**

  In `TPlan.cpp`, find the start of `BuildRegion` (line ~669):
  ```cpp
    unsigned Level = P.Depth - 1 - Idx;
    std::string LevelStr = std::to_string(Level);
  ```

  Add `DimIdx` immediately after `Level`:
  ```cpp
    unsigned Level  = P.Depth - 1 - Idx;
    unsigned DimIdx = Level; // innermost=0, outermost=Depth-1 (equals Level)
    std::string LevelStr = std::to_string(Level);
  ```

  > `DimIdx == Level` since both equal `Depth-1-Idx`. The separate name makes intent explicit.

- [ ] **Step 7: Replace `Idx` with `DimIdx` in the two induction recipe constructors**

  In `TPlan.cpp`, find lines 727–733 (both branches of the induction phi handling):
  ```cpp
          R = new TPWidenPointerInductionRecipe(
              &Phi, StartTP,
              StartTP /* placeholder; patched after body */, Idx);
        else
          R = new TPWidenIntOrFpInductionRecipe(
              &Phi, StartTP,
              StartTP /* placeholder; patched after body */, Idx);
  ```

  Change both trailing `Idx` arguments to `DimIdx`:
  ```cpp
          R = new TPWidenPointerInductionRecipe(
              &Phi, StartTP,
              StartTP /* placeholder; patched after body */, DimIdx);
        else
          R = new TPWidenIntOrFpInductionRecipe(
              &Phi, StartTP,
              StartTP /* placeholder; patched after body */, DimIdx);
  ```

- [ ] **Step 8: Replace `P.DimPFs[Idx]` with `P.DimPFs[DimIdx]` in both branches**

  There are two occurrences of `P.DimPFs[Idx].get()` in `buildInitial()`:

  **Non-innermost branch (line ~812):**
  ```cpp
      auto *IncrRecipe = new TPCanonicalIVIncrRecipe(CanonIV, P.DimPFs[Idx].get());
  ```
  Change to:
  ```cpp
      auto *IncrRecipe = new TPCanonicalIVIncrRecipe(CanonIV, P.DimPFs[DimIdx].get());
  ```

  **Innermost branch (line ~866):**
  ```cpp
      auto *IncrRecipe = new TPCanonicalIVIncrRecipe(CanonIV, P.DimPFs[Idx].get());
  ```
  Change to:
  ```cpp
      auto *IncrRecipe = new TPCanonicalIVIncrRecipe(CanonIV, P.DimPFs[DimIdx].get());
  ```

- [ ] **Step 9: Populate new `TPRegionBlock` fields in the non-innermost branch**

  In `TPlan.cpp`, non-innermost branch, find the block ending with `return Region` (line ~842). Before that return, add:
  ```cpp
      // Populate named structural fields.
      Region->setHeaderForLoop(L, HeaderBB);
      Region->setLatchForLoop(L, LatchBB);
      // Note: setMiddle/setScalar here sets the child-level (intermediate) region's
      // MiddleBB and ScalarPH as allocated inside this BuildRegion(Idx) call. For
      // whatever level becomes Outer (Idx==0), Step 11 will overwrite these with the
      // outermost MiddleBB/ScalarPH created in the top-level wiring block.
      Region->setMiddle(MiddleBB);
      Region->setScalar(ScalarPH);
      Region->setInner(Child);

      // Register region — innermost pushes first (recursion unwinds inner → outer).
      P.Regions.push_back(Region);
      P.LoopIdx2TPRB[L] = Region;

      return Region;
  ```
  Remove the old bare `return Region;` that was there.

- [ ] **Step 10: Populate new `TPRegionBlock` fields in the innermost branch**

  In `TPlan.cpp`, innermost branch, find the `return Region` (line ~896). Before it, add:
  ```cpp
    // Populate named structural fields.
    Region->setHeaderForLoop(L, HeaderBB);
    Region->setLatchForLoop(L, LatchBB);
    // Middle, Scalar, Inner stay null for leaf region.

    P.Regions.push_back(Region);
    P.LoopIdx2TPRB[L] = Region;

    return Region;
  ```
  Remove the old bare `return Region;`.

- [ ] **Step 11: Set `Middle`/`Scalar` on the outermost region and create `Preheader`**

  In `TPlan.cpp`, top-level wiring block (lines ~899–923). Find:
  ```cpp
    P.setEntry(OuterPH);
  ```

  Add immediately after `P.setEntry(OuterPH)`:
  ```cpp
    // Wire Middle/Scalar on the outermost region (created outside BuildRegion).
    // For depth-1 plans Outer is also the leaf region — Middle/Scalar must stay null
    // (leaf-region invariant). Only set them for multi-level nests.
    if (P.Depth > 1) {
      Outer->setMiddle(MiddleBB);
      Outer->setScalar(ScalarPH);
    }

    // Create the SCEV-expansion preheader (empty; will be wired in a future commit).
    P.Preheader = P.createTPBasicBlock("tensor.preheader");
  ```

- [ ] **Step 12: Replace `ReductionDims` assignment with remapping loop**

  In `TPlan.cpp`, after the top-level wiring block, find (line ~925):
  ```cpp
  P.ReductionDims = Info.ReductionDims;
  ```

  Replace with:
  ```cpp
  // Remap ReductionDims from outermost=0 (LoopNestAnalyzer convention) to
  // innermost=0 (TPlan DimIdx convention). Always test Info.ReductionDims —
  // never P.ReductionDims — to avoid reading back previously written bits.
  P.ReductionDims.resize(P.Depth);
  for (unsigned I = 0; I < P.Depth; ++I)
    if (Info.ReductionDims.test(I))
      P.ReductionDims.set(P.Depth - 1 - I);
  ```

- [ ] **Step 13: Build**

  ```bash
  ninja -C build LLVMVectorize 2>&1 | tail -20
  ```
  Expected: zero errors, zero warnings about new fields.

- [ ] **Step 14: Run all LoopTensorize lit tests**

  ```bash
  build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/
  ```
  Expected: all tests pass. Pay special attention to `pf-dimset-gemm.ll` (must still emit `llvm.matrix.multiply`) and `pf-dimset-plain-reduction.ll` (must still emit `fadd`). No `tplan-print.ll` CHECK line changes expected.

- [ ] **Step 15: Commit**

  ```bash
  git add llvm/include/llvm/Transforms/Vectorize/TPlan.h \
          llvm/lib/Transforms/Vectorize/TPlan.cpp
  git commit -m "tplan: add multi-region fields (Middle/Scalar/Inner/Regions/Preheader) and DimIdx reversal"
  ```

---

## Task 2: Commit 2 — Inner-aware traversal + `printFlat`/`executeFlat`

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlan.h:460-465` (TPRegionBlock public methods)
- Modify: `llvm/lib/Transforms/Vectorize/TPlan.cpp:531-571` (print/execute)

**Build command:** `ninja -C build LLVMVectorize 2>&1 | tail -20`
**Test command:** `build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/`

---

- [ ] **Step 1: Add `intraRegionOrder` static function to `TPlan.cpp`**

  In `TPlan.cpp`, find the existing `constructionOrder` function (line ~271). Add `intraRegionOrder` immediately after it:

  ```cpp
  /// DFS pre-order from \p Start, treating TPRegionBlock nodes as opaque leaves
  /// (adds them to the result but does not descend into their internal CFG).
  /// Used to traverse a single region's own blocks without crossing into nested regions.
  static SmallVector<TPBlockBase *, 8>
  intraRegionOrder(TPBlockBase *Start) {
    SmallVector<TPBlockBase *, 8> Order;
    SmallPtrSet<TPBlockBase *, 8> Visited;
    SmallVector<TPBlockBase *, 8> Stack;
    Stack.push_back(Start);
    while (!Stack.empty()) {
      TPBlockBase *B = Stack.pop_back_val();
      if (!Visited.insert(B).second)
        continue;
      Order.push_back(B);
      // Do NOT descend into nested regions — treat them as opaque leaves.
      if (isa<TPRegionBlock>(B))
        continue;
      for (TPBlockBase *Succ : llvm::reverse(B->getSuccessors()))
        if (!Visited.count(Succ))
          Stack.push_back(Succ);
    }
    return Order;
  }
  ```

- [ ] **Step 2: Update `TPRegionBlock::print()` to use Inner-aware traversal**

  In `TPlan.cpp`, find the current `TPRegionBlock::print()` (lines 531–544):
  ```cpp
  void TPRegionBlock::print(raw_ostream &OS, const Twine &Indent,
                             TPSlotTracker &Tracker) const {
    OS << Indent << "<x1> " << getName() << ": {\n";
    if (Entry) {
      std::string InnerIndentStr = (Indent + "  ").str();
      for (TPBlockBase *B : constructionOrder(Entry))
        B->print(OS, InnerIndentStr, Tracker);
    }
    OS << Indent << "}\n";
    printBlockSuccessors(OS, Indent, this);
    OS << "\n";
  }
  ```

  Replace with:
  ```cpp
  void TPRegionBlock::print(raw_ostream &OS, const Twine &Indent,
                             TPSlotTracker &Tracker) const {
    OS << Indent << "<x1> " << getName() << ": {\n";
    if (Entry) {
      std::string InnerIndentStr = (Indent + "  ").str();
      for (TPBlockBase *B : intraRegionOrder(Entry)) {
        if (Inner && B == Inner)
          Inner->print(OS, InnerIndentStr, Tracker);
        else
          B->print(OS, InnerIndentStr, Tracker);
      }
    }
    OS << Indent << "}\n";
    printBlockSuccessors(OS, Indent, this);
    OS << "\n";
  }
  ```

- [ ] **Step 3: Update `TPRegionBlock::execute()` to use Inner-aware traversal**

  In `TPlan.cpp`, find the current `TPRegionBlock::execute()` (lines 564–571):
  ```cpp
  void TPRegionBlock::execute(TPTransformState &State) {
    if (Entry)
      for (TPBlockBase *B : constructionOrder(Entry))
        B->execute(State);
  }
  ```

  Replace with:
  ```cpp
  void TPRegionBlock::execute(TPTransformState &State) {
    if (!Entry)
      return;
    for (TPBlockBase *B : intraRegionOrder(Entry)) {
      if (Inner && B == Inner)
        Inner->execute(State);
      else
        B->execute(State);
    }
  }
  ```

- [ ] **Step 4: Declare `printFlat()` and `executeFlat()` in `TPlan.h`**

  In `TPlan.h`, find the existing `TPRegionBlock` method declarations (around line 462):
  ```cpp
    void execute(TPTransformState &State) override;
    void print(raw_ostream &OS, const Twine &Indent,
               TPSlotTracker &Tracker) const override;
  ```

  Add the flat alternatives:
  ```cpp
    void execute(TPTransformState &State) override;
    void print(raw_ostream &OS, const Twine &Indent,
               TPSlotTracker &Tracker) const override;

    /// Flat alternatives using constructionOrder (original behavior, for debugging).
    void printFlat(raw_ostream &OS, const Twine &Indent,
                   TPSlotTracker &Tracker) const;
    void executeFlat(TPTransformState &State);
  ```

- [ ] **Step 5: Add `printFlat()` and `executeFlat()` implementations to `TPlan.cpp`**

  Immediately after the new `TPRegionBlock::execute()`, add:

  ```cpp
  void TPRegionBlock::printFlat(raw_ostream &OS, const Twine &Indent,
                                 TPSlotTracker &Tracker) const {
    OS << Indent << "<x1> " << getName() << ": {\n";
    if (Entry) {
      std::string InnerIndentStr = (Indent + "  ").str();
      for (TPBlockBase *B : constructionOrder(Entry))
        B->print(OS, InnerIndentStr, Tracker);
    }
    OS << Indent << "}\n";
    printBlockSuccessors(OS, Indent, this);
    OS << "\n";
  }

  void TPRegionBlock::executeFlat(TPTransformState &State) {
    if (Entry)
      for (TPBlockBase *B : constructionOrder(Entry))
        B->execute(State);
  }
  ```

- [ ] **Step 6: Build**

  ```bash
  ninja -C build LLVMVectorize 2>&1 | tail -20
  ```
  Expected: zero errors.

- [ ] **Step 7: Run all LoopTensorize lit tests**

  ```bash
  build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/
  ```
  Expected: all tests pass, especially `tplan-print.ll` — output format must be identical to before Commit 2 (the Inner-aware traversal visits blocks in the same order as `constructionOrder` for the region topology built by `buildInitial()`).

  If `tplan-print.ll` fails, compare the actual output against CHECK lines:
  ```bash
  build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tplan-print.ll 2>&1
  ```
  Any ordering difference reveals a bug in `intraRegionOrder` — verify that `TPRegionBlock` successor edges are still followed correctly by checking `InnerPH`'s successors include `Child` (the Inner region).

- [ ] **Step 8: Commit**

  ```bash
  git add llvm/include/llvm/Transforms/Vectorize/TPlan.h \
          llvm/lib/Transforms/Vectorize/TPlan.cpp
  git commit -m "tplan: add Inner-aware print/execute; keep printFlat/executeFlat as alternatives"
  ```
