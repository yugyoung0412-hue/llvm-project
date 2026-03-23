# TPlan → VPlan-Style Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat `TPLoopRegion` tree with a VPlan-style `TPBlockBase`/`TPBasicBlock`/`TPIRBasicBlock`/`TPRegionBlock`/`TPBlockUtils` hierarchy, rewrite `buildInitial()` and `print()`, stub lowering for compile, then rewire lowering in commit 2.

**Architecture:** Two commits. Commit 1: new block classes in `TPlan.h`, rewritten `buildInitial()` + `print()` in `TPlan.cpp`, no-op stubs in `TPlanWidener.cpp`/`TPlanLowering.cpp`, updated lit test. Commit 2: implement `execute()` on all block types, rewire `TPlanLowering_lower()` and `TPlanWidener_widen()` to walk the block CFG.

**Tech Stack:** C++17, LLVM ADT (`SmallVector`, `SmallPtrSet`, `iplist`, `ilist_node`), LLVM Loop/BasicBlock IR APIs, FileCheck lit tests, ninja build.

**Spec:** `docs/superpowers/specs/2026-03-24-tplan-vplan-refactor-design.md`

---

## File Map

| File | Commit | Change |
|------|--------|--------|
| `llvm/include/llvm/Transforms/Vectorize/TPlan.h` | 1 | Add block classes; remove `TPLoopRegion`; update `TPlan` |
| `llvm/lib/Transforms/Vectorize/TPlan.cpp` | 1 | Implement constructors, `print()`, rewrite `buildInitial()` |
| `llvm/lib/Transforms/Vectorize/TPlanWidener.cpp` | 1 | Replace region-tree walk with no-op stub |
| `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` | 1 | Replace region-tree walk with no-op stub |
| `llvm/test/Transforms/LoopTensorize/basic/tplan-print.ll` | 1 | Update CHECK lines for new block format |
| `llvm/lib/Transforms/Vectorize/TPlan.cpp` | 2 | Implement `execute()` for all block types |
| `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` | 2 | Rewire to block CFG walk |
| `llvm/lib/Transforms/Vectorize/TPlanWidener.cpp` | 2 | Rewire to `tp_all_basicblocks()` traversal |
| `llvm/include/llvm/Transforms/Vectorize/TPlan.h` | 2 | No change — `execute()` is already declared pure virtual from Commit 1; the spec's "Finalize execute() as pure virtual" is satisfied by the Commit 1 header |

---

## ── COMMIT 1: DATA MODEL + PRINT ──

---

### Task 1: Add block-class forward decls + `TPBlockBase` to `TPlan.h`

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlan.h`

- [ ] **Step 1.1: Add forward declarations**

After the existing forward declarations (around line 24–38), add:

```cpp
class BasicBlock;
class TPBasicBlock;
class TPBlockBase;
class TPBlockUtils;
class TPIRBasicBlock;
class TPRegionBlock;
```

- [ ] **Step 1.2: Add `TPBlockBase` class**

After the existing `class TPSyntheticValue` block (around line 116), add the following. Note the `SubclassID` enum values and the `friend TPBlockUtils` declaration.

```cpp
//===----------------------------------------------------------------------===//
// TPBlockBase — base for all TPlan CFG blocks (mirrors VPBlockBase)
//===----------------------------------------------------------------------===//
class TPBlockBase {
  friend class TPBlockUtils;

public:
  using TPBlockTy = enum { TPRegionBlockSC, TPBasicBlockSC, TPIRBasicBlockSC };
  using TPBlocksTy = SmallVectorImpl<TPBlockBase *>;

protected:
  TPBlockBase(unsigned char SC, StringRef N) : SubclassID(SC), Name(N.str()) {}

public:
  virtual ~TPBlockBase() = default;

  unsigned getTPBlockID() const { return SubclassID; }
  const std::string &getName() const { return Name; }
  void setName(StringRef N) { Name = N.str(); }

  TPRegionBlock *getParent() { return Parent; }
  const TPRegionBlock *getParent() const { return Parent; }
  void setParent(TPRegionBlock *P) { Parent = P; }

  const TPBlocksTy &getSuccessors() const { return Successors; }
  TPBlocksTy &getSuccessors() { return Successors; }
  const TPBlocksTy &getPredecessors() const { return Predecessors; }
  TPBlocksTy &getPredecessors() { return Predecessors; }

  TPBlockBase *getSingleSuccessor() const {
    return Successors.size() == 1 ? Successors[0] : nullptr;
  }
  TPBlockBase *getSinglePredecessor() const {
    return Predecessors.size() == 1 ? Predecessors[0] : nullptr;
  }
  size_t getNumSuccessors() const { return Successors.size(); }
  size_t getNumPredecessors() const { return Predecessors.size(); }

  /// Only valid on the plan's entry block.
  TPlan *getPlan() const { return Plan; }
  void setPlan(TPlan *P) { Plan = P; }

  void setOneSuccessor(TPBlockBase *S) {
    assert(Successors.empty() && "Successor already set");
    assert(S->getParent() == getParent() && "Blocks must share parent");
    appendSuccessor(S);
  }
  void setTwoSuccessors(TPBlockBase *S0, TPBlockBase *S1) {
    assert(Successors.empty() && "Successors already set");
    assert(S0->getParent() == getParent() && "Blocks must share parent");
    assert(S1->getParent() == getParent() && "Blocks must share parent");
    appendSuccessor(S0);
    appendSuccessor(S1);
  }
  void setPredecessors(ArrayRef<TPBlockBase *> Preds) {
    assert(Predecessors.empty() && "Predecessors already set");
    for (auto *P : Preds) appendPredecessor(P);
  }
  void setSuccessors(ArrayRef<TPBlockBase *> Succs) {
    assert(Successors.empty() && "Successors already set");
    for (auto *S : Succs) appendSuccessor(S);
  }
  void clearPredecessors() { Predecessors.clear(); }
  void clearSuccessors() { Successors.clear(); }

  virtual void execute(TPTransformState &State) = 0;
  virtual void print(raw_ostream &OS, const Twine &Indent,
                     TPSlotTracker &Tracker) const = 0;

private:
  const unsigned char SubclassID;
  std::string Name;
  TPRegionBlock *Parent = nullptr;
  SmallVector<TPBlockBase *, 1> Predecessors;
  SmallVector<TPBlockBase *, 1> Successors;

  TPlan *Plan = nullptr; ///< Only set on the plan's entry block.

  void appendSuccessor(TPBlockBase *S) { Successors.push_back(S); }
  void appendPredecessor(TPBlockBase *P) { Predecessors.push_back(P); }
  void removeSuccessor(TPBlockBase *S) {
    auto It = llvm::find(Successors, S);
    assert(It != Successors.end()); Successors.erase(It);
  }
  void removePredecessor(TPBlockBase *P) {
    auto It = llvm::find(Predecessors, P);
    assert(It != Predecessors.end()); Predecessors.erase(It);
  }
  void replacePredecessor(TPBlockBase *Old, TPBlockBase *New) {
    auto It = llvm::find(Predecessors, Old);
    assert(It != Predecessors.end()); *It = New;
  }
};
```

- [ ] **Step 1.3: Verify it compiles so far (header-only check)**

```bash
ninja -C build LLVMVectorize 2>&1 | grep -E "error:|TPlan" | head -20
```

Expected: errors only about missing `TPBasicBlock` etc. (not yet defined). No syntax errors in the new block.

---

### Task 2: Add `TPBasicBlock`, `TPIRBasicBlock`, `TPRegionBlock`, `TPBlockUtils` to `TPlan.h`

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlan.h`

- [ ] **Step 2.1: Add `TPBasicBlock`**

Immediately after the `TPBlockBase` class, add:

```cpp
//===----------------------------------------------------------------------===//
// TPBasicBlock — named block owning a recipe list (mirrors VPBasicBlock)
//===----------------------------------------------------------------------===//
class TPBasicBlock : public TPBlockBase {
public:
  using RecipeListTy = iplist<TPRecipeBase>;

  explicit TPBasicBlock(StringRef Name = "")
      : TPBlockBase(TPBasicBlockSC, Name) {}

  static bool classof(const TPBlockBase *B) {
    return B->getTPBlockID() == TPBasicBlockSC ||
           B->getTPBlockID() == TPIRBasicBlockSC;
  }

  using iterator = RecipeListTy::iterator;
  using const_iterator = RecipeListTy::const_iterator;
  iterator begin() { return Recipes.begin(); }
  iterator end() { return Recipes.end(); }
  const_iterator begin() const { return Recipes.begin(); }
  const_iterator end() const { return Recipes.end(); }
  bool empty() const { return Recipes.empty(); }
  RecipeListTy &getRecipeList() { return Recipes; }

  void appendRecipe(TPRecipeBase *R) { Recipes.push_back(R); }

  /// Returns a pointer to the recipe list for ilist_node parent access.
  static RecipeListTy TPBasicBlock::*getSublistAccess(TPRecipeBase *) {
    return &TPBasicBlock::Recipes;
  }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &OS, const Twine &Indent,
             TPSlotTracker &Tracker) const override;

protected:
  explicit TPBasicBlock(unsigned char SC, StringRef Name)
      : TPBlockBase(SC, Name) {}

  RecipeListTy Recipes;
};
```

- [ ] **Step 2.2: Add `TPIRBasicBlock`**

Immediately after `TPBasicBlock`:

```cpp
//===----------------------------------------------------------------------===//
// TPIRBasicBlock — wraps an IR BasicBlock (mirrors VPIRBasicBlock)
//===----------------------------------------------------------------------===//
class TPIRBasicBlock : public TPBasicBlock {
public:
  explicit TPIRBasicBlock(BasicBlock *BB)
      : TPBasicBlock(TPIRBasicBlockSC,
                     (Twine("ir-bb<") + BB->getName() + ">").str()),
        IRBB(BB) {}

  BasicBlock *getIRBasicBlock() const { return IRBB; }

  static bool classof(const TPBlockBase *B) {
    return B->getTPBlockID() == TPIRBasicBlockSC;
  }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &OS, const Twine &Indent,
             TPSlotTracker &Tracker) const override;

private:
  BasicBlock *IRBB;
};
```

- [ ] **Step 2.3: Add `TPRegionBlock`**

Immediately after `TPIRBasicBlock`:

```cpp
//===----------------------------------------------------------------------===//
// TPRegionBlock — SESE loop region with Entry + Exiting (mirrors VPRegionBlock)
//===----------------------------------------------------------------------===//
class TPRegionBlock : public TPBlockBase {
public:
  explicit TPRegionBlock(StringRef Name = "", bool IsReplicator = false)
      : TPBlockBase(TPRegionBlockSC, Name), IsReplicator(IsReplicator) {}

  static bool classof(const TPBlockBase *B) {
    return B->getTPBlockID() == TPRegionBlockSC;
  }

  TPBlockBase *getEntry() { return Entry; }
  const TPBlockBase *getEntry() const { return Entry; }
  TPBlockBase *getExiting() { return Exiting; }
  const TPBlockBase *getExiting() const { return Exiting; }

  void setEntry(TPBlockBase *B) {
    assert(B->getPredecessors().empty() && "Entry must have no predecessors");
    Entry = B;
    B->setParent(this);
  }
  void setExiting(TPBlockBase *B) {
    assert(B->getSuccessors().empty() && "Exiting must have no successors");
    Exiting = B;
    B->setParent(this);
  }

  bool isReplicator() const { return IsReplicator; }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &OS, const Twine &Indent,
             TPSlotTracker &Tracker) const override;

private:
  TPBlockBase *Entry = nullptr;
  TPBlockBase *Exiting = nullptr;
  bool IsReplicator = false;
};
```

- [ ] **Step 2.4: Add `TPBlockUtils`**

Immediately after `TPRegionBlock`. Note `#include "llvm/ADT/STLExtras.h"` is needed for `llvm::find` — add to `TPlan.h` includes if not present (check current includes first; `llvm/ADT/SmallVector.h` already pulls in range utilities).

```cpp
//===----------------------------------------------------------------------===//
// TPBlockUtils — block wiring utilities (mirrors VPBlockUtils)
//===----------------------------------------------------------------------===//
class TPBlockUtils {
public:
  /// Bidirectional connect: adds S to From's successors and From to S's
  /// predecessors. Both blocks must have the same parent.
  static void connectBlocks(TPBlockBase *From, TPBlockBase *To) {
    assert(From->getParent() == To->getParent() &&
           "connectBlocks: blocks must share parent (nullptr == nullptr for "
           "top-level blocks)");
    From->appendSuccessor(To);
    To->appendPredecessor(From);
  }

  /// Bidirectional disconnect.
  static void disconnectBlocks(TPBlockBase *From, TPBlockBase *To) {
    From->removeSuccessor(To);
    To->removePredecessor(From);
  }

  /// Insert New after After: New inherits After's successors, After → New.
  static void insertBlockAfter(TPBlockBase *New, TPBlockBase *After) {
    assert(New->getSuccessors().empty() && New->getPredecessors().empty());
    New->setParent(After->getParent());
    transferSuccessors(After, New);
    connectBlocks(After, New);
  }

  /// Transfer all successors from Old to New (updates predecessor lists too).
  static void transferSuccessors(TPBlockBase *Old, TPBlockBase *New) {
    SmallVector<TPBlockBase *, 4> Succs(Old->getSuccessors());
    Old->clearSuccessors();
    for (auto *S : Succs) {
      auto It = llvm::find(S->getPredecessors(), Old);
      assert(It != S->getPredecessors().end());
      *It = New;
      New->appendSuccessor(S);
    }
  }
};
```

- [ ] **Step 2.5: Add `#include "llvm/ADT/Twine.h"` to `TPlan.h` includes if not already present**

`TPIRBasicBlock` uses `Twine`. Check if it's transitively included via `IRBuilder.h` — if not, add it explicitly at the top of the includes block.

- [ ] **Step 2.6: Declare `constructionOrder` free function in `TPlan.h`**

After the `TPBlockUtils` class, add this declaration so both `TPlan.cpp` and `TPlanLowering.cpp` can call it without duplicating the implementation:

```cpp
/// DFS pre-order traversal from \p Start, following successors in insertion
/// order. Used by print() and block-driven lowering.
SmallVector<TPBlockBase *, 8> constructionOrder(TPBlockBase *Start);
```

This replaces the `static` keyword on the definition in `TPlan.cpp` (Task 4, Step 4.1) — change `static SmallVector<...> constructionOrder(...)` to a non-static definition matching this declaration.

- [ ] **Step 2.7: Build check**

```bash
ninja -C build LLVMVectorize 2>&1 | grep "error:" | head -20
```

Expected: only errors about `TPLoopRegion` still being referenced (we haven't removed it yet) and missing `execute()`/`print()` implementations. No errors in the new class declarations.

---

### Task 3: Remove `TPLoopRegion` and update `TPlan` class in `TPlan.h`

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlan.h`

- [ ] **Step 3.1: Remove `TPLoopRegion` class**

Delete the entire `TPLoopRegion` class definition (currently lines 451–476):
```cpp
//===----------------------------------------------------------------------===//
// TPLoopRegion — one per nesting level, owns recipes + child region
...
};
```

Also remove `class TPLoopRegion;` from the forward declarations at the top.

- [ ] **Step 3.2: Replace `TPplan` private fields and add factory methods**

In the `TPlan` class, make the following changes:

**Remove** these private fields:
```cpp
std::unique_ptr<TPLoopRegion> RootRegion;
```

**Remove** this public method (if present as inline):
```cpp
TPLoopRegion *getRootRegion() const { return RootRegion.get(); }
```

**Add** to the `TPlan` private section:
```cpp
  TPBlockBase *Entry = nullptr;                 ///< Outermost preheader block.
  SmallVector<TPBlockBase *> CreatedBlocks;      ///< Owns all blocks.
```

**Add** to the `TPlan` public section (after the existing `getDimPF` / `getPFForDim` / `setDimPF` accessors):
```cpp
  /// Entry block (outermost preheader, a TPBasicBlock).
  TPBlockBase *getEntry() const { return Entry; }
  void setEntry(TPBlockBase *B) { Entry = B; }

  /// Factory methods — allocate and track blocks.
  TPBasicBlock *createTPBasicBlock(StringRef Name) {
    auto *B = new TPBasicBlock(Name);
    CreatedBlocks.push_back(B);
    return B;
  }
  TPRegionBlock *createTPRegionBlock(StringRef Name) {
    auto *B = new TPRegionBlock(Name);
    CreatedBlocks.push_back(B);
    return B;
  }
  TPIRBasicBlock *createTPIRBasicBlock(BasicBlock *IRBB) {
    auto *B = new TPIRBasicBlock(IRBB);
    CreatedBlocks.push_back(B);
    return B;
  }
```

**Add** a destructor to `TPlan` (in the public section):
```cpp
  ~TPlan() {
    for (auto *B : CreatedBlocks)
      delete B;
  }
```

Since `TPlan` now has a destructor and owns raw pointers, also delete the copy constructor/assignment to prevent double-free:
```cpp
  TPlan(const TPlan &) = delete;
  TPlan &operator=(const TPlan &) = delete;
  TPlan(TPlan &&) = default;
  TPlan &operator=(TPlan &&) = default;
```

- [ ] **Step 3.3: Build check — expect stub errors only**

```bash
ninja -C build LLVMVectorize 2>&1 | grep "error:" | head -30
```

Expected errors: `TPplan::getRootRegion()` called from `TPlanWidener.cpp` and `TPlanLowering.cpp` — these will be fixed in Task 6. Also missing `execute()`/`print()` virtual implementations. No header-level syntax errors expected.

---

### Task 4: Implement block `print()` in `TPlan.cpp`

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlan.cpp`

- [ ] **Step 4.1: Add `constructionOrder` helper and `printBlockSuccessors` helper**

After the existing `printIndent` function (around line 91), add. Note: **not** `static` — the function is declared in `TPlan.h` (added in Task 2, Step 2.6) so `TPlanLowering.cpp` can also call it.

```cpp
/// DFS pre-order traversal starting from \p Start, following successors in
/// insertion order. Visited tracking prevents re-visiting latchBB.
SmallVector<TPBlockBase *, 8>
llvm::constructionOrder(TPBlockBase *Start) {
  SmallVector<TPBlockBase *, 8> Order;
  SmallPtrSet<TPBlockBase *, 8> Visited;
  SmallVector<TPBlockBase *, 8> Stack;
  Stack.push_back(Start);
  while (!Stack.empty()) {
    TPBlockBase *B = Stack.pop_back_val();
    if (!Visited.insert(B).second)
      continue;
    Order.push_back(B);
    // Push successors reversed so successor[0] is visited first (LIFO).
    for (TPBlockBase *Succ : llvm::reverse(B->getSuccessors()))
      if (!Visited.count(Succ))
        Stack.push_back(Succ);
  }
  return Order;
}

static void printBlockSuccessors(raw_ostream &OS, const Twine &Indent,
                                  const TPBlockBase *B) {
  OS << Indent;
  if (B->getSuccessors().empty()) {
    OS << "No successors\n";
    return;
  }
  OS << "Successor(s):";
  for (const TPBlockBase *S : B->getSuccessors())
    OS << " " << S->getName();
  OS << "\n";
}
```

Also add these includes at the top of `TPlan.cpp` if not already present:
```cpp
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"   // for llvm::reverse
```

- [ ] **Step 4.2: Implement `TPBasicBlock::print()`**

After the existing recipe `print()` implementations (after `TPCanonicalIVExitCmpRecipe::print`), add:

```cpp
//===----------------------------------------------------------------------===//
// TPBlockBase subclass print() implementations
//===----------------------------------------------------------------------===//

void TPBasicBlock::print(raw_ostream &OS, const Twine &Indent,
                          TPSlotTracker &Tracker) const {
  OS << Indent << getName() << ":\n";
  // Recipes use unsigned Indent (existing API); compute depth from Twine length.
  unsigned RecipeDepth = Indent.str().size() / 2 + 1;
  for (const TPRecipeBase &R : Recipes)
    R.print(OS, RecipeDepth, Tracker);
  printBlockSuccessors(OS, Indent, this);
  OS << "\n";
}
```

- [ ] **Step 4.3: Implement `TPIRBasicBlock::print()`**

```cpp
void TPIRBasicBlock::print(raw_ostream &OS, const Twine &Indent,
                            TPSlotTracker &Tracker) const {
  OS << Indent << getName() << ":\n";
  unsigned RecipeDepth = Indent.str().size() / 2 + 1;
  for (const TPRecipeBase &R : Recipes)
    R.print(OS, RecipeDepth, Tracker);
  printBlockSuccessors(OS, Indent, this);
  OS << "\n";
}
```

- [ ] **Step 4.4: Implement `TPRegionBlock::print()`**

```cpp
void TPRegionBlock::print(raw_ostream &OS, const Twine &Indent,
                           TPSlotTracker &Tracker) const {
  OS << Indent << "<x1> " << getName() << ": {\n";
  if (Entry) {
    // Materialise the indent string — Twine is a non-owning ref type and
    // must not be stored as a named variable (dangling reference UB).
    std::string InnerIndentStr = (Indent + "  ").str();
    for (TPBlockBase *B : constructionOrder(Entry))
      B->print(OS, InnerIndentStr, Tracker);
  }
  OS << Indent << "}\n";
  printBlockSuccessors(OS, Indent, this);
  OS << "\n";
}
```

- [ ] **Step 4.5: Add stub `execute()` implementations for all three block types**

These stubs allow commit 1 to compile cleanly:

```cpp
//===----------------------------------------------------------------------===//
// TPBlockBase subclass execute() stubs (rewired in commit 2)
//===----------------------------------------------------------------------===//

void TPBasicBlock::execute(TPTransformState &) {
  // TODO: rewire in commit 2
}

void TPIRBasicBlock::execute(TPTransformState &) {
  // TODO: rewire in commit 2
}

void TPRegionBlock::execute(TPTransformState &) {
  // TODO: rewire in commit 2
}
```

- [ ] **Step 4.6: Remove `TPLoopRegion::print()` from `TPlan.cpp`**

Delete lines 219–238 (the `TPLoopRegion::print` implementation).

---

### Task 5: Rewrite `buildInitial()` in `TPlan.cpp`

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlan.cpp`

- [ ] **Step 5.1: Replace the `BuildRegion` lambda signature**

In `TPlan::buildInitial()`, replace the existing recursive lambda:
```cpp
std::function<std::unique_ptr<TPLoopRegion>(unsigned)> BuildRegion = ...
```

with a new lambda returning `TPRegionBlock*`:

```cpp
std::function<TPRegionBlock *(unsigned)> BuildRegion =
    [&](unsigned Idx) -> TPRegionBlock * {
```

- [ ] **Step 5.2: Replace region creation and block setup at the start of the lambda**

Replace:
```cpp
auto Region = std::make_unique<TPLoopRegion>(Idx, L, TC);
```

With:
```cpp
unsigned Level = P.Depth - 1 - Idx;
std::string LevelStr = std::to_string(Level);

auto *headerBB = P.createTPIRBasicBlock(L->getHeader());
auto *latchBB = P.createTPBasicBlock("tensor.latch" + LevelStr);
```

- [ ] **Step 5.3: Redirect recipe emission from `Region` to `headerBB`**

All calls to `Region->appendRecipe(...)` for the CanonIV, PHI recipes, and header instructions should become `headerBB->appendRecipe(...)`. All `Region->setIV(...)` calls can be removed (the IV is tracked via `P.ValueMap` already).

Change:
```cpp
auto *CanonIV = new TPCanonicalIVRecipe(ZeroTP, ZeroTP);
Region->appendRecipe(CanonIV);
```
to:
```cpp
auto *CanonIV = new TPCanonicalIVRecipe(ZeroTP, ZeroTP);
headerBB->appendRecipe(CanonIV);
```

And in the PHI loop, change:
```cpp
Region->appendRecipe(R);
...
Region->setIV(R);
```
to:
```cpp
headerBB->appendRecipe(R);
// (remove Region->setIV(R) — no longer needed)
```

- [ ] **Step 5.4: Route `EmitBlock` output to the correct block**

The existing `EmitBlock` lambda appends recipes to `Region`. Change it to accept a target block:

```cpp
auto EmitBlock = [&](BasicBlock *BB, TPBasicBlock *Target) {
  for (Instruction &Inst : *BB) {
    // ... same recipe creation logic ...
    Target->appendRecipe(R);
    // ... same ValueMap updates ...
  }
};
```

Then call it as:
- `EmitBlock(Header, headerBB)` — header non-PHI instructions go to headerBB
- `EmitBlock(LatchBB, latchBB)` — latch instructions go to latchBB (new, see below)
- For innermost body blocks: `EmitBlock(BB, bodyBB)`
- For non-innermost preheader blocks: `EmitBlock(BB, innerPH)`

Add latch emission explicitly: before the `if (Idx + 1 < AllLoops.size())` check, add:
```cpp
// Emit latch non-PHI instructions to latchBB.
if (BasicBlock *Latch = L->getLoopLatch())
  EmitBlock(Latch, latchBB);
```

- [ ] **Step 5.5: Replace child recursion and add inner-preheader / body blocks**

Replace:
```cpp
if (Idx + 1 < AllLoops.size()) {
  auto Child = BuildRegion(Idx + 1);
  Region->setChild(std::move(Child));
}
```

With:
```cpp
if (Idx + 1 < AllLoops.size()) {
  // Non-innermost: create inner preheader + recurse
  unsigned ChildLevel = Level - 1;
  std::string ChildStr = std::to_string(ChildLevel);
  auto *innerPH = P.createTPBasicBlock("tensor.ph" + ChildStr);
  auto *child = BuildRegion(Idx + 1);
  auto *middleBB = P.createTPBasicBlock("middle.block" + ChildStr);
  BasicBlock *ExitBB = AllLoops[Idx + 1]->getExitBlock();
  auto *cleanupBB = ExitBB ? P.createTPIRBasicBlock(ExitBB) : nullptr;
  auto *scalarPH = P.createTPBasicBlock("scalar.ph" + ChildStr);

  // Emit body blocks (not header, not latch, not in inner loops) to innerPH.
  Loop *InnerLoop = AllLoops[Idx + 1];
  for (BasicBlock *BB : L->blocks()) {
    if (BB == L->getHeader() || BB == L->getLoopLatch()) continue;
    if (InnerLoop->contains(BB)) continue;
    EmitBlock(BB, innerPH);
  }

  // Wire intra-region CFG.
  TPBlockUtils::connectBlocks(headerBB, latchBB);
  TPBlockUtils::connectBlocks(headerBB, innerPH);
  TPBlockUtils::connectBlocks(innerPH, child);
  TPBlockUtils::connectBlocks(child, middleBB);
  if (cleanupBB) {
    TPBlockUtils::connectBlocks(middleBB, cleanupBB);
    TPBlockUtils::connectBlocks(middleBB, scalarPH);
    TPBlockUtils::connectBlocks(cleanupBB, scalarPH);
  } else {
    TPBlockUtils::connectBlocks(middleBB, scalarPH);
  }
  TPBlockUtils::connectBlocks(scalarPH, latchBB);

  // Carve out region.
  auto *region = P.createTPRegionBlock("tensor loop" + LevelStr);
  region->setEntry(headerBB);   // sets headerBB->Parent = region
  region->setExiting(latchBB);  // sets latchBB->Parent = region
  innerPH->setParent(region);
  child->setParent(region);
  middleBB->setParent(region);
  if (cleanupBB) cleanupBB->setParent(region);
  scalarPH->setParent(region);

  // Append canonical IV companion recipes to latchBB.
  if (BoundTP) {
    auto *IncrRecipe = new TPCanonicalIVIncrRecipe(CanonIV, P.DimPFs[Idx].get());
    latchBB->appendRecipe(IncrRecipe);
    CanonIV->setOperand(1, IncrRecipe);
    IncrRecipe->addUser(CanonIV);
    auto *CmpRecipe = new TPCanonicalIVExitCmpRecipe(IncrRecipe, BoundTP);
    latchBB->appendRecipe(CmpRecipe);
  }

  return region;

} else {
  // Innermost: create body block.
  auto *bodyBB = P.createTPBasicBlock("tensor.body.0");

  // Emit body blocks (not header, not latch) to bodyBB.
  for (BasicBlock *BB : L->blocks()) {
    if (BB == L->getHeader() || BB == L->getLoopLatch()) continue;
    EmitBlock(BB, bodyBB);
  }

  TPBlockUtils::connectBlocks(headerBB, latchBB);
  TPBlockUtils::connectBlocks(headerBB, bodyBB);

  auto *region = P.createTPRegionBlock("tensor loop" + LevelStr);
  region->setEntry(headerBB);
  region->setExiting(latchBB);
  bodyBB->setParent(region);

  // Append canonical IV companion recipes to latchBB.
  if (BoundTP) {
    auto *IncrRecipe = new TPCanonicalIVIncrRecipe(CanonIV, P.DimPFs[Idx].get());
    latchBB->appendRecipe(IncrRecipe);
    CanonIV->setOperand(1, IncrRecipe);
    IncrRecipe->addUser(CanonIV);
    auto *CmpRecipe = new TPCanonicalIVExitCmpRecipe(IncrRecipe, BoundTP);
    latchBB->appendRecipe(CmpRecipe);
  }

  return region;
}
```

- [ ] **Step 5.6: Patch widen-induction step after body (unchanged logic, update target)**

The existing step-patch loop reads:
```cpp
for (TPRecipeBase &R : Region->getRecipes()) {
```
Change this to search `headerBB`:
```cpp
for (TPRecipeBase &R : *headerBB) {
```

- [ ] **Step 5.7: Replace top-level `P.RootRegion = BuildRegion(0)` with block wiring**

Replace:
```cpp
if (!AllLoops.empty())
  P.RootRegion = BuildRegion(0);
```

With:
```cpp
if (!AllLoops.empty()) {
  unsigned OuterLevel = P.Depth - 1;
  std::string OuterStr = std::to_string(OuterLevel);

  auto *outerPH   = P.createTPBasicBlock("tensor.ph" + OuterStr);
  auto *outer     = BuildRegion(0);
  auto *middleBB  = P.createTPBasicBlock("middle.block" + OuterStr);
  BasicBlock *ExitBB = AllLoops[0]->getExitBlock();
  auto *cleanupBB = ExitBB ? P.createTPIRBasicBlock(ExitBB) : nullptr;
  auto *scalarPH  = P.createTPBasicBlock("scalar.ph" + OuterStr);

  // Top-level blocks all have parent = nullptr (top-level plan).
  TPBlockUtils::connectBlocks(outerPH, outer);
  TPBlockUtils::connectBlocks(outer, middleBB);
  if (cleanupBB) {
    TPBlockUtils::connectBlocks(middleBB, cleanupBB);
    TPBlockUtils::connectBlocks(middleBB, scalarPH);
    TPBlockUtils::connectBlocks(cleanupBB, scalarPH);
  } else {
    TPBlockUtils::connectBlocks(middleBB, scalarPH);
  }
  // scalarPH: no successors at top level.

  P.setEntry(outerPH);
}
```

---

### Task 6: Update `TPlan::print()` in `TPlan.cpp`

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlan.cpp`

- [ ] **Step 6.1: Replace `TPlan::print()`**

Replace the existing `TPlan::print()` (currently lines 509–537) with:

```cpp
void TPlan::print(raw_ostream &OS) const {
  OS << "TPlan '" << FuncName << "' {\n";

  // Pre-assign PF[d] slots before any lazy recipe-slot assignment.
  Tracker.reset();
  for (const auto &DP : DimPFs)
    Tracker.preAssignSynthetic(DP.get());

  // Print per-dim synthetic live-ins.
  for (const auto &DP : DimPFs) {
    OS << "Live-in ";
    DP->printAsOperand(OS, Tracker);
    OS << " = " << DP->getName() << "\n";
  }

  // Print IR-backed live-ins.
  for (const auto &LI : LiveIns) {
    OS << "Live-in ";
    LI->printAsOperand(OS, Tracker);
    OS << "\n";
  }
  OS << "\n";

  if (Entry) {
    for (TPBlockBase *B : constructionOrder(Entry))
      B->print(OS, "", Tracker);
  }

  OS << "}\n";
}
```

---

### Task 7: Stub out `TPlanWidener.cpp` and `TPlanLowering.cpp`

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlanWidener.cpp`
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp`

- [ ] **Step 7.1: Replace `walkRecipes` and the phase-1/phase-2 body in `TPlanWidener.cpp`**

Replace the entire `walkRecipes` static function and `TPlanWidener_widen` body with:

```cpp
void llvm::TPlanWidener_widen(TPlan &Plan) {
  // TODO (commit 2): rewire to walk TPBasicBlock recipe lists via
  // tp_all_basicblocks(Plan).
  (void)Plan;
}
```

Keep all `#include` lines unchanged.

- [ ] **Step 7.2: Replace `lowerRegion` and `TPlanLowering_lower` body in `TPlanLowering.cpp`**

Replace `lowerRegion` static function and the body of `TPlanLowering_lower` with:

```cpp
bool llvm::TPlanLowering_lower(TPlan &Plan, Function &F, LoopInfo &LI,
                                ScalarEvolution &SE, DominatorTree &DT) {
  // TODO (commit 2): rewire to call TPlanWidener_widen, pattern match,
  // then walk Plan.getEntry() block CFG calling block.execute(State).
  (void)Plan; (void)F; (void)LI; (void)SE; (void)DT;
  return false;
}
```

Keep all existing `#include` lines. Keep all recipe `execute()` implementations (`TPCanonicalIVRecipe::execute`, `TPWidenRecipe::execute`, etc.) — they are unchanged.

- [ ] **Step 7.3: Build to verify commit 1 compiles cleanly**

```bash
ninja -C build LLVMVectorize 2>&1 | grep "error:" | head -20
```

Expected: **zero errors**. Warnings are acceptable.

---

### Task 8: Update lit test and verify commit 1

**Files:**
- Modify: `llvm/test/Transforms/LoopTensorize/basic/tplan-print.ll`

- [ ] **Step 8.1: Run the test to see actual output**

```bash
build/bin/opt -passes=loop-tensorize --disable-verify \
  -debug-only=loop-tensorize \
  llvm/test/Transforms/LoopTensorize/basic/tplan-print.ll \
  -o /dev/null 2>&1
```

Note the actual block names and recipe order from the new output.

- [ ] **Step 8.2: Update CHECK lines**

Replace the existing CHECK block (lines 7–25) with patterns matching the new block-structured output. Use loose patterns that survive minor recipe changes:

```llvm
; CHECK: TPlan 'gemm' {
; CHECK: Live-in {{.*}} = PF[0]
; CHECK: Live-in {{.*}} = PF[1]
; CHECK: Live-in {{.*}} = PF[2]
; CHECK: Live-in
; CHECK: tensor.ph2:
; CHECK: Successor(s): tensor loop2
; CHECK: <x1> tensor loop2: {
; CHECK: ir-bb<outer>:
; CHECK: CANONICAL-INDUCTION
; CHECK: WIDEN-INDUCTION
; CHECK: tensor.latch2:
; CHECK: tensor.ph1:
; CHECK: <x1> tensor loop1: {
; CHECK: ir-bb<middle>:
; CHECK: CANONICAL-INDUCTION
; CHECK: WIDEN-INDUCTION
; CHECK: tensor.ph0:
; CHECK: <x1> tensor loop0: {
; CHECK: ir-bb<inner>:
; CHECK: CANONICAL-INDUCTION
; CHECK: WIDEN-INDUCTION
; CHECK: WIDEN{{.*}} = fmul
; CHECK: WIDEN store
; CHECK: tensor.latch0:
; CHECK: CANONICAL-INDUCTION-INC
; CHECK: CANONICAL-INDUCTION-CMP
```

Adjust block names to match the actual output from Step 8.1 if they differ.

- [ ] **Step 8.3: Run the lit test**

```bash
build/bin/llvm-lit -v \
  llvm/test/Transforms/LoopTensorize/basic/tplan-print.ll
```

Expected:
```
PASS: LLVM :: Transforms/LoopTensorize/basic/tplan-print.ll (1 of 1)
```

- [ ] **Step 8.4: Commit 1**

```bash
git add llvm/include/llvm/Transforms/Vectorize/TPlan.h \
        llvm/lib/Transforms/Vectorize/TPlan.cpp \
        llvm/lib/Transforms/Vectorize/TPlanWidener.cpp \
        llvm/lib/Transforms/Vectorize/TPlanLowering.cpp \
        llvm/test/Transforms/LoopTensorize/basic/tplan-print.ll
git commit -m "$(cat <<'EOF'
tplan: replace TPLoopRegion with VPlan-style block hierarchy

Add TPBlockBase/TPBasicBlock/TPIRBasicBlock/TPRegionBlock/TPBlockUtils
mirroring VPlan's block model. Rewrite buildInitial() to build a block
CFG with named blocks (tensor.phN, tensor.latchN, tensor.loopN, etc.)
and wire them with TPBlockUtils::connectBlocks(). Update print() to
walk the block CFG in DFS pre-order matching do_conv2d.txt format.
Stub lowering in TPlanWidener/TPlanLowering for commit 2.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## ── COMMIT 2: LOWERING REWIRE ──

---

### Task 9: Implement `execute()` for all block types

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlan.cpp`

- [ ] **Step 9.1: Replace `TPBasicBlock::execute()` stub**

```cpp
void TPBasicBlock::execute(TPTransformState &State) {
  // Each recipe handles its own IR emission using State.Builder.
  for (TPRecipeBase &R : Recipes)
    R.execute(State);
}
```

- [ ] **Step 9.2: Replace `TPIRBasicBlock::execute()` stub**

```cpp
void TPIRBasicBlock::execute(TPTransformState &State) {
  // Recipes are inserted before the first non-phi instruction of IRBB.
  Instruction *InsertPt = &*IRBB->getFirstNonPHIIt();
  State.Builder.SetInsertPoint(InsertPt);
  for (TPRecipeBase &R : Recipes)
    R.execute(State);
}
```

- [ ] **Step 9.3: Replace `TPRegionBlock::execute()` stub**

The spec says `rpo_order(Entry)`. For this refactor, `constructionOrder` is safe for execution because the latch block has no successors within the region and its recipes (IV increment + comparison) do not produce values consumed by sibling blocks — they only feed back into the header PHI, which already exists in IR. If a future change requires true RPO, implement a post-order walk and reverse it; for now `constructionOrder` is correct and simpler.

```cpp
void TPRegionBlock::execute(TPTransformState &State) {
  // Walk internal CFG in construction order (DFS pre-order from Entry).
  // The latch block's recipes (IV incr + cmp) have no successors within the
  // region, so execution order matches the def-use requirements.
  if (Entry)
    for (TPBlockBase *B : constructionOrder(Entry))
      B->execute(State);
}
```

---

### Task 10: Rewire `TPlanLowering.cpp` and `TPlanWidener.cpp`

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp`
- Modify: `llvm/lib/Transforms/Vectorize/TPlanWidener.cpp`

- [ ] **Step 10.1: Add `tp_all_basicblocks` helper to `TPlanWidener.cpp`**

Replace the stub body of `TPlanWidener_widen` with:

```cpp
/// Collect all TPBasicBlock instances in the plan by recursively walking
/// block successors and descending into TPRegionBlock interiors.
static void collectBasicBlocks(TPBlockBase *Start,
                                SmallVectorImpl<TPBasicBlock *> &Out,
                                SmallPtrSetImpl<TPBlockBase *> &Visited) {
  if (!Start || !Visited.insert(Start).second)
    return;
  if (auto *BB = dyn_cast<TPBasicBlock>(Start))
    Out.push_back(BB);
  if (auto *R = dyn_cast<TPRegionBlock>(Start))
    if (R->getEntry())
      collectBasicBlocks(R->getEntry(), Out, Visited);
  for (TPBlockBase *Succ : Start->getSuccessors())
    collectBasicBlocks(Succ, Out, Visited);
}

void llvm::TPlanWidener_widen(TPlan &Plan) {
  SmallVector<TPBasicBlock *, 32> AllBBs;
  SmallPtrSet<TPBlockBase *, 32> Visited;
  if (Plan.getEntry())
    collectBasicBlocks(Plan.getEntry(), AllBBs, Visited);

  SmallVector<TPSingleDefRecipe *, 32> Worklist;

  // Phase 1: Seed from TPWidenInductionRecipe.
  for (TPBasicBlock *BB : AllBBs) {
    for (TPRecipeBase &R : *BB) {
      if (auto *WI = dyn_cast<TPWidenInductionRecipe>(&R)) {
        unsigned Dim = WI->getDimIndex();
        WI->DimSet.resize(std::max(WI->DimSet.size(), (size_t)(Dim + 1)));
        WI->DimSet.set(Dim);
        Worklist.push_back(WI);
      }
    }
  }

  // Phase 2: BFS union propagation (unchanged logic).
  while (!Worklist.empty()) {
    TPSingleDefRecipe *V = Worklist.pop_back_val();
    for (TPUser *U : V->users()) {
      auto *Recipe = dyn_cast<TPRecipeBase>(U);
      if (!Recipe) continue;
      TPSingleDefRecipe *DV = Recipe->getDefinedValue();
      if (!DV) continue;
      unsigned NeedSize = V->DimSet.size();
      if (DV->DimSet.size() < NeedSize) DV->DimSet.resize(NeedSize);
      SmallBitVector Before = DV->DimSet;
      DV->DimSet |= V->DimSet;
      if (DV->DimSet != Before) Worklist.push_back(DV);
    }
  }
}
```

- [ ] **Step 10.2: Replace stub in `TPlanLowering_lower`**

Replace the stub body with the original logic, but using block CFG walk:

```cpp
bool llvm::TPlanLowering_lower(TPlan &Plan, Function &F, LoopInfo &LI,
                                ScalarEvolution &SE, DominatorTree &DT) {
  // 1. Propagate DimSets via BFS.
  TPlanWidener_widen(Plan);

  // 2. Classify every recipe by DimSet patterns.
  RecipeClassMap CM;
  TPRecipePatternMatcher_match(Plan, CM);

  // 3. Lower: walk block CFG in construction order.
  IRBuilder<> Builder(F.getContext());
  if (!F.empty())
    Builder.SetInsertPoint(&F.getEntryBlock().front());

  TPTransformState State(Builder, Plan);
  State.ClassMap = &CM;

  if (Plan.getEntry()) {
    // Collect and execute all blocks in top-level construction order.
    // TPRegionBlock::execute() recurses into its interior.
    for (TPBlockBase *B : constructionOrder(Plan.getEntry()))
      B->execute(State);
  }
  return true;
}
```

**Declaration note**: `constructionOrder` is defined in `TPlan.cpp` and also needed in `TPlanLowering.cpp`. Declare it as a free function in `TPlan.h` (after `TPBlockUtils`) so both translation units can use it without duplication:

```cpp
/// DFS pre-order traversal starting from \p Start, following successors in
/// insertion order. Used by print() and lowering.
SmallVector<TPBlockBase *, 8> constructionOrder(TPBlockBase *Start);
```

Change the definition in `TPlan.cpp` from `static` to non-static (remove the `static` keyword) so the declaration in the header can refer to it.

- [ ] **Step 10.3: Build and run all LoopTensorize lit tests**

```bash
ninja -C build LLVMVectorize opt 2>&1 | grep "error:" | head -10
build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/
```

Expected build output: **zero errors**.

Expected lit summary (all tests must pass; the exact count depends on how many tests exist):
```
Testing Time: N.Ns
  Passed: N
```

If any lowering test fails, run it in isolation to see the full output:
```bash
build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/<failing-test>.ll
```

The key expected behavior from the lowering path: `TPplanLowering_lower` now returns `true` instead of `false`, and `@llvm.matrix.multiply` should appear in the IR output for tests that exercise the Contraction path. Verify one representative lowering test produces IR containing:
```
; CHECK: call {{.*}} @llvm.matrix.multiply
```

- [ ] **Step 10.4: Commit 2**

```bash
git add llvm/lib/Transforms/Vectorize/TPlan.cpp \
        llvm/lib/Transforms/Vectorize/TPlanLowering.cpp \
        llvm/lib/Transforms/Vectorize/TPlanWidener.cpp
git commit -m "$(cat <<'EOF'
tplan: rewire execute() and lowering to walk block CFG

Implement TPBasicBlock/TPIRBasicBlock/TPRegionBlock::execute() to
dispatch recipe lowering via the block graph. Replace the region-tree
walk in TPlanLowering_lower() and TPlanWidener_widen() with
constructionOrder(Plan.getEntry()) and tp_all_basicblocks() traversals.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Final Verification

- [ ] Run full LoopTensorize lit suite:
```bash
build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/
```
Expected: all pass.

- [ ] Push both commits:
```bash
git push yg LoopTensorizebyClaude
```
