# TPlan → VPlan-Style Refactor Design Spec

**Date:** 2026-03-24
**Branch:** LoopTensorizebyClaude
**Goal:** Refactor TPlan to mirror VPlan's hierarchical CFG structure, replacing the flat `TPLoopRegion` tree with `TPBlockBase`/`TPBasicBlock`/`TPRegionBlock`/`TPIRBasicBlock`, and rewiring the print and lowering infrastructure to walk the new block graph.

---

## 1. Motivation

The current `TPLoopRegion` stores all recipes in a single flat `iplist` and tracks only one child region. This diverges from VPlan's `VPRegionBlock`/`VPBasicBlock` model and makes it hard to express the full per-level block structure (preheader, header, latch, body, middle-block, scalar-ph) that the pass needs for both printing and code generation. The target output format is documented in `do_conv2d.txt`.

---

## 2. Class Hierarchy

Mirrors VPlan (`VP*` → `TP*`) with the same ownership and connection semantics.

```
TPBlockBase                     ← VPBlockBase
  ├── TPBasicBlock               ← VPBasicBlock
  │     └── TPIRBasicBlock       ← VPIRBasicBlock
  └── TPRegionBlock              ← VPRegionBlock

TPBlockUtils                    ← VPBlockUtils
```

### TPBlockBase

- Fields:
  - `const unsigned char SubclassID` — set at construction time, used for `isa<>`/`dyn_cast<>` (mirrors VPlan's `VPBlockBase(const unsigned char SC, const std::string &N)` constructor)
  - `std::string Name`
  - `TPRegionBlock *Parent` — immediate enclosing region, null at top level
  - `SmallVector<TPBlockBase*, 1> Predecessors`, `SmallVector<TPBlockBase*, 1> Successors`
- Subclass ID enum (used as `unsigned char` values): `TPBasicBlockSC`, `TPIRBasicBlockSC`, `TPRegionBlockSC`
- `getTPBlockID() const → unsigned` — returns `SubclassID`
- `getName()`, `setName()`
- `getParent()`, `setParent(TPRegionBlock*)` — public, used by `setEntry`/`setExiting` and `TPBlockUtils`
- `getPlan()` / `setPlan(TPlan*)` — only valid on the plan's entry block
- Public connection API: `setOneSuccessor`, `setTwoSuccessors`, `setPredecessors`, `setSuccessors`, `clearSuccessors`, `clearPredecessors`
- Pure virtual: `execute(TPTransformState&)`, `print(raw_ostream&, const Twine& Indent, TPSlotTracker&)`
- Private (friend `TPBlockUtils`): `appendSuccessor`, `appendPredecessor`, `removeSuccessor`, `removePredecessor`, `replacePredecessor`

### TPBasicBlock : TPBlockBase
- Constructor passes `TPBasicBlockSC` to `TPBlockBase`
- Adds `iplist<TPRecipeBase> Recipes`
- `appendRecipe(TPRecipeBase*)`, `insert(iterator, TPRecipeBase*)`, recipe iterators (`begin`/`end`/`phis()`)
- `getTerminator()`, `isExiting()`
- `classof`: `SubclassID == TPBasicBlockSC || SubclassID == TPIRBasicBlockSC`
- Used for: `tensor.phN`, `tensor.latchN`, `middle.blockN`, `scalar.phN`, `tensor.body.*`

### TPIRBasicBlock : TPBasicBlock
- Constructor passes `TPIRBasicBlockSC` to `TPBlockBase`
- Adds `BasicBlock* IRBB`
- Name is always `"ir-bb<" + IRBB->getName() + ">"` — derived from the IR block's actual name
- `classof`: `SubclassID == TPIRBasicBlockSC`
- Used for: loop header blocks (`ir-bb<for.condX.preheader>`), cleanup blocks (`ir-bb<for.cond.cleanupX.loopexit>`)
- `execute()`: inserts recipes before first non-phi of `IRBB`

### TPRegionBlock : TPBlockBase
- Constructor passes `TPRegionBlockSC` to `TPBlockBase`
- Fields: `TPBlockBase* Entry`, `TPBlockBase* Exiting`, `bool IsReplicator` (kept for structural parity, unused initially)
- `setEntry(TPBlockBase*)` — asserts `Entry->getPredecessors().empty()`; calls `Entry->setParent(this)`
- `setExiting(TPBlockBase*)` — asserts `Exiting->getSuccessors().empty()`; calls `Exiting->setParent(this)`
- `classof`: `SubclassID == TPRegionBlockSC`
- Named `tensor loopN`
- `execute()`: walks internal CFG in RPO from `Entry`, dispatches `block.execute(State)`

### TPBlockUtils (free class, static methods)
- Friend of `TPBlockBase` — accesses private `appendSuccessor`, `appendPredecessor`, `removeSuccessor`, `removePredecessor`
- `connectBlocks(From, To)` — calls `From->appendSuccessor(To)` + `To->appendPredecessor(From)`; asserts `From->getParent() == To->getParent()` (same-parent invariant; see Section 3 note). For top-level blocks both parents are `nullptr`, so the assertion `nullptr == nullptr` passes automatically — no special-casing is required.
- `disconnectBlocks(From, To)` — bidirectional removal
- `insertBlockAfter(New, After)` — rewires successors
- `transferSuccessors(Old, New)` — moves all successor edges

### TPlan changes
- Removes `unique_ptr<TPLoopRegion> RootRegion` and the `TPLoopRegion` class entirely
- Adds `TPBlockBase* Entry = nullptr` — outermost preheader block
- Adds `SmallVector<TPBlockBase*> CreatedBlocks` — owns all blocks; destructor calls `delete` on each
- Factory methods (call `new`, push to `CreatedBlocks`, return raw pointer):
  - `createTPBasicBlock(StringRef Name) → TPBasicBlock*`
  - `createTPRegionBlock(StringRef Name) → TPRegionBlock*`
  - `createTPIRBasicBlock(BasicBlock* IRBB) → TPIRBasicBlock*`
- `getEntry() → TPBlockBase*`, `setEntry(TPBlockBase*)`
- All other fields (`LiveIns`, `DimPFs`, `Tracker`, `DimPFMap`, `ValueMap`, etc.) unchanged

---

## 3. buildInitial() Rewrite

Replaces the current region-tree builder with a **single recursive pass** over the loop nest.

### Same-parent invariant note
`TPBlockUtils::connectBlocks` asserts `From->getParent() == To->getParent()`. This is satisfied throughout the algorithm: `scalar.ph<d-1>` and `tensor.latch<d>` are both children of the same `tensor loop<d>` region (siblings within it), so the back-edge `connectBlocks(scalarPH, latchBB)` always connects two blocks with the same parent. There are **no cross-region successor edges** in this design.

### Block parent assignment
After carving out a region, `setParent(region)` is called explicitly on these blocks:
- Non-innermost: `headerBB`, `latchBB`, `innerPH`, `child` (the nested `TPRegionBlock`), `middleBB`, `cleanupBB`, `scalarPH`
- Innermost: `headerBB`, `latchBB`, `bodyBB`

`setEntry` and `setExiting` set the parent of their argument automatically; the remaining blocks are set via explicit `block->setParent(region)` calls.

### Algorithm

```
BuildRegion(Loop L, depth d, TPlan& P) → TPRegionBlock*:

  headerBB = P.createTPIRBasicBlock(L.header)
              // name = "ir-bb<" + L.header->getName() + ">"
  latchBB  = P.createTPBasicBlock("tensor.latch" + d)
              // recipes: CANONICAL-IV-INCR, branch-on-count, etc.

  [Populate headerBB and latchBB recipes — same logic as current buildInitial()]

  if L.hasChild (non-innermost):
    innerPH   = P.createTPBasicBlock("tensor.ph" + (d-1))
    child     = BuildRegion(L.child, d-1, P)               // recurse
    middleBB  = P.createTPBasicBlock("middle.block" + (d-1))
    cleanupBB = P.createTPIRBasicBlock(L.child.exitBlock)
                // name = "ir-bb<" + exitBlock->getName() + ">"
    scalarPH  = P.createTPBasicBlock("scalar.ph" + (d-1))

    // Wire intra-region CFG (all blocks share same parent after setParent below)
    TPBlockUtils::connectBlocks(headerBB, latchBB)          // loop-exit branch
    TPBlockUtils::connectBlocks(headerBB, innerPH)          // fall-through to inner
    TPBlockUtils::connectBlocks(innerPH,  child)
    TPBlockUtils::connectBlocks(child,    middleBB)
    TPBlockUtils::connectBlocks(middleBB, cleanupBB)
    TPBlockUtils::connectBlocks(middleBB, scalarPH)
    TPBlockUtils::connectBlocks(cleanupBB, scalarPH)
    TPBlockUtils::connectBlocks(scalarPH, latchBB)          // back to latch

  else (innermost):
    bodyBB = P.createTPBasicBlock("tensor.body.0")
    TPBlockUtils::connectBlocks(headerBB, latchBB)
    TPBlockUtils::connectBlocks(headerBB, bodyBB)
    // latchBB and bodyBB have no successors within this region

  // Carve out region
  region = P.createTPRegionBlock("tensor loop" + d)
  region.setEntry(headerBB)    // sets headerBB->Parent = region
  region.setExiting(latchBB)   // sets latchBB->Parent = region
  // Set parent on remaining inner blocks explicitly:
  if non-innermost:
    for b in {innerPH, child, middleBB, cleanupBB, scalarPH}:
      b->setParent(region)
  else:
    bodyBB->setParent(region)

  return region

Top-level buildInitial():
  outerPH   = P.createTPBasicBlock("tensor.ph" + depth)
  outer     = BuildRegion(outerLoop, depth, P)
  middleBB  = P.createTPBasicBlock("middle.block" + depth)
  cleanupBB = P.createTPIRBasicBlock(outerLoop.exitBlock)
  scalarPH  = P.createTPBasicBlock("scalar.ph" + depth)

  // Top-level blocks have no parent (parent = null = top-level plan)
  TPBlockUtils::connectBlocks(outerPH,   outer)
  TPBlockUtils::connectBlocks(outer,     middleBB)
  TPBlockUtils::connectBlocks(middleBB,  cleanupBB)
  TPBlockUtils::connectBlocks(middleBB,  scalarPH)
  TPBlockUtils::connectBlocks(cleanupBB, scalarPH)
  // scalarPH: no successors at top level

  P.setEntry(outerPH)   // returns TPBlockBase*; outerPH is a TPBasicBlock
```

Note: top-level blocks (`outerPH`, `outer`, `middleBB`, `cleanupBB`, `scalarPH`) have `Parent = nullptr` because they belong directly to the plan, not to any region. The `connectBlocks` same-parent assertion `From->getParent() == To->getParent()` evaluates to `nullptr == nullptr`, which is true — no special-casing is required.

Recipe placement is unchanged — same recipes as today, now inserted into named `TPBasicBlock` recipe lists.

---

## 4. print()

`TPBlockBase::print()` is pure virtual. Traversal within a region follows **block-construction order** (the order blocks were created and connected), not generic DFS, to produce the exact block ordering seen in `do_conv2d.txt`.

```
TPBasicBlock::print(Indent):
  OS << Indent << Name << ":\n"
  for recipe in Recipes: recipe.print(OS, Indent+"  ", Tracker)
  printSuccessors(OS, Indent)   // "Successor(s): X, Y" or "No successors"

TPIRBasicBlock::print(Indent):
  OS << Indent << "ir-bb<" << IRBB->getName() << ">:\n"
  for recipe in Recipes: recipe.print(OS, Indent+"  ", Tracker)
  printSuccessors(OS, Indent)

TPRegionBlock::print(Indent):
  OS << Indent << "<x1> " << Name << ": {\n"
  // Walk internal blocks in construction order from Entry
  for block in construction_order(Entry):
    block.print(OS, Indent+"  ", Tracker)
  OS << Indent << "}\n"
  printSuccessors(OS, Indent)

TPlan::print():
  Tracker.reset()
  pre-assign DimPF synthetic slots
  emit live-in lines
  // Walk top-level blocks in construction order from getEntry()
  for block in construction_order(getEntry()):
    block.print(OS, "", Tracker)
```

`construction_order` is defined as **DFS pre-order from Entry, visiting each block's successors in the order they appear in the successor list (insertion order)**. This produces the correct `do_conv2d.txt` ordering for this specific topology because:
- `headerBB` has successor[0] = `latchBB` and successor[1] = `innerPH` (or `bodyBB`). DFS visits `latchBB` first. Since `latchBB` has no successors within its region (it is the exiting block), DFS terminates immediately at `latchBB` and then descends into `innerPH`/`bodyBB` — exactly matching the block order in `do_conv2d.txt`.
- The same definition applies to the top-level walk in `TPlan::print()`, starting from `getEntry()` (the outer preheader block).

Note: `TPBlockBase::print()` uses `const Twine& Indent` for string concatenation (e.g., `Indent + "  "`), while `TPRecipeBase::print()` retains `unsigned Indent`. These are intentionally different; do not change the recipe signature.

---

## 5. execute() / Lowering

### Commit 1 compile strategy
`TPLoopRegion` is removed entirely in commit 1. `TPlanWidener.cpp` and `TPlanLowering.cpp` currently call `Plan.getRootRegion()` which will no longer exist. Commit 1 **also** minimally adapts these two files to compile: replace the region-tree traversal bodies with `// TODO: rewire in commit 2` no-op stubs. Lowering is temporarily non-functional in commit 1's build, but all other lit tests (print tests) still pass.

### Commit 2: block-driven dispatch

```
TPBasicBlock::execute(State):
  BasicBlock *BB = createEmptyBasicBlock(State)   // emit IR BB
  for recipe in Recipes: recipe.execute(State)

TPIRBasicBlock::execute(State):
  insert recipes before first non-phi of IRBB
  for recipe in Recipes: recipe.execute(State)

TPRegionBlock::execute(State):
  emit loop structure (preheader → header → latch back-edge)
  for block in rpo_order(Entry):
    block.execute(State)

TPlanLowering_lower():
  TPlanWidener_widen(Plan)
  TPRecipePatternMatcher_match(...)
  TPTransformState State(...)
  for block in construction_order(Plan.getEntry()):
    block.execute(State)

TPlanWidener_widen():
  same BFS union logic over DimSets
  traversal: tp_all_basicblocks(Plan) helper — collects all TPBasicBlock
  instances by walking construction_order recursively into regions
```

---

## 6. What Stays Unchanged

- All `TPRecipeBase` subclasses and their `print()` / `execute()` implementations
- `TPValue`, `TPUser`, `TPLiveIn`, `TPSyntheticValue`, `TPSlotTracker`
- `TPlanTypes.h`, `TensorOpKind`, `RecipeClassification`
- `TPRecipePatternMatcher`
- `TPTransformState` struct

---

## 7. Delivery Plan

### Commit 1 — Data model + print
| File | Change |
|------|--------|
| `llvm/include/llvm/Transforms/Vectorize/TPlan.h` | Add `TPBlockBase`, `TPBasicBlock`, `TPIRBasicBlock`, `TPRegionBlock`, `TPBlockUtils`; remove `TPLoopRegion`; update `TPlan` with factory methods + `CreatedBlocks` |
| `llvm/lib/Transforms/Vectorize/TPlan.cpp` | Implement block constructors, `print()`, rewrite `buildInitial()`, update `TPlan::print()` |
| `llvm/lib/Transforms/Vectorize/TPlanWidener.cpp` | Replace region-tree traversal with no-op stub (compile gate only) |
| `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` | Replace region-tree traversal with no-op stub (compile gate only) |
| `llvm/test/Transforms/LoopTensorize/basic/tplan-print.ll` | Update CHECK lines for new block-structured output |

Gate: `ninja -C build LLVMVectorize` + print lit tests pass.

### Commit 2 — Lowering rewire
| File | Change |
|------|--------|
| `llvm/include/llvm/Transforms/Vectorize/TPlan.h` | Finalize `execute()` as pure virtual on `TPBlockBase` |
| `llvm/lib/Transforms/Vectorize/TPlan.cpp` | Implement `execute()` for all block types |
| `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` | Replace no-op stub with block CFG walk |
| `llvm/lib/Transforms/Vectorize/TPlanWidener.cpp` | Replace no-op stub with `tp_all_basicblocks()` traversal |
| `llvm/test/Transforms/LoopTensorize/` | Verify all lowering lit tests pass |

Gate: full LoopTensorize lit suite passes.
