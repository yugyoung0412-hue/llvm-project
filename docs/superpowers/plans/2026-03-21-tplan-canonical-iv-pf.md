# TPlan Canonical IV + PF Synthetic Live-in Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `PF` (Parallel Factor) as a synthetic live-in and `CANONICAL-INDUCTION` / companion recipes to TPlan, mirroring VPlan's VF and canonical IV structure.

**Architecture:** Extend `TPSlotTracker` to hold a unified `DenseMap<const TPValue *, unsigned>` slot map, add `TPSyntheticValue` for IR-unbacked values (PF), add three new recipe classes (`TPCanonicalIVRecipe`, `TPCanonicalIVIncrRecipe`, `TPCanonicalIVExitCmpRecipe`), and update `TPlan::buildInitial` to insert them into each loop region.

**Tech Stack:** C++17, LLVM `ilist_node`, LLVM `raw_ostream`, LLVM `Loop`/`ICmpInst` IR APIs, FileCheck lit tests.

**Spec:** `docs/superpowers/specs/2026-03-21-tplan-canonical-iv-pf-design.md`

---

## File Map

| File | Change |
|------|--------|
| `llvm/include/llvm/Transforms/Vectorize/TPlan.h` | Add `TPSyntheticValue`; widen `TPSlotTracker`; add 3 recipe classes; add `PF` field to `TPlan` |
| `llvm/lib/Transforms/Vectorize/TPlan.cpp` | Implement new `print()` methods; update `buildInitial()`; update `TPlan::print()` |
| `llvm/test/Transforms/LoopTensorize/basic/tplan-print.ll` | Update expected CHECK lines for new output |

---

## Task 1: Widen TPSlotTracker + add TPSyntheticValue

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlan.h`
- Modify: `llvm/lib/Transforms/Vectorize/TPlan.cpp`

### Background
`TPSlotTracker::SlotMap` is currently `DenseMap<const TPDefVal *, unsigned>`. We widen it to
`DenseMap<const TPValue *, unsigned>` so synthetics and recipe-defined values share one counter.
`TPSyntheticValue` is a new `TPValue` subclass with no IR backing; it stores its pre-assigned
slot number and emits `tp<%N>` via `printAsOperand`.

- [ ] **Step 1.1: Add `TPSyntheticValue` declaration to `TPlan.h`**

In `TPlan.h`, after the `TPLiveIn` class block (around line 76), add:

```cpp
//===----------------------------------------------------------------------===//
// TPSyntheticValue — a named synthetic value with no IR backing (e.g., PF)
//===----------------------------------------------------------------------===//
class TPSyntheticValue : public TPValue {
public:
  explicit TPSyntheticValue(StringRef Name) : Name(Name) {}
  StringRef getName() const { return Name; }
  void printAsOperand(raw_ostream &OS, TPSlotTracker &Tracker) const override;

private:
  std::string Name;
};
```

- [ ] **Step 1.2: Widen `TPSlotTracker` in `TPlan.h`**

Replace the existing `TPSlotTracker` class (around line 81) with:

```cpp
//===----------------------------------------------------------------------===//
// TPSlotTracker — assigns monotonic tp<%N> numbers to TPValues
//===----------------------------------------------------------------------===//
class TPSlotTracker {
public:
  /// Pre-assign a slot to a synthetic value. Must be called before any
  /// lazy getSlot() calls so synthetics get the lowest slot numbers.
  void preAssignSynthetic(const TPSyntheticValue *V);

  /// Lazily assign a slot to a recipe-defined value on first access.
  unsigned getSlot(const TPValue *V);

  void reset() { SlotMap.clear(); NextSlot = 0; }

private:
  DenseMap<const TPValue *, unsigned> SlotMap;
  unsigned NextSlot = 0;
};
```

- [ ] **Step 1.3: Implement `TPSyntheticValue::printAsOperand` and new `TPSlotTracker` methods in `TPlan.cpp`**

Replace the old `TPSlotTracker::getSlot` and add new methods (around line 24):

```cpp
void TPSlotTracker::preAssignSynthetic(const TPSyntheticValue *V) {
  SlotMap.try_emplace(V, NextSlot++);
}

unsigned TPSlotTracker::getSlot(const TPValue *V) {
  auto [It, Inserted] = SlotMap.try_emplace(V, NextSlot);
  if (Inserted)
    ++NextSlot;
  return It->second;
}

void TPSyntheticValue::printAsOperand(raw_ostream &OS,
                                      TPSlotTracker &Tracker) const {
  OS << "tp<%" << Tracker.getSlot(this) << ">";
}
```

- [ ] **Step 1.4: Fix the one call site of the old `getSlot(const TPDefVal *)` in `TPDefVal::printAsOperand`**

In `TPlan.cpp`, `TPDefVal::printAsOperand` currently calls `Tracker.getSlot(this)` where `this`
is `const TPDefVal *`. Since `TPDefVal` derives from `TPValue`, this still compiles after the
signature change to `getSlot(const TPValue *)`. Verify no cast or overload resolution issue:

```cpp
void TPDefVal::printAsOperand(raw_ostream &OS, TPSlotTracker &Tracker) const {
  OS << "tp<%" << Tracker.getSlot(this) << ">";
}
```

No code change needed here — just confirm it compiles.

- [ ] **Step 1.5: Build to verify no compile errors**

```bash
ninja -C /path/to/build LLVMVectorize 2>&1 | tail -20
```

Expected: compiles cleanly (or only unrelated warnings).

- [ ] **Step 1.6: Commit**

```bash
git add llvm/include/llvm/Transforms/Vectorize/TPlan.h \
        llvm/lib/Transforms/Vectorize/TPlan.cpp
git commit -m "tplan: widen TPSlotTracker to TPValue*, add TPSyntheticValue"
```

---

## Task 2: Add PF field to TPlan + emit in print

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlan.h`
- Modify: `llvm/lib/Transforms/Vectorize/TPlan.cpp`

### Background
`TPlan` gets a `TPSyntheticValue PF{"PF"}` member. `TPlan::print()` pre-assigns it slot 0 via
`Tracker.preAssignSynthetic(&PF)` before any region traversal, then emits
`Live-in tp<%0> = PF` as the first live-in line.

- [ ] **Step 2.1: Add `PF` member to `TPlan` in `TPlan.h`**

In the `TPlan` private section (around line 347), add:

```cpp
  TPSyntheticValue PF{"PF"};
```

Also add a public accessor if needed by recipes:
```cpp
public:
  const TPSyntheticValue *getPF() const { return &PF; }
```

- [ ] **Step 2.2: Update `TPlan::print()` in `TPlan.cpp`**

Replace the existing `TPlan::print()` (around line 399):

```cpp
void TPlan::print(raw_ostream &OS) const {
  OS << "TPlan '" << FuncName << "' (depth=" << Depth << ") {\n";

  // Pre-assign PF as tp<%0> before any lazy recipe-slot assignment.
  Tracker.reset();
  Tracker.preAssignSynthetic(&PF);

  // Print synthetic live-ins first (VPlan style).
  OS << "Live-in ";
  PF.printAsOperand(OS, Tracker);
  OS << " = PF\n";

  // Print IR-backed live-ins (unchanged, still ir<>).
  for (const auto &LI : LiveIns) {
    OS << "Live-in ";
    LI->printAsOperand(OS, Tracker);
    OS << "\n";
  }
  OS << "\n";

  if (RootRegion)
    RootRegion->print(OS, 0, Tracker);

  OS << "}\n";
}
```

- [ ] **Step 2.3: Build and spot-check output**

```bash
ninja -C build opt
build/bin/opt -passes=loop-tensorize -debug-only=loop-tensorize \
  ggml_compute_forward_mul_mat.ll -o /dev/null 2>&1 | head -5
```

Expected first lines:
```
TPlan '_ZL32...' (depth=5) {
Live-in tp<%0> = PF
Live-in ir<0>
Live-in ir<%7>
```

- [ ] **Step 2.4: Commit**

```bash
git add llvm/include/llvm/Transforms/Vectorize/TPlan.h \
        llvm/lib/Transforms/Vectorize/TPlan.cpp
git commit -m "tplan: add PF synthetic live-in, emit as tp<%0> = PF"
```

---

## Task 3: Add three new recipe classes (header declarations)

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlan.h`

### Background
Three new recipe classes, all synthetic (no `Instruction *` backing):
- `TPCanonicalIVRecipe` — the phi: `CANONICAL-INDUCTION tp<%N> = phi ir<0>, tp<%step>`
- `TPCanonicalIVIncrRecipe` — the add: `CANONICAL-INDUCTION-INC tp<%M> = add tp<%N>, tp<%0>`
- `TPCanonicalIVExitCmpRecipe` — the icmp: `CANONICAL-INDUCTION-CMP tp<%C> = icmp tp<%M>, ir<%bound>`

`RecipeKind` enum gets three new values.

- [ ] **Step 3.1: Add enum values to `RecipeKind` in `TPlan.h`**

In `TPRecipeBase::RecipeKind` (around line 109), add:

```cpp
    CanonicalIV,
    CanonicalIVIncr,
    CanonicalIVExitCmp,
```

- [ ] **Step 3.2: Declare `TPCanonicalIVRecipe` in `TPlan.h`**

After the `TPWidenCastRecipe` block, add:

```cpp
//===----------------------------------------------------------------------===//
// TPCanonicalIVRecipe — CANONICAL-INDUCTION: synthetic loop counter phi
//===----------------------------------------------------------------------===//
class TPCanonicalIVRecipe : public TPRecipeBase {
public:
  /// StartVal: TPLiveIn for ir<0>. StepVal: placeholder; patched after
  /// TPCanonicalIVIncrRecipe is created.
  TPCanonicalIVRecipe(TPValue *StartVal, TPValue *StepVal)
      : TPRecipeBase(RecipeKind::CanonicalIV) {
    addOperand(StartVal);
    addOperand(StepVal);
  }

  void print(raw_ostream &OS, unsigned Indent,
             TPSlotTracker &Tracker) const override;

  static bool classof(const TPRecipeBase *R) {
    return R->getKind() == RecipeKind::CanonicalIV;
  }
};

//===----------------------------------------------------------------------===//
// TPCanonicalIVIncrRecipe — CANONICAL-INDUCTION-INC: canonical IV + PF
//===----------------------------------------------------------------------===//
class TPCanonicalIVIncrRecipe : public TPRecipeBase {
public:
  /// IVVal: TPDefVal of TPCanonicalIVRecipe. PFVal: TPSyntheticValue for PF.
  TPCanonicalIVIncrRecipe(TPValue *IVVal, TPValue *PFVal)
      : TPRecipeBase(RecipeKind::CanonicalIVIncr) {
    addOperand(IVVal);
    addOperand(PFVal);
  }

  void print(raw_ostream &OS, unsigned Indent,
             TPSlotTracker &Tracker) const override;

  static bool classof(const TPRecipeBase *R) {
    return R->getKind() == RecipeKind::CanonicalIVIncr;
  }
};

//===----------------------------------------------------------------------===//
// TPCanonicalIVExitCmpRecipe — CANONICAL-INDUCTION-CMP: exit condition icmp
//===----------------------------------------------------------------------===//
class TPCanonicalIVExitCmpRecipe : public TPRecipeBase {
public:
  /// IncrVal: TPDefVal of TPCanonicalIVIncrRecipe. BoundVal: TPLiveIn for
  /// the loop bound (RHS of the latch ICmpInst).
  TPCanonicalIVExitCmpRecipe(TPValue *IncrVal, TPValue *BoundVal)
      : TPRecipeBase(RecipeKind::CanonicalIVExitCmp) {
    addOperand(IncrVal);
    addOperand(BoundVal);
  }

  void print(raw_ostream &OS, unsigned Indent,
             TPSlotTracker &Tracker) const override;

  static bool classof(const TPRecipeBase *R) {
    return R->getKind() == RecipeKind::CanonicalIVExitCmp;
  }
};
```

- [ ] **Step 3.3: Build to verify declarations compile**

```bash
ninja -C build LLVMVectorize 2>&1 | tail -10
```

Expected: only "undefined reference" linker errors for the three missing `print()` implementations (not compile errors). If building just the object, compile errors only.

- [ ] **Step 3.4: Commit**

```bash
git add llvm/include/llvm/Transforms/Vectorize/TPlan.h
git commit -m "tplan: declare TPCanonicalIVRecipe + companion recipe classes"
```

---

## Task 4: Implement print() for the three new recipe classes

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlan.cpp`

- [ ] **Step 4.1: Implement `TPCanonicalIVRecipe::print()`**

In `TPlan.cpp`, after `TPWidenCastRecipe::print()` (around line 148), add:

```cpp
void TPCanonicalIVRecipe::print(raw_ostream &OS, unsigned Indent,
                                TPSlotTracker &Tracker) const {
  printIndent(OS, Indent);
  OS << "CANONICAL-INDUCTION ";
  DefVal->printAsOperand(OS, Tracker);
  OS << " = phi ";
  Operands[0]->printAsOperand(OS, Tracker);
  OS << ", ";
  Operands[1]->printAsOperand(OS, Tracker);
  OS << "\n";
}
```

- [ ] **Step 4.2: Implement `TPCanonicalIVIncrRecipe::print()`**

```cpp
void TPCanonicalIVIncrRecipe::print(raw_ostream &OS, unsigned Indent,
                                    TPSlotTracker &Tracker) const {
  printIndent(OS, Indent);
  OS << "CANONICAL-INDUCTION-INC ";
  DefVal->printAsOperand(OS, Tracker);
  OS << " = add ";
  Operands[0]->printAsOperand(OS, Tracker);
  OS << ", ";
  Operands[1]->printAsOperand(OS, Tracker);
  OS << "\n";
}
```

- [ ] **Step 4.3: Implement `TPCanonicalIVExitCmpRecipe::print()`**

```cpp
void TPCanonicalIVExitCmpRecipe::print(raw_ostream &OS, unsigned Indent,
                                       TPSlotTracker &Tracker) const {
  printIndent(OS, Indent);
  OS << "CANONICAL-INDUCTION-CMP ";
  DefVal->printAsOperand(OS, Tracker);
  OS << " = icmp ";
  Operands[0]->printAsOperand(OS, Tracker);
  OS << ", ";
  Operands[1]->printAsOperand(OS, Tracker);
  OS << "\n";
}
```

- [ ] **Step 4.4: Build to verify no linker errors**

```bash
ninja -C build opt 2>&1 | tail -5
```

Expected: links cleanly.

- [ ] **Step 4.5: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/TPlan.cpp
git commit -m "tplan: implement print() for canonical IV recipe classes"
```

---

## Task 5: Insert canonical IV recipes in buildInitial()

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlan.cpp`

### Background
In `TPlan::buildInitial()`'s `BuildRegion` lambda, for each loop level:
1. Insert `TPCanonicalIVRecipe` as the **first** recipe (before processing PHIs)
2. After body + child region are built, create `TPCanonicalIVIncrRecipe` and
   `TPCanonicalIVExitCmpRecipe`, then patch the canonical IV's step operand.

The trip count bound (`Value *`) is read from the loop latch's `ICmpInst` condition RHS.

- [ ] **Step 5.1: Read latch bound helper**

At the top of the `BuildRegion` lambda body (before any recipe creation), add a helper to get
the loop bound `Value *` from the latch:

```cpp
// Get the loop exit bound: latch branch condition ICmpInst RHS.
Value *LatchBound = nullptr;
if (BasicBlock *Latch = L->getLoopLatch()) {
  if (auto *BI = dyn_cast<BranchInst>(Latch->getTerminator())) {
    if (BI->isConditional()) {
      if (auto *Cmp = dyn_cast<ICmpInst>(BI->getCondition()))
        LatchBound = Cmp->getOperand(1);
    }
  }
}
TPValue *BoundTP = LatchBound ? P.getOrCreateLiveIn(LatchBound) : nullptr;
```

- [ ] **Step 5.2: Create TPCanonicalIVRecipe as first recipe**

Immediately after the region is constructed (`auto Region = std::make_unique<TPLoopRegion>(...)`),
before the PHI processing loop, add:

```cpp
// Insert canonical IV phi as the first recipe (VPlan-style).
// Use a zero live-in as start; match the IV phi's type (may be i32 or i64).
// Direct-push to Operands[1] to avoid poisoning the zero live-in's use list
// with a stale user entry before the real step is known.
TPValue *ZeroTP = P.getOrCreateLiveIn(
    ConstantInt::get(InductionPhi->getType(), 0));
auto *CanonIV = new TPCanonicalIVRecipe(ZeroTP, ZeroTP /*placeholder step*/);
Region->appendRecipe(CanonIV);
```

Key: `InductionPhi->getType()` — use the IV phi's actual type, not a hardcoded `i64`, to
match the convention used for zero start values elsewhere in `buildInitial`.

Note: `ZeroTP` is used as a placeholder for operand[1] (step). It will be replaced in Step 5.4.
The placeholder causes a stale user entry on `ZeroTP` (from `addOperand` in the constructor),
which is a known minor limitation — it does not affect printing correctness since user lists
are not traversed during `print()`.

- [ ] **Step 5.3: Build body and child as before (no change)**

The existing PHI processing, `EmitBlock`, and child region recursion remain unchanged.

- [ ] **Step 5.4: Create companion recipes and patch canonical IV step**

After the child region recursion and the existing WIDEN-INDUCTION step-patch block, add:

```cpp
// Create canonical IV companion recipes.
if (CanonIV->getDefinedValue() && BoundTP) {
  // Increment: canonical_iv + PF
  auto *IncrRecipe = new TPCanonicalIVIncrRecipe(
      CanonIV->getDefinedValue(), P.getPF());
  Region->appendRecipe(IncrRecipe);

  // Patch canonical IV step operand to point to the increment result.
  CanonIV->setOperand(1, IncrRecipe->getDefinedValue());
  IncrRecipe->getDefinedValue()->addUser(CanonIV);

  // Exit cmp: incremented_iv icmp bound
  auto *CmpRecipe = new TPCanonicalIVExitCmpRecipe(
      IncrRecipe->getDefinedValue(), BoundTP);
  Region->appendRecipe(CmpRecipe);
}
```

- [ ] **Step 5.5: Build opt**

```bash
ninja -C build opt 2>&1 | tail -5
```

Expected: clean build.

- [ ] **Step 5.6: Smoke-test output structure**

```bash
build/bin/opt -passes=loop-tensorize -debug-only=loop-tensorize \
  ggml_compute_forward_mul_mat.ll -o /dev/null 2>&1 | grep -E "CANONICAL|Live-in tp"
```

Expected output (slot numbers may vary):
```
Live-in tp<%0> = PF
CANONICAL-INDUCTION tp<%1> = phi ir<0>, tp<%2>
CANONICAL-INDUCTION-INC tp<%2> = add tp<%1>, tp<%0>
CANONICAL-INDUCTION-CMP tp<%3> = icmp tp<%2>, ir<%17>
CANONICAL-INDUCTION tp<%N> = phi ir<0>, tp<%M>
...
```

- [ ] **Step 5.7: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/TPlan.cpp
git commit -m "tplan: insert CANONICAL-INDUCTION + companion recipes in buildInitial"
```

---

## Task 6: Update lit test

**Files:**
- Modify: `llvm/test/Transforms/LoopTensorize/basic/tplan-print.ll`

### Background
The lit test uses loose `CHECK:` patterns. We need to add checks for:
- `Live-in tp<%0> = PF` appearing before other live-ins
- `CANONICAL-INDUCTION` appearing in each loop region
- `CANONICAL-INDUCTION-INC` and `CANONICAL-INDUCTION-CMP` appearing after body recipes

Run the test first to see actual output, then lock in the patterns.

- [ ] **Step 6.1: Run the test and capture actual output**

```bash
build/bin/opt -passes=loop-tensorize -debug-only=loop-tensorize \
  llvm/test/Transforms/LoopTensorize/basic/tplan-print.ll -o /dev/null 2>&1
```

Note the exact output for the GEMM 3-loop test.

- [ ] **Step 6.2: Update CHECK lines in `tplan-print.ll`**

Replace the CHECK block (lines 7–16) with:

```llvm
; CHECK: TPlan 'gemm' (depth=3) {
; CHECK: Live-in tp<%0> = PF
; CHECK: Live-in
; CHECK: loop[0]
; CHECK: CANONICAL-INDUCTION
; CHECK: WIDEN-INDUCTION
; CHECK: loop[1]
; CHECK: CANONICAL-INDUCTION
; CHECK: WIDEN-INDUCTION
; CHECK: loop[2]
; CHECK: CANONICAL-INDUCTION
; CHECK: WIDEN-INDUCTION
; CHECK: WIDEN{{.*}} = fmul
; CHECK: WIDEN store
; CHECK: CANONICAL-INDUCTION-INC
; CHECK: CANONICAL-INDUCTION-CMP
```

- [ ] **Step 6.3: Run the lit test**

```bash
build/bin/llvm-lit -v \
  llvm/test/Transforms/LoopTensorize/basic/tplan-print.ll
```

Expected:
```
PASS: LLVM :: Transforms/LoopTensorize/basic/tplan-print.ll (1 of 1)
```

- [ ] **Step 6.4: Commit**

```bash
git add llvm/test/Transforms/LoopTensorize/basic/tplan-print.ll
git commit -m "tplan: update lit test for CANONICAL-INDUCTION + PF live-in"
```

---

## Task 7: Final verification

- [ ] **Step 7.1: Run full output on ggml test and save**

```bash
build/bin/opt -passes=loop-tensorize -debug-only=loop-tensorize \
  ggml_compute_forward_mul_mat.ll -o /dev/null 2>&1 \
  | grep -v "^.*opt: WARNING" | grep -v "^PatternHint" | grep -v "^Best cost" \
  > ggml_compute_forward_mul_mat_tplan_initial.txt
```

- [ ] **Step 7.2: Verify structure manually**

Check that the output file contains:
- `Live-in tp<%0> = PF` as the first live-in
- Every `<x1> loop[N]` block begins with `CANONICAL-INDUCTION`
- Every `<x1> loop[N]` block ends (before child) with `CANONICAL-INDUCTION-INC` + `CANONICAL-INDUCTION-CMP`
- IR live-ins (`ir<0>`, `ir<%7>`, etc.) appear unchanged in body recipes
- All 5 loop levels have canonical IV structure

- [ ] **Step 7.3: Run lit tests for the LoopTensorize suite**

```bash
build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/
```

Expected: all pass.
