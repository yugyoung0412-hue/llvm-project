# TPlan PHI Recipes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a VPlan-mirrored PHI recipe hierarchy to TPlan in two commits: restructure + split existing recipes (Commit 1), then add six new stub recipes (Commit 2).

**Architecture:** Introduce `TPHeaderPHIRecipe` as an abstract base for all header PHI recipes; split the monolithic `TPWidenInductionRecipe` into `TPWidenIntOrFpInductionRecipe` + `TPWidenPointerInductionRecipe`; rebase `TPCanonicalIVRecipe` and `TPReductionPHIRecipe`; add six no-op stubs in Commit 2.

**Tech Stack:** C++17, LLVM ilist/SmallBitVector, FileCheck lit tests, ninja/clang build.

---

## File Map

| File | Role in this plan |
|------|-------------------|
| `llvm/include/llvm/Transforms/Vectorize/TPlan.h` | All class declarations, enum, `classof` |
| `llvm/lib/Transforms/Vectorize/TPlan.cpp` | `buildInitial()` branch, `print()` implementations |
| `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` | `execute()` implementations |
| `llvm/test/Transforms/LoopTensorize/basic/tplan-print.ll` | Lit test — no CHECK change needed. The spec's consumer table mentions updating "TPWidenInductionRecipe" CHECK lines, but the actual CHECK strings match the *print output token* (`WIDEN-INDUCTION`), not the class name. Since `TPWidenIntOrFpInductionRecipe::print` emits `WIDEN-INDUCTION` (same as the old monolithic class), no change is required. |

---

## Task 1: Update enum — replace `TPWidenInductionSC` with two new SCs

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlan.h:612-614`

- [ ] **Step 1: Replace `TPWidenInductionSC` with the two new SCs in the enum**

  In `TPlan.h`, find the enum block around line 612 and make this change:

  ```cpp
  // Before:
      TPCanonicalIVSC,
      TPWidenInductionSC,
      TPReductionPHISC,

  // After:
      TPCanonicalIVSC,
      TPWidenIntOrFpInductionSC,  // replaces TPWidenInductionSC
      TPWidenPointerInductionSC,  // NEW
      TPReductionPHISC,
  ```

  Range markers `TPLastHeaderPHISC = TPReductionPHISC` and `TPLastPHISC = TPReductionPHISC` are **unchanged** in Commit 1.

- [ ] **Step 2: Update `TPSingleDefRecipe::classof` for the new SCs**

  In `TPlan.h`, find the classof switch around line 671. Replace:
  ```cpp
      case TPWidenInductionSC:
  ```
  with:
  ```cpp
      case TPWidenIntOrFpInductionSC:
      case TPWidenPointerInductionSC:
  ```

- [ ] **Step 3: Build to confirm enum and classof compile**

  ```bash
  ninja -C build llvm-opt 2>&1 | head -40
  ```
  Expected: compile errors from `TPWidenInductionRecipe` still referencing `TPWidenInductionSC` — that's expected at this step; fix in Task 2.

---

## Task 2: Add `TPHeaderPHIRecipe` abstract base and restructure induction recipes

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlan.h:846-873`

- [ ] **Step 1: Add `TPHeaderPHIRecipe` class before `TPWidenInductionRecipe`**

  In `TPlan.h`, insert this new class **before** the `TPWidenInductionRecipe` comment block (~line 846):

  ```cpp
  //===----------------------------------------------------------------------===//
  // TPHeaderPHIRecipe — abstract base for all header PHI recipes (mirrors VPHeaderPHIRecipe)
  //===----------------------------------------------------------------------===//
  class TPHeaderPHIRecipe : public TPSingleDefRecipe, public TPPhiAccessors {
  protected:
    TPHeaderPHIRecipe(TPRecipeTy ID, ArrayRef<TPValue *> Operands)
        : TPSingleDefRecipe(ID, Operands) {}
    const TPRecipeBase *getAsRecipe() const override { return this; }
  public:
    static bool classof(const TPRecipeBase *R) {
      return R->getTPRecipeID() >= TPFirstHeaderPHISC &&
             R->getTPRecipeID() <= TPLastHeaderPHISC;
    }
    TPValue *getStartValue() const { return getOperand(0); }
    void setStartValue(TPValue *V) { setOperand(0, V); }
  };
  ```

- [ ] **Step 2: Replace the monolithic `TPWidenInductionRecipe` with an abstract base + two concrete classes**

  Remove the entire existing `TPWidenInductionRecipe` class block (lines ~846–873) and replace with the code below.

  **Important:** `IVPhi` and `DimIndex` were `private` in the original class. They must be `protected` in the new abstract base so the concrete subclasses can access them. The snippet below already reflects this.

  ```cpp
  //===----------------------------------------------------------------------===//
  // TPWidenInductionRecipe — abstract base for IV PHIs (mirrors VPWidenInductionRecipe)
  //===----------------------------------------------------------------------===//
  class TPWidenInductionRecipe : public TPHeaderPHIRecipe {
  protected:
    PHINode *IVPhi;    // was private in the old concrete class — now protected
    unsigned DimIndex; // same; no default initializer (constructor always receives an explicit Dim)
    TPWidenInductionRecipe(TPRecipeTy ID, PHINode *Phi,
                           TPValue *Start, TPValue *Step, unsigned Dim)
        : TPHeaderPHIRecipe(ID, {Start, Step}),
          IVPhi(Phi), DimIndex(Dim) {}
  public:
    static bool classof(const TPRecipeBase *R) {
      return R->getTPRecipeID() == TPWidenIntOrFpInductionSC ||
             R->getTPRecipeID() == TPWidenPointerInductionSC;
    }
    PHINode *getIVPhi()    const { return IVPhi; }
    unsigned getDimIndex() const { return DimIndex; }
  };

  //===----------------------------------------------------------------------===//
  // TPWidenIntOrFpInductionRecipe — integer or FP IV
  //===----------------------------------------------------------------------===//
  class TPWidenIntOrFpInductionRecipe : public TPWidenInductionRecipe {
  public:
    TPWidenIntOrFpInductionRecipe(PHINode *Phi, TPValue *Start,
                                   TPValue *Step, unsigned Dim)
        : TPWidenInductionRecipe(TPWidenIntOrFpInductionSC, Phi,
                                 Start, Step, Dim) {}
    static bool classof(const TPRecipeBase *R) {
      return R->getTPRecipeID() == TPWidenIntOrFpInductionSC;
    }
    void execute(TPTransformState &State) const override;
    void print(raw_ostream &OS, unsigned Indent,
               TPSlotTracker &Tracker) const override;
  };

  //===----------------------------------------------------------------------===//
  // TPWidenPointerInductionRecipe — pointer IV
  //===----------------------------------------------------------------------===//
  class TPWidenPointerInductionRecipe : public TPWidenInductionRecipe {
  public:
    TPWidenPointerInductionRecipe(PHINode *Phi, TPValue *Start,
                                   TPValue *Step, unsigned Dim)
        : TPWidenInductionRecipe(TPWidenPointerInductionSC, Phi,
                                 Start, Step, Dim) {}
    static bool classof(const TPRecipeBase *R) {
      return R->getTPRecipeID() == TPWidenPointerInductionSC;
    }
    void execute(TPTransformState &State) const override;
    void print(raw_ostream &OS, unsigned Indent,
               TPSlotTracker &Tracker) const override;
  };
  ```

- [ ] **Step 3: Build to check header compiles**

  ```bash
  ninja -C build llvm-opt 2>&1 | head -60
  ```
  Expected: errors in `TPlan.cpp` and `TPlanLowering.cpp` about missing `execute()`/`print()` — will fix next.

---

## Task 3: Rebase `TPCanonicalIVRecipe` and `TPReductionPHIRecipe`

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlan.h:1014,1022,878,884`

- [ ] **Step 1: Rebase `TPCanonicalIVRecipe` onto `TPHeaderPHIRecipe`**

  Find `TPCanonicalIVRecipe` class (~line 1014). Change its base class declaration:
  ```cpp
  // Before:
  class TPCanonicalIVRecipe : public TPSingleDefRecipe, public TPPhiAccessors {
  protected:
    const TPRecipeBase *getAsRecipe() const override { return this; }
  public:
    TPCanonicalIVRecipe(TPValue *StartVal, TPValue *StepVal)
        : TPSingleDefRecipe(TPCanonicalIVSC, {StartVal, StepVal}) {}

  // After:
  class TPCanonicalIVRecipe : public TPHeaderPHIRecipe {
  public:
    TPCanonicalIVRecipe(TPValue *StartVal, TPValue *StepVal)
        : TPHeaderPHIRecipe(TPCanonicalIVSC, {StartVal, StepVal}) {}
  ```
  Remove the `protected: const TPRecipeBase *getAsRecipe()` override — it's now inherited from `TPHeaderPHIRecipe`.

- [ ] **Step 2: Rebase `TPReductionPHIRecipe` onto `TPHeaderPHIRecipe`**

  Find `TPReductionPHIRecipe` class (~line 878). Change:
  ```cpp
  // Before:
  class TPReductionPHIRecipe : public TPSingleDefRecipe, public TPPhiAccessors {
  protected:
    const TPRecipeBase *getAsRecipe() const override { return this; }
  public:
    TPReductionPHIRecipe(PHINode *Phi, TPValue *InitVal, TPValue *LoopVal)
        : TPSingleDefRecipe(TPReductionPHISC, {InitVal, LoopVal}),

  // After:
  class TPReductionPHIRecipe : public TPHeaderPHIRecipe {
  public:
    TPReductionPHIRecipe(PHINode *Phi, TPValue *InitVal, TPValue *LoopVal)
        : TPHeaderPHIRecipe(TPReductionPHISC, {InitVal, LoopVal}),
  ```
  Remove the `protected: const TPRecipeBase *getAsRecipe()` override — inherited.

- [ ] **Step 3: Build to confirm no regressions from rebasing**

  ```bash
  ninja -C build llvm-opt 2>&1 | head -60
  ```
  Expected: errors from the `.cpp` files — `TPWidenIntOrFpInductionRecipe::execute` and `TPWidenIntOrFpInductionRecipe::print` (and their pointer counterparts) are declared in the header (Task 2) but not yet implemented in any `.cpp`. `TPCanonicalIVRecipe` and `TPReductionPHIRecipe` already have implementations — those should compile cleanly with the new base class.

---

## Task 4: Implement `execute()` and `print()` for concrete induction classes

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlan.cpp:302-312`
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp:178-181`

- [ ] **Step 1: Replace `TPWidenInductionRecipe::print` in `TPlan.cpp` with two concrete versions**

  Find `TPWidenInductionRecipe::print` at `TPlan.cpp:302`. Replace the whole function with:

  ```cpp
  void TPWidenIntOrFpInductionRecipe::print(raw_ostream &OS, unsigned Indent,
                                             TPSlotTracker &Tracker) const {
    printIndent(OS, Indent);
    OS << "WIDEN-INDUCTION ";
    printAsOperand(OS, Tracker);
    OS << " = phi ";
    Operands[0]->printAsOperand(OS, Tracker);
    OS << ", ";
    Operands[1]->printAsOperand(OS, Tracker);
    OS << "\n";
  }

  void TPWidenPointerInductionRecipe::print(raw_ostream &OS, unsigned Indent,
                                             TPSlotTracker &Tracker) const {
    printIndent(OS, Indent);
    OS << "WIDEN-POINTER-INDUCTION ";
    printAsOperand(OS, Tracker);
    OS << " = phi ";
    Operands[0]->printAsOperand(OS, Tracker);
    OS << ", ";
    Operands[1]->printAsOperand(OS, Tracker);
    OS << "\n";
  }
  ```

- [ ] **Step 2: Replace `TPWidenInductionRecipe::execute` in `TPlanLowering.cpp` with two concrete versions**

  Find `TPWidenInductionRecipe::execute` at `TPlanLowering.cpp:178`. Replace with:

  ```cpp
  void TPWidenIntOrFpInductionRecipe::execute(TPTransformState &State) const {
    State.setValue(this, IVPhi);
  }

  void TPWidenPointerInductionRecipe::execute(TPTransformState &State) const {
    State.setValue(this, IVPhi);
  }
  ```

- [ ] **Step 3: Build to confirm both files compile**

  ```bash
  ninja -C build llvm-opt 2>&1 | head -40
  ```
  Expected: remaining errors only from `buildInitial()` still using the old constructor.

---

## Task 5: Update `buildInitial()` to branch on pointer vs int/fp IV

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlan.cpp:665-668`

- [ ] **Step 1: Replace the `TPWidenInductionRecipe` construction in `buildInitial()`**

  Find `TPlan.cpp:665`:
  ```cpp
  // Before:
        auto *R = new TPWidenInductionRecipe(
            &Phi, StartTP,
            StartTP /* placeholder; patched after body */, Idx);

  // After:
        TPWidenInductionRecipe *R;
        if (Phi.getType()->isPointerTy())
          R = new TPWidenPointerInductionRecipe(
              &Phi, StartTP,
              StartTP /* placeholder; patched after body */, Idx);
        else
          R = new TPWidenIntOrFpInductionRecipe(
              &Phi, StartTP,
              StartTP /* placeholder; patched after body */, Idx);
  ```

- [ ] **Step 2: Build to confirm everything compiles cleanly**

  ```bash
  ninja -C build llvm-opt 2>&1 | head -20
  ```
  Expected: clean build (zero errors).

- [ ] **Step 3: Verify lit test and confirm no CHECK change is needed**

  ```bash
  llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tplan-print.ll
  ```
  Expected: PASS.

  **Why no CHECK update:** The spec's Consumer Updates table says "Update CHECK lines: `TPWidenInductionRecipe` → `TPWidenIntOrFpInductionRecipe`". That note refers to the *class name in the print output*. However, the actual `print()` function emits the string token `WIDEN-INDUCTION`, and the CHECK lines match that token — not the C++ class name. Since `TPWidenIntOrFpInductionRecipe::print` keeps the same `"WIDEN-INDUCTION"` token, no CHECK file change is needed. This is a plan-level clarification of an imprecise spec requirement.

- [ ] **Step 4: Commit and push**

  ```bash
  git add llvm/include/llvm/Transforms/Vectorize/TPlan.h \
          llvm/lib/Transforms/Vectorize/TPlan.cpp \
          llvm/lib/Transforms/Vectorize/TPlanLowering.cpp
  git commit -m "tplan: add TPHeaderPHIRecipe; split TPWidenInductionRecipe into int/fp + pointer subclasses"
  git push yg LoopTensorizebyClaude
  ```

---

## Task 6: Add six new recipe stubs (Commit 2)

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlan.h` (enum, 6 new classes, classof)
- Modify: `llvm/lib/Transforms/Vectorize/TPlan.cpp` (6 new print() bodies)
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` (6 new execute() bodies)

- [ ] **Step 1: Add new SCs to the enum and update range markers**

  In `TPlan.h`, find the enum. Insert after `TPReductionPHISC` and **before** `TPCanonicalIVIncrSC`. Three groups in order: (a) header PHI stubs (still inside the header-PHI range), (b) non-header PHI stubs (inside the overall PHI range, outside the header-PHI sub-range), (c) canonical IV companions (outside the PHI range, as today). Then update the range markers.

  ```cpp
      // Header PHI stubs (Commit 2)
      TPFirstOrderRecurrencePHISC,
      TPActiveLaneMaskPHISC,
      TPEVLBasedIVPHISC,

      // Non-header PHI stubs (Commit 2)
      TPPredInstPHISC,
      TPPhiSC,

      // Canonical IV companion recipes (outside PHI range)
      TPCanonicalIVIncrSC,
      TPCanonicalIVExitCmpSC,
  ```

  Update range markers:
  ```cpp
      TPLastHeaderPHISC  = TPEVLBasedIVPHISC,   // was TPReductionPHISC
      TPLastPHISC        = TPPhiSC,              // was TPReductionPHISC
  ```

- [ ] **Step 2: Update `TPSingleDefRecipe::classof`**

  - The current switch has `case TPWidenPHISC: return false;` as a standalone block (placeholder). **Edit that one existing case** — do NOT add a second `case TPWidenPHISC:`. Move it into the `return true` arm alongside the other new cases:
    ```cpp
    // Before:
        case TPReductionPHISC:
        case TPCanonicalIVIncrSC:
        case TPCanonicalIVExitCmpSC:
          return true;
        case TPWidenStoreSC:
        case TPWidenPHISC:  // no concrete class yet
          return false;

    // After:
        case TPReductionPHISC:
        case TPWidenPHISC:
        case TPFirstOrderRecurrencePHISC:
        case TPActiveLaneMaskPHISC:
        case TPEVLBasedIVPHISC:
        case TPPredInstPHISC:
        case TPPhiSC:
        case TPCanonicalIVIncrSC:
        case TPCanonicalIVExitCmpSC:
          return true;
        case TPWidenStoreSC:
          return false;
    ```

- [ ] **Step 3: Add the six stub class declarations in `TPlan.h`**

  After the `TPReductionPHIRecipe` class block, add:

  ```cpp
  //===----------------------------------------------------------------------===//
  // TPFirstOrderRecurrencePHIRecipe — first-order recurrence PHI stub
  //===----------------------------------------------------------------------===//
  class TPFirstOrderRecurrencePHIRecipe : public TPHeaderPHIRecipe {
  public:
    explicit TPFirstOrderRecurrencePHIRecipe(PHINode *Phi, TPValue *StartVal)
        : TPHeaderPHIRecipe(TPFirstOrderRecurrencePHISC, {StartVal}) {}
    static bool classof(const TPRecipeBase *R) {
      return R->getTPRecipeID() == TPFirstOrderRecurrencePHISC;
    }
    void execute(TPTransformState &State) const override;
    void print(raw_ostream &OS, unsigned Indent,
               TPSlotTracker &Tracker) const override;
  };

  //===----------------------------------------------------------------------===//
  // TPActiveLaneMaskPHIRecipe — active-lane-mask PHI stub
  //===----------------------------------------------------------------------===//
  class TPActiveLaneMaskPHIRecipe : public TPHeaderPHIRecipe {
  public:
    explicit TPActiveLaneMaskPHIRecipe(TPValue *StartVal)
        : TPHeaderPHIRecipe(TPActiveLaneMaskPHISC, {StartVal}) {}
    static bool classof(const TPRecipeBase *R) {
      return R->getTPRecipeID() == TPActiveLaneMaskPHISC;
    }
    void execute(TPTransformState &State) const override;
    void print(raw_ostream &OS, unsigned Indent,
               TPSlotTracker &Tracker) const override;
  };

  //===----------------------------------------------------------------------===//
  // TPEVLBasedIVPHIRecipe — EVL-based IV PHI stub
  //===----------------------------------------------------------------------===//
  class TPEVLBasedIVPHIRecipe : public TPHeaderPHIRecipe {
  public:
    explicit TPEVLBasedIVPHIRecipe(TPValue *StartVal)
        : TPHeaderPHIRecipe(TPEVLBasedIVPHISC, {StartVal}) {}
    static bool classof(const TPRecipeBase *R) {
      return R->getTPRecipeID() == TPEVLBasedIVPHISC;
    }
    void execute(TPTransformState &State) const override;
    void print(raw_ostream &OS, unsigned Indent,
               TPSlotTracker &Tracker) const override;
  };

  //===----------------------------------------------------------------------===//
  // TPWidenPHIRecipe — generic widen-PHI (fills existing placeholder)
  //===----------------------------------------------------------------------===//
  class TPWidenPHIRecipe : public TPSingleDefRecipe, public TPPhiAccessors {
  protected:
    const TPRecipeBase *getAsRecipe() const override { return this; }
  public:
    explicit TPWidenPHIRecipe(PHINode *Phi, TPValue *StartVal)
        : TPSingleDefRecipe(TPWidenPHISC, {StartVal}) {}
    static bool classof(const TPRecipeBase *R) {
      return R->getTPRecipeID() == TPWidenPHISC;
    }
    void execute(TPTransformState &State) const override;
    void print(raw_ostream &OS, unsigned Indent,
               TPSlotTracker &Tracker) const override;
  };

  //===----------------------------------------------------------------------===//
  // TPPredInstPHIRecipe — predicated-instruction result PHI stub
  //===----------------------------------------------------------------------===//
  class TPPredInstPHIRecipe : public TPSingleDefRecipe {
  public:
    explicit TPPredInstPHIRecipe(TPValue *PredVal)
        : TPSingleDefRecipe(TPPredInstPHISC, {PredVal}) {}
    static bool classof(const TPRecipeBase *R) {
      return R->getTPRecipeID() == TPPredInstPHISC;
    }
    void execute(TPTransformState &State) const override;
    void print(raw_ostream &OS, unsigned Indent,
               TPSlotTracker &Tracker) const override;
  };

  //===----------------------------------------------------------------------===//
  // TPPhi — generic PHI stub
  //===----------------------------------------------------------------------===//
  class TPPhi : public TPSingleDefRecipe, public TPPhiAccessors {
  protected:
    const TPRecipeBase *getAsRecipe() const override { return this; }
  public:
    explicit TPPhi(ArrayRef<TPValue *> Operands)
        : TPSingleDefRecipe(TPPhiSC, Operands) {}
    static bool classof(const TPRecipeBase *R) {
      return R->getTPRecipeID() == TPPhiSC;
    }
    void execute(TPTransformState &State) const override;
    void print(raw_ostream &OS, unsigned Indent,
               TPSlotTracker &Tracker) const override;
  };
  ```

- [ ] **Step 4: Implement `print()` for all six stubs in `TPlan.cpp`**

  Add after the existing `TPReductionPHIRecipe::print` body:

  ```cpp
  void TPFirstOrderRecurrencePHIRecipe::print(raw_ostream &OS, unsigned Indent,
                                               TPSlotTracker &Tracker) const {
    printIndent(OS, Indent);
    OS << "FIRST-ORDER-RECURRENCE-PHI ";
    printAsOperand(OS, Tracker);
    OS << "\n";
  }

  void TPActiveLaneMaskPHIRecipe::print(raw_ostream &OS, unsigned Indent,
                                         TPSlotTracker &Tracker) const {
    printIndent(OS, Indent);
    OS << "ACTIVE-LANE-MASK-PHI ";
    printAsOperand(OS, Tracker);
    OS << "\n";
  }

  void TPEVLBasedIVPHIRecipe::print(raw_ostream &OS, unsigned Indent,
                                     TPSlotTracker &Tracker) const {
    printIndent(OS, Indent);
    OS << "EVL-BASED-IV-PHI ";
    printAsOperand(OS, Tracker);
    OS << "\n";
  }

  void TPWidenPHIRecipe::print(raw_ostream &OS, unsigned Indent,
                                TPSlotTracker &Tracker) const {
    printIndent(OS, Indent);
    OS << "WIDEN-PHI ";
    printAsOperand(OS, Tracker);
    OS << "\n";
  }

  void TPPredInstPHIRecipe::print(raw_ostream &OS, unsigned Indent,
                                   TPSlotTracker &Tracker) const {
    printIndent(OS, Indent);
    OS << "PRED-PHI ";
    printAsOperand(OS, Tracker);
    OS << "\n";
  }

  void TPPhi::print(raw_ostream &OS, unsigned Indent,
                    TPSlotTracker &Tracker) const {
    printIndent(OS, Indent);
    OS << "PHI ";
    printAsOperand(OS, Tracker);
    OS << "\n";
  }
  ```

- [ ] **Step 5: Implement no-op `execute()` for all six stubs in `TPlanLowering.cpp`**

  Add after the existing `TPReductionPHIRecipe::execute`:

  ```cpp
  void TPFirstOrderRecurrencePHIRecipe::execute(TPTransformState &) const {}
  void TPActiveLaneMaskPHIRecipe::execute(TPTransformState &) const {}
  void TPEVLBasedIVPHIRecipe::execute(TPTransformState &) const {}
  void TPWidenPHIRecipe::execute(TPTransformState &) const {}
  void TPPredInstPHIRecipe::execute(TPTransformState &) const {}
  void TPPhi::execute(TPTransformState &) const {}
  ```

- [ ] **Step 6: Build**

  ```bash
  ninja -C build llvm-opt 2>&1 | head -20
  ```
  Expected: clean build.

- [ ] **Step 7: Run lit test**

  ```bash
  llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tplan-print.ll
  ```
  Expected: PASS (no new visible output from stubs).

- [ ] **Step 8: Commit and push**

  ```bash
  git add llvm/include/llvm/Transforms/Vectorize/TPlan.h \
          llvm/lib/Transforms/Vectorize/TPlan.cpp \
          llvm/lib/Transforms/Vectorize/TPlanLowering.cpp
  git commit -m "tplan: add 6 new PHI recipe stubs (TPWidenPHIRecipe, TPFirstOrderRecurrencePHIRecipe, TPActiveLaneMaskPHIRecipe, TPEVLBasedIVPHIRecipe, TPPredInstPHIRecipe, TPPhi)"
  git push yg LoopTensorizebyClaude
  ```
