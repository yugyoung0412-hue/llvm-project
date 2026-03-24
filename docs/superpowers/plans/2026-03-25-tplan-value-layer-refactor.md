# TPlan Value-Layer Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor TPlan's value/def/use classes to mirror VPlan's `VPValue`/`VPDef`/`VPUser`/`VPRecipeValue` design, replacing the conflated `TPSingleDefRecipe : TPRecipeBase, TPValue` dual-inheritance with the clean three-way split; add `TPDef`, `TPRecipeValue`, `TPConstantInt`, `TPPhiAccessors`, `TPIRFlags`, `TPIRMetadata`, `TPRecipeWithIRFlags`, and a full `TPRecipeTy` SubclassID enum with PHI-range markers.

**Architecture:** Four git commits delivered across Tasks 1–4. Tasks 1 and 2 together compose "Commit 1" from the spec — split into two smaller commits here for safer incremental review. Task 3 is spec Commit 2 (TPPhiAccessors). Task 4 is spec Commit 3 (flags/metadata). Each task gates on a build + lit-test pass before committing.

**Intentional spec deviations:**
- `TPRecipeWithIRFlags` constructor takes `Instruction &FlagSrc` (extracts flags inline) rather than a pre-built `const TPIRFlags &Flags`. Call sites always have the instruction available — this is strictly more ergonomic.
- `TPPhiAccessors` block-related accessors (`getIncomingBlock`, `incoming_blocks`, `incoming_values_and_blocks`, `getIncomingValueForBlock`) are deferred — no current consumer in TPlan uses them. `removeIncomingValueFor` takes `unsigned Idx` (simpler; block-based overload deferred).
- Tasks 1+2 are two commits instead of one to keep diffs reviewable.

**Tech Stack:** C++17, LLVM ADT (`TinyPtrVector`, `SmallBitVector`, `DenseMap`, `STLExtras`), LLVM `ilist_node_with_parent`, LLVM lit.

---

## File Map

| File | Role / Change |
|---|---|
| `llvm/include/llvm/Transforms/Vectorize/TPlan.h` | All class declarations — primary change file |
| `llvm/lib/Transforms/Vectorize/TPlan.cpp` | Method implementations; `buildInitial()` update |
| `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` | Update `TPTransformState` ValueMap key; `execute()` flag calls |
| `llvm/lib/Transforms/Vectorize/TPlanWidener.cpp` | Update renamed-class references |
| `llvm/lib/Transforms/Vectorize/TPRecipeMatcher.cpp` | Update renamed-class references |
| `llvm/test/Transforms/LoopTensorize/basic/tplan-print.ll` | Lit test — verify print format unchanged |

**Build command throughout:** `ninja -C build LLVMVectorize 2>&1 | tail -20`
**Test command throughout:** `build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/`
*(Substitute your actual build directory path if it differs from `build/`.)*

---

## Task 1: Value layer — rename TPLiveIn→TPIRValue, TPSyntheticValue→TPSymbolicValue; replace ValueKind with SubclassID; add UnderlyingVal and TPConstantInt; add TPValue iterator/utility API

**Spec reference:** §3.1, §3.2, §3.3, §3.4

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlan.h`
- Modify: `llvm/lib/Transforms/Vectorize/TPlan.cpp`
- Modify: `llvm/lib/Transforms/Vectorize/TPlanWidener.cpp`
- Modify: `llvm/lib/Transforms/Vectorize/TPRecipeMatcher.cpp`
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp`

- [ ] **Step 1: Add new includes to `TPlan.h`**

  Add after the existing `#include` block:
  ```cpp
  #include "llvm/ADT/APInt.h"
  #include "llvm/ADT/STLExtras.h"
  #include "llvm/IR/Constants.h"
  ```

- [ ] **Step 2: Replace `TPValue` class in `TPlan.h`**

  Replace the entire `TPValue` class (lines 48–66 in the current header) with:
  ```cpp
  class TPValue {
  public:
    // SubclassIDs for the value layer (independent of TPRecipeTy on the recipe layer).
    enum {
      TPVIRValueSC,      // wraps IR Value* (was LiveIn)
      TPVSymbolicSC,     // named synthetic, no IR backing (was Synthetic)
      TPVRecipeValueSC,  // value defined by a recipe
    };

    unsigned getTPValueID() const { return SubclassID; }

    // Non-null only for TPIRValue / TPConstantInt.
    Value *getUnderlyingValue() const { return UnderlyingVal; }

    // User-list management.
    void addUser(class TPUser *U) { Users.push_back(U); }
    void removeUser(class TPUser *U) {
      auto It = llvm::find(Users, U);
      if (It != Users.end()) Users.erase(It);
    }
    ArrayRef<class TPUser *> users() const { return Users; }

    // User-list query helpers (mirror VPValue API).
    bool hasOneUse() const { return Users.size() == 1; }
    bool hasMoreThanOneUniqueUser() const {
      if (Users.size() <= 1) return false;
      SmallPtrSet<TPUser *, 4> Seen(Users.begin(), Users.end());
      return Seen.size() > 1;
    }
    TPUser *getSingleUser() const {
      assert(hasOneUse() && "Not a single-use value");
      return Users[0];
    }

    // Replace all uses of this value with New.
    void replaceAllUsesWith(TPValue *New);
    // Conditional replacement: call Fn(User, OperandIdx); replace if true.
    void replaceUsesWithIf(TPValue *New,
                            function_ref<bool(TPUser &, unsigned)> Fn);

    // Returns the recipe that defines this value, or nullptr for live-ins/symbolics.
    class TPRecipeBase *getDefiningRecipe();
    const class TPRecipeBase *getDefiningRecipe() const;

    // True for live-ins and symbolics — values produced outside any loop region.
    bool isDefinedOutsideLoopRegions() const {
      return getTPValueID() != TPVRecipeValueSC;
    }

    // User iterators.
    using user_iterator       = SmallVectorImpl<TPUser *>::iterator;
    using const_user_iterator = SmallVectorImpl<TPUser *>::const_iterator;
    using user_range          = iterator_range<user_iterator>;
    using const_user_range    = iterator_range<const_user_iterator>;
    user_range       users_range()       { return make_range(Users.begin(), Users.end()); }
    const_user_range users_range() const { return make_range(Users.begin(), Users.end()); }

    virtual void printAsOperand(raw_ostream &OS, TPSlotTracker &Tracker) const = 0;
    virtual ~TPValue() = default;

  protected:
    explicit TPValue(unsigned SC, Value *UV = nullptr)
        : SubclassID(SC), UnderlyingVal(UV) {}

    SmallVector<class TPUser *, 2> Users;

  private:
    const unsigned char SubclassID;
    Value *UnderlyingVal = nullptr;
  };
  ```

- [ ] **Step 3: Replace `TPLiveIn` with `TPIRValue` in `TPlan.h`**

  Replace the `TPLiveIn` class:
  ```cpp
  class TPIRValue : public TPValue {
  public:
    explicit TPIRValue(Value *UV)
        : TPValue(TPVIRValueSC, UV) {
      assert(UV && "TPIRValue requires an underlying IR value");
    }
    Value *getValue() const { return getUnderlyingValue(); }
    Type  *getType()  const { return getUnderlyingValue()->getType(); }
    void printAsOperand(raw_ostream &OS, TPSlotTracker &Tracker) const override;

    static bool classof(const TPValue *V) {
      return V->getTPValueID() == TPVIRValueSC;
    }
  };
  ```

- [ ] **Step 4: Add `TPConstantInt` after `TPIRValue` in `TPlan.h`**

  ```cpp
  class TPConstantInt : public TPIRValue {
  public:
    explicit TPConstantInt(ConstantInt *CI) : TPIRValue(CI) {}
    bool isOne()  const;
    bool isZero() const;
    const APInt &getAPInt()    const;
    unsigned getBitWidth()     const;
    uint64_t getZExtValue()    const;

    // Both overloads required for isa<TPConstantInt> to work from
    // either TPIRValue* or TPValue* source type.
    static bool classof(const TPIRValue *V) {
      return isa<ConstantInt>(V->getUnderlyingValue());
    }
    static bool classof(const TPValue *V) {
      if (const auto *IR = dyn_cast<TPIRValue>(V))
        return classof(IR);
      return false;
    }
  };
  ```

- [ ] **Step 5: Replace `TPSyntheticValue` with `TPSymbolicValue` in `TPlan.h`**

  ```cpp
  class TPSymbolicValue : public TPValue {
  public:
    explicit TPSymbolicValue(StringRef Name)
        : TPValue(TPVSymbolicSC), Name(Name) {}
    StringRef getName() const { return Name; }
    void printAsOperand(raw_ostream &OS, TPSlotTracker &Tracker) const override;

    static bool classof(const TPValue *V) {
      return V->getTPValueID() == TPVSymbolicSC;
    }
  private:
    std::string Name;
  };
  ```

- [ ] **Step 6: Update `TPSlotTracker` in `TPlan.h`**

  Change `preAssignSynthetic` parameter:
  ```cpp
  void preAssignSynthetic(const TPSymbolicValue *V);
  ```

- [ ] **Step 7: Update `TPlan` field types in `TPlan.h`**

  ```cpp
  // In TPlan private section:
  SmallVector<std::unique_ptr<TPSymbolicValue>> DimPFs;   // was TPSyntheticValue
  SmallVector<std::unique_ptr<TPIRValue>>       LiveIns;  // was TPLiveIn

  // Public accessor:
  TPSymbolicValue *getDimPF(unsigned D) const {           // was TPSyntheticValue*
    assert(D < DimPFs.size() && "Dim out of range");
    return DimPFs[D].get();
  }

  // Private helper:
  TPIRValue *getOrCreateLiveIn(Value *V);                 // was TPLiveIn*
  ```

- [ ] **Step 8: Update `TPlan.cpp` — rename classes and methods**

  In `TPlan.cpp` do a text search-and-replace:
  - `TPLiveIn` → `TPIRValue`
  - `TPSyntheticValue` → `TPSymbolicValue`
  - `ValueKind::LiveIn` → `TPVIRValueSC`
  - `ValueKind::Synthetic` → `TPVSymbolicSC`
  - `ValueKind::Def` → `TPVRecipeValueSC`
  - `getValueKind()` → `getTPValueID()`

  Then rename the `printAsOperand` implementations:
  - `void TPLiveIn::printAsOperand(...)` → `void TPIRValue::printAsOperand(...)`
  - `void TPSyntheticValue::printAsOperand(...)` → `void TPSymbolicValue::printAsOperand(...)`

  Add `TPConstantInt` method implementations:
  ```cpp
  bool TPConstantInt::isOne()  const { return cast<ConstantInt>(getUnderlyingValue())->isOne(); }
  bool TPConstantInt::isZero() const { return cast<ConstantInt>(getUnderlyingValue())->isZero(); }
  const APInt &TPConstantInt::getAPInt() const {
    return cast<ConstantInt>(getUnderlyingValue())->getValue();
  }
  unsigned TPConstantInt::getBitWidth() const { return getAPInt().getBitWidth(); }
  uint64_t TPConstantInt::getZExtValue() const { return getAPInt().getZExtValue(); }
  ```

  Add `TPValue::replaceAllUsesWith` and `replaceUsesWithIf` implementations:
  ```cpp
  void TPValue::replaceAllUsesWith(TPValue *New) {
    for (TPUser *U : SmallVector<TPUser *, 4>(Users))
      for (unsigned I = 0, E = U->getNumOperands(); I != E; ++I)
        if (U->getOperand(I) == this)
          U->setOperand(I, New);
  }

  void TPValue::replaceUsesWithIf(TPValue *New,
                                   function_ref<bool(TPUser &, unsigned)> Fn) {
    for (TPUser *U : SmallVector<TPUser *, 4>(Users))
      for (unsigned I = 0, E = U->getNumOperands(); I != E; ++I)
        if (U->getOperand(I) == this && Fn(*U, I))
          U->setOperand(I, New);
  }
  ```

  Add stub `TPValue::getDefiningRecipe()` (returns nullptr for the base class;
  `TPRecipeValue` overrides this — defined after TPSingleDefRecipe below):
  ```cpp
  TPRecipeBase *TPValue::getDefiningRecipe() { return nullptr; }
  const TPRecipeBase *TPValue::getDefiningRecipe() const { return nullptr; }
  ```

- [ ] **Step 9: Update `TPlanWidener.cpp` references**

  Replace `TPLiveIn` → `TPIRValue`, `TPSyntheticValue` → `TPSymbolicValue`,
  `getValueKind()` → `getTPValueID()`, `ValueKind::*` → `TP*SC`.

- [ ] **Step 10: Update `TPRecipeMatcher.cpp` references**

  Same rename pass.

- [ ] **Step 11: Update `TPlanLowering.cpp` references**

  Same rename pass; any `TPLiveIn *` casts → `TPIRValue *`.

- [ ] **Step 12: Build**

  ```bash
  ninja -C build LLVMVectorize 2>&1 | tail -30
  ```
  Expected: 0 errors.

- [ ] **Step 13: Run lit test**

  ```bash
  build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/
  ```

- [ ] **Step 14: Commit**

  ```bash
  git add llvm/include/llvm/Transforms/Vectorize/TPlan.h \
          llvm/lib/Transforms/Vectorize/TPlan.cpp \
          llvm/lib/Transforms/Vectorize/TPlanWidener.cpp \
          llvm/lib/Transforms/Vectorize/TPRecipeMatcher.cpp \
          llvm/lib/Transforms/Vectorize/TPlanLowering.cpp
  git commit -m "tplan: rename TPLiveIn→TPIRValue, TPSyntheticValue→TPSymbolicValue; add SubclassID/UnderlyingVal to TPValue; add TPConstantInt and value utility API"
  ```

---

## Task 2: Add TPDef, TPRecipeValue, TPUser refactor, TPRecipeTy enum; wire into TPRecipeBase + TPSingleDefRecipe; update all recipe subclasses

**Spec reference:** §3.5, §3.6, §3.7, §3.8, §3.9

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlan.h`
- Modify: `llvm/lib/Transforms/Vectorize/TPlan.cpp`
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp`
- Modify: `llvm/lib/Transforms/Vectorize/TPlanWidener.cpp`
- Modify: `llvm/lib/Transforms/Vectorize/TPRecipeMatcher.cpp`

Add to `TPlan.h` includes: `#include "llvm/ADT/TinyPtrVector.h"`

- [ ] **Step 1: Add `TPDef` class in `TPlan.h` (before `TPRecipeBase`)**

  ```cpp
  //===----------------------------------------------------------------------===//
  // TPDef — owns the TPRecipeValue instances produced by a recipe (mirrors VPDef)
  //===----------------------------------------------------------------------===//
  class TPDef {
    friend class TPRecipeValue;
    TinyPtrVector<TPRecipeValue *> DefinedValues;

    void addDefinedValue(TPRecipeValue *V) { DefinedValues.push_back(V); }
    void removeDefinedValue(TPRecipeValue *V) {
      auto It = llvm::find(DefinedValues, V);
      assert(It != DefinedValues.end());
      DefinedValues.erase(It);
    }

  public:
    TPDef() = default;
    virtual ~TPDef() {
      // DefinedValues is already empty here: TPRecipeValue::~TPRecipeValue calls
      // removeDefinedValue, which runs before this destructor in MRO order.
      // The assert catches any future multi-def allocation that forgets to delete.
      assert(DefinedValues.empty() &&
             "TPDef destroyed with live DefinedValues — standalone values must be deleted by subclass");
    }

    TPValue       *getTPSingleValue()       { assert(DefinedValues.size()==1); return DefinedValues[0]; }
    const TPValue *getTPSingleValue() const { assert(DefinedValues.size()==1); return DefinedValues[0]; }
    TPValue       *getTPValue(unsigned I)       { return DefinedValues[I]; }
    const TPValue *getTPValue(unsigned I) const { return DefinedValues[I]; }

    // Return type is non-const element for both overloads (matches VPlan pattern;
    // avoids pointer-to-pointer conversion issues with TinyPtrVector).
    ArrayRef<TPRecipeValue *> definedValues()       { return DefinedValues; }
    ArrayRef<TPRecipeValue *> definedValues() const { return DefinedValues; }

    unsigned getNumDefinedValues() const { return DefinedValues.size(); }
  };
  ```

- [ ] **Step 2: Add `TPRecipeValue` class in `TPlan.h` (after `TPDef`, before `TPRecipeBase`)**

  ```cpp
  //===----------------------------------------------------------------------===//
  // TPRecipeValue — a TPValue produced by a recipe (mirrors VPRecipeValue)
  //===----------------------------------------------------------------------===//
  class TPRecipeValue : public TPValue {
    TPDef *Def;
  public:
    // Called by TPSingleDefRecipe ctor with `this` as Def.
    TPRecipeValue(TPDef *D, Value *UV = nullptr)
        : TPValue(TPVRecipeValueSC, UV), Def(D) {
      Def->addDefinedValue(this);
    }
    // Removes itself from Def->DefinedValues. By the time TPDef::~TPDef runs,
    // this destructor has already fired (MRO), so DefinedValues is empty.
    virtual ~TPRecipeValue() {
      Def->removeDefinedValue(this);
    }

    TPDef *getDef() const { return Def; }

    // Returns the enclosing recipe. Safe because the only TPDef subclass that
    // creates TPRecipeValue is TPRecipeBase (static_cast is valid).
    TPRecipeBase *getDefiningRecipe() {
      return static_cast<TPRecipeBase *>(Def);
    }
    const TPRecipeBase *getDefiningRecipe() const {
      return static_cast<const TPRecipeBase *>(Def);
    }

    static bool classof(const TPValue *V) {
      return V->getTPValueID() == TPVRecipeValueSC;
    }
  };
  ```

- [ ] **Step 3: Replace `TPUser` class in `TPlan.h`**

  ```cpp
  class TPUser {
    friend class TPPhiAccessors;  // forward-declared below; grants removeOperand access
  public:
    TPUser() = delete;
    virtual ~TPUser() {
      for (TPValue *Op : Operands)
        Op->removeUser(this);
    }

    void addOperand(TPValue *V) {
      Operands.push_back(V);
      V->addUser(this);
    }
    void setOperand(unsigned I, TPValue *V) {
      assert(I < Operands.size());
      Operands[I]->removeUser(this);
      Operands[I] = V;
      V->addUser(this);
    }
    void swapOperands() {
      assert(Operands.size() == 2 && "swapOperands requires exactly 2 operands");
      std::swap(Operands[0], Operands[1]);
    }
    void replaceUsesOfWith(TPValue *From, TPValue *To) {
      for (unsigned I = 0, E = Operands.size(); I != E; ++I)
        if (Operands[I] == From)
          setOperand(I, To);
    }

    ArrayRef<TPValue *> operands() const { return Operands; }
    TPValue *getOperand(unsigned I) const { return Operands[I]; }
    unsigned getNumOperands() const { return Operands.size(); }

    using operand_iterator       = SmallVectorImpl<TPValue *>::iterator;
    using const_operand_iterator = SmallVectorImpl<TPValue *>::const_iterator;
    using operand_range          = iterator_range<operand_iterator>;
    using const_operand_range    = iterator_range<const_operand_iterator>;
    operand_range       operands_range()       { return make_range(Operands.begin(), Operands.end()); }
    const_operand_range operands_range() const { return make_range(Operands.begin(), Operands.end()); }

    // Stub methods mirroring VPUser API (not used in TPlan today).
    virtual bool usesScalars(const TPValue *) const { return false; }
    virtual bool usesFirstLaneOnly(const TPValue *) const { return false; }
    virtual bool usesFirstPartOnly(const TPValue *) const { return false; }

    static bool classof(const TPUser *) { return true; }

  protected:
    explicit TPUser(ArrayRef<TPValue *> Ops) {
      for (TPValue *V : Ops) addOperand(V);
    }

    void removeOperand(unsigned I) {
      assert(I < Operands.size());
      Operands[I]->removeUser(this);
      Operands.erase(Operands.begin() + I);
    }

    SmallVector<TPValue *, 4> Operands;
  };
  ```

- [ ] **Step 4: Replace `TPRecipeBase` class in `TPlan.h`**

  Key changes: `ilist_node` → `ilist_node_with_parent`, add `TPDef` base, replace
  `RecipeKind` with `TPRecipeTy`, add `Parent` field (required by
  `ilist_node_with_parent`), add `isPhi()`, add insertion helpers.

  ```cpp
  class TPRecipeBase
      : public ilist_node_with_parent<TPRecipeBase, TPBasicBlock>,
        public TPDef,
        public TPUser {

    friend class TPBasicBlock;
    friend class TPBlockUtils;

    const unsigned char SubclassID;  // Holds a TPRecipeTy value
    TPBasicBlock *Parent = nullptr;  // Required by ilist_node_with_parent

  public:
    // TPRecipeTy: mirrors VPRecipeTy. Access as TPRecipeBase::TPWidenSC.
    using TPRecipeTy = enum {
      // Non-PHI recipes
      TPWidenSC,
      TPWidenGEPSC,
      TPWidenLoadSC,
      TPWidenStoreSC,
      TPWidenCastSC,

      // PHI-like recipes (range: [TPFirstPHISC, TPLastPHISC])
      TPWidenPHISC,           // placeholder — no concrete class yet; classof returns false

      // Header PHI recipes (subset: [TPFirstHeaderPHISC, TPLastHeaderPHISC])
      TPCanonicalIVSC,
      TPWidenInductionSC,
      TPReductionPHISC,

      // Canonical IV companions (outside PHI range)
      TPCanonicalIVIncrSC,
      TPCanonicalIVExitCmpSC,

      // Range markers
      TPFirstPHISC       = TPWidenPHISC,
      TPFirstHeaderPHISC = TPCanonicalIVSC,
      TPLastHeaderPHISC  = TPReductionPHISC,
      TPLastPHISC        = TPReductionPHISC,
    };

    unsigned getTPRecipeID() const { return SubclassID; }

    bool isPhi() const {
      return getTPRecipeID() >= TPFirstPHISC &&
             getTPRecipeID() <= TPLastPHISC;
    }

    TPBasicBlock *getParent()       { return Parent; }
    const TPBasicBlock *getParent() const { return Parent; }

    TPSingleDefRecipe       *getDefinedValue();
    const TPSingleDefRecipe *getDefinedValue() const;

    // Insertion/movement helpers (set/clear Parent as needed).
    void insertBefore(TPRecipeBase *InsertPos);
    void insertAfter(TPRecipeBase *InsertPos);
    void removeFromParent();
    iplist<TPRecipeBase>::iterator eraseFromParent();

    virtual void print(raw_ostream &OS, unsigned Indent,
                       TPSlotTracker &Tracker) const = 0;
    virtual void execute(TPTransformState &State) const = 0;
    virtual ~TPRecipeBase() = default;

    static bool classof(const TPUser *) { return true; }

  protected:
    explicit TPRecipeBase(unsigned char SC, ArrayRef<TPValue *> Operands = {})
        : TPDef(), TPUser(Operands), SubclassID(SC) {}
  };
  ```

  Also update `TPBasicBlock::appendRecipe` to set `Parent`:
  ```cpp
  void appendRecipe(TPRecipeBase *R) {
    R->Parent = this;
    Recipes.push_back(R);
  }
  ```

- [ ] **Step 5: Replace `TPSingleDefRecipe` in `TPlan.h`**

  ```cpp
  class TPSingleDefRecipe : public TPRecipeBase,   // first — ilist layout
                            public TPRecipeValue {  // second — sub-object
  public:
    SmallBitVector DimSet;

    void printAsOperand(raw_ostream &OS, TPSlotTracker &Tracker) const override;

    static bool classof(const TPRecipeBase *R) {
      switch (R->getTPRecipeID()) {
      case TPWidenSC:
      case TPWidenGEPSC:
      case TPWidenLoadSC:
      case TPWidenCastSC:
      case TPCanonicalIVSC:
      case TPWidenInductionSC:
      case TPReductionPHISC:
      case TPCanonicalIVIncrSC:
      case TPCanonicalIVExitCmpSC:
        return true;
      case TPWidenStoreSC:
      case TPWidenPHISC:  // no concrete class yet
        return false;
      }
      llvm_unreachable("unknown TPRecipeTy");
    }
    static bool classof(const TPValue *V) {
      return V->getTPValueID() == TPVRecipeValueSC;
    }

  protected:
    explicit TPSingleDefRecipe(unsigned char SC,
                               ArrayRef<TPValue *> Ops = {},
                               Value *UV = nullptr)
        : TPRecipeBase(SC, Ops), TPRecipeValue(this, UV) {}
  };
  ```

- [ ] **Step 6: Add inline `getDefinedValue()` after `TPSingleDefRecipe`**

  ```cpp
  inline TPSingleDefRecipe *TPRecipeBase::getDefinedValue() {
    return dyn_cast<TPSingleDefRecipe>(this);
  }
  inline const TPSingleDefRecipe *TPRecipeBase::getDefinedValue() const {
    return dyn_cast<TPSingleDefRecipe>(this);
  }
  ```

- [ ] **Step 7: Update all concrete recipe subclasses in `TPlan.h`**

  Update each class: change base `RecipeKind` arg to the new SC constant,
  pass operand list directly to the base constructor, update `classof`:

  **TPWidenInductionRecipe:**
  ```cpp
  TPWidenInductionRecipe(PHINode *IV, TPValue *StartVal, TPValue *StepVal,
                          unsigned DimIdx = 0)
      : TPSingleDefRecipe(TPWidenInductionSC, {StartVal, StepVal}),
        IVPhi(IV), DimIndex(DimIdx) {}
  static bool classof(const TPRecipeBase *R) {
    return R->getTPRecipeID() == TPWidenInductionSC;
  }
  ```

  **TPReductionPHIRecipe:**
  ```cpp
  TPReductionPHIRecipe(PHINode *Phi, TPValue *InitVal, TPValue *LoopVal)
      : TPSingleDefRecipe(TPReductionPHISC, {InitVal, LoopVal}), RedPhi(Phi) {}
  static bool classof(const TPRecipeBase *R) {
    return R->getTPRecipeID() == TPReductionPHISC;
  }
  ```

  **TPWidenRecipe:**
  ```cpp
  TPWidenRecipe(Instruction *I, SmallVectorImpl<TPValue *> &Ops)
      : TPSingleDefRecipe(TPWidenSC, Ops), Inst(I) {}
  static bool classof(const TPRecipeBase *R) {
    return R->getTPRecipeID() == TPWidenSC;
  }
  ```

  **TPWidenGEPRecipe:**
  ```cpp
  TPWidenGEPRecipe(Instruction *GEP, SmallVectorImpl<TPValue *> &Ops)
      : TPSingleDefRecipe(TPWidenGEPSC, Ops), GEPInst(GEP) {}
  static bool classof(const TPRecipeBase *R) {
    return R->getTPRecipeID() == TPWidenGEPSC;
  }
  ```

  **TPWidenLoadRecipe:**
  ```cpp
  TPWidenLoadRecipe(Instruction *Load, TPValue *PtrOp)
      : TPSingleDefRecipe(TPWidenLoadSC, {PtrOp}), LoadInst(Load) {}
  static bool classof(const TPRecipeBase *R) {
    return R->getTPRecipeID() == TPWidenLoadSC;
  }
  ```

  **TPWidenStoreRecipe** (non-def — base is `TPRecipeBase`):
  ```cpp
  TPWidenStoreRecipe(Instruction *Store, TPValue *PtrOp, TPValue *ValOp)
      : TPRecipeBase(TPWidenStoreSC, {PtrOp, ValOp}), StoreInst(Store) {}
  static bool classof(const TPRecipeBase *R) {
    return R->getTPRecipeID() == TPWidenStoreSC;
  }
  ```

  **TPWidenCastRecipe:**
  ```cpp
  TPWidenCastRecipe(Instruction *Cast, TPValue *SrcOp)
      : TPSingleDefRecipe(TPWidenCastSC, {SrcOp}), CastInst(Cast) {}
  static bool classof(const TPRecipeBase *R) {
    return R->getTPRecipeID() == TPWidenCastSC;
  }
  ```

  **TPCanonicalIVRecipe:**
  ```cpp
  TPCanonicalIVRecipe(TPValue *StartVal, TPValue *StepVal)
      : TPSingleDefRecipe(TPCanonicalIVSC, {StartVal, StepVal}) {}
  static bool classof(const TPRecipeBase *R) {
    return R->getTPRecipeID() == TPCanonicalIVSC;
  }
  ```

  **TPCanonicalIVIncrRecipe:**
  ```cpp
  TPCanonicalIVIncrRecipe(TPValue *IVVal, TPValue *PFVal)
      : TPSingleDefRecipe(TPCanonicalIVIncrSC, {IVVal, PFVal}) {}
  static bool classof(const TPRecipeBase *R) {
    return R->getTPRecipeID() == TPCanonicalIVIncrSC;
  }
  ```

  **TPCanonicalIVExitCmpRecipe:**
  ```cpp
  TPCanonicalIVExitCmpRecipe(TPValue *IncrVal, TPValue *BoundVal)
      : TPSingleDefRecipe(TPCanonicalIVExitCmpSC, {IncrVal, BoundVal}) {}
  static bool classof(const TPRecipeBase *R) {
    return R->getTPRecipeID() == TPCanonicalIVExitCmpSC;
  }
  ```

- [ ] **Step 8: Update `TPTransformState` in `TPlan.h`**

  Change ValueMap key and accessors:
  ```cpp
  DenseMap<const TPRecipeValue *, Value *> ValueMap;

  Value *getValue(const TPRecipeValue *V) const { return ValueMap.lookup(V); }
  void   setValue(const TPRecipeValue *V, Value *IRV) { ValueMap[V] = IRV; }
  ```
  Existing call sites using `TPSingleDefRecipe*` compile unchanged (IS-A relationship).

- [ ] **Step 9: Update `TPlan.cpp` — `buildInitial()` and `RecipeKind` references**

  - Replace all `RecipeKind::*` enum references with the new SC constants.
  - Remove any manual `addOperand` calls whose operands are now passed via the
    constructor list (the constructors call `addOperand` internally via `TPUser`).
  - Keep the `setOperand` call that patches the canonical IV step operand — it now
    also calls `removeUser`/`addUser` automatically via the updated `setOperand`.
  - `getOrCreateLiveIn` return type: `TPIRValue *`.

- [ ] **Step 10: Update `TPlanWidener.cpp` and `TPRecipeMatcher.cpp`**

  Replace `getKind() == RecipeKind::WidenInduction` style checks with
  `isa<TPWidenInductionRecipe>(&R)` or `R->getTPRecipeID() == TPWidenInductionSC`.

- [ ] **Step 11: Update `TPlanLowering.cpp`**

  Replace all `getKind()` / `RecipeKind::` references with `getTPRecipeID()` /
  `TPRecipeBase::TP*SC` constants. Any `dyn_cast<TPSingleDefRecipe>` calls
  continue to work (classof updated in Step 5).

- [ ] **Step 12: Add `TPRecipeBase` insertion-helper implementations to `TPlan.cpp`**

  ```cpp
  void TPRecipeBase::insertBefore(TPRecipeBase *InsertPos) {
    TPBasicBlock *BB = InsertPos->getParent();
    assert(BB && "InsertPos has no parent");
    BB->getRecipeList().insert(BB->getRecipeList().getIterator(*InsertPos), this);
    Parent = BB;
  }

  void TPRecipeBase::insertAfter(TPRecipeBase *InsertPos) {
    TPBasicBlock *BB = InsertPos->getParent();
    assert(BB && "InsertPos has no parent");
    auto It = BB->getRecipeList().getIterator(*InsertPos);
    ++It;
    BB->getRecipeList().insert(It, this);
    Parent = BB;
  }

  void TPRecipeBase::removeFromParent() {
    assert(Parent && "Recipe has no parent");
    Parent->getRecipeList().remove(this);
    Parent = nullptr;
  }

  iplist<TPRecipeBase>::iterator TPRecipeBase::eraseFromParent() {
    assert(Parent && "Recipe has no parent");
    return Parent->getRecipeList().erase(this);
  }
  ```

- [ ] **Step 13: Build**

  ```bash
  ninja -C build LLVMVectorize 2>&1 | tail -30
  ```

- [ ] **Step 14: Run lit test**

  ```bash
  build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/
  ```

- [ ] **Step 15: Commit**

  ```bash
  git add llvm/include/llvm/Transforms/Vectorize/TPlan.h \
          llvm/lib/Transforms/Vectorize/TPlan.cpp \
          llvm/lib/Transforms/Vectorize/TPlanWidener.cpp \
          llvm/lib/Transforms/Vectorize/TPRecipeMatcher.cpp \
          llvm/lib/Transforms/Vectorize/TPlanLowering.cpp
  git commit -m "tplan: add TPDef/TPRecipeValue/TPRecipeTy enum; refactor TPUser/TPRecipeBase/TPSingleDefRecipe; update all recipe subclasses"
  ```

---

## Task 3: Add TPPhiAccessors mixin; apply to three header-PHI recipes

**Spec reference:** §3.10

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlan.h`
- Modify: `llvm/lib/Transforms/Vectorize/TPlan.cpp`

Note: Block-related accessors (`getIncomingBlock`, `incoming_blocks`,
`incoming_values_and_blocks`, block-based `removeIncomingValueFor`) are
deferred — no current TPlan consumer uses them.

- [ ] **Step 1: Add `TPPhiAccessors` forward declaration to `TPlan.h` (near top forward decls)**

  The `friend class TPPhiAccessors` in `TPUser` (added in Task 2) needs the
  class to exist as a complete type at the use site. Add the declaration
  after `TPUser` (before `TPRecipeBase`) or as a forward declaration if needed.

- [ ] **Step 2: Add `TPPhiAccessors` class in `TPlan.h` (after `TPSingleDefRecipe`, before concrete recipe classes)**

  ```cpp
  //===----------------------------------------------------------------------===//
  // TPPhiAccessors — mixin for PHI-like recipes (mirrors VPPhiAccessors)
  //===----------------------------------------------------------------------===//
  class TPPhiAccessors {
  protected:
    virtual const TPRecipeBase *getAsRecipe() const = 0;
  public:
    virtual ~TPPhiAccessors() = default;

    unsigned getNumIncoming() const {
      return getAsRecipe()->getNumOperands();
    }

    TPValue *getIncomingValue(unsigned Idx) const {
      return getAsRecipe()->getOperand(Idx);
    }

    TPUser::const_operand_range incoming_values() const {
      return getAsRecipe()->operands_range();
    }

    // Remove operand at index Idx (calls TPUser::removeOperand via friend).
    // Block-based overload deferred.
    void removeIncomingValueFor(unsigned Idx) {
      const_cast<TPRecipeBase *>(getAsRecipe())->removeOperand(Idx);
    }

    void printPhiOperands(raw_ostream &O, TPSlotTracker &SlotTracker) const;
  };
  ```

- [ ] **Step 3: Apply `TPPhiAccessors` to `TPCanonicalIVRecipe`**

  Change declaration to:
  ```cpp
  class TPCanonicalIVRecipe : public TPSingleDefRecipe, public TPPhiAccessors {
  protected:
    const TPRecipeBase *getAsRecipe() const override { return this; }
  public:
    ...
  ```

- [ ] **Step 4: Apply `TPPhiAccessors` to `TPWidenInductionRecipe`**

  ```cpp
  class TPWidenInductionRecipe : public TPSingleDefRecipe, public TPPhiAccessors {
  protected:
    const TPRecipeBase *getAsRecipe() const override { return this; }
  public:
    ...
  ```

- [ ] **Step 5: Apply `TPPhiAccessors` to `TPReductionPHIRecipe`**

  ```cpp
  class TPReductionPHIRecipe : public TPSingleDefRecipe, public TPPhiAccessors {
  protected:
    const TPRecipeBase *getAsRecipe() const override { return this; }
  public:
    ...
  ```

- [ ] **Step 6: Implement `TPPhiAccessors::printPhiOperands` in `TPlan.cpp`**

  ```cpp
  void TPPhiAccessors::printPhiOperands(raw_ostream &O,
                                         TPSlotTracker &SlotTracker) const {
    for (unsigned I = 0, E = getNumIncoming(); I != E; ++I) {
      if (I > 0) O << ", ";
      getIncomingValue(I)->printAsOperand(O, SlotTracker);
    }
  }
  ```

- [ ] **Step 7: Build**

  ```bash
  ninja -C build LLVMVectorize 2>&1 | tail -30
  ```

- [ ] **Step 8: Run lit test**

  ```bash
  build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/
  ```

- [ ] **Step 9: Commit**

  ```bash
  git add llvm/include/llvm/Transforms/Vectorize/TPlan.h \
          llvm/lib/Transforms/Vectorize/TPlan.cpp
  git commit -m "tplan: add TPPhiAccessors mixin; apply to canonical-IV, widen-induction, reduction-PHI recipes"
  ```

---

## Task 4: Add TPIRFlags, TPIRMetadata, TPRecipeWithIRFlags; rebase widen recipes

**Spec reference:** §3.11, §3.12, §3.13, §4

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlan.h`
- Modify: `llvm/lib/Transforms/Vectorize/TPlan.cpp`
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp`

Add to `TPlan.h` includes:
```cpp
#include "llvm/IR/FMF.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Metadata.h"
```

- [ ] **Step 1: Add `TPIRFlags` class in `TPlan.h` (after `TPPhiAccessors`)**

  Mirror VPIRFlags. The union approach avoids per-recipe flag-field bloat:

  ```cpp
  class TPIRFlags {
  public:
    enum class OperationType : unsigned char {
      Cmp, FCmp, OverflowingBinOp, Trunc,
      DisjointOp, PossiblyExactOp, GEPOp, FPMathOp, NonNegOp, Other,
    };

  private:
    OperationType OpType = OperationType::Other;
    union {
      unsigned char AllFlags = 0;
      struct { bool HasNUW : 1; bool HasNSW : 1; } OvflowFlags;
      struct { bool IsExact : 1; } ExactFlags;
      struct { bool IsDisjoint : 1; } DisjointFlags;
      struct { bool IsNonNeg : 1; } NonNegFlags;
    };
    FastMathFlags FMF;
    CmpInst::Predicate CmpPred = CmpInst::BAD_ICMP_PREDICATE;

  public:
    TPIRFlags() = default;
    explicit TPIRFlags(Instruction &I);
    void applyFlags(Instruction &I) const;
    void dropPoisonGeneratingFlags() {
      // Clear poison-generating flags per operation type.
      // NUW/NSW/exact/disjoint/nonNeg bits are in the union; FMF requires
      // explicit per-flag clearing for nnan/ninf.
      switch (OpType) {
      case OperationType::OverflowingBinOp:
        OvflowFlags.HasNUW = OvflowFlags.HasNSW = false; break;
      case OperationType::PossiblyExactOp:
        ExactFlags.IsExact = false; break;
      case OperationType::DisjointOp:
        DisjointFlags.IsDisjoint = false; break;
      case OperationType::NonNegOp:
        NonNegFlags.IsNonNeg = false; break;
      case OperationType::FPMathOp:
      case OperationType::FCmp:
        FMF.setNoNaNs(false); FMF.setNoInfs(false); break;
      default: break;
      }
    }

    OperationType getOperationType()       const { return OpType; }
    bool hasNoUnsignedWrap()               const { return OvflowFlags.HasNUW; }
    bool hasNoSignedWrap()                 const { return OvflowFlags.HasNSW; }
    bool isDisjoint()                      const { return DisjointFlags.IsDisjoint; }
    bool isNonNeg()                        const { return NonNegFlags.IsNonNeg; }
    bool isExact()                         const { return ExactFlags.IsExact; }
    FastMathFlags   getFastMathFlags()     const { return FMF; }
    CmpInst::Predicate getPredicate()      const { return CmpPred; }
  };
  ```

- [ ] **Step 2: Add `TPIRMetadata` class in `TPlan.h` (after `TPIRFlags`)**

  ```cpp
  class TPIRMetadata {
    SmallVector<std::pair<unsigned, MDNode *>, 4> Metadata;
  public:
    TPIRMetadata() = default;
    explicit TPIRMetadata(Instruction &I);
    void applyMetadata(Instruction &I) const;
    void setMetadata(unsigned Kind, MDNode *Node);
    MDNode *getMetadata(unsigned Kind) const;
    void intersect(const TPIRMetadata &Other);
  };
  ```

- [ ] **Step 3: Add `TPRecipeWithIRFlags` class in `TPlan.h` (after `TPIRMetadata`)**

  Constructor takes `Instruction &FlagSrc` and extracts flags inline — more
  ergonomic than a pre-built `TPIRFlags` object at all call sites.

  ```cpp
  struct TPRecipeWithIRFlags : public TPSingleDefRecipe, public TPIRFlags {
    TPRecipeWithIRFlags(unsigned char SC, ArrayRef<TPValue *> Ops,
                        Instruction &FlagSrc, Value *UV = nullptr)
        : TPSingleDefRecipe(SC, Ops, UV), TPIRFlags(FlagSrc) {}

    static bool classof(const TPRecipeBase *R) {
      switch (R->getTPRecipeID()) {
      case TPWidenSC:
      case TPWidenGEPSC:
      case TPWidenCastSC:
        return true;
      default:
        return false;
      }
    }
  };
  ```

- [ ] **Step 4: Rebase `TPWidenRecipe` onto `TPRecipeWithIRFlags`**

  ```cpp
  class TPWidenRecipe : public TPRecipeWithIRFlags {
  public:
    TPWidenRecipe(Instruction *I, SmallVectorImpl<TPValue *> &Ops)
        : TPRecipeWithIRFlags(TPWidenSC, Ops, *I), Inst(I) {}
    ...
  ```

- [ ] **Step 5: Rebase `TPWidenGEPRecipe` onto `TPRecipeWithIRFlags`**

  ```cpp
  class TPWidenGEPRecipe : public TPRecipeWithIRFlags {
  public:
    TPWidenGEPRecipe(Instruction *GEP, SmallVectorImpl<TPValue *> &Ops)
        : TPRecipeWithIRFlags(TPWidenGEPSC, Ops, *GEP), GEPInst(GEP) {}
    ...
  ```

- [ ] **Step 6: Rebase `TPWidenCastRecipe` onto `TPRecipeWithIRFlags`**

  ```cpp
  class TPWidenCastRecipe : public TPRecipeWithIRFlags {
  public:
    TPWidenCastRecipe(Instruction *Cast, TPValue *SrcOp)
        : TPRecipeWithIRFlags(TPWidenCastSC, {SrcOp}, *Cast), CastInst(Cast) {}
    ...
  ```

- [ ] **Step 7: Implement `TPIRFlags::TPIRFlags(Instruction &I)` in `TPlan.cpp`**

  ```cpp
  TPIRFlags::TPIRFlags(Instruction &I) {
    if (auto *OBO = dyn_cast<OverflowingBinaryOperator>(&I)) {
      OpType = OperationType::OverflowingBinOp;
      OvflowFlags.HasNUW = OBO->hasNoUnsignedWrap();
      OvflowFlags.HasNSW = OBO->hasNoSignedWrap();
    } else if (auto *TI = dyn_cast<TruncInst>(&I)) {
      OpType = OperationType::Trunc;
      // TruncInst has no NUW/NSW in LLVM IR (flags not exposed yet).
    } else if (auto *CI = dyn_cast<ICmpInst>(&I)) {
      OpType = OperationType::Cmp;
      CmpPred = CI->getPredicate();
    } else if (auto *FCI = dyn_cast<FCmpInst>(&I)) {
      OpType = OperationType::FCmp;
      CmpPred = FCI->getPredicate();
      FMF = FCI->getFastMathFlags();
    } else if (auto *FPO = dyn_cast<FPMathOperator>(&I)) {
      OpType = OperationType::FPMathOp;
      FMF = FPO->getFastMathFlags();
    } else if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
      OpType = OperationType::GEPOp;
    } else if (auto *PEO = dyn_cast<PossiblyExactOperator>(&I)) {
      OpType = OperationType::PossiblyExactOp;
      ExactFlags.IsExact = PEO->isExact();
    } else if (auto *PDO = dyn_cast<PossiblyDisjointInst>(&I)) {
      OpType = OperationType::DisjointOp;
      DisjointFlags.IsDisjoint = PDO->isDisjoint();
    }
  }
  ```

- [ ] **Step 8: Implement `TPIRFlags::applyFlags(Instruction &I) const` in `TPlan.cpp`**

  ```cpp
  void TPIRFlags::applyFlags(Instruction &I) const {
    switch (OpType) {
    case OperationType::OverflowingBinOp:
      // Use Instruction's public setters — OverflowingBinaryOperator's setters
      // are private (friend Instruction only).
      I.setHasNoUnsignedWrap(OvflowFlags.HasNUW);
      I.setHasNoSignedWrap(OvflowFlags.HasNSW);
      break;
    case OperationType::FCmp:
    case OperationType::FPMathOp:
      // Use Instruction::setFastMathFlags (replaces flags) not
      // FPMathOperator::setFastMathFlags (which ORs, leaving stale bits).
      I.setFastMathFlags(FMF);
      break;
    case OperationType::PossiblyExactOp:
      // Instruction::setIsExact is public.
      I.setIsExact(ExactFlags.IsExact);
      break;
    case OperationType::DisjointOp:
      cast<PossiblyDisjointInst>(&I)->setIsDisjoint(DisjointFlags.IsDisjoint);
      break;
    default:
      break;
    }
  }
  ```

- [ ] **Step 9: Implement `TPIRMetadata` methods in `TPlan.cpp`**

  ```cpp
  TPIRMetadata::TPIRMetadata(Instruction &I) {
    SmallVector<std::pair<unsigned, MDNode *>, 8> All;
    I.getAllMetadataOtherThanDebugLoc(All);
    for (auto &[Kind, Node] : All) {
      // Keep only propagatable kinds (same filter as VPIRMetadata).
      if (Kind == LLVMContext::MD_tbaa ||
          Kind == LLVMContext::MD_fpmath ||
          Kind == LLVMContext::MD_access_group)
        Metadata.push_back({Kind, Node});
    }
  }

  void TPIRMetadata::applyMetadata(Instruction &I) const {
    for (auto &[Kind, Node] : Metadata)
      I.setMetadata(Kind, Node);
  }

  void TPIRMetadata::setMetadata(unsigned Kind, MDNode *Node) {
    for (auto &P : Metadata) { if (P.first == Kind) { P.second = Node; return; } }
    Metadata.push_back({Kind, Node});
  }

  MDNode *TPIRMetadata::getMetadata(unsigned Kind) const {
    for (auto &P : Metadata) if (P.first == Kind) return P.second;
    return nullptr;
  }

  void TPIRMetadata::intersect(const TPIRMetadata &Other) {
    Metadata.erase(llvm::remove_if(Metadata, [&](auto &P) {
      return Other.getMetadata(P.first) == nullptr;
    }), Metadata.end());
  }
  ```

- [ ] **Step 10: Update `execute()` in `TPlanLowering.cpp` to call `applyFlags()`**

  In `TPWidenRecipe::execute()`, `TPWidenGEPRecipe::execute()`, and
  `TPWidenCastRecipe::execute()`, after cloning the instruction:
  ```cpp
  Value *Result = State.Builder.Insert(Inst->clone());
  applyFlags(*cast<Instruction>(Result));   // ← add this line
  State.setValue(this, Result);
  ```

- [ ] **Step 11: Build**

  ```bash
  ninja -C build LLVMVectorize 2>&1 | tail -30
  ```

- [ ] **Step 12: Run lit test**

  ```bash
  build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/
  ```

- [ ] **Step 13: Commit**

  ```bash
  git add llvm/include/llvm/Transforms/Vectorize/TPlan.h \
          llvm/lib/Transforms/Vectorize/TPlan.cpp \
          llvm/lib/Transforms/Vectorize/TPlanLowering.cpp
  git commit -m "tplan: add TPIRFlags, TPIRMetadata, TPRecipeWithIRFlags; rebase widen/GEP/cast recipes"
  ```

---

## Task 5: Commit spec + plan docs, then push

- [ ] **Step 1: Commit the spec and plan documents**

  ```bash
  git add docs/superpowers/specs/2026-03-25-tplan-value-layer-refactor-design.md \
          docs/superpowers/plans/2026-03-25-tplan-value-layer-refactor.md
  git commit -m "docs: add TPlan value-layer refactor spec (rev 3) and implementation plan"
  ```

- [ ] **Step 2: Push all commits to remote**

  ```bash
  git push origin LoopTensorizebyClaude
  ```
  Expected: all 5 commits appear on the remote branch.
