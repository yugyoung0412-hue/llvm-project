# TPlan Value-Layer Refactor: Mirror VPlan's TPValue/TPDef/TPUser Design

**Date:** 2026-03-25 (rev 3)
**Branch:** LoopTensorizebyClaude
**Goal:** Refactor TPlan's value/def/use infrastructure to mirror VPlan's
`VPValue`/`VPDef`/`VPUser`/`VPRecipeValue` design pattern, replacing the conflated
`TPSingleDefRecipe : TPRecipeBase, TPValue` dual-inheritance with the clean
separation `TPSingleDefRecipe : TPRecipeBase, TPRecipeValue`. Also add
`TPConstantInt`, `TPPhiAccessors`, `TPIRFlags`, `TPIRMetadata`, and
`TPRecipeWithIRFlags`, and introduce a full `TPRecipeTy` SubclassID enum with
PHI-range markers mirroring VPlan's `VPRecipeTy`.

**Precondition:** Block hierarchy refactor
(`2026-03-24-tplan-vplan-refactor-design.md`) is complete.
`TPBlockBase`/`TPBasicBlock`/`TPRegionBlock`/`TPIRBasicBlock`/`TPBlockUtils`
are already implemented.

**Invariant:** The nested-loop TPlan print format (as documented in
`do_conv2d_stage.txt` and `ggml_compute_forward_mul_mat_stage.txt`) must not
change.

---

## 1. Motivation

Currently `TPSingleDefRecipe` inherits directly from `TPValue`, making each
recipe simultaneously a value. This collapses VPlan's clean three-way
separation:

- `TPRecipeBase` does not inherit `TPDef` — no standard "what values does this
  recipe define?" API.
- `TPSingleDefRecipe : TPValue` prevents future multi-def recipes.
- `TPRecipeBase` uses a loose `RecipeKind` enum field rather than a
  `const unsigned char SubclassID` — no isa/dyn\_cast efficiency.
- VPlan's PHI-recipe range markers (`VPFirstPHISC`, `VPFirstHeaderPHISC`,
  `VPLastHeaderPHISC`, `VPLastPHISC`) have no TPlan equivalent.

---

## 2. Class Mapping

| VPlan class | TPlan equivalent | Status |
|---|---|---|
| `VPValue` | `TPValue` | Refactor |
| `VPIRValue` | `TPIRValue` | Rename from `TPLiveIn` |
| `VPConstantInt` | `TPConstantInt` | **New** |
| `VPSymbolicValue` | `TPSymbolicValue` | Rename from `TPSyntheticValue` |
| `VPRecipeValue` | `TPRecipeValue` | **New** |
| `VPUser` | `TPUser` | Refactor |
| `VPDef` | `TPDef` | **New** |
| `VPSlotTracker` | `TPSlotTracker` | Keep number-based (simpler) |
| `VPPhiAccessors` | `TPPhiAccessors` | **New** |
| `VPIRFlags` | `TPIRFlags` | **New** |
| `VPIRMetadata` | `TPIRMetadata` | **New** |
| `VPRecipeBase` | `TPRecipeBase` | Refactor: `TPDef`+`TPUser`; `TPRecipeTy` enum |
| `VPSingleDefRecipe` | `TPSingleDefRecipe` | Refactor: second base `TPValue`→`TPRecipeValue` |
| `VPRecipeWithIRFlags` | `TPRecipeWithIRFlags` | **New** |

**Not mirrored (VPlan-specific):**
`VPLane` (lane indexing), `VPCostContext` (cost model out of scope).
`VPTransformState` exists as `TPTransformState` — updated in place.

---

## 3. Detailed Design

### 3.1 TPValue

Mirrors `VPValue`. Two separate `SubclassID` fields exist in the system —
one on `TPValue` (value layer) and one on `TPRecipeBase` (recipe layer) —
exactly as in VPlan. They are independent enums in independent class
hierarchies and never conflict.

Changes from current `TPValue`:

- Replace `ValueKind Kind` with `const unsigned char SubclassID` (same three
  values, new names for consistency):

  ```
  TPVIRValueSC      // wraps IR Value* (was LiveIn)
  TPVSymbolicSC     // named synthetic, no IR backing (was Synthetic)
  TPVRecipeValueSC  // value defined by a recipe (was Def)
  ```

- Replace per-subclass `IRVal` field with `Value *UnderlyingVal = nullptr`
  stored on `TPValue` (matches VPlan; `TPIRValue` reads it via
  `getUnderlyingValue()`).
- User list: `SmallVector<TPUser*, 1>` (VPlan uses 1 as default).
- Add iterator typedefs: `user_iterator`, `const_user_iterator`,
  `user_range`, `const_user_range`.
- Add `hasMoreThanOneUniqueUser()`, `hasOneUse()`, `getSingleUser()`.
- Add `replaceAllUsesWith(TPValue *New)`.
- Add `replaceUsesWithIf(TPValue*, function_ref<bool(TPUser&, unsigned)>)`.
- Add `getDefiningRecipe() → TPRecipeBase*` (nullptr for live-ins/synthetics).
- Add `isDefinedOutsideLoopRegions() const`.
- `printAsOperand(raw_ostream&, TPSlotTracker&)` remains pure virtual;
  each concrete leaf (`TPIRValue`, `TPSymbolicValue`, `TPSingleDefRecipe`)
  provides its own override — the exact same three code paths as today.
  `TPIRValue` retains the `ir<...>` format; `TPSymbolicValue` retains
  `tp<%N>`; `TPSingleDefRecipe` retains `ir<%name>` or `tp<%N>`.

### 3.2 TPIRValue (rename from TPLiveIn)

Mirrors `VPIRValue`. Reads its `IR Value*` from `TPValue::UnderlyingVal`:

```cpp
struct TPIRValue : public TPValue {
  TPIRValue(Value *UV) : TPValue(TPVIRValueSC, UV) {
    assert(UV && "TPIRValue requires an underlying IR value");
  }
  Value *getValue() const { return getUnderlyingValue(); }
  Type  *getType()  const;          // new: returns UV->getType()
  static bool classof(const TPValue *V) {
    return V->getTPValueID() == TPVIRValueSC;
  }
};
```

`TPlan::LiveIns` type changes from
`SmallVector<unique_ptr<TPLiveIn>>` to `SmallVector<unique_ptr<TPIRValue>>`.
`TPSlotTracker::preAssignSynthetic` signature changes from
`const TPSyntheticValue*` to `const TPSymbolicValue*`.

### 3.3 TPConstantInt (new)

Mirrors `VPConstantInt`. Overlay on `TPIRValue` for `ConstantInt` values:

```cpp
struct TPConstantInt : public TPIRValue {
  TPConstantInt(ConstantInt *CI) : TPIRValue(CI) {}
  bool isOne()  const;
  bool isZero() const;
  const APInt &getAPInt() const;
  unsigned getBitWidth() const;
  uint64_t getZExtValue() const;
  // Accept TPIRValue* as the immediate base — required for isa<TPConstantInt>
  // to work when the source type is TPIRValue*.
  static bool classof(const TPIRValue *V) {
    return isa<ConstantInt>(V->getUnderlyingValue());
  }
  // Also accept TPValue* for isa<TPConstantInt>(tpvalue_ptr).
  static bool classof(const TPValue *V) {
    if (auto *IR = dyn_cast<TPIRValue>(V))
      return classof(IR);
    return false;
  }
};
```

Used for DimPF values, loop bounds, and step values that are known constants.

### 3.4 TPSymbolicValue (rename from TPSyntheticValue)

Mirrors `VPSymbolicValue` with the addition of a `Name` field (VPlan omits
it; TPlan needs it for `PF[0]`, `PF[1]`, … print output):

```cpp
struct TPSymbolicValue : public TPValue {
  explicit TPSymbolicValue(StringRef Name)
      : TPValue(TPVSymbolicSC), Name(Name) {}
  StringRef getName() const { return Name; }
  static bool classof(const TPValue *V) {
    return V->getTPValueID() == TPVSymbolicSC;
  }
private:
  std::string Name;
};
```

`TPlan::DimPFs` type changes from
`SmallVector<unique_ptr<TPSyntheticValue>>` to
`SmallVector<unique_ptr<TPSymbolicValue>>`.

### 3.5 TPRecipeValue (new)

Mirrors `VPRecipeValue`. A `TPValue` produced by a recipe:

```cpp
class TPRecipeValue : public TPValue {
  friend class TPValue;
  friend class TPDef;
  TPDef *Def;
public:
  // Constructor called from TPSingleDefRecipe's ctor with `this`.
  // Registers itself into Def->DefinedValues via TPDef::addDefinedValue.
  TPRecipeValue(TPDef *Def, Value *UV = nullptr);
  virtual ~TPRecipeValue();  // calls TPDef::removeDefinedValue; does NOT delete self
  static bool classof(const TPValue *V) {
    return V->getTPValueID() == TPVRecipeValueSC;
  }
};
```

**Memory ownership (critical):** In `TPSingleDefRecipe`, `TPRecipeValue` is a
*base-class sub-object*, not a separately heap-allocated object.

`TPDef::~TPDef` **must NOT call `delete` on any element of `DefinedValues`**
in the single-def recipe case. Calling `delete` on a `TPRecipeValue*`
that is a non-first base sub-object (i.e., the address differs from the
allocation address of the most-derived `TPSingleDefRecipe`) is undefined
behaviour — `operator delete` would receive the wrong pointer value.

The correct behaviour, following VPlan exactly: `TPDef::~TPDef` iterates
`DefinedValues` and calls `delete D` **only** for `TPRecipeValue` objects
that were separately heap-allocated (e.g., multi-def recipes where extra
values are `new`-ed independently). For the `TPSingleDefRecipe` case all
`TPRecipeValue` entries in `DefinedValues` are sub-objects of the recipe;
their destruction is handled automatically when the most-derived recipe
object is destroyed. The destructor body for the single-def case is
therefore a no-op (or an empty loop that skips sub-objects).

Implementation strategy: `TPDef` can track whether each `TPRecipeValue` is
a standalone allocation via a parallel `bool` vector, or (simpler, matching
VPlan's current approach) rely on the invariant that `DefinedValues.size()`
is always 1 for `TPSingleDefRecipe` and never call `delete` there. A
`DEBUG`-only assertion `assert(DefinedValues.empty())` in `TPDef::~TPDef` is
sufficient for the current single-def-only codebase.

### 3.6 TPUser

Mirrors `VPUser`. Key changes from current `TPUser`:

- **Default constructor deleted** (`TPUser() = delete`), mirroring VPlan.
  The following constructors are provided:

  ```cpp
  protected:
    // Full-list constructor — recipe passes all operands at construction.
    explicit TPUser(ArrayRef<TPValue *> Operands);
    // Incremental constructor — recipe adds operands via addOperand().
    explicit TPUser(std::initializer_list<TPValue *> Operands);
  ```

  All existing recipe constructors are audited and migrated as part of
  Commit 1: each recipe's constructor must call one of these forms. A
  concrete example for `TPWidenRecipe`:
  ```cpp
  // Before (invalid after TPUser() = delete):
  TPWidenRecipe(Instruction *I) : TPSingleDefRecipe(TPWidenSC, {}), Inst(I) {}
  // After:
  TPWidenRecipe(Instruction *I)
      : TPSingleDefRecipe(TPWidenSC, {}), Inst(I) {}
  // TPSingleDefRecipe calls TPRecipeBase(SC, {}),
  // which calls TPUser({}) — the empty-list ArrayRef form.
  ```
- `setOperand(I, New)` now calls `Operands[I]->removeUser(*this)` then
  `New->addUser(*this)` before assigning — matches VPlan. **Call-site audit**
  (Commit 1): all existing `setOperand` usages (canonical IV step patching in
  `buildInitial()`) must be verified to pass live, non-freed values.
- `swapOperands()` added (asserts `size() == 2`).
- `replaceUsesOfWith(TPValue *From, TPValue *To)` added.
- Destructor: calls `Op->removeUser(*this)` for all operands.
- Iterator typedefs: `op_begin/op_end`, `operand_iterator`,
  `const_operand_iterator`, `operand_range`, `const_operand_range`.
- `usesScalars()`, `usesFirstLaneOnly()`, `usesFirstPartOnly()` virtual
  stubs returning `false` (mirrors VPlan; not used in TPlan today but
  present for API completeness).

### 3.7 TPDef (new)

Mirrors `VPDef`. Owns the `TPRecipeValue` instances a recipe defines:

```cpp
class TPDef {
  friend class TPRecipeValue;
  TinyPtrVector<TPRecipeValue *> DefinedValues;

  // Called from TPRecipeValue constructor — private.
  void addDefinedValue(TPRecipeValue *V);
  // Called from TPRecipeValue destructor — private.
  void removeDefinedValue(TPRecipeValue *V);

public:
  TPDef() = default;
  virtual ~TPDef();
  // See §3.5 memory note: destructor calls delete only on non-sub-object values.

  TPValue       *getTPSingleValue();        // asserts size == 1
  const TPValue *getTPSingleValue() const;
  TPValue       *getTPValue(unsigned I);
  const TPValue *getTPValue(unsigned I) const;
  ArrayRef<TPRecipeValue *>       definedValues();
  ArrayRef<const TPRecipeValue *> definedValues() const;
  unsigned getNumDefinedValues() const;
};
```

`TinyPtrVector` stores the pointer inline for the common single-def case —
zero heap allocation.

### 3.8 TPRecipeBase — TPRecipeTy enum

`TPRecipeBase` gains a `const unsigned char SubclassID` field (independent
of `TPValue::SubclassID`) and a full `TPRecipeTy` enum mirroring VPlan's
`VPRecipeTy`. The old `RecipeKind` enum and `getKind()` are **replaced** by
`TPRecipeTy` and `getTPRecipeID()` throughout.

```cpp
class TPRecipeBase
    : public ilist_node_with_parent<TPRecipeBase, TPBasicBlock>,
      public TPDef,
      public TPUser {

  friend TPBasicBlock;
  friend TPBlockUtils;

  // SubclassID holds a TPRecipeTy value.
  const unsigned char SubclassID;
  // NOTE: No explicit `TPBasicBlock *Parent` field. `ilist_node_with_parent`
  // provides getParent() by walking the containing ilist — storing a separate
  // `Parent` pointer would be redundant, add per-recipe size overhead, and
  // risk divergence. The current code's explicit `Parent` field is removed.

public:
  // TPRecipeTy is a typedef for an anonymous enum declared inside the class.
  // Members are accessed unqualified inside TPRecipeBase, or as
  // `TPRecipeBase::TPWidenSC` (not `TPRecipeBase::TPRecipeTy::TPWidenSC`)
  // from external code — consistent with VPlan's top-level VPRecipeTy usage.
  using TPRecipeTy = enum {
    // Non-PHI recipes
    TPWidenSC,
    TPWidenGEPSC,
    TPWidenLoadSC,
    TPWidenStoreSC,
    TPWidenCastSC,

    // PHI-like recipes — kept together (mirrors VPlan's PHI range)
    TPWidenPHISC,              // generic widen-phi (future use)

    // Header PHI recipes — subset of PHI-like (mirrors VPlan's header-PHI range)
    TPCanonicalIVSC,           // was TPCanonicalIVRecipe
    TPWidenInductionSC,        // was TPWidenInductionRecipe
    TPReductionPHISC,          // was TPReductionPHIRecipe

    // Canonical IV companion recipes (not PHIs, follow latch)
    TPCanonicalIVIncrSC,       // was TPCanonicalIVIncrRecipe
    TPCanonicalIVExitCmpSC,    // was TPCanonicalIVExitCmpRecipe

    // Range markers (mirrors VPlan's VPFirstPHISC / VPLastPHISC pattern)
    TPFirstPHISC        = TPWidenPHISC,
    TPFirstHeaderPHISC  = TPCanonicalIVSC,
    TPLastHeaderPHISC   = TPReductionPHISC,
    TPLastPHISC         = TPReductionPHISC,
  };

  TPRecipeBase(unsigned char SC, ArrayRef<TPValue *> Operands)
      : TPDef(), TPUser(Operands), SubclassID(SC) {}

  unsigned getTPRecipeID() const { return SubclassID; }

  // getParent() is provided by ilist_node_with_parent — no stored field needed.
  // (ilist_node_with_parent<TPRecipeBase, TPBasicBlock>::getParent())

  // Insertion/removal (mirrors VPRecipeBase)
  void insertBefore(TPRecipeBase *InsertPos);
  void insertAfter(TPRecipeBase *InsertPos);
  void moveAfter(TPRecipeBase *MovePos);
  void removeFromParent();
  iplist<TPRecipeBase>::iterator eraseFromParent();

  // isPhi(): true if getTPRecipeID() in [TPFirstPHISC, TPLastPHISC]
  bool isPhi() const {
    return getTPRecipeID() >= TPFirstPHISC &&
           getTPRecipeID() <= TPLastPHISC;
  }

  // Convenience: getDefinedValue() → getTPSingleValue() cast
  TPSingleDefRecipe       *getDefinedValue();
  const TPSingleDefRecipe *getDefinedValue() const;

  virtual void print(raw_ostream &, unsigned Indent, TPSlotTracker &) const = 0;
  virtual void execute(TPTransformState &State) const = 0;
};
```

**PHI range semantics (mirrors VPlan exactly):**

| Marker | Value | Meaning |
|---|---|---|
| `TPFirstPHISC` | `TPWidenPHISC` | First PHI-like recipe subclass ID |
| `TPFirstHeaderPHISC` | `TPCanonicalIVSC` | First header-PHI recipe subclass ID |
| `TPLastHeaderPHISC` | `TPReductionPHISC` | Last header-PHI recipe subclass ID |
| `TPLastPHISC` | `TPReductionPHISC` | Last PHI-like recipe subclass ID |

All three header-PHI recipes (`TPCanonicalIVRecipe`, `TPWidenInductionRecipe`,
`TPReductionPHIRecipe`) also inherit `TPPhiAccessors` (§3.10).

**`classof` pattern:** Recipes use `getTPRecipeID()` exclusively going forward.
No `RecipeKind`/`getKind()` references survive.

### 3.9 TPSingleDefRecipe

Second base changes from `TPValue` to `TPRecipeValue`. Inheritance ordering
is fixed (see §3.5 memory note):

```cpp
class TPSingleDefRecipe : public TPRecipeBase,   // ← first
                          public TPRecipeValue {  // ← second (sub-object)
public:
  SmallBitVector DimSet;   // tensor-specific, unchanged

  // Passes `this` as TPDef* to TPRecipeValue constructor
  TPSingleDefRecipe(unsigned char SC, ArrayRef<TPValue *> Ops,
                    Value *UV = nullptr)
      : TPRecipeBase(SC, Ops), TPRecipeValue(this, UV) {}

  // printAsOperand override stays here (same output as today)
  void printAsOperand(raw_ostream &OS, TPSlotTracker &Tracker) const override;

  static bool classof(const TPRecipeBase *R) {
    // Enumerate all single-def subclass IDs explicitly (not a negative check).
    // TPWidenPHISC is intentionally excluded: no concrete TPSingleDefRecipe
    // subclass implements it yet (future use). Returning true for an ID with
    // no concrete class would allow isa<TPSingleDefRecipe> to succeed while
    // no safe downcast exists. Add it here once TPWidenPHIRecipe is implemented.
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
    case TPWidenPHISC:
      return false;
    }
    llvm_unreachable("unknown TPRecipeTy");
  }
  static bool classof(const TPValue *V) {
    return V->getTPValueID() == TPVRecipeValueSC;
  }
};
```

### 3.10 TPPhiAccessors (new)

Mirrors `VPPhiAccessors`. Mixin for PHI-like recipes; requires
`friend class TPPhiAccessors` in `TPUser` (to access `removeOperand`):

```cpp
class TPPhiAccessors {
protected:
  virtual const TPRecipeBase *getAsRecipe() const = 0;
public:
  virtual ~TPPhiAccessors() = default;

  TPValue *getIncomingValue(unsigned Idx) const;
  const TPBasicBlock *getIncomingBlock(unsigned Idx) const;
  TPValue *getIncomingValueForBlock(const TPBasicBlock *BB) const;

  virtual unsigned getNumIncoming() const {
    return getAsRecipe()->getNumOperands();
  }

  TPUser::const_operand_range incoming_values() const;

  // Range over const TPBasicBlock*
  using const_incoming_blocks_range = ...;
  const_incoming_blocks_range incoming_blocks() const;

  // zip range of (value, block) pairs — mirrors VPPhiAccessors exactly
  auto incoming_values_and_blocks() const;

  // Removes the incoming value for IncomingBlock (calls TPUser::removeOperand).
  // Non-const: mutates the operand list.
  void removeIncomingValueFor(TPBlockBase *IncomingBlock);

  void printPhiOperands(raw_ostream &O, TPSlotTracker &SlotTracker) const;
};
```

Applied to: `TPCanonicalIVRecipe`, `TPWidenInductionRecipe`,
`TPReductionPHIRecipe`.

`TPUser` gains `friend class TPPhiAccessors` to grant access to
`removeOperand`.

### 3.11 TPIRFlags (new)

Mirrors `VPIRFlags`. Captures IR instruction flags for flag-preserving
lowering. Includes `Trunc` (needed by `TPWidenCastRecipe` on `TruncInst`):

```cpp
class TPIRFlags {
  enum class OperationType : unsigned char {
    Cmp, FCmp, OverflowingBinOp, Trunc,   // ← Trunc included
    DisjointOp, PossiblyExactOp, GEPOp,
    FPMathOp, NonNegOp, Other
  };
  // WrapFlagsTy, TruncFlagsTy, DisjointFlagsTy, ExactFlagsTy,
  // FastMathFlagsTy, FCmpFlagsTy, NonNegFlagsTy — same as VPIRFlags
  // ReductionFlagsTy omitted (no reduction op classification in TPlan)
  union { ... };
public:
  TPIRFlags();
  TPIRFlags(Instruction &I);           // extracts flags from IR instruction
  void applyFlags(Instruction &I) const;
  void dropPoisonGeneratingFlags();
  // Accessors: getPredicate(), getFastMathFlags(), hasNoUnsignedWrap(),
  //            hasNoSignedWrap(), isDisjoint(), isNonNeg(), etc.
};
```

### 3.12 TPIRMetadata (new)

Mirrors `VPIRMetadata`. Captures `!tbaa`, `!llvm.access.group`, etc.:

```cpp
class TPIRMetadata {
  SmallVector<std::pair<unsigned, MDNode *>> Metadata;
public:
  TPIRMetadata() = default;
  TPIRMetadata(Instruction &I);        // extracts propagatable metadata
  void applyMetadata(Instruction &I) const;
  void setMetadata(unsigned Kind, MDNode *Node);
  MDNode *getMetadata(unsigned Kind) const;
  void intersect(const TPIRMetadata &Other);
};
```

### 3.13 TPRecipeWithIRFlags (new)

Mirrors `VPRecipeWithIRFlags`. Combines `TPSingleDefRecipe + TPIRFlags`:

```cpp
struct TPRecipeWithIRFlags : public TPSingleDefRecipe, public TPIRFlags {
  TPRecipeWithIRFlags(unsigned char SC, ArrayRef<TPValue *> Ops,
                      const TPIRFlags &Flags, Value *UV = nullptr)
      : TPSingleDefRecipe(SC, Ops, UV), TPIRFlags(Flags) {}

  static bool classof(const TPRecipeBase *R) {
    switch (R->getTPRecipeID()) {
    case TPWidenSC:
    case TPWidenGEPSC:
    case TPWidenCastSC:
      return true;
    default: return false;
    }
  }
};
```

`TPWidenRecipe`, `TPWidenGEPRecipe`, and `TPWidenCastRecipe` become
subclasses of `TPRecipeWithIRFlags`.

---

## 4. Migration of Existing Recipe Subclasses

| Recipe class | SubclassID | Change |
|---|---|---|
| `TPWidenInductionRecipe` | `TPWidenInductionSC` | Add `TPPhiAccessors` base |
| `TPReductionPHIRecipe` | `TPReductionPHISC` | Add `TPPhiAccessors` base |
| `TPCanonicalIVRecipe` | `TPCanonicalIVSC` | Add `TPPhiAccessors` base |
| `TPWidenRecipe` | `TPWidenSC` | Base: `TPSingleDefRecipe` → `TPRecipeWithIRFlags` |
| `TPWidenGEPRecipe` | `TPWidenGEPSC` | Base: `TPSingleDefRecipe` → `TPRecipeWithIRFlags` |
| `TPWidenCastRecipe` | `TPWidenCastSC` | Base: `TPSingleDefRecipe` → `TPRecipeWithIRFlags` (includes `Trunc` flags) |
| `TPWidenLoadRecipe` | `TPWidenLoadSC` | Stays `TPSingleDefRecipe`; ctor creates `TPIRMetadata` for `!tbaa` |
| `TPWidenStoreRecipe` | `TPWidenStoreSC` | Stays `TPRecipeBase` (non-def); ctor creates `TPIRMetadata` |
| `TPWidenPHIRecipe` *(new, future)* | `TPWidenPHISC` | Placeholder for generic PHI widening; no implementation in this refactor |
| `TPCanonicalIVIncrRecipe` | `TPCanonicalIVIncrSC` | Stays `TPSingleDefRecipe` |
| `TPCanonicalIVExitCmpRecipe` | `TPCanonicalIVExitCmpSC` | Stays `TPSingleDefRecipe` |

`getKind()` / `RecipeKind` references are eliminated; all code uses
`getTPRecipeID()` and `TPRecipeTy` enum values.

---

## 5. What Stays Unchanged

- `TPSlotTracker` — number-based slot assignment kept.
- **Print format** — `printAsOperand` overrides on `TPIRValue`, `TPSymbolicValue`,
  `TPSingleDefRecipe` produce identical output; `DimSet` annotation in
  `TPBasicBlock::print` is unchanged.
- `TPlanTypes.h` — `TensorOpKind`, `RecipeClassification`, `RecipeClassMap`.
- `TPlanWidener.cpp` — DimSet BFS logic unchanged; update
  `TPLiveIn`/`TPSyntheticValue` name references.
- `TPRecipeMatcher.cpp` — classification logic unchanged; update name references.
- `TPTransformState` — `ValueMap` key type changes from
  `const TPSingleDefRecipe*` to `const TPRecipeValue*`. The public accessors
  are updated accordingly:
  ```cpp
  // Before:
  Value *getValue(const TPSingleDefRecipe *R) const;
  void   setValue(const TPSingleDefRecipe *R, Value *V);
  // After:
  Value *getValue(const TPRecipeValue *R) const;
  void   setValue(const TPRecipeValue *R, Value *V);
  ```
  Call sites in `TPlanLowering.cpp` that call `State.getValue(SomeSingleDefRecipe*)`
  continue to work because `TPSingleDefRecipe` IS a `TPRecipeValue` (via
  inheritance), so the pointer implicitly converts.
- `TPlan` container: `DimPFs` becomes `SmallVector<unique_ptr<TPSymbolicValue>>`,
  `LiveIns` becomes `SmallVector<unique_ptr<TPIRValue>>`, all other fields
  unchanged.

---

## 6. Key Invariants

1. **Inheritance ordering preserved**: `TPSingleDefRecipe : TPRecipeBase,
   TPRecipeValue` — `TPRecipeBase` must be listed first so `ilist_node`
   memory layout is correct.
2. **`TPDef::~TPDef` does NOT delete sub-objects**: `TPDef::~TPDef` must not
   call `delete` on any `TPRecipeValue*` that is a base-class sub-object of
   a recipe, because `operator delete` on a non-first-base sub-object pointer
   is undefined behaviour. Sub-objects are destroyed by the recipe's own
   destructor. The destructor body is a no-op (or asserts `DefinedValues`
   is empty) for the current single-def-only codebase.
3. **TinyPtrVector zero-cost**: all current recipes define 0 or 1 value;
   inline-pointer path, no heap.
4. **Two independent SubclassIDs**: `TPValue::SubclassID` (3 values,
   value layer) and `TPRecipeBase::SubclassID` (TPRecipeTy, recipe layer)
   are unrelated fields in separate class hierarchies.
5. **PHI range contiguity**: `TPWidenPHISC`, `TPCanonicalIVSC`,
   `TPWidenInductionSC`, `TPReductionPHISC` are adjacent in the enum so
   range checks `[TPFirstPHISC, TPLastPHISC]` work with a single
   comparison.
6. **No VPlan headers included**: all TP* classes are fully independent.

---

## 7. File Map

| File | Change |
|---|---|
| `llvm/include/llvm/Transforms/Vectorize/TPlan.h` | Add `TPDef`, `TPRecipeValue`, `TPConstantInt`, `TPPhiAccessors`, `TPIRFlags`, `TPIRMetadata`, `TPRecipeWithIRFlags`; rename `TPLiveIn`→`TPIRValue`, `TPSyntheticValue`→`TPSymbolicValue`; refactor `TPValue`, `TPUser`, `TPRecipeBase` (add `TPRecipeTy`), `TPSingleDefRecipe`; add `friend class TPPhiAccessors` in `TPUser` |
| `llvm/lib/Transforms/Vectorize/TPlan.cpp` | Implement new class methods; update `buildInitial()` to use new names and `TPRecipeWithIRFlags` subclasses; audit all `setOperand` call sites |
| `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` | Update `TPTransformState::ValueMap` key to `TPRecipeValue*`; update `execute()` to call `applyFlags()` / `applyMetadata()` |
| `llvm/lib/Transforms/Vectorize/TPlanWidener.cpp` | Update `TPLiveIn`/`TPSyntheticValue` name references; DimSet BFS logic unchanged |
| `llvm/lib/Transforms/Vectorize/TPRecipeMatcher.cpp` | Update name references; classification logic unchanged |
| `llvm/test/Transforms/LoopTensorize/basic/tplan-print.ll` | Verify CHECK lines still pass (print format unchanged) |

---

## 8. Delivery Plan

### Commit 1 — Value/Def/User layer + TPRecipeTy enum

**Files:** `TPlan.h`, `TPlan.cpp`, `TPlanWidener.cpp`, `TPlanLowering.cpp`,
`TPRecipeMatcher.cpp`

- Add `TPDef`, `TPRecipeValue`, `TPConstantInt`, `TPSymbolicValue` (rename),
  `TPIRValue` (rename).
- Refactor `TPValue`, `TPUser` (delete default ctor; update `setOperand`;
  add iterator typedefs).
- Refactor `TPRecipeBase`: add `TPDef`+`TPUser` bases; replace `RecipeKind`
  with `TPRecipeTy`; add `getTPRecipeID()`, `isPhi()`, insertion methods.
- Refactor `TPSingleDefRecipe`: second base `TPValue`→`TPRecipeValue`;
  explicit `classof` switch.
- Update `TPlan::DimPFs` and `TPlan::LiveIns` element types.
- Update `TPSlotTracker::preAssignSynthetic` signature.
- Audit and fix all `setOperand` call sites.
- **Gate:** `ninja -C build LLVMVectorize` + `tplan-print.ll` lit test passes;
  print format unchanged.

### Commit 2 — TPPhiAccessors

**Files:** `TPlan.h`, `TPlan.cpp`

- Add `TPPhiAccessors` (all methods including `removeIncomingValueFor` and
  `incoming_values_and_blocks()`).
- Add `friend class TPPhiAccessors` in `TPUser`.
- Apply to `TPCanonicalIVRecipe`, `TPWidenInductionRecipe`,
  `TPReductionPHIRecipe`.
- **Gate:** build clean; print unchanged.

### Commit 3 — TPIRFlags + TPIRMetadata + TPRecipeWithIRFlags

**Files:** `TPlan.h`, `TPlan.cpp`, `TPlanLowering.cpp`

- Add `TPIRFlags` (with `Trunc` op-type), `TPIRMetadata`,
  `TPRecipeWithIRFlags`.
- Rebase `TPWidenRecipe`, `TPWidenGEPRecipe`, `TPWidenCastRecipe` onto
  `TPRecipeWithIRFlags`.
- Update `execute()` implementations to call `applyFlags()` and
  `applyMetadata()`.
- **Gate:** build clean; lowering produces correct IR with flags/metadata
  preserved.
