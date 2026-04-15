#ifndef LLVM_TRANSFORMS_TENSORIZE_TPLAN_VALUE_H
#define LLVM_TRANSFORMS_TENSORIZE_TPLAN_VALUE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/ADT/iterator_range.h"

namespace llvm {

// Forward declarations.
class raw_ostream;
class Value;
class TPDef;
class TPSlotTracker;
class TPUser;
class TPRecipeBase;

// This is the base class of the VPlan Def/Use graph, used for modeling the data
// flow into, within and out of the VPlan. VPValues can stand for live-ins
// coming from the input IR, instructions which VPlan will generate if executed
// and live-outs which the VPlan will need to fix accordingly.
class TPValue {
  friend class TPBuilder;
  friend class TPDef;
  friend class TPInstruction;
  friend struct TPlanTransforms;
  friend class TPBasicBlock;
  friend class TPInterleavedAccessInfo;
  friend struct TPSymbolicValue;
  friend class TPSlotTracker;
  friend class TPRecipeBase;

  const unsigned char SubclassID; ///< Subclass identifier (for isa/dyn_cast).

  SmallVector<TPUser *, 1> Users;

  // Parallel Factor
  // ElementCount PF = ElementCount::getFixed(1);

protected:
  // Hold the underlying Value, if any, attached to this VPValue.
  Value *UnderlyingVal;

  /// Pointer to the VPDef that defines this VPValue. If it is nullptr, the
  /// VPValue is not defined by any recipe modeled in VPlan.
  TPDef *Def;

  TPValue(const unsigned char SC, Value *UV = nullptr, TPDef *Def = nullptr);

  // DESIGN PRINCIPLE: Access to the underlying IR must be strictly limited to
  // the front-end and back-end of VPlan so that the middle-end is as
  // independent as possible of the underlying IR. We grant access to the
  // underlying IR using friendship. In that way, we should be able to use VPlan
  // for multiple underlying IRs (Polly?) by providing a new VPlan front-end,
  // back-end and analysis information for the new IR.

public:
  enum class ValueKind { LiveIn, Synthetic, Def };

  ValueKind getValueKind() const { return Kind; }

  /// Return the underlying Value attached to this VPValue.
  Value *getUnderlyingValue() const { return UnderlyingVal; }

  // /// Apply PF
  // void applyPF(ElementCount newPF) { this->PF = newPF; }

  /// An enumeration for keeping track of the concrete subclass of VPValue that
  /// are actually instantiated.
  enum {
    // Below is llvm/release19.x
    TPValueSC, /// A generic VPValue, like live-in values or defined by a recipe
               /// that defines multiple values.
    TPVRecipeSC, /// A VPValue sub-class that is a VPRecipeBase.

    // Below is llvm/release22.x
    TPIRValueSC, /// A live-in VPValue wrapping an IR value.
    TPTSymbolicSC, /// A symbolic live-in TPValue without IR backing.
    TPRecipeValueSC, /// A TPValue defined by a recipe.
  };

  /// Create a live-in VPValue.
  TPValue(Value *UV = nullptr) : TPValue(TPValueSC, UV, nullptr) {}
  /// Create a VPValue for a \p Def which is a subclass of VPValue.
  TPValue(TPDef *Def, Value *UV = nullptr) : TPValue(TPVRecipeSC, UV, Def) {}
  /// Create a VPValue for a \p Def which defines multiple values.
  TPValue(Value *UV, TPDef *Def) : TPValue(TPValueSC, UV, Def) {}
  TPValue(const TPValue &) = delete;
  TPValue &operator=(const TPValue &) = delete;

  virtual ~TPValue();

  /// \return an ID for the concrete type of this object.
  /// This is used to implement the classof checks. This should not be used
  /// for any other purpose, as the values may change as LLVM evolves.
  unsigned getTPValueID() const { return SubclassID; }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void printAsOperand(raw_ostream &OS, TPSlotTracker &Tracker) const;
  void print(raw_ostream &OS, TPSlotTracker &Tracker) const;

  /// Dump the value to stderr (for debugging).
  void dump() const;
#endif

  unsigned getNumUsers() const { return Users.size(); }
  void addUser(TPUser &User) { Users.push_back(&User); }

  /// Remove a single \p User from the list of users.
  void removeUser(TPUser &User) {
    // The same user can be added multiple times, e.g. because the same VPValue
    // is used twice by the same VPUser. Remove a single one.
    auto *I = find(Users, &User);
    if (I != Users.end())
      Users.erase(I);
  }

  typedef SmallVectorImpl<TPUser *>::iterator user_iterator;
  typedef SmallVectorImpl<TPUser *>::const_iterator const_user_iterator;
  typedef iterator_range<user_iterator> user_range;
  typedef iterator_range<const_user_iterator> const_user_range;

  user_iterator user_begin() { return Users.begin(); }
  const_user_iterator user_begin() const { return Users.begin(); }
  user_iterator user_end() { return Users.end(); }
  const_user_iterator user_end() const { return Users.end(); }
  user_range users() { return user_range(user_begin(), user_end()); }
  const_user_range users() const {
    return const_user_range(user_begin(), user_end());
  }

  /// Returns true if the value has more than one unique user.
  bool hasMoreThanOneUniqueUser() {
    if (getNumUsers() == 0)
      return false;

    // Check if all users match the first user.
    auto Current = std::next(user_begin());
    while (Current != user_end() && *user_begin() == *Current)
      Current++;
    return Current != user_end();
  }

  void replaceAllUsesWith(TPValue *New);

  /// Go through the uses list for this VPValue and make each use point to \p
  /// New if the callback ShouldReplace returns true for the given use specified
  /// by a pair of (VPUser, the use index).
  void replaceUsesWithIf(
      TPValue *New,
      llvm::function_ref<bool(TPUser &U, unsigned Idx)> ShouldReplace);

  /// Returns the recipe defining this VPValue or nullptr if it is not defined
  /// by a recipe, i.e. is a live-in.
  TPRecipeBase *getDefiningRecipe();
  const TPRecipeBase *getDefiningRecipe() const;

  /// Returns true if this VPValue is defined by a recipe.
  bool hasDefiningRecipe() const { return getDefiningRecipe(); }

  /// Returns true if this VPValue is a live-in, i.e. defined outside the VPlan.
  bool isLiveIn() const { return !hasDefiningRecipe(); }

  /// Returns the underlying IR value, if this VPValue is defined outside the
  /// scope of VPlan. Returns nullptr if the VPValue is defined by a VPDef
  /// inside a VPlan.
  Value *getLiveInIRValue() {
    assert(isLiveIn() &&
           "VPValue is not a live-in; it is defined by a VPDef inside a VPlan");
    return getUnderlyingValue();
  }
  const Value *getLiveInIRValue() const {
    assert(isLiveIn() &&
           "VPValue is not a live-in; it is defined by a VPDef inside a VPlan");
    return getUnderlyingValue();
  }

  /// Returns true if the VPValue is defined outside any vector regions, i.e. it
  /// is a live-in value.
  /// TODO: Also handle recipes defined in pre-header blocks.
  bool isDefinedOutsideVectorRegions() const { return !hasDefiningRecipe(); }

  // Set \p Val as the underlying Value of this VPValue.
  void setUnderlyingValue(Value *Val) {
    assert(!UnderlyingVal && "Underlying Value is already set.");
    UnderlyingVal = Val;
  }

  private:
    ValueKind Kind = ValueKind::Def;

};

typedef DenseMap<Value *, TPValue *> Value2TPValueTy;
typedef DenseMap<TPValue *, Value *> TPValue2ValueTy;

raw_ostream &operator<<(raw_ostream &OS, const TPValue &V);

/// This class augments VPValue with operands which provide the inverse def-use
/// edges from VPValue's users to their defs.
class TPUser {
public:
  /// Subclass identifier (for isa/dyn_cast).
  enum class TPUserID {
    Recipe,
    LiveOut,
  };

private:
  SmallVector<TPValue *, 2> Operands;

  TPUserID ID;

protected:
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the operands to \p O.
  void printOperands(raw_ostream &O, TPSlotTracker &SlotTracker) const;
#endif

  TPUser(ArrayRef<TPValue *> Operands, TPUserID ID) : ID(ID) {
    for (TPValue *Operand : Operands)
      addOperand(Operand);
  }

  TPUser(std::initializer_list<TPValue *> Operands, TPUserID ID)
      : TPUser(ArrayRef<TPValue *>(Operands), ID) {}

  template <typename IterT>
  TPUser(iterator_range<IterT> Operands, TPUserID ID) : ID(ID) {
    for (TPValue *Operand : Operands)
      addOperand(Operand);
  }

public:
  TPUser() = delete;
  TPUser(const TPUser &) = delete;
  TPUser &operator=(const TPUser &) = delete;
  virtual ~TPUser() {
    for (TPValue *Op : operands())
      Op->removeUser(*this);
  }

  TPUserID getTPUserID() const { return ID; }

  void addOperand(TPValue *Operand) {
    Operands.push_back(Operand);
    Operand->addUser(*this);
  }

  unsigned getNumOperands() const { return Operands.size(); }
  inline TPValue *getOperand(unsigned N) const {
    assert(N < Operands.size() && "Operand index out of bounds");
    return Operands[N];
  }

  void setOperand(unsigned I, TPValue *New) {
    Operands[I]->removeUser(*this);
    Operands[I] = New;
    New->addUser(*this);
  }

  typedef SmallVectorImpl<TPValue *>::iterator operand_iterator;
  typedef SmallVectorImpl<TPValue *>::const_iterator const_operand_iterator;
  typedef iterator_range<operand_iterator> operand_range;
  typedef iterator_range<const_operand_iterator> const_operand_range;

  operand_iterator op_begin() { return Operands.begin(); }
  const_operand_iterator op_begin() const { return Operands.begin(); }
  operand_iterator op_end() { return Operands.end(); }
  const_operand_iterator op_end() const { return Operands.end(); }
  operand_range operands() { return operand_range(op_begin(), op_end()); }
  const_operand_range operands() const {
    return const_operand_range(op_begin(), op_end());
  }

  /// Returns true if the VPUser uses scalars of operand \p Op. Conservatively
  /// returns if only first (scalar) lane is used, as default.
  virtual bool usesScalars(const TPValue *Op) const {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return onlyFirstLaneUsed(Op);
  }

  /// Returns true if the VPUser only uses the first lane of operand \p Op.
  /// Conservatively returns false.
  virtual bool onlyFirstLaneUsed(const TPValue *Op) const {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return false;
  }

  /// Returns true if the VPUser only uses the first part of operand \p Op.
  /// Conservatively returns false.
  virtual bool onlyFirstPartUsed(const TPValue *Op) const {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return false;
  }
};

/// A symbolic live-in TPValue, used for values like vector trip count, VF, and
/// VFxUF.
/// - named synthetic value with no IR baking (mirrors VPSymbolicValue)
struct TPSymbolicValue : public TPValue {
  TPSymbolicValue(StringRef Name) : TPValue(TPTSymbolicSC, nullptr), Name(Name) {}

  StringRef getName() const { return Name; }

  // Below cannot use, cause TPlan has no virtual function for printAsOpernad.
  // void printAsOperand(raw_ostream &OS, TPSlotTracker &Tracker) const override;

  static bool classof(const TPValue *V) {
    return V->getTPValueID() == TPTSymbolicSC;
  }
private:
  std::string Name;
};

/// This class augments a recipe with a set of TPValues defined by the recipe.
/// It allows recipes to define zero, one or multiple TPValues. A TPDef owns
/// the TPValues it defines and is responsible for deleting its defined values.
/// Single-value TPDefs that also inherit from TPValue must make sure to inherit
/// from TPDef before TPValue.
class TPDef {
  friend class TPValue;

  /// Subclass identifier (for isa/dyn_cast).
  const unsigned char SubclassID;

  /// The VPValues defined by this VPDef.
  TinyPtrVector<TPValue *> DefinedValues;

  /// Add \p V as a defined value by this VPDef.
  void addDefinedValue(TPValue *V) {
    assert(V->Def == this &&
           "can only add VPValue already linked with this VPDef");
    DefinedValues.push_back(V);
  }

  /// Remove \p V from the values defined by this VPDef. \p V must be a defined
  /// value of this VPDef.
  void removeDefinedValue(TPValue *V) {
    assert(V->Def == this && "can only remove VPValue linked with this VPDef");
    assert(is_contained(DefinedValues, V) &&
           "VPValue to remove must be in DefinedValues");
    llvm::erase(DefinedValues, V);
    V->Def = nullptr;
  }

public:
  /// An enumeration for keeping track of the concrete subclass of VPRecipeBase
  /// that is actually instantiated. Values of this enumeration are kept in the
  /// SubclassID field of the VPRecipeBase objects. They are used for concrete
  /// type identification.
  using TPRecipeTy = enum {
    TPBranchOnMaskSC,
    TPDerivedIVSC,
    TPExpandSCEVSC,
    TPInstructionSC,
    TPInterleaveSC,
    TPReductionEVLSC,
    TPReductionSC,
    TPReplicateSC,
    TPNewInstrSC,
    TPScalarCastSC,
    TPScalarIVStepsSC,
    TPVectorPointerSC,
    TPWidenCallSC,
    TPMatrixCallSC,
    TPWidenCanonicalIVSC,
    TPWidenCastSC,
    TPWidenGEPSC,
    TPWidenLoadEVLSC,
    TPWidenLoadSC,
    TPWidenStoreEVLSC,
    TPWidenStoreSC,
    TPWidenSC,
    TPWidenSelectSC,
    TPBlendSC,
    // START: Phi-like recipes. Need to be kept together.
    TPWidenPHISC,
    TPPredInstPHISC,
    // START: SubclassID for recipes that inherit VPHeaderPHIRecipe.
    // VPHeaderPHIRecipe need to be kept together.
    TPCanonicalIVPHISC,
    TPActiveLaneMaskPHISC,
    TPEVLBasedIVPHISC,
    TPFirstOrderRecurrencePHISC,
    TPWidenIntOrFpInductionSC,
    TPWidenPointerInductionSC,
    TPReductionPHISC,
    // END: SubclassID for recipes that inherit VPHeaderPHIRecipe
    // END: Phi-like recipes
    TPFirstPHISC = TPWidenPHISC,
    TPFirstHeaderPHISC = TPCanonicalIVPHISC,
    TPLastHeaderPHISC = TPReductionPHISC,
    TPLastPHISC = TPReductionPHISC,
  };

  TPDef(const unsigned char SC) : SubclassID(SC) {}

  virtual ~TPDef() {
    for (TPValue *D : make_early_inc_range(DefinedValues)) {
      assert(D->Def == this &&
             "all defined VPValues should point to the containing VPDef");
      assert(D->getNumUsers() == 0 &&
             "all defined VPValues should have no more users");
      D->Def = nullptr;
      delete D;
    }
  }

  /// Returns the only VPValue defined by the VPDef. Can only be called for
  /// VPDefs with a single defined value.
  TPValue *getTPSingleValue() {
    assert(DefinedValues.size() == 1 && "must have exactly one defined value");
    assert(DefinedValues[0] && "defined value must be non-null");
    return DefinedValues[0];
  }
  const TPValue *getTPSingleValue() const {
    assert(DefinedValues.size() == 1 && "must have exactly one defined value");
    assert(DefinedValues[0] && "defined value must be non-null");
    return DefinedValues[0];
  }

  /// Returns the VPValue with index \p I defined by the VPDef.
  TPValue *getTPValue(unsigned I) {
    assert(DefinedValues[I] && "defined value must be non-null");
    return DefinedValues[I];
  }
  const TPValue *getTPValue(unsigned I) const {
    assert(DefinedValues[I] && "defined value must be non-null");
    return DefinedValues[I];
  }

  /// Returns an ArrayRef of the values defined by the VPDef.
  ArrayRef<TPValue *> definedValues() { return DefinedValues; }
  /// Returns an ArrayRef of the values defined by the VPDef.
  ArrayRef<TPValue *> definedValues() const { return DefinedValues; }

  /// Returns the number of values defined by the VPDef.
  unsigned getNumDefinedValues() const { return DefinedValues.size(); }

  /// \return an ID for the concrete type of this object.
  /// This is used to implement the classof checks. This should not be used
  /// for any other purpose, as the values may change as LLVM evolves.
  unsigned getTPDefID() const { return SubclassID; }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Dump the VPDef to stderr (for debugging).
  void dump() const;

  /// Each concrete VPDef prints itself.
  virtual void print(raw_ostream &O, const Twine &Indent,
                     TPSlotTracker &SlotTracker) const = 0;
#endif
};

class TPlan;
class TPBasicBlock;

/// This class can be used to assign names to VPValues. For VPValues without
/// underlying value, assign consecutive numbers and use those as names (wrapped
/// in vp<>). Otherwise, use the name from the underlying value (wrapped in
/// ir<>), appending a .V version number if there are multiple uses of the same
/// name. Allows querying names for VPValues for printing, similar to the
/// ModuleSlotTracker for IR values.
class TPSlotTracker { // yuxin.an: L449
  /// Keep track of versioned names assigned to VPValues with underlying IR
  /// values.
  DenseMap<const TPValue *, std::string> TPValue2Name;
  /// Keep track of the next number to use to version the base name.
  StringMap<unsigned> BaseName2Version;

  /// Number to assign to the next VPValue without underlying value.
  unsigned NextSlot = 0;

  void assignName(const TPValue *V);
  void assignNames(const TPlan &Plan);
  void assignNames(const TPBasicBlock *TPBB);

public:
  TPSlotTracker(const TPlan *Plan = nullptr) {
    if (Plan)
      assignNames(*Plan);
  }

  /// Returns the name assigned to \p V, if there is one, otherwise try to
  /// construct one from the underlying value, if there's one; else return
  /// <badref>.
  std::string getOrCreateName(const TPValue *V) const;
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_TENSORIZE_TPLAN_VALUE_H
