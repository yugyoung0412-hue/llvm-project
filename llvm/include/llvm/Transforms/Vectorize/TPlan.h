//===- TPlan.h - Tensor Plan IR for LoopTensorize -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_VECTORIZE_TPLAN_H
#define LLVM_TRANSFORMS_VECTORIZE_TPLAN_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Vectorize/LoopNestAnalyzer.h"
#include "llvm/Transforms/Vectorize/TPlanTypes.h"
#include <memory>

namespace llvm {

class DominatorTree;
class Function;
class Instruction;
class Loop;
class LoopInfo;
class ScalarEvolution;
class SCEV;
class TPDefVal;
class TPlan;
class TPLoopRegion;
class TPRecipeBase;
class TPSlotTracker;
class TPSyntheticValue;
class Value;
struct TPTransformState;

//===----------------------------------------------------------------------===//
// TPValue — base SSA value node with use tracking
//===----------------------------------------------------------------------===//
class TPValue {
public:
  enum class ValueKind { LiveIn, Synthetic, Def };

  explicit TPValue(ValueKind K) : Kind(K) {}
  virtual ~TPValue() = default;
  virtual void printAsOperand(raw_ostream &OS, TPSlotTracker &Tracker) const = 0;

  ValueKind getValueKind() const { return Kind; }

  void addUser(class TPUser *U) { Users.push_back(U); }
  ArrayRef<class TPUser *> users() const { return Users; }

protected:
  SmallVector<class TPUser *, 2> Users;

private:
  ValueKind Kind;
};

//===----------------------------------------------------------------------===//
// TPUser — base for anything that consumes TPValues
//===----------------------------------------------------------------------===//
class TPUser {
public:
  virtual ~TPUser() = default;

  void addOperand(TPValue *V) {
    Operands.push_back(V);
    V->addUser(this);
  }
  void setOperand(unsigned I, TPValue *V) { Operands[I] = V; }
  ArrayRef<TPValue *> operands() const { return Operands; }
  TPValue *getOperand(unsigned I) const { return Operands[I]; }

protected:
  SmallVector<TPValue *, 4> Operands;
};

//===----------------------------------------------------------------------===//
// TPLiveIn — loop-invariant value from outside the nest
//===----------------------------------------------------------------------===//
class TPLiveIn : public TPValue {
public:
  explicit TPLiveIn(Value *V) : TPValue(ValueKind::LiveIn), IRVal(V) {}
  Value *getIRValue() const { return IRVal; }
  void printAsOperand(raw_ostream &OS, TPSlotTracker &Tracker) const override;

  static bool classof(const TPValue *V) {
    return V->getValueKind() == ValueKind::LiveIn;
  }

private:
  Value *IRVal;
};

//===----------------------------------------------------------------------===//
// TPSyntheticValue — a named synthetic value with no IR backing (e.g., PF)
//===----------------------------------------------------------------------===//
class TPSyntheticValue : public TPValue {
public:
  explicit TPSyntheticValue(StringRef Name)
      : TPValue(ValueKind::Synthetic), Name(Name) {}
  StringRef getName() const { return Name; }
  void printAsOperand(raw_ostream &OS, TPSlotTracker &Tracker) const override;

  static bool classof(const TPValue *V) {
    return V->getValueKind() == ValueKind::Synthetic;
  }

private:
  std::string Name;
};

//===----------------------------------------------------------------------===//
// TPSlotTracker — assigns monotonic tp<%N> numbers to TPValues
//===----------------------------------------------------------------------===//
class TPSlotTracker {
public:
  /// Pre-assign a slot to a synthetic value. Must be called before any
  /// lazy getSlot() calls so synthetics get the lowest slot numbers.
  void preAssignSynthetic(const TPSyntheticValue *V);

  /// Lazily assign a slot to any TPValue on first access.
  unsigned getSlot(const TPValue *V);

  void reset() { SlotMap.clear(); NextSlot = 0; }

private:
  DenseMap<const TPValue *, unsigned> SlotMap;
  unsigned NextSlot = 0;
};

//===----------------------------------------------------------------------===//
// TPDefVal — value defined by a recipe, printed as tp<%N>
//===----------------------------------------------------------------------===//
class TPDefVal : public TPValue {
public:
  explicit TPDefVal(TPRecipeBase *R)
      : TPValue(ValueKind::Def), DefRecipe(R) {}
  TPRecipeBase *getDefiningRecipe() const { return DefRecipe; }
  void printAsOperand(raw_ostream &OS, TPSlotTracker &Tracker) const override;

  SmallBitVector DimSet; ///< Loop dim indices this value spans; set by TPlanWidener_widen().

  static bool classof(const TPValue *V) {
    return V->getValueKind() == ValueKind::Def;
  }

private:
  TPRecipeBase *DefRecipe;
};

//===----------------------------------------------------------------------===//
// TPRecipeBase — base for all TPlan recipe nodes
//===----------------------------------------------------------------------===//
class TPRecipeBase : public ilist_node<TPRecipeBase>, public TPUser {
public:
  enum class RecipeKind {
    WidenInduction,
    ReductionPHI,
    Widen,
    WidenGEP,
    WidenLoad,
    WidenStore,
    WidenCast,
    CanonicalIV,
    CanonicalIVIncr,
    CanonicalIVExitCmp,
  };

  RecipeKind getKind() const { return Kind; }

  /// Returns the value defined by this recipe, or nullptr for stores.
  TPDefVal *getDefinedValue() const { return DefVal.get(); }

  virtual void print(raw_ostream &OS, unsigned Indent,
                     TPSlotTracker &Tracker) const = 0;
  virtual void execute(TPTransformState &State) const = 0;
  virtual ~TPRecipeBase() = default;

  /// All TPUser instances in a TPlan are TPRecipeBase instances.
  static bool classof(const TPUser *) { return true; }

protected:
  explicit TPRecipeBase(RecipeKind K, bool DefinesValue = true) : Kind(K) {
    if (DefinesValue)
      DefVal = std::make_unique<TPDefVal>(this);
  }

  RecipeKind Kind;
  std::unique_ptr<TPDefVal> DefVal; // null for void recipes (store)
};

//===----------------------------------------------------------------------===//
// TPWidenInductionRecipe — WIDEN-INDUCTION: loop IV PHIs
//===----------------------------------------------------------------------===//
class TPWidenInductionRecipe : public TPRecipeBase {
public:
  TPWidenInductionRecipe(PHINode *IV, TPValue *StartVal, TPValue *StepVal,
                          unsigned DimIdx = 0)
      : TPRecipeBase(RecipeKind::WidenInduction), IVPhi(IV), DimIndex(DimIdx) {
    addOperand(StartVal);
    addOperand(StepVal);
  }

  PHINode *getIVPhi() const { return IVPhi; }
  unsigned getDimIndex() const { return DimIndex; }

  void print(raw_ostream &OS, unsigned Indent,
             TPSlotTracker &Tracker) const override;
  void execute(TPTransformState &State) const override;

  static bool classof(const TPRecipeBase *R) {
    return R->getKind() == RecipeKind::WidenInduction;
  }

private:
  PHINode *IVPhi;
  unsigned DimIndex = 0; ///< Index in LoopNestInfo::IVs (0 = outermost).
};

//===----------------------------------------------------------------------===//
// TPReductionPHIRecipe — WIDEN-REDUCTION-PHI: accumulator PHIs
//===----------------------------------------------------------------------===//
class TPReductionPHIRecipe : public TPRecipeBase {
public:
  TPReductionPHIRecipe(PHINode *Phi, TPValue *InitVal, TPValue *LoopVal)
      : TPRecipeBase(RecipeKind::ReductionPHI), RedPhi(Phi) {
    addOperand(InitVal);
    addOperand(LoopVal);
  }

  PHINode *getPhi() const { return RedPhi; }

  void print(raw_ostream &OS, unsigned Indent,
             TPSlotTracker &Tracker) const override;
  void execute(TPTransformState &State) const override;

  static bool classof(const TPRecipeBase *R) {
    return R->getKind() == RecipeKind::ReductionPHI;
  }

private:
  PHINode *RedPhi;
};

//===----------------------------------------------------------------------===//
// TPWidenRecipe — WIDEN: arithmetic/icmp/generic instructions
//===----------------------------------------------------------------------===//
class TPWidenRecipe : public TPRecipeBase {
public:
  TPWidenRecipe(Instruction *I, SmallVectorImpl<TPValue *> &Ops)
      : TPRecipeBase(RecipeKind::Widen), Inst(I) {
    for (TPValue *Op : Ops)
      addOperand(Op);
  }

  Instruction *getInstruction() const { return Inst; }

  void print(raw_ostream &OS, unsigned Indent,
             TPSlotTracker &Tracker) const override;
  void execute(TPTransformState &State) const override;

  static bool classof(const TPRecipeBase *R) {
    return R->getKind() == RecipeKind::Widen;
  }

private:
  Instruction *Inst;
};

//===----------------------------------------------------------------------===//
// TPWidenGEPRecipe — WIDEN-GEP: getelementptr
//===----------------------------------------------------------------------===//
class TPWidenGEPRecipe : public TPRecipeBase {
public:
  TPWidenGEPRecipe(Instruction *GEP, SmallVectorImpl<TPValue *> &Ops)
      : TPRecipeBase(RecipeKind::WidenGEP), GEPInst(GEP) {
    for (TPValue *Op : Ops)
      addOperand(Op);
  }

  Instruction *getInstruction() const { return GEPInst; }

  void print(raw_ostream &OS, unsigned Indent,
             TPSlotTracker &Tracker) const override;
  void execute(TPTransformState &State) const override;

  static bool classof(const TPRecipeBase *R) {
    return R->getKind() == RecipeKind::WidenGEP;
  }

private:
  Instruction *GEPInst;
};

//===----------------------------------------------------------------------===//
// TPWidenLoadRecipe — WIDEN: load instruction
//===----------------------------------------------------------------------===//
class TPWidenLoadRecipe : public TPRecipeBase {
public:
  TPWidenLoadRecipe(Instruction *Load, TPValue *PtrOp)
      : TPRecipeBase(RecipeKind::WidenLoad), LoadInst(Load) {
    addOperand(PtrOp);
  }

  Instruction *getInstruction() const { return LoadInst; }

  void print(raw_ostream &OS, unsigned Indent,
             TPSlotTracker &Tracker) const override;
  void execute(TPTransformState &State) const override;

  static bool classof(const TPRecipeBase *R) {
    return R->getKind() == RecipeKind::WidenLoad;
  }

private:
  Instruction *LoadInst;
};

//===----------------------------------------------------------------------===//
// TPWidenStoreRecipe — WIDEN store: defines no value
//===----------------------------------------------------------------------===//
class TPWidenStoreRecipe : public TPRecipeBase {
public:
  TPWidenStoreRecipe(Instruction *Store, TPValue *PtrOp, TPValue *ValOp)
      : TPRecipeBase(RecipeKind::WidenStore, /*DefinesValue=*/false),
        StoreInst(Store) {
    addOperand(PtrOp);
    addOperand(ValOp);
  }

  Instruction *getInstruction() const { return StoreInst; }

  void print(raw_ostream &OS, unsigned Indent,
             TPSlotTracker &Tracker) const override;
  void execute(TPTransformState &State) const override;

  static bool classof(const TPRecipeBase *R) {
    return R->getKind() == RecipeKind::WidenStore;
  }

private:
  Instruction *StoreInst;
};

//===----------------------------------------------------------------------===//
// TPWidenCastRecipe — WIDEN-CAST: bitcast, sext, zext
//===----------------------------------------------------------------------===//
class TPWidenCastRecipe : public TPRecipeBase {
public:
  TPWidenCastRecipe(Instruction *Cast, TPValue *SrcOp)
      : TPRecipeBase(RecipeKind::WidenCast), CastInst(Cast) {
    addOperand(SrcOp);
  }

  Instruction *getInstruction() const { return CastInst; }

  void print(raw_ostream &OS, unsigned Indent,
             TPSlotTracker &Tracker) const override;
  void execute(TPTransformState &State) const override;

  static bool classof(const TPRecipeBase *R) {
    return R->getKind() == RecipeKind::WidenCast;
  }

private:
  Instruction *CastInst;
};

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
  void execute(TPTransformState &State) const override;

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
  void execute(TPTransformState &State) const override;

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
  void execute(TPTransformState &State) const override;

  static bool classof(const TPRecipeBase *R) {
    return R->getKind() == RecipeKind::CanonicalIVExitCmp;
  }
};

//===----------------------------------------------------------------------===//
// TPLoopRegion — one per nesting level, owns recipes + child region
//===----------------------------------------------------------------------===//
class TPLoopRegion {
public:
  TPLoopRegion(unsigned Level, Loop *L, const SCEV *TripCount)
      : Level(Level), OwnerLoop(L), TripCount(TripCount) {}

  void appendRecipe(TPRecipeBase *R) { Recipes.push_back(R); }
  void setIV(TPDefVal *V) { IV = V; }
  TPDefVal *getIV() const { return IV; }
  void setChild(std::unique_ptr<TPLoopRegion> C) { Child = std::move(C); }
  TPLoopRegion *getChild() const { return Child.get(); }
  unsigned getLevel() const { return Level; }
  Loop *getLoop() const { return OwnerLoop; }
  const SCEV *getTripCount() const { return TripCount; }
  iplist<TPRecipeBase> &getRecipes() { return Recipes; }
  const iplist<TPRecipeBase> &getRecipes() const { return Recipes; }

  void print(raw_ostream &OS, unsigned Indent, TPSlotTracker &Tracker) const;

private:
  unsigned Level;
  Loop *OwnerLoop;
  const SCEV *TripCount;
  TPDefVal *IV = nullptr;
  iplist<TPRecipeBase> Recipes;
  std::unique_ptr<TPLoopRegion> Child;
};

//===----------------------------------------------------------------------===//
// TPlan — top-level container: root region, live-ins, slot tracker
//===----------------------------------------------------------------------===//
class TPlan {
public:
  /// Build an initial TPlan by walking the loop nest IR.
  static TPlan buildInitial(const LoopNestInfo &Info);

  void print(raw_ostream &OS) const;

  TPLoopRegion *getRootRegion() const { return RootRegion.get(); }

  /// Returns the per-dimension parallel-factor synthetic value for dim \p D.
  TPSyntheticValue *getDimPF(unsigned D) const {
    assert(D < DimPFs.size() && "Dim out of range");
    return DimPFs[D].get();
  }

  const SmallBitVector &getReductionDims() const { return ReductionDims; }

  /// Returns the parallel factor for dimension \p Dim. Default: 1 (scalar).
  /// Set by LoopTensorize via setDimPF() before lowering.
  unsigned getPFForDim(unsigned Dim) const {
    auto It = DimPFMap.find(Dim);
    return It != DimPFMap.end() ? It->second : 1u;
  }
  void setDimPF(unsigned Dim, unsigned PF) { DimPFMap[Dim] = PF; }

private:
  SmallVector<std::unique_ptr<TPSyntheticValue>> DimPFs; ///< PF[0]…PF[Depth-1]
  std::string FuncName;
  unsigned Depth = 0;
  SmallBitVector ReductionDims;              ///< Dims not in any store IndexExpr.
  DenseMap<unsigned, unsigned> DimPFMap;     ///< dim index → parallel factor.
  SmallVector<std::unique_ptr<TPLiveIn>> LiveIns;
  std::unique_ptr<TPLoopRegion> RootRegion;
  mutable TPSlotTracker Tracker;

  // Map from IR Value* to the TPValue* representing it (live-in or def).
  DenseMap<Value *, TPValue *> ValueMap;

  TPLiveIn *getOrCreateLiveIn(Value *V);
  TPValue *getTPValue(Value *V);
};

/// State passed to execute() during TPlan lowering.
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

/// Propagates DimSets from induction variables through the def-use graph
/// using BFS with union rule. Must be called before TPRecipePatternMatcher_match().
void TPlanWidener_widen(TPlan &Plan);

/// Lower all recipes in Plan to LLVM IR using the DimSet-driven dispatch.
/// Calls TPlanWidener_widen() and TPRecipePatternMatcher_match() internally.
bool TPlanLowering_lower(TPlan &Plan, Function &F, LoopInfo &LI,
                          ScalarEvolution &SE, DominatorTree &DT);

} // namespace llvm
#endif // LLVM_TRANSFORMS_VECTORIZE_TPLAN_H
