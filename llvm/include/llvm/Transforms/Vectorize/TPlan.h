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

class BasicBlock;
class DominatorTree;
class Function;
class Instruction;
class Loop;
class LoopInfo;
class ScalarEvolution;
class SCEV;
class TPBasicBlock;
class TPBlockBase;
class TPBlockUtils;
class TPIRBasicBlock;
class TPLoopRegion;
class TPRecipeBase;
class TPRegionBlock;
class TPSingleDefRecipe;
class TPSlotTracker;
class TPSyntheticValue;
class TPlan;
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

  /// Only valid on the plan's entry block.
  TPlan *getPlan() const { return Plan; }
  void setPlan(TPlan *P) { Plan = P; }

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
  TPlan *Plan = nullptr; ///< Only set on the plan's entry block.
  SmallVector<TPBlockBase *, 1> Predecessors;
  SmallVector<TPBlockBase *, 1> Successors;

  void appendSuccessor(TPBlockBase *S) { Successors.push_back(S); }
  void appendPredecessor(TPBlockBase *P) { Predecessors.push_back(P); }
  void removeSuccessor(TPBlockBase *S) {
    auto *It = llvm::find(Successors, S);
    assert(It != Successors.end()); Successors.erase(It);
  }
  void removePredecessor(TPBlockBase *P) {
    auto *It = llvm::find(Predecessors, P);
    assert(It != Predecessors.end()); Predecessors.erase(It);
  }
  void replacePredecessor(TPBlockBase *Old, TPBlockBase *New) {
    auto *It = llvm::find(Predecessors, Old);
    assert(It != Predecessors.end()); *It = New;
  }
};

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

//===----------------------------------------------------------------------===//
// TPBlockUtils — block wiring utilities (mirrors VPBlockUtils)
//===----------------------------------------------------------------------===//
class TPBlockUtils {
public:
  /// Bidirectional connect: adds To to From's successors and From to To's
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

  /// Insert New after After: New inherits After's successors, After -> New.
  static void insertBlockAfter(TPBlockBase *New, TPBlockBase *After) {
    assert(New->getSuccessors().empty() && New->getPredecessors().empty());
    New->setParent(After->getParent());
    transferSuccessors(After, New);
    connectBlocks(After, New);
  }

  /// Transfer all successors from Old to New (updates predecessor lists too).
  static void transferSuccessors(TPBlockBase *Old, TPBlockBase *New) {
    SmallVector<TPBlockBase *, 4> Succs;
    Succs.append(Old->getSuccessors().begin(), Old->getSuccessors().end());
    Old->clearSuccessors();
    for (auto *S : Succs) {
      auto *It = llvm::find(S->getPredecessors(), Old);
      assert(It != S->getPredecessors().end());
      *It = New;
      New->appendSuccessor(S);
    }
  }
};

/// DFS pre-order traversal from \p Start, following successors in insertion
/// order. Used by print() and block-driven lowering.
SmallVector<TPBlockBase *, 8> constructionOrder(TPBlockBase *Start);

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
  TPSingleDefRecipe *getDefinedValue();
  const TPSingleDefRecipe *getDefinedValue() const;

  virtual void print(raw_ostream &OS, unsigned Indent,
                     TPSlotTracker &Tracker) const = 0;
  virtual void execute(TPTransformState &State) const = 0;
  virtual ~TPRecipeBase() = default;

  /// All TPUser instances in a TPlan are TPRecipeBase instances.
  static bool classof(const TPUser *) { return true; }

protected:
  explicit TPRecipeBase(RecipeKind K) : Kind(K) {}

  RecipeKind Kind;
};

//===----------------------------------------------------------------------===//
// TPSingleDefRecipe — recipe that IS the value it defines (dual inheritance)
//===----------------------------------------------------------------------===//
class TPSingleDefRecipe : public TPRecipeBase, public TPValue {
public:
  /// Loop dim indices this value spans; set by TPlanWidener_widen().
  SmallBitVector DimSet;

  void printAsOperand(raw_ostream &OS, TPSlotTracker &Tracker) const override;

  static bool classof(const TPRecipeBase *R) {
    return R->getKind() != RecipeKind::WidenStore;
  }
  static bool classof(const TPValue *V) {
    return V->getValueKind() == ValueKind::Def;
  }

protected:
  explicit TPSingleDefRecipe(RecipeKind K)
      : TPRecipeBase(K), TPValue(ValueKind::Def) {}
};

//===----------------------------------------------------------------------===//
// TPRecipeBase::getDefinedValue — inline after TPSingleDefRecipe is defined
//===----------------------------------------------------------------------===//
inline TPSingleDefRecipe *TPRecipeBase::getDefinedValue() {
  return dyn_cast<TPSingleDefRecipe>(this);
}
inline const TPSingleDefRecipe *TPRecipeBase::getDefinedValue() const {
  return dyn_cast<TPSingleDefRecipe>(this);
}

//===----------------------------------------------------------------------===//
// TPWidenInductionRecipe — WIDEN-INDUCTION: loop IV PHIs
//===----------------------------------------------------------------------===//
class TPWidenInductionRecipe : public TPSingleDefRecipe {
public:
  TPWidenInductionRecipe(PHINode *IV, TPValue *StartVal, TPValue *StepVal,
                          unsigned DimIdx = 0)
      : TPSingleDefRecipe(RecipeKind::WidenInduction), IVPhi(IV),
        DimIndex(DimIdx) {
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
class TPReductionPHIRecipe : public TPSingleDefRecipe {
public:
  TPReductionPHIRecipe(PHINode *Phi, TPValue *InitVal, TPValue *LoopVal)
      : TPSingleDefRecipe(RecipeKind::ReductionPHI), RedPhi(Phi) {
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
class TPWidenRecipe : public TPSingleDefRecipe {
public:
  TPWidenRecipe(Instruction *I, SmallVectorImpl<TPValue *> &Ops)
      : TPSingleDefRecipe(RecipeKind::Widen), Inst(I) {
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
class TPWidenGEPRecipe : public TPSingleDefRecipe {
public:
  TPWidenGEPRecipe(Instruction *GEP, SmallVectorImpl<TPValue *> &Ops)
      : TPSingleDefRecipe(RecipeKind::WidenGEP), GEPInst(GEP) {
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
class TPWidenLoadRecipe : public TPSingleDefRecipe {
public:
  TPWidenLoadRecipe(Instruction *Load, TPValue *PtrOp)
      : TPSingleDefRecipe(RecipeKind::WidenLoad), LoadInst(Load) {
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
      : TPRecipeBase(RecipeKind::WidenStore), StoreInst(Store) {
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
class TPWidenCastRecipe : public TPSingleDefRecipe {
public:
  TPWidenCastRecipe(Instruction *Cast, TPValue *SrcOp)
      : TPSingleDefRecipe(RecipeKind::WidenCast), CastInst(Cast) {
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
class TPCanonicalIVRecipe : public TPSingleDefRecipe {
public:
  /// StartVal: TPLiveIn for ir<0>. StepVal: placeholder; patched after
  /// TPCanonicalIVIncrRecipe is created.
  TPCanonicalIVRecipe(TPValue *StartVal, TPValue *StepVal)
      : TPSingleDefRecipe(RecipeKind::CanonicalIV) {
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
class TPCanonicalIVIncrRecipe : public TPSingleDefRecipe {
public:
  /// IVVal: TPCanonicalIVRecipe (is TPValue). PFVal: TPSyntheticValue for PF.
  TPCanonicalIVIncrRecipe(TPValue *IVVal, TPValue *PFVal)
      : TPSingleDefRecipe(RecipeKind::CanonicalIVIncr) {
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
class TPCanonicalIVExitCmpRecipe : public TPSingleDefRecipe {
public:
  /// IncrVal: TPCanonicalIVIncrRecipe (is TPValue). BoundVal: TPLiveIn for
  /// the loop bound (RHS of the latch ICmpInst).
  TPCanonicalIVExitCmpRecipe(TPValue *IncrVal, TPValue *BoundVal)
      : TPSingleDefRecipe(RecipeKind::CanonicalIVExitCmp) {
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
  void setIV(TPSingleDefRecipe *V) { IV = V; }
  TPSingleDefRecipe *getIV() const { return IV; }
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
  TPSingleDefRecipe *IV = nullptr;
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
  DenseMap<const TPSingleDefRecipe *, Value *> ValueMap;

  TPTransformState(IRBuilder<> &B, const TPlan &P) : Builder(B), Plan(P) {}

  Value *getValue(const TPSingleDefRecipe *V) const { return ValueMap.lookup(V); }
  void setValue(const TPSingleDefRecipe *V, Value *IRV) { ValueMap[V] = IRV; }

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
