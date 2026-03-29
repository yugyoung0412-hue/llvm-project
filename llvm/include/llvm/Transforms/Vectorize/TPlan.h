//===- TPlan.h - Tensor Plan IR for LoopTensorize -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_VECTORIZE_TPLAN_H
#define LLVM_TRANSFORMS_VECTORIZE_TPLAN_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/FMF.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Metadata.h"
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
class SCEVExpander;
class SCEV;
class TPBasicBlock;
class TPBlockBase;
class TPBlockUtils;
class TPIRBasicBlock;
class TPRecipeBase;
class TPRegionBlock;
class TPSingleDefRecipe;
class TPConstantInt;
class TPDef;
class TPIRValue;
class TPPhiAccessors;
class TPRecipeValue;
class TPSlotTracker;
class TPSymbolicValue;
class TPlan;
class Value;
struct TPTransformState;

//===----------------------------------------------------------------------===//
// TPValue — base SSA value node with use tracking (mirrors VPValue)
//===----------------------------------------------------------------------===//
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
  // Conditional replacement: replaces operand I in user U if Fn(U, I) is true.
  void replaceUsesWithIf(TPValue *New,
                          function_ref<bool(TPUser &, unsigned)> Fn);

  // Returns the recipe that defines this value, or nullptr for live-ins/symbolics.
  class TPRecipeBase *getDefiningRecipe();
  const class TPRecipeBase *getDefiningRecipe() const;

  // True for live-ins and symbolics (produced outside any loop region).
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

//===----------------------------------------------------------------------===//
// TPUser — base for anything that consumes TPValues (mirrors VPUser)
//===----------------------------------------------------------------------===//
class TPUser {
  friend class TPPhiAccessors;
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
    // User lists are unordered (index-agnostic), so no user-list update needed.
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

  // Stubs mirroring VPUser API (not used in TPlan today).
  virtual bool usesScalars(const TPValue *)     const { return false; }
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

//===----------------------------------------------------------------------===//
// TPIRValue — loop-invariant value from outside the nest (mirrors VPIRValue)
//===----------------------------------------------------------------------===//
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

//===----------------------------------------------------------------------===//
// TPConstantInt — overlay on TPIRValue for ConstantInt values (mirrors VPConstantInt)
//===----------------------------------------------------------------------===//
class TPConstantInt : public TPIRValue {
public:
  explicit TPConstantInt(ConstantInt *CI) : TPIRValue(CI) {}
  bool isOne()  const;
  bool isZero() const;
  const APInt &getAPInt()    const;
  unsigned getBitWidth()     const;
  uint64_t getZExtValue()    const;

  static bool classof(const TPIRValue *V) {
    return isa<ConstantInt>(V->getUnderlyingValue());
  }
  static bool classof(const TPValue *V) {
    if (const auto *IR = dyn_cast<TPIRValue>(V))
      return classof(IR);
    return false;
  }
};

//===----------------------------------------------------------------------===//
// TPSymbolicValue — named synthetic value with no IR backing (mirrors VPSymbolicValue)
//===----------------------------------------------------------------------===//
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

//===----------------------------------------------------------------------===//
// TPSlotTracker — assigns monotonic tp<%N> numbers to TPValues
//===----------------------------------------------------------------------===//
class TPSlotTracker {
public:
  /// Pre-assign a slot to a synthetic value. Must be called before any
  /// lazy getSlot() calls so synthetics get the lowest slot numbers.
  void preAssignSynthetic(const TPSymbolicValue *V);

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

  void appendRecipe(TPRecipeBase *R); // defined inline after TPRecipeBase

  /// Returns a pointer to the recipe list for ilist_node parent access.
  static RecipeListTy TPBasicBlock::*getSublistAccess(TPRecipeBase *) {
    return &TPBasicBlock::Recipes;
  }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &OS, const Twine &Indent,
             TPSlotTracker &Tracker) const override;

  /// Set the IR basic block at which recipes in this block should be inserted.
  /// When set, execute() repositions the builder to this block's first non-PHI
  /// before executing recipes.  Used for synthetic latch blocks that correspond
  /// to a real IR latch but keep a "tensor.latch*" name for TPlan printing.
  void setInsertionBB(BasicBlock *BB) { InsertionBB = BB; }
  BasicBlock *getInsertionBB() const { return InsertionBB; }

protected:
  explicit TPBasicBlock(unsigned char SC, StringRef Name)
      : TPBlockBase(SC, Name) {}

  RecipeListTy Recipes;
  BasicBlock *InsertionBB = nullptr; ///< Optional IR BB for builder positioning.
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
  void print(raw_ostream &OS, const Twine &Indent,
             TPSlotTracker &Tracker) const override;

  /// Flat alternatives using constructionOrder (original behavior, for debugging).
  void printFlat(raw_ostream &OS, const Twine &Indent,
                 TPSlotTracker &Tracker) const;
  void executeFlat(TPTransformState &State);

private:
  TPBlockBase *Entry    = nullptr;
  TPBlockBase *Exiting  = nullptr;
  TPBlockBase *Middle   = nullptr;  ///< Epilogue-check block; null for innermost.
  TPBlockBase *Scalar   = nullptr;  ///< Scalar preheader; null for innermost.
  TPRegionBlock *Inner  = nullptr;  ///< Next-inner nested region; null if leaf.
  DenseMap<Loop *, TPBlockBase *> Loop2HeaderTPB;
  DenseMap<Loop *, TPBlockBase *> Loop2LatchTPB;
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
// TPDef — owns the TPRecipeValue instances produced by a recipe (mirrors VPDef)
//===----------------------------------------------------------------------===//
class TPDef {
  friend class TPRecipeValue;
  // Most recipes define exactly one value, so TinyPtrVector stores the pointer
  // inline with no heap allocation for the common case.  A vector is only
  // needed when a single recipe must produce multiple outputs — for example,
  // an interleaved-load group (factor N) that issues one wide load + shuffles
  // but exposes N independent results to the rest of the IR:
  //
  //   for (int i = 0; i < N; i++) {
  //     float r = data[i].r;  // \
  //     float g = data[i].g;  //  > one interleave recipe, 3 defined values
  //     float b = data[i].b;  // /
  //   }
  //
  // TPlan does not currently generate such recipes, but the infrastructure
  // is kept general (matching VPlan's VPDef design) to allow it in the future.
  TinyPtrVector<TPRecipeValue *> DefinedValues;

  void addDefinedValue(TPRecipeValue *V) { DefinedValues.push_back(V); }
  void removeDefinedValue(TPRecipeValue *V) {
    auto *It = llvm::find(DefinedValues, V);
    assert(It != DefinedValues.end());
    DefinedValues.erase(It);
  }

public:
  TPDef() = default;
  virtual ~TPDef() {
    // DefinedValues is empty here: TPRecipeValue::~TPRecipeValue calls
    // removeDefinedValue before this destructor runs (MRO order).
    assert(DefinedValues.empty() &&
           "TPDef destroyed with live DefinedValues");
  }

  // Return TPRecipeValue* (not TPValue*) to avoid incomplete-type conversion at
  // the point of TPDef definition.  Callers can upcast to TPValue* implicitly.
  TPRecipeValue *getTPSingleValue() {
    assert(DefinedValues.size() == 1); return DefinedValues[0];
  }
  const TPRecipeValue *getTPSingleValue() const {
    assert(DefinedValues.size() == 1); return DefinedValues[0];
  }
  TPRecipeValue *getTPValue(unsigned I) { return DefinedValues[I]; }
  const TPRecipeValue *getTPValue(unsigned I) const { return DefinedValues[I]; }
  // Non-const element type for both overloads (avoids pointer-to-pointer issues).
  ArrayRef<TPRecipeValue *> definedValues() { return DefinedValues; }
  ArrayRef<TPRecipeValue *> definedValues() const { return DefinedValues; }
  unsigned getNumDefinedValues() const { return DefinedValues.size(); }
};

//===----------------------------------------------------------------------===//
// TPRecipeValue — a TPValue produced by a recipe (mirrors VPRecipeValue)
//===----------------------------------------------------------------------===//
class TPRecipeValue : public TPValue {
  TPDef *Def;
public:
  TPRecipeValue(TPDef *D, Value *UV = nullptr)
      : TPValue(TPVRecipeValueSC, UV), Def(D) {
    Def->addDefinedValue(this);
  }
  virtual ~TPRecipeValue() {
    Def->removeDefinedValue(this);
  }

  TPDef *getDef() const { return Def; }

  // Defined inline after TPRecipeBase is complete (below).
  TPRecipeBase *getDefiningRecipe();
  const TPRecipeBase *getDefiningRecipe() const;

  static bool classof(const TPValue *V) {
    return V->getTPValueID() == TPVRecipeValueSC;
  }
};

//===----------------------------------------------------------------------===//
// TPRecipeBase — base for all TPlan recipe nodes (mirrors VPRecipeBase)
//===----------------------------------------------------------------------===//
class TPRecipeBase
    : public ilist_node_with_parent<TPRecipeBase, TPBasicBlock>,
      public TPDef,
      public TPUser {

  friend class TPBasicBlock;
  friend class TPBlockUtils;

  const unsigned char SubclassID;  // holds a TPRecipeTy value
  TPBasicBlock *Parent = nullptr;  // required by ilist_node_with_parent

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
    TPWidenPHISC,           // TPWidenPHIRecipe

    // Header PHI recipes (subset: [TPFirstHeaderPHISC, TPLastHeaderPHISC])
    TPCanonicalIVSC,
    TPWidenIntOrFpInductionSC,  // replaces TPWidenInductionSC
    TPWidenPointerInductionSC,  // NEW
    TPReductionPHISC,

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

    // Range markers
    TPFirstPHISC       = TPWidenPHISC,
    TPFirstHeaderPHISC = TPCanonicalIVSC,
    TPLastHeaderPHISC  = TPEVLBasedIVPHISC,   // was TPReductionPHISC
    TPLastPHISC        = TPPhiSC,              // was TPReductionPHISC
  };

  unsigned getTPRecipeID() const { return SubclassID; }

  bool isPhi() const {
    return getTPRecipeID() >= TPFirstPHISC &&
           getTPRecipeID() <= TPLastPHISC;
  }

  TPBasicBlock *getParent()       { return Parent; }
  const TPBasicBlock *getParent() const { return Parent; }

  /// Returns the single-def value, or nullptr for stores.
  TPSingleDefRecipe       *getDefinedValue();
  const TPSingleDefRecipe *getDefinedValue() const;

  // Insertion/movement helpers.
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

//===----------------------------------------------------------------------===//
// TPSingleDefRecipe — recipe that defines exactly one value (mirrors VPSingleDefRecipe)
//===----------------------------------------------------------------------===//
class TPSingleDefRecipe : public TPRecipeBase,   // first — ilist layout
                          public TPRecipeValue {  // second — sub-object
public:
  /// Loop dim indices this value spans; set by TPlanWidener_widen().
  SmallBitVector DimSet;

  /// Per-dim memory stride overrides (load/store recipes only).
  /// Key: dim index (innermost=0). Value: SCEV stride expression in elements.
  /// Absent entry → dense default expressed as a SCEV constant.
  /// Populated by TPRecipePatternMatcher_match() via SCEV GEP-index analysis.
  DenseMap<unsigned, const SCEV *> MemStrides;

  /// Returns the effective memory stride for \p Dim as a SCEV expression.
  /// Returns MemStrides[Dim] if set, else SE.getConstant(getDenseStrideForDim(Dim)).
  const SCEV *getMemStride(unsigned Dim, const TPlan &Plan,
                            ScalarEvolution &SE) const;

  void printAsOperand(raw_ostream &OS, TPSlotTracker &Tracker) const override;

  static bool classof(const TPRecipeBase *R) {
    switch (R->getTPRecipeID()) {
    case TPWidenSC:
    case TPWidenGEPSC:
    case TPWidenLoadSC:
    case TPWidenCastSC:
    case TPCanonicalIVSC:
    case TPWidenIntOrFpInductionSC:
    case TPWidenPointerInductionSC:
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

//===----------------------------------------------------------------------===//
// Inline definitions after TPSingleDefRecipe is complete
//===----------------------------------------------------------------------===//
inline TPSingleDefRecipe *TPRecipeBase::getDefinedValue() {
  return dyn_cast<TPSingleDefRecipe>(this);
}
inline const TPSingleDefRecipe *TPRecipeBase::getDefinedValue() const {
  return dyn_cast<TPSingleDefRecipe>(this);
}
// TPRecipeValue::getDefiningRecipe — safe because only TPRecipeBase subtypes
// TPDef and creates TPRecipeValue (i.e., Def always points at a TPRecipeBase).
inline TPRecipeBase *TPRecipeValue::getDefiningRecipe() {
  return static_cast<TPRecipeBase *>(getDef());
}
inline const TPRecipeBase *TPRecipeValue::getDefiningRecipe() const {
  return static_cast<const TPRecipeBase *>(getDef());
}

// TPBasicBlock::appendRecipe — defined after TPRecipeBase is complete so we
// can set R->Parent.
inline void TPBasicBlock::appendRecipe(TPRecipeBase *R) {
  R->Parent = this;
  Recipes.push_back(R);
}

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

  /// Remove operand at index \p Idx.
  /// Block-based overload is deferred — no current consumer uses it.
  void removeIncomingValueFor(unsigned Idx) {
    const_cast<TPRecipeBase *>(getAsRecipe())->removeOperand(Idx);
  }

  void printPhiOperands(raw_ostream &OS, TPSlotTracker &SlotTracker) const;
};

//===----------------------------------------------------------------------===//
// TPIRFlags — per-instruction poison/flags storage (mirrors VPIRFlags)
//===----------------------------------------------------------------------===//
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

  OperationType      getOperationType()   const { return OpType; }
  bool hasNoUnsignedWrap()               const { return OvflowFlags.HasNUW; }
  bool hasNoSignedWrap()                 const { return OvflowFlags.HasNSW; }
  bool isDisjoint()                      const { return DisjointFlags.IsDisjoint; }
  bool isNonNeg()                        const { return NonNegFlags.IsNonNeg; }
  bool isExact()                         const { return ExactFlags.IsExact; }
  FastMathFlags   getFastMathFlags()     const { return FMF; }
  CmpInst::Predicate getPredicate()      const { return CmpPred; }
};

//===----------------------------------------------------------------------===//
// TPIRMetadata — per-instruction propagatable metadata (mirrors VPIRMetadata)
//===----------------------------------------------------------------------===//
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

//===----------------------------------------------------------------------===//
// TPRecipeWithIRFlags — single-def recipe that also stores IR flags
//===----------------------------------------------------------------------===//
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

//===----------------------------------------------------------------------===//
// TPWidenInductionRecipe — abstract base for IV PHIs (mirrors VPWidenInductionRecipe)
//===----------------------------------------------------------------------===//
class TPWidenInductionRecipe : public TPHeaderPHIRecipe {
protected:
  PHINode *IVPhi;    // was private in the old concrete class — now protected
  unsigned DimIndex; // no default initializer (constructor always receives an explicit Dim)
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

//===----------------------------------------------------------------------===//
// TPReductionPHIRecipe — WIDEN-REDUCTION-PHI: accumulator PHIs
//===----------------------------------------------------------------------===//
class TPReductionPHIRecipe : public TPHeaderPHIRecipe {
public:
  TPReductionPHIRecipe(PHINode *Phi, TPValue *InitVal, TPValue *LoopVal)
      : TPHeaderPHIRecipe(TPReductionPHISC, {InitVal, LoopVal}),
        RedPhi(Phi) {}

  PHINode *getPhi() const { return RedPhi; }

  void print(raw_ostream &OS, unsigned Indent,
             TPSlotTracker &Tracker) const override;
  void execute(TPTransformState &State) const override;

  static bool classof(const TPRecipeBase *R) {
    return R->getTPRecipeID() == TPReductionPHISC;
  }

private:
  PHINode *RedPhi;
};

//===----------------------------------------------------------------------===//
// TPFirstOrderRecurrencePHIRecipe — first-order recurrence PHI stub
//===----------------------------------------------------------------------===//
class TPFirstOrderRecurrencePHIRecipe : public TPHeaderPHIRecipe {
public:
  explicit TPFirstOrderRecurrencePHIRecipe(TPValue *StartVal)
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
  explicit TPWidenPHIRecipe(TPValue *StartVal)
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

//===----------------------------------------------------------------------===//
// TPWidenRecipe — WIDEN: arithmetic/icmp/generic instructions
//===----------------------------------------------------------------------===//
class TPWidenRecipe : public TPRecipeWithIRFlags {
public:
  TPWidenRecipe(Instruction *I, SmallVectorImpl<TPValue *> &Ops)
      : TPRecipeWithIRFlags(TPWidenSC, Ops, *I), Inst(I) {}

  Instruction *getInstruction() const { return Inst; }

  void print(raw_ostream &OS, unsigned Indent,
             TPSlotTracker &Tracker) const override;
  void execute(TPTransformState &State) const override;

  static bool classof(const TPRecipeBase *R) {
    return R->getTPRecipeID() == TPWidenSC;
  }

private:
  Instruction *Inst;
};

//===----------------------------------------------------------------------===//
// TPWidenGEPRecipe — WIDEN-GEP: getelementptr
//===----------------------------------------------------------------------===//
class TPWidenGEPRecipe : public TPRecipeWithIRFlags {
public:
  TPWidenGEPRecipe(Instruction *GEP, SmallVectorImpl<TPValue *> &Ops)
      : TPRecipeWithIRFlags(TPWidenGEPSC, Ops, *GEP), GEPInst(GEP) {}

  Instruction *getInstruction() const { return GEPInst; }

  void print(raw_ostream &OS, unsigned Indent,
             TPSlotTracker &Tracker) const override;
  void execute(TPTransformState &State) const override;

  static bool classof(const TPRecipeBase *R) {
    return R->getTPRecipeID() == TPWidenGEPSC;
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
      : TPSingleDefRecipe(TPWidenLoadSC, {PtrOp}), LoadInst(Load) {}

  Instruction *getInstruction() const { return LoadInst; }

  void print(raw_ostream &OS, unsigned Indent,
             TPSlotTracker &Tracker) const override;
  void execute(TPTransformState &State) const override;

  static bool classof(const TPRecipeBase *R) {
    return R->getTPRecipeID() == TPWidenLoadSC;
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
      : TPRecipeBase(TPWidenStoreSC, {PtrOp, ValOp}), StoreInst(Store) {}

  Instruction *getInstruction() const { return StoreInst; }

  /// Dimensions this store participates in. Copied from the stored-value
  /// operand's DimSet by TPRecipePatternMatcher_match().
  SmallBitVector DimSet;

  /// Per-dim memory stride overrides in elements.
  /// Populated by TPRecipePatternMatcher_match() via SCEV GEP-index analysis.
  DenseMap<unsigned, const SCEV *> MemStrides;

  /// Returns the effective memory stride for \p Dim as a SCEV expression.
  const SCEV *getMemStride(unsigned Dim, const TPlan &Plan,
                            ScalarEvolution &SE) const;

  void print(raw_ostream &OS, unsigned Indent,
             TPSlotTracker &Tracker) const override;
  void execute(TPTransformState &State) const override;

  static bool classof(const TPRecipeBase *R) {
    return R->getTPRecipeID() == TPWidenStoreSC;
  }

private:
  Instruction *StoreInst;
};

//===----------------------------------------------------------------------===//
// TPWidenCastRecipe — WIDEN-CAST: bitcast, sext, zext, trunc
//===----------------------------------------------------------------------===//
class TPWidenCastRecipe : public TPRecipeWithIRFlags {
public:
  TPWidenCastRecipe(Instruction *Cast, TPValue *SrcOp)
      : TPRecipeWithIRFlags(TPWidenCastSC, {SrcOp}, *Cast), CastInst(Cast) {}

  Instruction *getInstruction() const { return CastInst; }

  void print(raw_ostream &OS, unsigned Indent,
             TPSlotTracker &Tracker) const override;
  void execute(TPTransformState &State) const override;

  static bool classof(const TPRecipeBase *R) {
    return R->getTPRecipeID() == TPWidenCastSC;
  }

private:
  Instruction *CastInst;
};

//===----------------------------------------------------------------------===//
// TPCanonicalIVRecipe — CANONICAL-INDUCTION: synthetic loop counter phi
//===----------------------------------------------------------------------===//
class TPCanonicalIVRecipe : public TPHeaderPHIRecipe {
public:
  /// StartVal: TPIRValue for ir<0>. StepVal: patched after
  /// TPCanonicalIVIncrRecipe is created.
  TPCanonicalIVRecipe(TPValue *StartVal, TPValue *StepVal)
      : TPHeaderPHIRecipe(TPCanonicalIVSC, {StartVal, StepVal}) {}

  void print(raw_ostream &OS, unsigned Indent,
             TPSlotTracker &Tracker) const override;
  void execute(TPTransformState &State) const override;

  static bool classof(const TPRecipeBase *R) {
    return R->getTPRecipeID() == TPCanonicalIVSC;
  }
};

//===----------------------------------------------------------------------===//
// TPCanonicalIVIncrRecipe — CANONICAL-INDUCTION-INC: canonical IV + PF
//===----------------------------------------------------------------------===//
class TPCanonicalIVIncrRecipe : public TPSingleDefRecipe {
public:
  /// IVVal: TPCanonicalIVRecipe. PFVal: TPSymbolicValue for PF[D].
  TPCanonicalIVIncrRecipe(TPValue *IVVal, TPValue *PFVal)
      : TPSingleDefRecipe(TPCanonicalIVIncrSC, {IVVal, PFVal}) {}

  void print(raw_ostream &OS, unsigned Indent,
             TPSlotTracker &Tracker) const override;
  void execute(TPTransformState &State) const override;

  static bool classof(const TPRecipeBase *R) {
    return R->getTPRecipeID() == TPCanonicalIVIncrSC;
  }
};

//===----------------------------------------------------------------------===//
// TPCanonicalIVExitCmpRecipe — CANONICAL-INDUCTION-CMP: exit condition icmp
//===----------------------------------------------------------------------===//
class TPCanonicalIVExitCmpRecipe : public TPSingleDefRecipe {
public:
  /// IncrVal: TPCanonicalIVIncrRecipe. BoundVal: TPIRValue for the loop bound.
  TPCanonicalIVExitCmpRecipe(TPValue *IncrVal, TPValue *BoundVal)
      : TPSingleDefRecipe(TPCanonicalIVExitCmpSC, {IncrVal, BoundVal}) {}

  void print(raw_ostream &OS, unsigned Indent,
             TPSlotTracker &Tracker) const override;
  void execute(TPTransformState &State) const override;

  static bool classof(const TPRecipeBase *R) {
    return R->getTPRecipeID() == TPCanonicalIVExitCmpSC;
  }
};

//===----------------------------------------------------------------------===//
// TPlan — top-level container: entry block, live-ins, slot tracker
//===----------------------------------------------------------------------===//
class TPlan {
public:
  /// Build an initial TPlan by walking the loop nest IR.
  static TPlan buildInitial(const LoopNestInfo &Info);

  TPlan() = default;
  TPlan(const TPlan &) = delete;
  TPlan &operator=(const TPlan &) = delete;
  TPlan(TPlan &&) = default;
  TPlan &operator=(TPlan &&) = default;

  ~TPlan() {
    for (auto *B : CreatedBlocks)
      delete B;
  }

  void print(raw_ostream &OS) const;

  /// Returns the per-dimension parallel-factor symbolic value for dim \p D.
  TPSymbolicValue *getDimPF(unsigned D) const {
    assert(D < DimPFs.size() && "Dim out of range");
    return DimPFs[D].get();
  }

  const SmallBitVector &getReductionDims() const { return ReductionDims; }

  /// Returns the parallel factor for dimension \p Dim. Default: 1 (scalar).
  /// Set by LoopTensorize via setDimPF() before lowering.
  /// \p Dim uses the DimIdx convention (innermost=0, outermost=Depth-1).
  unsigned getPFForDim(unsigned Dim) const {
    auto It = DimPFMap.find(Dim);
    return It != DimPFMap.end() ? It->second : 1u;
  }
  /// \p Dim uses the DimIdx convention (innermost=0, outermost=Depth-1).
  void setDimPF(unsigned Dim, unsigned PF) { DimPFMap[Dim] = PF; }
  /// Returns the dense (packed) stride for dimension \p Dim.
  /// Dense stride(D) = product of getPFForDim(d) for all d < D.
  /// Dim 0 (innermost) always returns 1.
  /// \p Dim uses DimIdx convention (innermost=0, outermost=Depth-1).
  uint64_t getDenseStrideForDim(unsigned Dim) const {
    uint64_t Stride = 1;
    for (unsigned D = 0; D < Dim; ++D)
      Stride *= static_cast<uint64_t>(getPFForDim(D));
    return Stride;
  }

  /// Entry block (outermost preheader, a TPBasicBlock).
  TPBlockBase *getEntry() const { return Entry; }
  /// Set the plan's outermost entry block.
  void setEntry(TPBlockBase *B) { Entry = B; }

  /// Preheader block reserved for SCEV expansions (empty in initial plan).
  /// Not connected to the plan CFG until the SCEV-expansion wiring commit;
  /// do not traverse to it via successor edges before then.
  TPBasicBlock *getPreheader() const           { return Preheader; }
  void          setPreheader(TPBasicBlock *B)  { Preheader = B; }

  /// All TPRegionBlocks: Regions[0]=innermost, Regions[N-1]=outermost.
  ArrayRef<TPRegionBlock *> getRegions() const { return Regions; }

  /// Returns the TPRegionBlock that owns loop \p L, or nullptr.
  TPRegionBlock *getRegionForLoop(Loop *L) const {
    return LoopIdx2TPRB.lookup(L);
  }

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

private:
  SmallVector<std::unique_ptr<TPSymbolicValue>> DimPFs; ///< Per-dim PFs; DimPFs[0]=innermost (DimIdx convention).
  std::string FuncName;
  unsigned Depth = 0;
  SmallBitVector ReductionDims;              ///< Dims not in any store IndexExpr.
  DenseMap<unsigned, unsigned> DimPFMap;     ///< dim index (DimIdx, innermost=0) → parallel factor.
  SmallVector<std::unique_ptr<TPIRValue>> LiveIns;
  TPBlockBase *Entry = nullptr;                 ///< Outermost preheader block.
  TPBasicBlock *Preheader = nullptr;               ///< Reserved for SCEV expansions.
  SmallVector<TPRegionBlock *, 4> Regions;         ///< [0]=innermost, [N-1]=outermost.
  MapVector<Loop *, TPRegionBlock *> LoopIdx2TPRB; ///< Insertion-order for deterministic iteration.
  SmallVector<TPBlockBase *> CreatedBlocks;      ///< Owns all blocks.
  mutable TPSlotTracker Tracker;

  // Map from IR Value* to the TPValue* representing it (live-in or def).
  DenseMap<Value *, TPValue *> ValueMap;

  TPIRValue *getOrCreateLiveIn(Value *V);
  TPValue *getTPValue(Value *V);
};

/// State passed to execute() during TPlan lowering.
struct TPTransformState {
  IRBuilder<> &Builder;
  const TPlan &Plan;
  const RecipeClassMap *ClassMap = nullptr;
  DenseMap<const TPRecipeValue *, Value *> ValueMap;
  /// Maps each original IR instruction to the most recently emitted clone.
  /// Used by remapClone() to fix intra-block operand dominance: when a recipe
  /// clones an instruction whose operands were defined later in the same BB,
  /// remapClone() replaces those operands with the already-emitted clones.
  DenseMap<Value *, Value *> EmittedMap;
  /// Set by TPlanLowering_lower() before execute() loop.
  ScalarEvolution *SE = nullptr;
  /// Set by TPlanLowering_lower() before execute() loop. Owned by the caller.
  SCEVExpander *Expander = nullptr;

  TPTransformState(IRBuilder<> &B, const TPlan &P) : Builder(B), Plan(P) {}

  Value *getValue(const TPRecipeValue *V) const { return ValueMap.lookup(V); }
  void setValue(const TPRecipeValue *V, Value *IRV) { ValueMap[V] = IRV; }

  /// For each operand of Clone that is an original IR instruction tracked in
  /// EmittedMap, replaces it with the corresponding emitted clone. Call this
  /// after Instruction::clone() and before Builder.Insert() to avoid
  /// use-before-def when the original instruction appears later in the same BB.
  void remapClone(Instruction *Clone) const {
    for (unsigned I = 0, E = Clone->getNumOperands(); I != E; ++I) {
      auto It = EmittedMap.find(Clone->getOperand(I));
      if (It != EmittedMap.end())
        Clone->setOperand(I, It->second);
    }
  }

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

//===----------------------------------------------------------------------===//
// Inline definitions requiring complete TPlan
//===----------------------------------------------------------------------===//
/// Propagates DimSets from induction variables through the def-use graph
/// using BFS with union rule. Must be called before TPRecipePatternMatcher_match().
void TPlanWidener_widen(TPlan &Plan);

/// Lower all recipes in Plan to LLVM IR using the DimSet-driven dispatch.
/// Calls TPlanWidener_widen() and TPRecipePatternMatcher_match() internally.
bool TPlanLowering_lower(TPlan &Plan, Function &F, LoopInfo &LI,
                          ScalarEvolution &SE, DominatorTree &DT);

} // namespace llvm
#endif // LLVM_TRANSFORMS_VECTORIZE_TPLAN_H
