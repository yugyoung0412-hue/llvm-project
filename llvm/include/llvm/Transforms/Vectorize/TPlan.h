//===- TPlan.h - TPlan CFG IR for LoopTensorize ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines the CFG skeleton of the TPlan IR used in LoopTensorize:
///   - TPTransformState  : carries IRBuilder + analysis passes + value map
///   - TPBasicBlock      : an iplist of TPRecipeBase nodes
///   - TPRegionBlock     : loop region with header + latch blocks
///   - TPlan             : the top-level IR object (Entry/VectorBody/Middle/Tail)
///
/// Free functions (defined in separate .cpp files) are declared at the bottom.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_VECTORIZE_TPLAN_H
#define LLVM_TRANSFORMS_VECTORIZE_TPLAN_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ilist.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Vectorize/LoopNestAnalyzer.h"
#include "llvm/Transforms/Vectorize/TensorISAInfo.h"
#include "llvm/Transforms/Vectorize/TensorPatternClassifier.h"
#include "llvm/Transforms/Vectorize/TensorTransformSpace.h"
#include "llvm/Transforms/Vectorize/TPRecipe.h"
#include <memory>

namespace llvm {

class TPlan;
class Function;

//===----------------------------------------------------------------------===//
// TPTransformState — carries the IR emitting context during lowering.
//===----------------------------------------------------------------------===//
struct TPTransformState {
  IRBuilder<> &Builder;
  LoopInfo &LI;
  DominatorTree &DT;
  ScalarEvolution &SE;
  DenseMap<const TPValue *, Value *> ValueMap;
  const TPlan *Plan = nullptr;

  TPTransformState(IRBuilder<> &B, LoopInfo &LI, DominatorTree &DT,
                   ScalarEvolution &SE, const TPlan *P)
      : Builder(B), LI(LI), DT(DT), SE(SE), Plan(P) {}

  Value *getValue(const TPValue *V) const {
    return ValueMap.lookup(V);
  }
  void setValue(const TPValue *V, Value *IRV) { ValueMap[V] = IRV; }

  /// Returns the parallel factor for dimension \p Dim from the plan.
  unsigned getPF(unsigned Dim) const;
};

//===----------------------------------------------------------------------===//
// TPBlockBase — common base for TPBasicBlock and TPRegionBlock.
//===----------------------------------------------------------------------===//
class TPBlockBase {
protected:
  TPlan *Plan = nullptr;
  friend class TPlan;
};

//===----------------------------------------------------------------------===//
// TPBasicBlock — a sequence of recipes inside a TPlan.
//===----------------------------------------------------------------------===//
class TPBasicBlock : public TPBlockBase {
  iplist<TPRecipeBase> Recipes;
  friend struct ilist_traits<TPRecipeBase>;

public:
  TPBasicBlock() = default;

  /// Append \p R and set its parent to this block.
  void appendRecipe(TPRecipeBase *R) {
    R->Parent = this;
    Recipes.push_back(R);
  }

  using iterator = iplist<TPRecipeBase>::iterator;
  using const_iterator = iplist<TPRecipeBase>::const_iterator;
  using reverse_iterator = iplist<TPRecipeBase>::reverse_iterator;

  iterator begin() { return Recipes.begin(); }
  iterator end() { return Recipes.end(); }
  const_iterator begin() const { return Recipes.begin(); }
  const_iterator end() const { return Recipes.end(); }
  reverse_iterator rbegin() { return Recipes.rbegin(); }
  reverse_iterator rend() { return Recipes.rend(); }

  bool empty() const { return Recipes.empty(); }
  unsigned size() const { return Recipes.size(); }

  /// Execute all recipes in order.
  void execute(TPTransformState &State);
};

//===----------------------------------------------------------------------===//
// TPRegionBlock — a loop region with header and latch basic blocks.
//===----------------------------------------------------------------------===//
class TPRegionBlock : public TPBlockBase {
  std::unique_ptr<TPBasicBlock> Header;
  std::unique_ptr<TPBasicBlock> Latch;

public:
  TPRegionBlock()
      : Header(std::make_unique<TPBasicBlock>()),
        Latch(std::make_unique<TPBasicBlock>()) {}

  TPBasicBlock *getHeader() { return Header.get(); }
  const TPBasicBlock *getHeader() const { return Header.get(); }
  TPBasicBlock *getLatch() { return Latch.get(); }
  const TPBasicBlock *getLatch() const { return Latch.get(); }
};

//===----------------------------------------------------------------------===//
// TPlan — the top-level CFG IR object.
//===----------------------------------------------------------------------===//
class TPlan {
  std::unique_ptr<TPBasicBlock>  Entry;
  std::unique_ptr<TPRegionBlock> VectorBody;
  std::unique_ptr<TPBasicBlock>  MiddleBlock;
  std::unique_ptr<TPBasicBlock>  ScalarTail; // nullable

  DenseMap<unsigned, unsigned> PFMap; // dim → parallel factor
  PatternKind Pattern = PatternKind::Generic;
  TensorOpDesc SelectedOp;

public:
  TPlan()
      : Entry(std::make_unique<TPBasicBlock>()),
        VectorBody(std::make_unique<TPRegionBlock>()),
        MiddleBlock(std::make_unique<TPBasicBlock>()),
        ScalarTail(nullptr) {}

  //--- Parallel factor ---
  void setPF(unsigned Dim, unsigned F) { PFMap[Dim] = F; }
  unsigned getPF(unsigned Dim) const {
    auto It = PFMap.find(Dim);
    return It != PFMap.end() ? It->second : 1u;
  }
  /// Resolve PF from tile sizes in beam-search transforms, explicit caller
  /// values, and ISA op constraints.
  void resolvePF(const TensorOpDesc &Op, ArrayRef<unsigned> ExplicitPF,
                 ArrayRef<Transform> BeamTiles);

  //--- Structural block accessors ---
  TPBasicBlock *getEntry() { return Entry.get(); }
  const TPBasicBlock *getEntry() const { return Entry.get(); }

  TPRegionBlock *getVectorBody() { return VectorBody.get(); }
  const TPRegionBlock *getVectorBody() const { return VectorBody.get(); }

  TPBasicBlock *getMiddleBlock() { return MiddleBlock.get(); }
  const TPBasicBlock *getMiddleBlock() const { return MiddleBlock.get(); }

  TPBasicBlock *getScalarTail() { return ScalarTail.get(); }
  const TPBasicBlock *getScalarTail() const { return ScalarTail.get(); }

  void enableScalarTail() {
    if (!ScalarTail)
      ScalarTail = std::make_unique<TPBasicBlock>();
  }

  //--- Pattern / op info ---
  void setPattern(PatternKind K) { Pattern = K; }
  PatternKind getPattern() const { return Pattern; }
  void setSelectedOp(const TensorOpDesc &Op) { SelectedOp = Op; }
  const TensorOpDesc &getSelectedOp() const { return SelectedOp; }

  //--- Debug ---
  void print(raw_ostream &O) const;
  bool verify() const;
};

//===----------------------------------------------------------------------===//
// Free function declarations (implemented in respective .cpp files)
//===----------------------------------------------------------------------===//

/// Build an initial scalar TPlan from a LoopNestInfo.
/// Returns nullptr if the loop nest cannot be represented.
std::unique_ptr<TPlan> TPlanBuilder_build(const LoopNestInfo &Info,
                                           const PatternHint &Hint,
                                           const TensorOpDesc &Op,
                                           ArrayRef<unsigned> ExplicitPF,
                                           LLVMContext &Ctx);

/// Propagate parallel factors from header PHIs through the def-use graph.
void TPlanWidener_widen(TPlan &Plan);

/// Lower TPlan to LLVM IR, inserting BBs into \p F.
bool TPlanLowering_lower(TPlan &Plan, Function &F, LoopInfo &LI,
                          ScalarEvolution &SE, DominatorTree &DT);

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_TPLAN_H
