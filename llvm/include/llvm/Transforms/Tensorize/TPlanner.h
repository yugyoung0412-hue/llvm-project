//===- LoopVectorizationPlanner.h - Planner for LoopVectorization ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_TENSORIZE_LOOPTENSORIZEPLANNER_H
#define LLVM_TRANSFORMS_TENSORIZE_LOOPTENSORIZEPLANNER_H

#include "TPattern.h"
#include "TPlan.h"
#include "TPlanDecisionLogic.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Tensorize/LoopTensorizationLegality.h"
#include "llvm/Transforms/Tensorize/TensorizeCommon.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/Transforms/Utils/SizeOpts.h"

namespace llvm {

class LoopTensorizeCostModel;
class Loop;
class LoopInfo;
class DominatorTree;
class LoopVectorizationLegality;
class LoopVectorizationCostModel;
class PredicatedScalarEvolution;
class LoopVectorizeHints;
class OptimizationRemarkEmitter;
class TargetTransformInfo;
class TargetLibraryInfo;
class VPRecipeBuilder;
class InterleavedAccessInfo;
class Instruction;
class TPlan;
class TPlanDecisionLogic;
class TPBuilder;

class TPBuilder {
  TPBasicBlock *BB = nullptr;
  TPBasicBlock::iterator InsertPt = TPBasicBlock::iterator();

  /// Insert \p VPI in BB at InsertPt if BB is set.
  // TPInstruction *tryInsertInstruction(TPInstruction *TPI) {
  //   if (BB)
  //     BB->insert(TPI, InsertPt);
  //   return TPI;
  // }
  /// Insert \p TPI in BB at InsertPt if BB is set.
  template <typename T> T *tryInsertInstruction(T *R) {
    // YYG:REMOVE
    errs() << "[tryInsertInstruction] \n";
    if (BB)
      BB->insert(R, InsertPt);
    // YYG:REMOVE
    errs() << "[tryInsertInstruction] \n";
    return R;
  }

  TPInstruction *createInstruction(unsigned Opcode,
                                   ArrayRef<TPValue *> Operands, DebugLoc DL,
                                   const Twine &Name = "") {
    return tryInsertInstruction(new TPInstruction(Opcode, Operands, DL, Name));
  }

  TPInstruction *createInstruction(unsigned Opcode,
                                   std::initializer_list<TPValue *> Operands,
                                   DebugLoc DL, const Twine &Name = "") {
    return createInstruction(Opcode, ArrayRef<TPValue *>(Operands), DL, Name);
  }

public:
  TPBuilder() = default;
  TPBuilder(TPBasicBlock *InsertBB) { setInsertPoint(InsertBB); }
  TPBuilder(TPRecipeBase *InsertPt) { setInsertPoint(InsertPt); }

  /// Clear the insertion point: created instructions will not be inserted into
  /// a block.
  void clearInsertionPoint() {
    BB = nullptr;
    InsertPt = TPBasicBlock::iterator();
  }

  TPBasicBlock *getInsertBlock() const { return BB; }
  TPBasicBlock::iterator getInsertPoint() const { return InsertPt; }

  /// Create a VPBuilder to insert after \p R.
  static TPBuilder getToInsertAfter(TPRecipeBase *R) {
    TPBuilder B;
    B.setInsertPoint(R->getParent(), std::next(R->getIterator()));
    return B;
  }

  class TPInsertPoint {
    TPBasicBlock *Block = nullptr;
    TPBasicBlock::iterator Point;

  public:
    /// Creates a new insertion point which doesn't point to anything.
    TPInsertPoint() = default;

    /// Creates a new insertion point at the given location.
    TPInsertPoint(TPBasicBlock *InsertBlock, TPBasicBlock::iterator InsertPoint)
        : Block(InsertBlock), Point(InsertPoint) {}

    /// Returns true if this insert point is set.
    bool isSet() const { return Block != nullptr; }

    TPBasicBlock *getBlock() const { return Block; }
    TPBasicBlock::iterator getPoint() const { return Point; }
  };

  /// Sets the current insert point to a previously-saved location.
  void restoreIP(TPInsertPoint IP) {
    if (IP.isSet())
      setInsertPoint(IP.getBlock(), IP.getPoint());
    else
      clearInsertionPoint();
  }

  /// This specifies that created VPInstructions should be appended to the end
  /// of the specified block.
  void setInsertPoint(TPBasicBlock *TheBB) {
    assert(TheBB && "Attempting to set a null insert point");
    BB = TheBB;
    InsertPt = BB->end();
  }

  /// This specifies that created instructions should be inserted at the
  /// specified point.
  void setInsertPoint(TPBasicBlock *TheBB, TPBasicBlock::iterator IP) {
    BB = TheBB;
    InsertPt = IP;
  }

  /// This specifies that created instructions should be inserted at the
  /// specified point.
  void setInsertPoint(TPRecipeBase *IP) {
    BB = IP->getParent();
    InsertPt = IP->getIterator();
  }

  /// Create an N-ary operation with \p Opcode, \p Operands and set \p Inst as
  /// its underlying Instruction.
  TPInstruction *createNaryOp(unsigned Opcode, ArrayRef<TPValue *> Operands,
                              Instruction *Inst = nullptr,
                              const Twine &Name = "") {
    DebugLoc DL;
    if (Inst)
      DL = Inst->getDebugLoc();
    TPInstruction *NewTPInst = createInstruction(Opcode, Operands, DL, Name);
    NewTPInst->setUnderlyingValue(Inst);
    return NewTPInst;
  }

  TPInstruction *createNaryOp(unsigned Opcode, ArrayRef<TPValue *> Operands,
                              DebugLoc DL, const Twine &Name = "") {
    return createInstruction(Opcode, Operands, DL, Name);
  }

  TPInstruction *createOverflowingOp(unsigned Opcode,
                                     std::initializer_list<TPValue *> Operands,
                                     TPRecipeWithIRFlags::WrapFlagsTy WrapFlags,
                                     DebugLoc DL = {}, const Twine &Name = "") {
    return tryInsertInstruction(
        new TPInstruction(Opcode, Operands, WrapFlags, DL, Name));
  }
  TPValue *createNot(TPValue *Operand, DebugLoc DL = {},
                     const Twine &Name = "") {
    return createInstruction(TPInstruction::Not, {Operand}, DL, Name);
  }

  TPValue *createAnd(TPValue *LHS, TPValue *RHS, DebugLoc DL = {},
                     const Twine &Name = "") {
    return createInstruction(Instruction::BinaryOps::And, {LHS, RHS}, DL, Name);
  }

  TPValue *createOr(TPValue *LHS, TPValue *RHS, DebugLoc DL = {},
                    const Twine &Name = "") {

    return tryInsertInstruction(new TPInstruction(
        Instruction::BinaryOps::Or, {LHS, RHS},
        TPRecipeWithIRFlags::DisjointFlagsTy(false), DL, Name));
  }

  TPValue *createLogicalAnd(TPValue *LHS, TPValue *RHS, DebugLoc DL = {},
                            const Twine &Name = "") {
    return tryInsertInstruction(
        new TPInstruction(TPInstruction::LogicalAnd, {LHS, RHS}, DL, Name));
  }

  TPValue *createSelect(TPValue *Cond, TPValue *TrueVal, TPValue *FalseVal,
                        DebugLoc DL = {}, const Twine &Name = "",
                        std::optional<FastMathFlags> FMFs = std::nullopt) {
    auto *Select =
        FMFs ? new TPInstruction(Instruction::Select, {Cond, TrueVal, FalseVal},
                                 *FMFs, DL, Name)
             : new TPInstruction(Instruction::Select, {Cond, TrueVal, FalseVal},
                                 DL, Name);
    return tryInsertInstruction(Select);
  }

  /// Create a new ICmp VPInstruction with predicate \p Pred and operands \p A
  /// and \p B.
  /// TODO: add createFCmp when needed.
  TPValue *createICmp(CmpInst::Predicate Pred, TPValue *A, TPValue *B,
                      DebugLoc DL = {}, const Twine &Name = "");

  //===--------------------------------------------------------------------===//
  // RAII helpers.
  //===--------------------------------------------------------------------===//

  /// RAII object that stores the current insertion point and restores it when
  /// the object is destroyed.
  class InsertPointGuard {
    TPBuilder &Builder;
    TPBasicBlock *Block;
    TPBasicBlock::iterator Point;

  public:
    InsertPointGuard(TPBuilder &B)
        : Builder(B), Block(B.getInsertBlock()), Point(B.getInsertPoint()) {}

    InsertPointGuard(const InsertPointGuard &) = delete;
    InsertPointGuard &operator=(const InsertPointGuard &) = delete;

    ~InsertPointGuard() { Builder.restoreIP(TPInsertPoint(Block, Point)); }
  };
};

/// Result of createTensorizedLoopSkeleton().
///
/// After creation, the CFG looks like:
///
///   [OrigPred] → [GuardBB: TC >=u PF ?]
///                    |  true                   false
///                    ↓                           ↓
///              [TensorPreheader]         [ScalarPreheader]
///               → [tensor loop] →         → [scalar clone] →
///                          [MergeBB (original loop exit)]
///
/// GuardBB         — emits `icmp uge RuntimeTC, PF` + `condbr`.
/// TensorPreheader — the original loop's preheader; emitContraction() fills it.
/// ScalarPreheader — the cloned loop's preheader; left unmodified (scalar).
/// MergeBB         — the original loop's single exit block.
/// Valid           — false if any precondition was not met.
///
/// The VMap (original → scalar-clone mapping) is returned via the out-param
/// passed to createTensorizedLoopSkeleton(); it is NOT stored here because
/// ValueToValueMapTy is neither copyable nor movable.
struct TensorizedLoopSkeleton {
  BasicBlock *GuardBB         = nullptr; ///< Runtime TC >=u PF check.
  BasicBlock *TensorPreheader = nullptr; ///< Original loop's preheader.
  BasicBlock *ScalarPreheader = nullptr; ///< Clone's preheader.
  BasicBlock *MergeBB         = nullptr; ///< Common exit block.
  bool Valid = false;
};

struct TensorizationFactor {
  /// Vector width with best cost.
  SmallVector<ElementCount> Width;

  /// Cost of the loop with that width.
  InstructionCost Cost;

  /// Cost of the scalar loop.
  InstructionCost ScalarCost;

  /// The minimum trip count required to make vectorization profitable, e.g. due
  /// to runtime checks.
  ElementCount MinProfitableTripCount;

  TensorizationFactor(SmallVector<ElementCount> Width, InstructionCost Cost,
                      InstructionCost ScalarCost)
      : Width(Width), Cost(Cost), ScalarCost(ScalarCost) {}

  /// Width 1 means no vectorization, cost 0 means uncomputed cost.
  static TensorizationFactor Disabled() {
    return {{ElementCount::getFixed(1), ElementCount::getFixed(1),
             ElementCount::getFixed(1)},
            0,
            0};
  }

  bool operator==(const TensorizationFactor &rhs) const {
    return Width == rhs.Width && Cost == rhs.Cost;
  }

  bool operator!=(const TensorizationFactor &rhs) const {
    return !(*this == rhs);
  }
};

using TPlanPtr = std::unique_ptr<TPlan>;

/// The following IndVarVectorizationFactor was pulled out of
/// Information about vectorization costs.
struct IndVarVectorizationFactor {
  /// Vector width with best cost.
  ElementCount Width;

  /// Cost of the loop with that width.
  InstructionCost VecCost;

  /// Cost of the scalar loop.
  InstructionCost ScalarCost;

  /// The minimum trip count required to make vectorization profitable, e.g. due
  /// to runtime checks.
  ElementCount MinProfitableTripCount;

  IndVarVectorizationFactor(ElementCount Width, InstructionCost Cost,
                            InstructionCost ScalarCost)
      : Width(Width), VecCost(Cost), ScalarCost(ScalarCost) {}

  /// Width 1 means no vectorization, cost 0 means uncomputed cost.
  static IndVarVectorizationFactor Disabled() {
    return {ElementCount::getFixed(1), 0, 0};
  }

  bool operator==(const IndVarVectorizationFactor &rhs) const {
    return Width == rhs.Width && VecCost == rhs.VecCost;
  }

  bool operator!=(const IndVarVectorizationFactor &rhs) const {
    return !(*this == rhs);
  }
};

struct InductionVariableInfo {
  PHINode *Phi;
  unsigned LoopDepth;
  std::vector<Instruction *> Uses;
  IndVarVectorizationFactor IndVF;

  InductionVariableInfo(PHINode *P, unsigned Depth,
                        std::vector<Instruction *> U, ElementCount Width,
                        InstructionCost Cost, InstructionCost ScalarCost)
      : Phi(P), LoopDepth(Depth), Uses(std::move(U)),
        IndVF(Width, Cost, ScalarCost) {}
};

struct CanonicalizedLoopInfo {
  Loop *L;
  BasicBlock *Preheader = nullptr;
  BasicBlock *Header = nullptr;
  BasicBlock *Latch = nullptr;
  BasicBlock *Exit = nullptr;
  Loop *ClonedeLoop = nullptr;
  // exclusive blocks that current Loop have
  SmallVector<BasicBlock *, 8> OwnBody;
  /// ---------------------------------------------------------------
  /// getLoop – depth 에 해당하는 Loop* 를 반환한다.
  ///   * 이미 L 이 있으면 그대로 반환.
  ///   * 없으면 (필수 매개변수) LoopInfo 를 사용해 새 Loop 을 만든다.
  /// ---------------------------------------------------------------
  // Loop *getLoop(LoopInfo *LI) const {
  //   errs() << "getLoop start!\n";
  //   // ---------------------------------------------------------------
  //   // 3) 새 Loop 을 만든다.
  //   //    Loop 클래스의 생성자는   Loop(LoopInfo *LI = nullptr,
  //   //                                 BasicBlock *Header = nullptr);
  //   //    여기서는 이미 알고 있는 Header 를 넘겨준다.
  //   // ---------------------------------------------------------------
  //   // ValueToValueMapTy VMap;
  //   // BasicBLock *NewHeader = CloneBasicBLock(Header, VMap, ".cloned",
  //   Header->getparent());

  //   // Loop *NewLoop = LI->AllocateLoop();

  //   // LI->addTopLevelLoop(NewLoop);
  //   // NewLoop->addBasicBlockToLoop(Header, *LI);
  //   // for(BasicBlock *bodyBB : OwnBody)
  //   //   NewLoop->addBasicBlockToLoop(bodyBB, *LI);
  //   // NewLoop->addBasicBlockToLoop(Latch, *LI);
  //   return NewLoop;
  // }
};

/// Planner drives the vectorization process after having passed
/// Legality checks.
class LoopTensorizePlanner {
public:
  /// The loop that we evaluate.
  SmallVector<Loop *> Loops;

  /// Loop Info analysis.
  LoopInfo *LI;

  /// The dominator tree.
  DominatorTree *DT;

  /// Target Library Info.
  const TargetLibraryInfo *TLI;

  /// Target Transform Info.
  const TargetTransformInfo &TTI;

  /// The legality analysis.
  LoopTensorizationLegality *Legal;

  /// The profitability analysis.
  LoopTensorizeCostModel &CM;

  /// The interleaved access analysis.
  MapVector<Loop *, InterleavedAccessInfo *> Loop2IAI;

  MapVector<Loop *, PredicatedScalarEvolution *> Loop2PSE;

  ScalarEvolution *SE;

  const LoopTensorizeHints &Hints;

  OptimizationRemarkEmitter *ORE;

  TPlanDecisionLogic &TPlanDL;

  TPlanPtr tplan;

  // Below is for MPVPlan, so it'll be deprecated.
  SmallVector<TPlanPtr, 4> TPlans;

  void printExclusiveLoops();

  // @yg0412.yun.
  // ExclusiveLoops[LoopDegree] = CanonicalizedLoopInfo obj;
  DenseMap<unsigned, CanonicalizedLoopInfo> ExclusiveLoops;

  Loop *extractSingleLoop(Loop *OrigLoop, LLVMContext &Ctx);

  void attachBlockNumber(BasicBlock *BB, unsigned Num);

  void CloneLoop(CanonicalizedLoopInfo *Info);

  void collectOwnBody(Loop *L, SmallVectorImpl<BasicBlock *> &OwnBody);

  void fillExclusiveLoops();

  bool collectLoopsAtDepth();

  DenseMap<PHINode *, InductionVariableInfo> IndVarChainMap;

  // The loop induction variable chains
  DenseMap<PHINode *, SmallVector<IndVarVectorizationFactor, 8>>
      VFperIndVarChains;
  /// Profitable vector factors per induction variable chains
  SmallVector<IndVarVectorizationFactor, 8> ProfitableVFs;

  /// A builder used to construct the current plan.
  TPBuilder Builder;

  /// Computes the cost of \p Plan for vectorization factor \p VF.
  ///
  /// The current implementation requires access to the
  /// LoopVectorizationLegality to handle inductions and reductions, which is
  /// why it is kept separate from the VPlan-only cost infrastructure.
  ///
  /// TODO: Move to VPlan::cost once the use of LoopVectorizationLegality has
  /// been retired.
  InstructionCost cost(TPlan &Plan, ElementCount VF) const;

  LoopTensorizePlanner(SmallVector<Loop *> Loops, LoopInfo *LI,
                       DominatorTree *DT, const TargetLibraryInfo *TLI,
                       const TargetTransformInfo &TTI,
                       LoopTensorizationLegality *Legal,
                       LoopTensorizeCostModel &CM, TPlanDecisionLogic &DL,
                       MapVector<Loop *, InterleavedAccessInfo *> Loop2IAI,
                       MapVector<Loop *, PredicatedScalarEvolution *> Loop2PSE, ScalarEvolution *SE,
                       const LoopTensorizeHints &Hints,
                       OptimizationRemarkEmitter *ORE)
      : Loops(Loops), LI(LI), DT(DT), TLI(TLI), TTI(TTI), Legal(Legal), CM(CM),
        TPlanDL(DL), Loop2IAI(Loop2IAI), Loop2PSE(Loop2PSE), Hints(Hints),
        ORE(ORE), SE(SE) {}

  TFTy MaxTF;
  TUFTy MaxTIC;
  bool SuccessToApplyPattern;

  TPlan Transform(PHINode *IndPhi, ElementCount width);

  bool ApplyPattern(TPlanPtr &tplan, TPRecipeBuilder *RecipeBuilder,
                    TPBasicBlock *TPBB, bool UseTensorType);

  void printIndVPlanMap();

  void printIndVarChain();

  void getIndVarChain(const Loop *L);

  bool createNestedLoopTPlan(bool UseTensorType);
  /// Plan how to best vectorize, return the best VF and its cost, or
  /// std::nullopt if vectorization and interleaving should be avoided up front.
  std::optional<IndVarVectorizationFactor>
  plan(PHINode *Indvar, ElementCount UserVF, unsigned UserIC);

  /// Use the TPlan-native path to plan how to best vectorize, return the best
  /// VF and its cost.
  IndVarVectorizationFactor planInTPlanNativePath(ElementCount UserVF);

  /// Use the TPlan-native path to plan how to best vectorize, return the best
  /// VF and its cost.
  TensorizationFactor planInTPlanNativePath(SmallVector<ElementCount> TF);

  void setMaxTF(TensorizePattern *tp, TFTy *MaxTFMap);

  /// Generate the IR code for the vectorized loop captured in VPlan \p BestPlan
  /// according to the best selected \p VF and  \p UF.
  ///
  /// TODO: \p IsEpilogueVectorization is needed to avoid issues due to epilogue
  /// vectorization re-using plans for both the main and epilogue vector loops.
  /// It should be removed once the re-use issue has been fixed.
  /// \p ExpandedSCEVs is passed during execution of the plan for epilogue loop
  /// to re-use expansion results generated during main plan execution.
  ///
  /// Returns a mapping of SCEVs to their expanded IR values and a mapping for
  /// the reduction resume values. Note that this is a temporary workaround
  /// needed due to the current epilogue handling.
  void
  executePlan(TFTy BestTF, TUFTy BestUF, TPlan &BestPlan, LoopTensorizer &LT,
              DominatorTree *DT, bool UseTensorType,
              bool IsEpilogueVectorization,
              const DenseMap<const SCEV *, Value *> *ExpandedSCEVs = nullptr);
  
/// Creates a tensorized loop skeleton around \p OutermostLoop.
///
/// Clones \p OutermostLoop as a scalar fallback and inserts a GuardBB that
/// checks `RuntimeTC >=u PF` before the loop. True → tensorized path (original
/// loop, subsequently modified by emitContraction). False → scalar clone.
///
/// The original-to-clone block/instruction mapping is written into \p VMap.
///
/// Preconditions:
///   - OutermostLoop must have a unique preheader with a single predecessor.
///   - OutermostLoop must have a single exit block.
///   - RuntimeTC must dominate the preheader.
///
/// \param LI   LoopInfo, updated in place.
/// \param DT   DominatorTree, updated in place.
/// \param VMap Output: maps original blocks/instructions to their clones.
/// Returns Valid=false on any precondition failure.
TensorizedLoopSkeleton createTensorizedLoopSkeleton(Loop *OutermostLoop,
                                                     Value *RuntimeTC,
                                                     unsigned PF,
                                                     LoopInfo &LI,
                                                     DominatorTree &DT,
                                                     ValueToValueMapTy &VMap);

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void printPlans(raw_ostream &O);
#endif

  /// Look through the existing plans and return true if we have one with
  /// vectorization factor \p VF.
  bool hasPlanWithTF(TFTy TF) const { llvm_unreachable(""); }

  /// Test a \p Predicate on a \p Range of VF's. Return the value of applying
  /// \p Predicate on Range.Start, possibly decreasing Range.End such that the
  /// returned value holds for the entire \p Range.

  // static bool getDecisionAndClampRange(
  //     const std::function<bool(DenseMap<Loop *, ElementCount>)> &Predicate,
  //     TFRange &Range);
  static bool getDecisionAndClampRange(
        const std::function<bool(llvm::ElementCount)> &Predicate,
        TFRange &Range,
        llvm::Loop *CurLoop);

  /// Builds an EmissionPolicy by classifying every dimension known to the plan.
  ///
  /// Classification rules (per dim):
  ///   - TC unknown                   → skip (Inline, not added to Specs).
  ///   - TC constant, RealTC <= PF    → skip (Inline, not added).
  ///   - TC constant, RealTC >  PF    → StaticTiled.
  ///   - TC dynamic, output dim       → StaticTiled (umin-bounded tiling loop).
  ///   - TC dynamic, contraction dim  → DynamicTiled (fixed-tile body loop).
  ///
  /// Contraction dims are identified from \p CM (RecipeClassMap), which gives
  /// the authoritative per-recipe ContractDim from pattern matching. This is
  /// more reliable than Plan.getReductionDims(), which uses loop-nest analysis
  /// that may misclassify dims with combined GEP indices (e.g. C[i+j]).
  ///
  /// For DynamicTiled dims the PF in the Policy comes from Plan.getPFForDim().
  /// emitContraction() may refine via TTI at recipe time (when ElemTy/ranks
  /// are known). The Policy PF is used as the guard threshold in
  /// createTensorizedLoopSkeleton().
  EmissionPolicy buildEmissionPolicy(const TPlan &Plan);

  /// \return The most profitable vectorization factor and the cost of that VF
  /// for vectorizing the epilogue. Returns IndVarVectorizationFactor::Disabled
  /// if epilogue vectorization is not supported for the loop.
  IndVarVectorizationFactor
  selectEpilogueIndVarVectorizationFactor(const ElementCount MaxVF,
                                          unsigned IC);

protected:
  /// Build VPlans for power-of-2 VF's between \p MinVF and \p MaxVF inclusive,
  /// according to the information gathered by Legal when it checked if it is
  /// legal to vectorize the loop.
  void buildTPlans(SmallVector<ElementCount> MinTF,
                   SmallVector<ElementCount> MaxTF);

private:
  /// Build a VPlan according to the information gathered by Legal. \return a
  /// VPlan for vectorization factors \p Range.Start and up to \p Range.End
  /// exclusive, possibly decreasing \p Range.End.
  TPlanPtr buildTPlan(SmallVector<TFRange> &Range);

  Loop *outermostLoopContainingBB(BasicBlock *BB);

  /// Build a VPlan using VPRecipes according to the information gather by
  /// Legal. This method is only used for the legacy inner loop vectorizer.
  /// \p Range's largest included VF is restricted to the maximum VF the
  /// returned VPlan is valid for. If no VPlan can be built for the input range,
  /// set the largest included VF to the maximum VF for which no plan could be
  /// built.
  TPlanPtr tryToBuildTPlanWithTPRecipes(TFRange &Range, bool UseTensorType);

  /// Build VPlans for power-of-2 VF's between \p MinVF and \p MaxVF inclusive,
  /// according to the information gathered by Legal when it checked if it is
  /// legal to vectorize the loop. This method creates VPlans using VPRecipes.
  void buildTPlansWithTPRecipes(TFTy MinTF, TFTy MaxTF, bool UseTensorType);

  // Adjust the recipes for reductions. For in-loop reductions the chain of
  // instructions leading from the loop exit instr to the phi need to be
  // converted to reductions, with one operand being vector and the other being
  // the scalar reduction chain. For other reductions, a select is introduced
  // between the phi and live-out recipes when folding the tail.
  void adjustRecipesForReductions(TPlanPtr &Plan,
                                  VPRecipeBuilder &RecipeBuilder,
                                  ElementCount MinVF);

  /// \return The most profitable vectorization factor for the available VPlans
  /// and the cost of that VF.
  /// This is now only used to verify the decisions by the new VPlan-based
  /// cost-model and will be retired once the VPlan-based cost-model is
  /// stabilized.
  IndVarVectorizationFactor selectIndVarVectorizationFactor();

  /// Returns true if the per-lane cost of IndVarVectorizationFactor A is lower
  /// than that of B.
  bool isMoreProfitable(const IndVarVectorizationFactor &A,
                        const IndVarVectorizationFactor &B) const;

  /// Determines if we have the infrastructure to vectorize the loop and its
  /// epilogue, assuming the main loop is vectorized by \p VF.
  bool isCandidateForEpilogueVectorization(const ElementCount VF) const;
};

class GeneratedRTChecks {
  /// Basic block which contains the generated SCEV checks, if any.
  BasicBlock *SCEVCheckBlock = nullptr;

  /// The value representing the result of the generated SCEV checks. If it is
  /// nullptr, either no SCEV checks have been generated or they have been used.
  Value *SCEVCheckCond = nullptr;

  /// Basic block which contains the generated memory runtime checks, if any.
  BasicBlock *MemCheckBlock = nullptr;

  /// The value representing the result of the generated memory runtime checks.
  /// If it is nullptr, either no memory runtime checks have been generated or
  /// they have been used.
  Value *MemRuntimeCheckCond = nullptr;

  DominatorTree *DT;
  LoopInfo *LI;
  TargetTransformInfo *TTI;

  SCEVExpander SCEVExp;
  SCEVExpander MemCheckExp;

  bool CostTooHigh = false;
  const bool AddBranchWeights;

  Loop *OuterLoop = nullptr;

public:
  GeneratedRTChecks(ScalarEvolution &SE, DominatorTree *DT, LoopInfo *LI,
                    TargetTransformInfo *TTI, const DataLayout &DL,
                    bool AddBranchWeights)
      : DT(DT), LI(LI), TTI(TTI), SCEVExp(SE, DL, "scev.check"),
        MemCheckExp(SE, DL, "scev.check"), AddBranchWeights(AddBranchWeights) {}
};

using SCEV2ValueTy = DenseMap<const SCEV *, Value *>;

class LoopTensorizer {

public:
  LoopTensorizer(SmallVector<Loop *> Loops,
                 MapVector<Loop *, PredicatedScalarEvolution *> Loop2PSE,
                 LoopInfo *LI, DominatorTree *DT, const TargetLibraryInfo *TLI,
                 const TargetTransformInfo *TTI, AssumptionCache *AC,
                 OptimizationRemarkEmitter *ORE, TFTy TensorWidth,
                 TFTy MinProfitableTripCount, TUFTy UnrollFactor,
                 LoopTensorizationLegality *LTL, LoopTensorizeCostModel *CM,
                 BlockFrequencyInfo *BFI, ProfileSummaryInfo *PSI,
                 GeneratedRTChecks &RTChecks,
                 std::shared_ptr<TensorizePattern> Pattern,
                 Triple::ArchType ArchType)
      : Loops(Loops), Loop2PSE(Loop2PSE), LI(LI), DT(DT), TLI(TLI), TTI(TTI),
        AC(AC), ORE(ORE), TF(TensorWidth), UF(UnrollFactor),
        Builder(Loop2PSE.begin()->second->getSE()->getContext()), Legal(LTL),
        Cost(CM), BFI(BFI), PSI(PSI), RTChecks(RTChecks), Pattern(Pattern),
        ArchType(ArchType) {
    // Query this against the original loop and save it here because the profile
    // of the original loop header may change as the transformation happens.
    OptForSizeBasedOnProfile = llvm::shouldOptimizeForSize(
        Loops.front()->getHeader(), PSI, BFI, PGSOQueryType::IRPass);

    // !FIXME(yuxin.an)
    // for (auto Elem : MinProfitableTripCount) {
    //   Loop *CurLoop = Elem.first;
    //   if (Elem.second.isZero())
    //     this->MinProfitableTripCount[CurLoop] = TensorWidth[CurLoop];
    //   else
    //     this->MinProfitableTripCount[CurLoop] =
    //     MinProfitableTripCount[CurLoop];
    // }
  }

  virtual ~LoopTensorizer() = default;

  /// Create a new empty loop that will contain vectorized instructions later
  /// on, while the old loop will be used as the scalar remainder. Control flow
  /// is generated around the vectorized (and scalar epilogue) loops consisting
  /// of various checks and bypasses. Return the pre-header block of the new
  /// loop and the start value for the canonical induction, if it is != 0. The
  /// latter is the case when vectorizing the epilogue loop. In the case of
  /// epilogue vectorization, this function is overriden to handle the more
  /// complex control flow around the loops.  \p ExpandedSCEVs is used to
  /// look up SCEV expansions for expressions needed during skeleton creation.
  virtual std::pair<BasicBlock *, MapVector<Loop *, Value *>>
  createTensorizedLoopSkeleton(const SCEV2ValueTy &ExpandedSCEVs) {
    llvm_unreachable("not implemented");
  }

  /// Fix the vectorized code, taking care of header phi's, live-outs, and more.
  void fixTensorizedLoop(TPTransformState &State, TPlan &Plan);

  void adaptForTarget(TPTransformState &State, bool UseTensorType);

  /// Create a new phi node for the induction variable \p OrigPhi to resume
  /// iteration count in the scalar epilogue, from where the vectorized loop
  /// left off. \p Step is the SCEV-expanded induction step to use. In cases
  /// where the loop skeleton is more complicated (i.e., epilogue vectorization)
  /// and the resume values can come from an additional bypass block, the \p
  /// AdditionalBypass pair provides information about the bypass block and the
  /// end value on the edge from bypass to this loop.
  PHINode *createInductionResumeValue(
      PHINode *OrigPhi, const InductionDescriptor &ID, Value *Step,
      ArrayRef<BasicBlock *> BypassBlocks,
      std::pair<BasicBlock *, Value *> AdditionalBypass = {nullptr, nullptr});

protected:
  friend class LoopTensorizePlanner;

  /// A small list of PHINodes.
  using PhiVector = SmallVector<PHINode *, 4>;

  /// A type for scalarized values in the new loop. Each value from the
  /// original loop, when scalarized, is represented by UF x VF scalar values
  /// in the new unrolled loop, where UF is the unroll factor and VF is the
  /// vectorization factor.
  using ScalarParts = SmallVector<SmallVector<Value *, 4>, 2>;

  /// Returns (and creates if needed) the trip count of the widened loop.
  MapVector<Loop *, Value *>
  getOrCreateTensorTripCount(BasicBlock *InsertBlock);

  /// Emit a bypass check to see if the vector trip count is zero, including if
  /// it overflows.
  void emitIterationCountCheck(BasicBlock *Bypass);

  /// Emit a bypass check to see if all of the SCEV assumptions we've
  /// had to make are correct. Returns the block containing the checks or
  /// nullptr if no checks have been added.
  BasicBlock *emitSCEVChecks(BasicBlock *Bypass);

  /// Emit bypass checks to check any memory assumptions we may have made.
  /// Returns the block containing the checks or nullptr if no checks have been
  /// added.
  BasicBlock *emitMemRuntimeChecks(BasicBlock *Bypass);

  /// Emit basic blocks (prefixed with \p Prefix) for the iteration check,
  /// vector loop preheader, middle block and scalar preheader.
  void createTensorLoopSkeleton(StringRef Prefix);

  /// Create new phi nodes for the induction variables to resume iteration count
  /// in the scalar epilogue, from where the vectorized loop left off.
  /// In cases where the loop skeleton is more complicated (eg. epilogue
  /// vectorization) and the resume values can come from an additional bypass
  /// block, the \p AdditionalBypass pair provides information about the bypass
  /// block and the end value on the edge from bypass to this loop.
  void createInductionResumeValues(
      const SCEV2ValueTy &ExpandedSCEVs,
      std::pair<BasicBlock *, Value *> AdditionalBypass = {nullptr, nullptr});

  /// (maxim.o): Emit code to bypass the whole scalar part of original tensorized
  /// code. The SimplifyCFG then should be able to delete it altogether later.
  void emitScalarLoopBypassCode();

  /// Allow subclasses to override and print debug traces before/after vplan
  /// execution, when trace information is requested.
  virtual void printDebugTracesAtStart() {};
  virtual void printDebugTracesAtEnd() {};

  /// The original loop.
  SmallVector<Loop *> Loops;

  /// A wrapper around ScalarEvolution used to add runtime SCEV checks. Applies
  /// dynamic knowledge to simplify SCEV expressions and converts them to a
  /// more usable form.
  MapVector<Loop *, PredicatedScalarEvolution *> Loop2PSE;

  /// Loop Info.
  LoopInfo *LI;

  /// Dominator Tree.
  DominatorTree *DT;

  /// Target Library Info.
  const TargetLibraryInfo *TLI;

  /// Target Transform Info.
  const TargetTransformInfo *TTI;

  /// Assumption Cache.
  AssumptionCache *AC;

  /// Interface to emit optimization remarks.
  OptimizationRemarkEmitter *ORE;

  /// The vectorization SIMD factor to use. Each vector will have this many
  /// vector elements.
  TFTy TF;

  TFTy MinProfitableTripCount;

  /// The vectorization unroll factor to use. Each scalar is vectorized to this
  /// many different vector instructions.
  TUFTy UF;

  /// The builder that we use
  IRBuilder<> Builder;

  // --- Vectorization state ---

  BasicBlock *EntryB;

  /// The vector-loop preheader.
  BasicBlock *LoopTensorPreHeader;

  /// The scalar-loop preheader.
  BasicBlock *LoopScalarPreHeader;

  /// Middle Block between the vector and the scalar.
  BasicBlock *LoopMiddleBlock;

  /// The unique ExitBlock of the scalar loop if one exists.  Note that
  /// there can be multiple exiting edges reaching this block.
  BasicBlock *LoopExitBlock;

  /// The scalar loop body.
  BasicBlock *LoopScalarBody;

  /// A list of all bypass blocks. The first block is the entry of the loop.
  SmallVector<BasicBlock *, 4> LoopBypassBlocks;

  /// Store instructions that were predicated.
  SmallVector<Instruction *, 4> PredicatedInstructions;

  /// The legality analysis.
  LoopTensorizationLegality *Legal;

  /// The profitablity analysis.
  LoopTensorizeCostModel *Cost;

  // Record whether runtime checks are added.
  bool AddedSafetyChecks = false;

  // Holds the end values for each induction variable. We save the end values
  // so we can later fix-up the external users of the induction variables.
  DenseMap<PHINode *, Value *> IVEndValues;

  /// BFI and PSI are used to check for profile guided size optimizations.
  BlockFrequencyInfo *BFI;
  ProfileSummaryInfo *PSI;

  // Whether this loop should be optimized for size based on profile guided size
  // optimizatios.
  bool OptForSizeBasedOnProfile;

  /// Structure to hold information about generated runtime checks, responsible
  /// for cleaning the checks, if vectorization turns out unprofitable.
  GeneratedRTChecks &RTChecks;

  std::shared_ptr<TensorizePattern> Pattern;

  Triple::ArchType ArchType;

  // Holds the resume values for reductions in the loops, used to set the
  // correct start value of reduction PHIs when vectorizing the epilogue.
  SmallMapVector<const RecurrenceDescriptor *, PHINode *, 4>
      ReductionResumeValues;
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_TENSORIZE_LOOPTENSORIZEPLANNER_H
