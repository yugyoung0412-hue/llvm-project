#include "llvm/Transforms/Tensorize/TPlanTransforms.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Tensorize/TPRecipeBuilder.h"
#include "llvm/Transforms/Tensorize/TPlanAnalysis.h"
#include "llvm/Transforms/Tensorize/TPlanCFG.h"
#include "llvm/Transforms/Tensorize/TPlanDominatorTree.h"
#include "llvm/Transforms/Tensorize/TPlanPatternMatch.h"
#include "llvm/Transforms/Tensorize/TensorizeCommon.h"

using namespace llvm;

static bool sinkScalarOperands(TPlan &Plan) {
  auto Iter = tp_depth_first_deep(Plan.getEntry());
  bool Changed = false;
  // First, collect the operands of all recipes in replicate blocks as seeds for
  // sinking.
  SetVector<std::pair<TPBasicBlock *, TPSingleDefRecipe *>> WorkList;
  for (TPRegionBlock *TPR : TPBlockUtils::blocksOnly<TPRegionBlock>(Iter)) {
    TPBasicBlock *EntryVPBB = TPR->getEntryBasicBlock();
    if (!TPR->isReplicator() || EntryVPBB->getSuccessors().size() != 2)
      continue;
    TPBasicBlock *TPBB = dyn_cast<TPBasicBlock>(EntryVPBB->getSuccessors()[0]);
    if (!TPBB || TPBB->getSingleSuccessor() != TPR->getExitingBasicBlock())
      continue;
    for (auto &Recipe : *TPBB) {
      for (TPValue *Op : Recipe.operands())
        if (auto *Def =
                dyn_cast_or_null<TPSingleDefRecipe>(Op->getDefiningRecipe()))
          WorkList.insert(std::make_pair(TPBB, Def));
    }
  }

  bool ScalarVFOnly = Plan.hasScalarTFOnly();
  // Try to sink each replicate or scalar IV steps recipe in the worklist.
  for (unsigned I = 0; I != WorkList.size(); ++I) {
    TPBasicBlock *SinkTo;
    TPSingleDefRecipe *SinkCandidate;
    std::tie(SinkTo, SinkCandidate) = WorkList[I];
    if (SinkCandidate->getParent() == SinkTo ||
        SinkCandidate->mayHaveSideEffects() ||
        SinkCandidate->mayReadOrWriteMemory())
      continue;
    if (auto *RepR = dyn_cast<TPReplicateRecipe>(SinkCandidate)) {
      if (!ScalarVFOnly && RepR->isUniform())
        continue;
    } else if (!isa<TPScalarIVStepsRecipe>(SinkCandidate))
      continue;

    bool NeedsDuplicating = false;
    // All recipe users of the sink candidate must be in the same block SinkTo
    // or all users outside of SinkTo must be uniform-after-vectorization (
    // i.e., only first lane is used) . In the latter case, we need to duplicate
    // SinkCandidate.
    auto CanSinkWithUser = [SinkTo, &NeedsDuplicating,
                            SinkCandidate](TPUser *U) {
      auto *UI = dyn_cast<TPRecipeBase>(U);
      if (!UI)
        return false;
      if (UI->getParent() == SinkTo)
        return true;
      NeedsDuplicating = UI->onlyFirstLaneUsed(SinkCandidate);
      // We only know how to duplicate VPRecipeRecipes for now.
      return NeedsDuplicating && isa<TPReplicateRecipe>(SinkCandidate);
    };
    if (!all_of(SinkCandidate->users(), CanSinkWithUser))
      continue;

    if (NeedsDuplicating) {
      if (ScalarVFOnly)
        continue;
      Instruction *I = SinkCandidate->getUnderlyingInstr();
      auto *Clone = new TPReplicateRecipe(I, SinkCandidate->operands(), true);
      // TODO: add ".cloned" suffix to name of Clone's VPValue.

      Clone->insertBefore(SinkCandidate);
      SinkCandidate->replaceUsesWithIf(Clone, [SinkTo](TPUser &U, unsigned) {
        return cast<TPRecipeBase>(&U)->getParent() != SinkTo;
      });
    }
    SinkCandidate->moveBefore(*SinkTo, SinkTo->getFirstNonPhi());
    for (TPValue *Op : SinkCandidate->operands())
      if (auto *Def =
              dyn_cast_or_null<TPSingleDefRecipe>(Op->getDefiningRecipe()))
        WorkList.insert(std::make_pair(SinkTo, Def));
    Changed = true;
  }
  return Changed;
}

/// If \p R is a region with a VPBranchOnMaskRecipe in the entry block, return
/// the mask.
TPValue *getPredicatedMask(TPRegionBlock *R) {
  auto *EntryBB = dyn_cast<TPBasicBlock>(R->getEntry());
  if (!EntryBB || EntryBB->size() != 1 ||
      !isa<TPBranchOnMaskRecipe>(EntryBB->begin()))
    return nullptr;

  return cast<TPBranchOnMaskRecipe>(&*EntryBB->begin())->getOperand(0);
}

/// If \p R is a triangle region, return the 'then' block of the triangle.
static TPBasicBlock *getPredicatedThenBlock(TPRegionBlock *R) {
  auto *EntryBB = cast<TPBasicBlock>(R->getEntry());
  if (EntryBB->getNumSuccessors() != 2)
    return nullptr;

  auto *Succ0 = dyn_cast<TPBasicBlock>(EntryBB->getSuccessors()[0]);
  auto *Succ1 = dyn_cast<TPBasicBlock>(EntryBB->getSuccessors()[1]);
  if (!Succ0 || !Succ1)
    return nullptr;

  if (Succ0->getNumSuccessors() + Succ1->getNumSuccessors() != 1)
    return nullptr;
  if (Succ0->getSingleSuccessor() == Succ1)
    return Succ0;
  if (Succ1->getSingleSuccessor() == Succ0)
    return Succ1;
  return nullptr;
}

// Merge replicate regions in their successor region, if a replicate region
// is connected to a successor replicate region with the same predicate by a
// single, empty VPBasicBlock.
static bool mergeReplicateRegionsIntoSuccessors(TPlan &Plan) {
  SetVector<TPRegionBlock *> DeletedRegions;

  // Collect replicate regions followed by an empty block, followed by another
  // replicate region with matching masks to process front. This is to avoid
  // iterator invalidation issues while merging regions.
  SmallVector<TPRegionBlock *, 8> WorkList;
  for (TPRegionBlock *Region1 : TPBlockUtils::blocksOnly<TPRegionBlock>(
           tp_depth_first_deep(Plan.getEntry()))) {
    if (!Region1->isReplicator())
      continue;
    auto *MiddleBasicBlock =
        dyn_cast_or_null<TPBasicBlock>(Region1->getSingleSuccessor());
    if (!MiddleBasicBlock || !MiddleBasicBlock->empty())
      continue;

    auto *Region2 =
        dyn_cast_or_null<TPRegionBlock>(MiddleBasicBlock->getSingleSuccessor());
    if (!Region2 || !Region2->isReplicator())
      continue;

    TPValue *Mask1 = getPredicatedMask(Region1);
    TPValue *Mask2 = getPredicatedMask(Region2);
    if (!Mask1 || Mask1 != Mask2)
      continue;

    assert(Mask1 && Mask2 && "both region must have conditions");
    WorkList.push_back(Region1);
  }

  // Move recipes from Region1 to its successor region, if both are triangles.
  for (TPRegionBlock *Region1 : WorkList) {
    if (DeletedRegions.contains(Region1))
      continue;
    auto *MiddleBasicBlock = cast<TPBasicBlock>(Region1->getSingleSuccessor());
    auto *Region2 = cast<TPRegionBlock>(MiddleBasicBlock->getSingleSuccessor());

    TPBasicBlock *Then1 = getPredicatedThenBlock(Region1);
    TPBasicBlock *Then2 = getPredicatedThenBlock(Region2);
    if (!Then1 || !Then2)
      continue;

    // Note: No fusion-preventing memory dependencies are expected in either
    // region. Such dependencies should be rejected during earlier dependence
    // checks, which guarantee accesses can be re-ordered for vectorization.
    //
    // Move recipes to the successor region.
    for (TPRecipeBase &ToMove : make_early_inc_range(reverse(*Then1)))
      ToMove.moveBefore(*Then2, Then2->getFirstNonPhi());

    auto *Merge1 = cast<TPBasicBlock>(Then1->getSingleSuccessor());
    auto *Merge2 = cast<TPBasicBlock>(Then2->getSingleSuccessor());

    // Move VPPredInstPHIRecipes from the merge block to the successor region's
    // merge block. Update all users inside the successor region to use the
    // original values.
    for (TPRecipeBase &Phi1ToMove : make_early_inc_range(reverse(*Merge1))) {
      TPValue *PredInst1 =
          cast<TPPredInstPHIRecipe>(&Phi1ToMove)->getOperand(0);
      TPValue *Phi1ToMoveV = Phi1ToMove.getTPSingleValue();
      Phi1ToMoveV->replaceUsesWithIf(PredInst1, [Then2](TPUser &U, unsigned) {
        auto *UI = dyn_cast<TPRecipeBase>(&U);
        return UI && UI->getParent() == Then2;
      });

      // Remove phi recipes that are unused after merging the regions.
      if (Phi1ToMove.getTPSingleValue()->getNumUsers() == 0) {
        Phi1ToMove.eraseFromParent();
        continue;
      }
      Phi1ToMove.moveBefore(*Merge2, Merge2->begin());
    }

    // Finally, remove the first region.
    for (TPBlockBase *Pred : make_early_inc_range(Region1->getPredecessors())) {
      TPBlockUtils::disconnectBlocks(Pred, Region1);
      TPBlockUtils::connectBlocks(Pred, MiddleBasicBlock);
    }
    TPBlockUtils::disconnectBlocks(Region1, MiddleBasicBlock);
    DeletedRegions.insert(Region1);
  }

  for (TPRegionBlock *ToDelete : DeletedRegions)
    delete ToDelete;
  return !DeletedRegions.empty();
}

static TPRegionBlock *createReplicateRegion(TPReplicateRecipe *PredRecipe,
                                            TPlan &Plan) {
  Instruction *Instr = PredRecipe->getUnderlyingInstr();
  // Build the triangular if-then region.
  std::string RegionName = (Twine("pred.") + Instr->getOpcodeName()).str();
  assert(Instr->getParent() && "Predicated instruction not in any basic block");
  auto *BlockInMask = PredRecipe->getMask();
  auto *BOMRecipe = new TPBranchOnMaskRecipe(BlockInMask);
  auto *Entry = new TPBasicBlock(Twine(RegionName) + ".entry", BOMRecipe);

  // Replace predicated replicate recipe with a replicate recipe without a
  // mask but in the replicate region.
  auto *RecipeWithoutMask = new TPReplicateRecipe(
      PredRecipe->getUnderlyingInstr(),
      make_range(PredRecipe->op_begin(), std::prev(PredRecipe->op_end())),
      PredRecipe->isUniform());
  auto *Pred = new TPBasicBlock(Twine(RegionName) + ".if", RecipeWithoutMask);

  TPPredInstPHIRecipe *PHIRecipe = nullptr;
  if (PredRecipe->getNumUsers() != 0) {
    PHIRecipe = new TPPredInstPHIRecipe(RecipeWithoutMask);
    PredRecipe->replaceAllUsesWith(PHIRecipe);
    PHIRecipe->setOperand(0, RecipeWithoutMask);
  }
  PredRecipe->eraseFromParent();
  auto *Exiting = new TPBasicBlock(Twine(RegionName) + ".continue", PHIRecipe);
  TPRegionBlock *Region =
      new TPRegionBlock(Entry, Exiting, {{}}, {{}}, RegionName, true);

  // Note: first set Entry as region entry and then connect successors starting
  // from it in order, to propagate the "parent" of each VPBasicBlock.
  TPBlockUtils::insertTwoBlocksAfter(Pred, Exiting, Entry);
  TPBlockUtils::connectBlocks(Pred, Exiting);

  return Region;
}

static void addReplicateRegions(TPlan &Plan) {
  SmallVector<TPReplicateRecipe *> WorkList;
  for (TPBasicBlock *TPBB : TPBlockUtils::blocksOnly<TPBasicBlock>(
           tp_depth_first_deep(Plan.getEntry()))) {
    for (TPRecipeBase &R : *TPBB)
      if (auto *RepR = dyn_cast<TPReplicateRecipe>(&R)) {
        if (RepR->isPredicated())
          WorkList.push_back(RepR);
      }
  }

  unsigned BBNum = 0;
  for (TPReplicateRecipe *RepR : WorkList) {
    TPBasicBlock *CurrentBlock = RepR->getParent();
    TPBasicBlock *SplitBlock = CurrentBlock->splitAt(RepR->getIterator());

    BasicBlock *OrigBB = RepR->getUnderlyingInstr()->getParent();
    SplitBlock->setName(
        OrigBB->hasName() ? OrigBB->getName() + "." + Twine(BBNum++) : "");
    // Record predicated instructions for above packing optimizations.
    TPBlockBase *Region = createReplicateRegion(RepR, Plan);
    Region->setParent(CurrentBlock->getParent());
    TPBlockUtils::disconnectBlocks(CurrentBlock, SplitBlock);
    TPBlockUtils::connectBlocks(CurrentBlock, Region);
    TPBlockUtils::connectBlocks(Region, SplitBlock);
  }
}

/// Remove redundant VPBasicBlocks by merging them into their predecessor if
/// the predecessor has a single successor.
static bool mergeBlocksIntoPredecessors(TPlan &Plan) {
  SmallVector<TPBasicBlock *> WorkList;
  for (TPBasicBlock *TPBB : TPBlockUtils::blocksOnly<TPBasicBlock>(
           tp_depth_first_deep(Plan.getEntry()))) {
    // Don't fold the exit block of the Plan into its single predecessor for
    // now.
    // TODO: Remove restriction once more of the skeleton is modeled in VPlan.
    if (TPBB->getNumSuccessors() == 0 && !TPBB->getParent())
      continue;
    auto *PredVPBB =
        dyn_cast_or_null<TPBasicBlock>(TPBB->getSinglePredecessor());
    if (!PredVPBB || PredVPBB->getNumSuccessors() != 1)
      continue;
    WorkList.push_back(TPBB);
  }

  for (TPBasicBlock *TPBB : WorkList) {
    TPBasicBlock *PredTPBB = cast<TPBasicBlock>(TPBB->getSinglePredecessor());
    for (TPRecipeBase &R : make_early_inc_range(*TPBB))
      R.moveBefore(*PredTPBB, PredTPBB->end());
    TPBlockUtils::disconnectBlocks(PredTPBB, TPBB);
    auto *ParentRegion = cast_or_null<TPRegionBlock>(TPBB->getParent());
    if (ParentRegion && ParentRegion->getExiting() == TPBB)
      ParentRegion->setExiting(PredTPBB);
    for (auto *Succ : to_vector(TPBB->successors())) {
      TPBlockUtils::disconnectBlocks(TPBB, Succ);
      TPBlockUtils::connectBlocks(PredTPBB, Succ);
    }
    delete TPBB;
  }
  return !WorkList.empty();
}

void TPlanTransforms::createAndOptimizeReplicateRegions(TPlan &Plan) {
  // Convert masked VPReplicateRecipes to if-then region blocks.
  addReplicateRegions(Plan);

  bool ShouldSimplify = true;
  while (ShouldSimplify) {
    ShouldSimplify = sinkScalarOperands(Plan);
    ShouldSimplify |= mergeReplicateRegionsIntoSuccessors(Plan);
    ShouldSimplify |= mergeBlocksIntoPredecessors(Plan);
  }
}

/// Remove redundant casts of inductions.
///
/// Such redundant casts are casts of induction variables that can be ignored,
/// because we already proved that the casted phi is equal to the uncasted phi
/// in the vectorized loop. There is no need to vectorize the cast - the same
/// value can be used for both the phi and casts in the vector loop.
static void removeRedundantInductionCasts(TPlan &Plan) {
  for (auto &[Dim, HeaderTPBB] : Plan.LoopIdx2HeaderTPBB) {
    for (auto &Phi : HeaderTPBB->phis()) {
      auto *IV = dyn_cast<TPWidenIntOrFpInductionRecipe>(&Phi);
      if (!IV || IV->getTruncInst())
        continue;

      // When IV is widened, the cast chain is bypassed.
      // getCastInsts() returns casts innermost-first; walk outward to find the
      // last cast recipe in the chain and replace its uses with IV.
      ArrayRef<Instruction *> Casts =
          IV->getInductionDescriptor().getCastInsts();
      if (Casts.empty())
        continue;

      TPValue *LastCastRecipe = IV;
      for (Instruction *IRCast : reverse(Casts)) {
        auto *It = llvm::find_if(LastCastRecipe->users(), [IRCast](TPUser *U) {
          auto *UserCast = dyn_cast<TPSingleDefRecipe>(U);
          return UserCast && UserCast->getUnderlyingValue() == IRCast;
        });
        if (It == LastCastRecipe->users().end())
          break;
        LastCastRecipe = cast<TPSingleDefRecipe>(*It);
      }
      if (LastCastRecipe != IV)
        LastCastRecipe->replaceAllUsesWith(IV);
    }
  }
}

/// Try to replace VPWidenCanonicalIVRecipes with a widened canonical IV
/// recipe, if it exists.
static void removeRedundantCanonicalIVs(TPlan &Plan) {
  for (auto &[Dim, HeaderTPBB] : Plan.LoopIdx2HeaderTPBB) {
    if (HeaderTPBB->empty())
      continue;

    auto *FirstRecipe = &*HeaderTPBB->begin();
    auto *CanonicalIV = dyn_cast<TPCanonicalIVPHIRecipe>(FirstRecipe);
    if (!CanonicalIV)
      continue;

    // Find TPWidenCanonicalIVRecipe that uses CanonicalIV
    TPWidenCanonicalIVRecipe *WidenNewIV = nullptr;
    for (TPUser *U : CanonicalIV->users()) {
      WidenNewIV = dyn_cast<TPWidenCanonicalIVRecipe>(U);
      if (WidenNewIV)
        break;
    }
    if (!WidenNewIV)
      continue;

    // Find canonical TPWidenIntOrFpInductionRecipe in the same header
    for (TPRecipeBase &Phi : HeaderTPBB->phis()) {
      auto *WidenOriginalIV = dyn_cast<TPWidenIntOrFpInductionRecipe>(&Phi);
      if (!WidenOriginalIV || !WidenOriginalIV->isCanonical())
        continue;

      // Check if replacement is beneficial
      bool ShouldReplace = false;

      // Check if any user of WidenOriginalIV does not use scalars
      for (TPUser *U : WidenOriginalIV->users()) {
        if (!U->usesScalars(WidenOriginalIV)) {
          ShouldReplace = true;
          break;
        }
      }

      // Or if WidenNewIV only uses the first lane
      if (!ShouldReplace && tputils::onlyFirstLaneUsed(WidenNewIV)) {
        ShouldReplace = true;
      }

      if (ShouldReplace) {
        WidenNewIV->replaceAllUsesWith(WidenOriginalIV);
        WidenNewIV->eraseFromParent();
        break;
      }
    }
  }
}

static TPScalarIVStepsRecipe *createScalarIVSteps(
    TPlan &Plan, InductionDescriptor::InductionKind Kind,
    Instruction::BinaryOps InductionOpcode, FPMathOperator *FPBinOp,
    ScalarEvolution &SE, Instruction *TruncI, TPValue *StartV, TPValue *Step,
    TPBasicBlock::iterator IP, TPBasicBlock *HeaderTPBB, Loop *L) {
  TPCanonicalIVPHIRecipe *CanonicalIV =
      cast<TPCanonicalIVPHIRecipe>(&*HeaderTPBB->begin());
  TPSingleDefRecipe *BaseIV = CanonicalIV;

  if (!CanonicalIV->isCanonical(Kind, StartV, Step)) {
    BaseIV = new TPDerivedIVRecipe(Kind, FPBinOp, StartV, CanonicalIV, Step);
    HeaderTPBB->insert(BaseIV, IP);
  }

  // Truncate base induction if needed.
  TPTypeAnalysis TypeInfo(
      cast<TPCanonicalIVPHIRecipe>(&*HeaderTPBB->begin())->getScalarType(),
      SE.getContext());
  Type *ResultTy = TypeInfo.inferScalarType(BaseIV);
  if (TruncI) {
    Type *TruncTy = TruncI->getType();
    assert(ResultTy->getScalarSizeInBits() > TruncTy->getScalarSizeInBits() &&
           "Not truncating.");
    assert(ResultTy->isIntegerTy() && "Truncation requires an integer type");
    BaseIV = new TPScalarCastRecipe(Instruction::Trunc, BaseIV, TruncTy);
    HeaderTPBB->insert(BaseIV, IP);
    ResultTy = TruncTy;
  }

  // Truncate step if needed.
  Type *StepTy = TypeInfo.inferScalarType(Step);
  if (ResultTy != StepTy) {
    assert(StepTy->getScalarSizeInBits() > ResultTy->getScalarSizeInBits() &&
           "Not truncating.");
    assert(StepTy->isIntegerTy() && "Truncation requires an integer type");
    Step = new TPScalarCastRecipe(Instruction::Trunc, Step, ResultTy);
    auto *TensorPreheader =
        cast<TPBasicBlock>(HeaderTPBB->getSingleHierarchicalPredecessor());
    TensorPreheader->appendRecipe(Step->getDefiningRecipe());
  }

  TPScalarIVStepsRecipe *Steps = new TPScalarIVStepsRecipe(
      BaseIV, Step, InductionOpcode,
      FPBinOp ? FPBinOp->getFastMathFlags() : FastMathFlags(), L);
  HeaderTPBB->insert(Steps, IP);
  return Steps;
}

/// Returns true if \p R is dead and can be removed.
static bool isDeadRecipe(TPRecipeBase &R) {
  using namespace llvm::PatternMatch;
  // Do remove conditional assume instructions as their conditions may be
  // flattened.
  auto *RepR = dyn_cast<TPReplicateRecipe>(&R);
  bool IsConditionalAssume =
      RepR && RepR->isPredicated() &&
      match(RepR->getUnderlyingInstr(), m_Intrinsic<Intrinsic::assume>());
  if (IsConditionalAssume)
    return true;

  if (R.mayHaveSideEffects())
    return false;

  // Recipe is dead if no user keeps the recipe alive.
  return all_of(R.definedValues(),
                [](TPValue *V) { return V->getNumUsers() == 0; });
}

static void removeDeadRecipes(TPlan &Plan) {
  ReversePostOrderTraversal<TPBlockDeepTraversalWrapper<TPBlockBase *>> RPOT(
      Plan.getEntry());

  for (TPBasicBlock *TPBB :
       reverse(TPBlockUtils::blocksOnly<TPBasicBlock>(RPOT))) {
    // The recipes in the block are processed in reverse order, to catch chains
    // of dead recipes.
    for (TPRecipeBase &R : make_early_inc_range(reverse(*TPBB))) {
      if (isDeadRecipe(R))
        R.eraseFromParent();
    }
  }
}

/// Returns true if \p R can be pruned because all of its defined values are
/// only used by subsumed recipes (which will be no-ops at execute() time).
/// Subsumed recipes themselves are never prunable — they stay in the plan.
static bool isPrunableConsideringSubsumed(const TPRecipeBase &R) {
  if (R.isSubsumed())
    return false; // subsumed recipes stay in the plan
  if (R.mayHaveSideEffects())
    return false;
  if (R.mayReadOrWriteMemory())
    return false;
  return all_of(R.definedValues(), [](const TPValue *V) {
    // vacuously true if no users; prunable if every user is subsumed
    return all_of(V->users(), [](const TPUser *U) {
      const auto *Recipe = dyn_cast<TPRecipeBase>(U);
      return Recipe && Recipe->isSubsumed();
    });
  });
}

/// Remove recipes whose only consumers are subsumed recipes.
/// Uses the same reverse-RPOT scan as removeDeadRecipes so that erasing an
/// inner-loop def propagates up: once the user count drops to zero the outer
/// def also becomes prunable in the same pass.
static void pruneSubsumedCrossLoopDefs(TPlan &Plan) {
  // Fast path: skip if no subsumed recipes exist
  bool AnySubsumed = false;
  {
    ReversePostOrderTraversal<TPBlockDeepTraversalWrapper<TPBlockBase *>> Check(
        Plan.getEntry());
    for (auto *TPBB : TPBlockUtils::blocksOnly<TPBasicBlock>(Check)) {
      for (TPRecipeBase &R : *TPBB) {
        if (R.isSubsumed()) {
          AnySubsumed = true;
          break;
        }
      }
      if (AnySubsumed)
        break;
    }
  }
  if (!AnySubsumed)
    return;

  // Post-order scan (reverse RPOT): inner blocks before outer blocks.
  // Erasing an inner recipe reduces its outer def's user count → chain removal.
  ReversePostOrderTraversal<TPBlockDeepTraversalWrapper<TPBlockBase *>> RPOT(
      Plan.getEntry());
  SmallVector<TPBasicBlock *, 16> Blocks(
      TPBlockUtils::blocksOnly<TPBasicBlock>(RPOT).begin(),
      TPBlockUtils::blocksOnly<TPBasicBlock>(RPOT).end());
  for (auto *TPBB : reverse(Blocks)) {
    SmallVector<TPRecipeBase *, 8> Recipes;
    for (TPRecipeBase &R : *TPBB)
      Recipes.push_back(&R);
    for (auto *R : reverse(Recipes)) {
      if (isPrunableConsideringSubsumed(*R))
        R->eraseFromParent();
    }
  }
}

/// Returns true if \p TPBB is (transitively) inside \p TargetRegion.
/// Walks up the parent-region chain from TPBB.
static bool isInsideRegion(TPBasicBlock *TPBB, TPRegionBlock *TargetRegion) {
  for (TPRegionBlock *Parent = TPBB->getParent(); Parent;
       Parent = Parent->getParent())
    if (Parent == TargetRegion)
      return true;
  return false;
}

/// Hoist loop-invariant recipes from the loop body to the loop preheader.
/// Processes dimensions innermost-first so that recipes hoisted to an inner
/// preheader become candidates for hoisting in the outer loop in the next
/// iteration.
static void hoistLoopInvariantRecipes(TPlan &Plan) {
  // Process innermost (Dim=0) first → outermost (Dim=N-1) last
  SmallVector<unsigned, 4> Dims;
  for (auto &[Dim, _] : Plan.LoopIdx2TPRB)
    Dims.push_back(Dim);
  llvm::sort(Dims); // ascending = innermost first

  for (unsigned Dim : Dims) {
    auto *RegIt = Plan.LoopIdx2TPRB.find(Dim);
    auto *PreIt = Plan.LoopIdx2PreHeaderTPBB.find(Dim);
    if (RegIt == Plan.LoopIdx2TPRB.end() ||
        PreIt == Plan.LoopIdx2PreHeaderTPBB.end())
      continue;
    TPRegionBlock *Region = RegIt->second;
    TPBasicBlock *PreHdrTPBB = PreIt->second;

    auto *HeaderIt = Plan.LoopIdx2HeaderTPBB.find(Dim);
    if (HeaderIt == Plan.LoopIdx2HeaderTPBB.end())
      continue;
    TPBasicBlock *HeaderTPBB = HeaderIt->second;

    // Shallow traversal: only direct-child TPBasicBlocks (skip nested regions)
    ReversePostOrderTraversal<TPBlockShallowTraversalWrapper<TPBlockBase *>>
        ShallowRPOT(
            TPBlockShallowTraversalWrapper<TPBlockBase *>(Region->getEntry()));

    for (TPBasicBlock *TPBB :
         TPBlockUtils::blocksOnly<TPBasicBlock>(ShallowRPOT)) {
      if (TPBB == static_cast<TPBasicBlock *>(HeaderTPBB))
        continue;

      for (TPRecipeBase &R : make_early_inc_range(*TPBB)) {
        if (R.mayHaveSideEffects())
          continue;
        if (R.mayReadOrWriteMemory())
          continue;
        // PHI recipes must stay in the header
        if (isa<TPHeaderPHIRecipe>(R))
          continue;

        // Recipe is invariant if all operands are defined outside this loop
        auto IsOutside = [&](TPValue *Op) -> bool {
          auto *Def = dyn_cast_or_null<TPRecipeBase>(Op->getDefiningRecipe());
          if (!Def)
            return true; // IR live-in value
          TPBasicBlock *DefBB = Def->getParent();
          return DefBB == PreHdrTPBB ||        // already hoisted here
                 !isInsideRegion(DefBB, Region); // defined in outer scope
        };

        if (all_of(R.operands(), IsOutside))
          R.moveBefore(*PreHdrTPBB, PreHdrTPBB->end());
      }
    }
  }
}

/// Legalize VPWidenPointerInductionRecipe, by replacing it with a PtrAdd
/// (IndStart, ScalarIVSteps (0, Step)) if only its scalar values are used, as
/// VPWidenPointerInductionRecipe will generate vectors only. If some users
/// require vectors while other require scalars, the scalar uses need to extract
/// the scalars from the generated vectors (Note that this is different to how
/// int/fp inductions are handled). Also optimize VPWidenIntOrFpInductionRecipe,
/// if any of its users needs scalar values, by providing them scalar steps
/// built on the canonical scalar IV and update the original IV's users. This is
/// an optional optimization to reduce the needs of vector extracts.
static void legalizeAndOptimizeInductions(TPlan &Plan, ScalarEvolution &SE) {
  bool HasOnlyVectorVFs = !Plan.hasScalarTFOnly();

  for (auto &[Dim, HeaderTPBB] : Plan.LoopIdx2HeaderTPBB) {
    auto *LoopIt = Plan.LoopIdx2Loop.find(Dim);
    Loop *L = (LoopIt != Plan.LoopIdx2Loop.end()) ? LoopIt->second : nullptr;
    if (!L)
      continue;
    TPBasicBlock::iterator InsertPt = HeaderTPBB->getFirstNonPhi();

    for (TPRecipeBase &Phi : HeaderTPBB->phis()) {
      // TPWidenPointerInductionRecipe: not supported in TPlan — skip
      auto *WideIV = dyn_cast<TPWidenIntOrFpInductionRecipe>(&Phi);
      if (!WideIV)
        continue;

      // If there are no scalar users and we are only vectorizing, no scalar
      // steps are needed.
      if (HasOnlyVectorVFs &&
          none_of(WideIV->users(), [WideIV](TPUser *U) {
            return U->usesScalars(WideIV);
          }))
        continue;

      const InductionDescriptor &ID = WideIV->getInductionDescriptor();
      TPScalarIVStepsRecipe *Steps = createScalarIVSteps(
          Plan, ID.getKind(), ID.getInductionOpcode(),
          dyn_cast_or_null<FPMathOperator>(ID.getInductionBinOp()), SE,
          WideIV->getTruncInst(), WideIV->getStartValue(),
          WideIV->getStepValue(), InsertPt, HeaderTPBB, L);

      // For scalar-only plans: replace all uses of WideIV with scalar steps.
      // For vector+scalar plans: replace only scalar users (bug fix).
      if (!HasOnlyVectorVFs)
        WideIV->replaceAllUsesWith(Steps);
      else
        WideIV->replaceUsesWithIf(Steps, [WideIV](TPUser &U, unsigned) {
          return U.usesScalars(WideIV);
        });
    }
  }
}

/// Remove redundant TPExpandSCEVRecipes across all levels of \p Plan
/// (global entry preheader and all loop preheaders) by replacing duplicates
/// with the first-seen recipe expanding the same SCEV expression.
static void removeRedundantExpandSCEVRecipesAllLevels(TPlan &Plan) {
  DenseMap<const SCEV *, TPValue *> SCEV2VPV;

  // Helper: dedup TPExpandSCEVRecipes in a single block
  auto Dedup = [&](TPBasicBlock *BB) {
    for (TPRecipeBase &R : make_early_inc_range(*BB)) {
      auto *ExpR = dyn_cast<TPExpandSCEVRecipe>(&R);
      if (!ExpR)
        continue;
      auto [It, Inserted] = SCEV2VPV.insert({ExpR->getSCEV(), ExpR});
      if (!Inserted) {
        ExpR->replaceAllUsesWith(It->second);
        ExpR->eraseFromParent();
      }
    }
  };

  // 1) Global entry preheader (outside all loops)
  Dedup(Plan.getEntry()->getEntryBasicBlock());

  // 2) Loop preheaders: outermost (Dim=N-1) → innermost (Dim=0)
  //    Outer preheaders dominate inner ones, so register outer values first
  //    to avoid use-before-def when deduplicating inner duplicates.
  //    MapVector insertion order may not match dim order, so sort explicitly.
  SmallVector<unsigned, 4> Dims;
  for (auto &[Dim, _] : Plan.LoopIdx2PreHeaderTPBB)
    Dims.push_back(Dim);
  llvm::sort(Dims, std::greater<unsigned>());

  for (unsigned D : Dims)
    Dedup(Plan.LoopIdx2PreHeaderTPBB[D]);
}

/// Try to simplify recipe \p R.
static void simplifyRecipe(TPRecipeBase &R, TPTypeAnalysis &TypeInfo) {
  using namespace llvm::TPlanPatternMatch;

  // Pattern 1: Redundant Blend — all incoming values are the same, replace with Inc0
  if (auto *Blend = dyn_cast<TPBlendRecipe>(&R)) {
    TPValue *Inc0 = Blend->getIncomingValue(0);
    for (unsigned I = 1; I != Blend->getNumIncomingValues(); ++I)
      if (Inc0 != Blend->getIncomingValue(I) &&
          !match(Blend->getMask(I), m_False()))
        return;
    Blend->replaceAllUsesWith(Inc0);
    Blend->eraseFromParent();
    return;
  }

  // Pattern 2: Trunc(ZExtOrSExt(A)) simplification
  TPValue *A;
  if (match(&R, m_Trunc(m_ZExtOrSExt(m_TPValue(A))))) {
    TPValue *Trunc = R.getTPSingleValue();
    Type *TruncTy = TypeInfo.inferScalarType(Trunc);
    Type *ATy = TypeInfo.inferScalarType(A);
    if (TruncTy == ATy) {
      Trunc->replaceAllUsesWith(A);
    } else {
      // Do not replace scalarizing recipes with widened casts
      if (isa<TPReplicateRecipe>(&R))
        return;

      if (ATy->getScalarSizeInBits() < TruncTy->getScalarSizeInBits()) {
        unsigned ExtOpcode = match(R.getOperand(0), m_SExt(m_TPValue()))
                                 ? Instruction::SExt
                                 : Instruction::ZExt;
        auto *TPC =
            new TPWidenCastRecipe(Instruction::CastOps(ExtOpcode), A, TruncTy);
        if (auto *UnderlyingExt = R.getOperand(0)->getUnderlyingValue())
          TPC->setUnderlyingValue(UnderlyingExt);
        TPC->insertBefore(&R);
        Trunc->replaceAllUsesWith(TPC);
      } else if (ATy->getScalarSizeInBits() > TruncTy->getScalarSizeInBits()) {
        auto *VPC = new TPWidenCastRecipe(Instruction::Trunc, A, TruncTy);
        VPC->insertBefore(&R);
        Trunc->replaceAllUsesWith(VPC);
      }
    }
    return;
  }

  // Pattern 3: (X && Y) || (X && !Y) → X
  TPValue *X, *Y, *X1, *Y1;
  if (match(&R,
            m_c_BinaryOr(m_LogicalAnd(m_TPValue(X), m_TPValue(Y)),
                         m_LogicalAnd(m_TPValue(X1), m_Not(m_TPValue(Y1))))) &&
      X == X1 && Y == Y1) {
    R.getTPSingleValue()->replaceAllUsesWith(X);
    return;
  }

  // Pattern 4: A * 1 → A
  if (match(&R, m_c_Mul(m_TPValue(A), m_SpecificInt(1))))
    R.getTPSingleValue()->replaceAllUsesWith(A);
}

/// Try to simplify the recipes in \p Plan.
static void simplifyRecipes(TPlan &Plan, LLVMContext &Ctx) {
  ReversePostOrderTraversal<TPBlockDeepTraversalWrapper<TPBlockBase *>> RPOT(
      Plan.getEntry());
  TPTypeAnalysis TypeInfo(Plan.getCanonicalIV()->getScalarType(), Ctx);
  for (TPBasicBlock *TPBB : TPBlockUtils::blocksOnly<TPBasicBlock>(RPOT)) {
    for (TPRecipeBase &R : make_early_inc_range(*TPBB)) {
      simplifyRecipe(R, TypeInfo);
    }
  }
}

void TPlanTransforms::markSubsumedRecipes(TPBasicBlock *Body,
                                          unsigned TilingDim) {
  // A recipe is subsumed when the tensor intrinsic replaces it entirely.
  // Rules are dim-aware so the function works for any tiling dimension, not
  // just the GEMM K-loop (innermost, dim 0).
  for (TPRecipeBase &R : *Body) {
    TensorOpKind Kind = R.getTensorOpKind();
    switch (R.getTPDefID()) {

    case TPRecipeBase::TPWidenLoadSC:
    case TPRecipeBase::TPWidenStoreSC:
      // Subsume memory ops that belong to the loop being tiled.
      // Loads/stores on other dimensions are not replaced by this intrinsic.
      if (R.getDimIndex() == static_cast<int>(TilingDim))
        R.setSubsumed(true);
      break;

    case TPRecipeBase::TPWidenSC:
      // Subsume all arithmetic except the Contraction and PlainReduction
      // anchors — those must survive so execute() can emit the intrinsic or
      // the scalar reduction update respectively.
      if (Kind != TensorOpKind::Contraction &&
          Kind != TensorOpKind::PlainReduction) {
        R.setSubsumed(true);
      } else if (Kind == TensorOpKind::Contraction) {
        assert(R.getContractDim() == static_cast<int>(TilingDim) &&
               "Contraction recipe's reduction dim must match the tiling dim");
      }
      break;

    case TPRecipeBase::TPWidenIntOrFpInductionSC:
      // Subsume only the IV for the loop being tiled; IVs of other loops
      // must survive so their execute() registers the PHINode in ValueMap.
      // TPWidenPointerInductionSC is intentionally excluded: its execute()
      // unconditionally writes to ValueMap regardless of IsSubsumed.
      if (R.getDimIndex() == static_cast<int>(TilingDim))
        R.setSubsumed(true);
      break;

    default:
      // WIDEN-GEP (tile-corner pointer), ReductionPHI, and unknown recipes
      // are kept — they must emit IR.
      break;
    }
  }
}

TPBasicBlock *TPlanTransforms::buildScalarEpilogue(TPBasicBlock *Body) {
  // Clone the K-loop body for scalar K%PF remainder iterations.
  // Each cloned recipe has IsSubsumed=false so execute() always emits IR.
  // The clones share the same IR Instruction pointers as the originals;
  // remapClone() + EmittedMap[OrigKIVPhi]=ScIV fixes up operands at
  // execute()-time to use the scalar induction variable instead of the
  // original K-loop PHI.
  auto *Epi = new TPBasicBlock("scalar.epilogue");
  for (TPRecipeBase &R : *Body) {
    TPRecipeBase *C = R.clone();
    assert(!C->isSubsumed() && "clone() must not propagate IsSubsumed=true");
    Epi->appendRecipe(C);
  }
  // Plan.addCreatedBlock(Epi);
  // Scalar block에 넣어야 할 듯. 
  // TPlan2TPlan Transformation Lowering을 신경써야 할까?
  // 이 함수는 독립적으로 그런 함수이긴해..
  return Epi;
}

TPTilingRegion *TPlanTransforms::replaceWithTilingRegion(
    TPRegionBlock *Innermost, const DimEmissionSpec &Spec) {
  // The Entry block (ir-bb<K.header>) holds all body recipes for loops
  // where header = latch = body (the common single-block K-loop in GEMM).

  // TODO(yg0412.yun) Need to merge TPBBs inside TPRegionBlock
  // which has single successor only.
  TPBlockBase *Cur = Innermost->getEntry();
  while (Cur) {
    if (auto *TPBB = dyn_cast<TPBasicBlock>(Cur)) {
      markSubsumedRecipes(TPBB, Spec.Dim);

      TPBasicBlock *Epilogue =
          (Spec.Mode == DimEmitMode::DynamicTiled) ? buildScalarEpilogue(TPBB)
                                                    : nullptr;
      // for (TPRecipeBase &R : *TPBB) {
      // }
    }
    Cur = Cur->getSingleSuccessor(); // 단일 successor인 경우
  }

  // // TPlan 전체를 deep 순회하며 TPBasicBlock만 필터링
  // for (TPBasicBlock *TPBB : TPBlockUtils::blocksOnly<TPBasicBlock>(
  //         tp_depth_first_deep(Innermost->getEntry()))) {
  //   // YYG::REMOVE
  //   errs() << "[replaceWithTilingRegion] Innermost-TPBB:\n";
  //   TPBB->dump();
  //   markSubsumedRecipes(TPBB, Spec.Dim);

  //   // TPBB 순회
  //   // for (TPRecipeBase &R : *TPBB) {
  //   //   // R: 각 recipe 처리
  //   // }
  // }

  // TPBasicBlock *Epilogue =
  //     (Spec.Mode == DimEmitMode::DynamicTiled) ? buildScalarEpilogue(TPBB)
  //                                               : nullptr;

  // Locate the K-loop IV PHINode for EmittedMap registration in execute().
  // PHINode *KIVPhi = nullptr;
  // for (TPRecipeBase &R : *InnerHeaderBody) {
  //   if (auto *IV = dyn_cast<TPWidenRecipe>(&R)) {
  //     // YYG::REMOVE
  //     errs() << "[replaceWithTilingRegion] IV: " << *IV << "\n";

  //     if (IV->getDimIndex() == Spec.Dim) {
  //       // KIVPhi = IV->getIVPhi();
  //       // YYG::REMOVE
  //       errs() << "[replaceWithTilingRegion] R: \n";
  //       R.dump();
  //       break;
  //     }
  //   }
  // }

  // assert(KIVPhi && "TPlanTransformer: no IV recipe found for tiling dim");

  // auto *TR = new TPTilingRegion(Spec.Dim, Spec.PF, Spec.Mode, Body, Epilogue,
  //                                KIVPhi);
  // Plan.addCreatedBlock(TR);

  // // Install the tiling override — TPRegionBlock::execute() will delegate to TR.
  // Innermost->setTilingOverride(TR);
  // return TR;

  // Stub implementation - returning nullptr for now
  return nullptr;
}

void TPlanTransforms::transform(TPTransformState &State) {
  // Pick the first tiling spec (StaticTiled or DynamicTiled).
  const DimEmissionSpec *TilingSpec = nullptr;
  for (const auto &S : Policy.Specs)
    if (S.Mode != DimEmitMode::Inline)
      TilingSpec = &S;
  if (!TilingSpec)
    return; // Inline only — no tiling needed.

  const SCEV *TCSCEV = Plan.getTCForDim(TilingSpec->Dim);
  if (!TCSCEV)
    return;

  // Expand the backedge-taken count SCEV before the outermost loop so it
  // dominates all newly created tiling blocks.
  BasicBlock *InsertBB = nullptr;
  if (OutermostLoop)
    if (BasicBlock *PH = OutermostLoop->getLoopPreheader())
      InsertBB = PH->getSinglePredecessor();
  if (!InsertBB)
    return;

  // TC = BTC + 1 (SCEV stores backedge-taken count, not trip count).
  Value *BTC =
      Expander.expandCodeFor(TCSCEV,  Type::getInt64Ty(InsertBB->getContext()),
                             InsertBB->getTerminator());
  IRBuilder<> PredB(InsertBB->getTerminator());
  Value *TCVal = PredB.CreateAdd(BTC, PredB.getInt64(1), "tc.tiling");
  State.TilingTCVal = TCVal;

  // Replace the region for the tiling dim with a TPTilingRegion.
  // Use TilingSpec->Dim (not hardcoded 0) so this works for any loop depth,
  // not just GEMM's innermost K-loop.
  TPRegionBlock *TargetRegion = Plan.LoopIdx2TPRB[TilingSpec->Dim];
  assert(TargetRegion && "LoopIdx2TPRB must contain an entry for every tiling dim");
  if (!TargetRegion)
    return;
  replaceWithTilingRegion(TargetRegion, *TilingSpec);

  // // For DynamicTiled dims, insert a runtime profitability guard.
  // if (TilingSpec->Mode == DimEmitMode::DynamicTiled)
  //   insertGuardBlock(*TilingSpec, TCVal);

  // LLVM_DEBUG(dbgs() << "TPlanTransformer: transformed Plan for dim="
  //                   << TilingSpec->Dim << " PF=" << TilingSpec->PF << "\n");
}

void TPlanTransforms::optimize(TPlan &Plan, ScalarEvolution &SE) {
  // [B] VPlan-derived passes — adapted for nested-loop TPlan
  removeRedundantCanonicalIVs(Plan);
  removeRedundantInductionCasts(Plan);
  simplifyRecipes(Plan, SE.getContext());
  legalizeAndOptimizeInductions(Plan, SE);
  removeDeadRecipes(Plan);

  // VPlan-derived pass — deep traversal keeps nested-loop compatibility
  createAndOptimizeReplicateRegions(Plan);

  // [B→C] Extended to all loop preheaders (outermost→innermost)
  removeRedundantExpandSCEVRecipesAllLevels(Plan);

  // [C] Nested-loop specific passes
  hoistLoopInvariantRecipes(Plan);
  pruneSubsumedCrossLoopDefs(Plan);

  // VPlan-derived pass — kept as-is
  mergeBlocksIntoPredecessors(Plan);
}

void TPlanTransforms::optimizeForTFAndUF(TPlan &Plan, ScalarEvolution &SE) {
  // assert(Plan.hasTF() && "BestTF is not avialable in Plan.");
  // assert(Plan.hasUF() && "BestUF is not avialable in Plan");

  // bool MadeChange = tryToReplaceALMWithWideALM(Plan, BestVF, BestUF);
  // MadeChange |= simplifyBranchConditionForVFAndUF(Plan, BestVF, BestUF, PSE);
  // MadeChange |= optimizeVectorInductionWidthForTCAndVFUF(Plan, BestVF, BestUF);
  // MadeChange |= simplifyKnownEVL(Plan, BestVF, PSE);

  // if (MadeChange) {
  //   Plan.setTF(BestVF);
  //   assert(Plan.getUF() == BestUF && "BestUF must match the Plan's UF");
  // }
  llvm_unreachable("Please implement TPlanTransforms::optimizeForTFAndUF!\n");
}
