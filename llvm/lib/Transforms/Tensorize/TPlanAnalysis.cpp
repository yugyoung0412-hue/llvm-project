#include "llvm/Transforms/Tensorize/TPlanAnalysis.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Tensorize/TPlan.h"
#include "llvm/Transforms/Tensorize/TPlanCFG.h"

using namespace llvm;

#define DEBUG_TYPE "tplan"

Type *TPTypeAnalysis::inferScalarTypeForRecipe(const TPBlendRecipe *R) {
  llvm_unreachable("");
}

Type *TPTypeAnalysis::inferScalarTypeForRecipe(const TPInstruction *R) {
  llvm_unreachable("");
}

Type *TPTypeAnalysis::inferScalarTypeForRecipe(const TPWidenRecipe *R) {
  llvm_unreachable("");
}

Type *TPTypeAnalysis::inferScalarTypeForRecipe(const TPWidenCallRecipe *R) {
  llvm_unreachable("");
}

Type *TPTypeAnalysis::inferScalarTypeForRecipe(const TPWidenMemoryRecipe *R) {
  llvm_unreachable("");
}

Type *TPTypeAnalysis::inferScalarTypeForRecipe(const TPWidenSelectRecipe *R) {
  llvm_unreachable("");
}

Type *TPTypeAnalysis::inferScalarTypeForRecipe(const TPReplicateRecipe *R) {
  llvm_unreachable("");
}

Type *TPTypeAnalysis::inferScalarType(const TPValue *V) {

  if (Type *CachedTy = CachedTypes.lookup(V))
    return CachedTy;

  if (V->isLiveIn()) {
    if (auto *IRValue = V->getLiveInIRValue())
      return IRValue->getType();
    // All VPValues without any underlying IR value (like the vector trip count
    // or the backedge-taken count) have the same type as the canonical IV.
    return CanonicalIVTy;
  }

  Type *ResultTy =
      TypeSwitch<const TPRecipeBase *, Type *>(V->getDefiningRecipe())
          .Case<TPActiveLaneMaskPHIRecipe, TPCanonicalIVPHIRecipe,
                TPReductionPHIRecipe>([this](const auto *R) {
            // Handle header phi recipes, except VPWidenIntOrFpInduction
            // which needs special handling due it being possibly truncated.
            // TODO: consider inferring/caching type of siblings, e.g.,
            // backedge value, here and in cases below.
            return inferScalarType(R->getStartValue());
          })
          .Case<TPWidenIntOrFpInductionRecipe, TPDerivedIVRecipe>(
              [](const auto *R) { return R->getScalarType(); })
          .Case<TPPredInstPHIRecipe, TPScalarIVStepsRecipe, TPWidenGEPRecipe,
                TPVectorPointerRecipe, TPWidenCanonicalIVRecipe>(
              [this](const TPRecipeBase *R) {
                return inferScalarType(R->getOperand(0));
              })
          .Case<TPBlendRecipe, TPInstruction, TPWidenRecipe, TPReplicateRecipe,
                TPWidenCallRecipe, TPWidenMemoryRecipe, TPWidenSelectRecipe>(
              [this](const auto *R) { return inferScalarTypeForRecipe(R); })
          .Case<TPWidenCastRecipe>(
              [](const TPWidenCastRecipe *R) { return R->getResultType(); })
          .Case<TPScalarCastRecipe>(
              [](const TPScalarCastRecipe *R) { return R->getResultType(); })
          .Case<TPExpandSCEVRecipe>([](const TPExpandSCEVRecipe *R) {
            return R->getSCEV()->getType();
          });

  assert(ResultTy && "could not infer type for the given TPValue");
  CachedTypes[V] = ResultTy;
  return ResultTy;
  // TODO(yuxin.an)
}

void llvm::collectEphemeralRecipesForTPlan(
    TPlan &Plan, DenseSet<TPRecipeBase *> &EphRecipes) {
  // First, collect seed recipes which are operands of assumes.
  SmallVector<TPRecipeBase *> Worklist;
  for (TPBasicBlock *TPBB : TPBlockUtils::blocksOnly<TPBasicBlock>(
           tp_depth_first_deep(Plan.getTensorLoopRegion()->getEntry()))) {
    for (TPRecipeBase &R : *TPBB) {
      auto *RepR = dyn_cast<TPReplicateRecipe>(&R);
      if (!RepR || !match(RepR->getUnderlyingInstr(),
                          PatternMatch::m_Intrinsic<Intrinsic::assume>()))
        continue;
      Worklist.push_back(RepR);
      EphRecipes.insert(RepR);
    }
  }

  // Process operands of candidates in worklist and add them to the set of
  // ephemeral recipes, if they don't have side-effects and are only used by
  // other ephemeral recipes.
  while (!Worklist.empty()) {
    TPRecipeBase *Cur = Worklist.pop_back_val();
    for (TPValue *Op : Cur->operands()) {
      auto *OpR = Op->getDefiningRecipe();
      if (!OpR || OpR->mayHaveSideEffects() || EphRecipes.contains(OpR))
        continue;
      if (any_of(Op->users(), [EphRecipes](TPUser *U) {
            auto *UR = dyn_cast<TPRecipeBase>(U);
            return !UR || !EphRecipes.contains(UR);
          }))
        continue;
      EphRecipes.insert(OpR);
      Worklist.push_back(OpR);
    }
  }
}
