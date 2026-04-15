#include "llvm/Transforms/Tensorize/TPlanDecisionLogic.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Tensorize/TPattern.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

using namespace llvm;

bool TPlanDecisionLogic::checkSubOptimal(TPlanPtr *tplan) {
  if (SubOptimalIdx == SIZE_MAX)
    return true;

  // TODO(yg0412.yun)
  // Need to insert algorithm for checking if this tplan is optimal
  // by cost-model?

  return false;
}

// bool TPlanDecisionLogic::ApplyPattern(SmallVector<TPlanPtr, 4> &TPlans) {
//   // Check if user specify pattern for LoopTensorizer
//   TensorizePattern *rawPtr = pattern.get();
//   if (isa<TargetAutoPattern>(rawPtr)) return false;

//   // FIXME (yg0412.yun)
//   // need to fix passing TPlans[0]
//   TPlanPtr TPlanWithPattern =
//   pattern->tryToBuildTPlanWithTPRecipes(TPlans[0].get()); return true;
// }

bool TPlanDecisionLogic::ISAAwareSearch(TPlanPtr *tplan) {
  // TODO (yg0412.yun)
  // Make ISW-aware search process & explore the search space

  // Traverse the BB



  /// YYG::REMOVE
  // unsigned opsize = TTI.getOp0Size();
  // errs() << "opsize : " << opsize << "\n";


  return false;
}

bool TPlanDecisionLogic::findBestTPlan(SmallVector<TPlanPtr, 4> &TPlans,
                                       bool DoSearch) {
  assert(!TPlans.empty() &&
         "Please, build TPlans first! call LTP.createNestedLoopTPlan()!");
  // check if the current TPlan is optimal among the TPlans
  // if theres, starting from it

  // FIXME (yg0412.yun). current `checkSubOptimal` returns true only for first
  // tplan of TPlnas. Because current LTP.TPlans only has single elements from
  // createNestedLoopTPlan(). need to fix this mechanism to more general-cases.
  for (size_t i = 0; i < TPlans.size(); ++i) {
    if (checkSubOptimal(&TPlans[i])) {
      SubOptimalIdx = i;
    }
  }

  if (DoSearch) {
    // Search Algorithm working for AutoTensorization
    /// FIXME(yg0412.yun)
    /// need to solve not picking fixed TPlans[0]
    bool SuccessToSearch = ISAAwareSearch(&TPlans[0]);
  }

  // if SubOptimalIdx is not SIZE_MAX, then SubOptimal TPlan is picked among the
  // TPlans.
  return SubOptimalIdx != SIZE_MAX;
}

unsigned TPlanDecisionLogic::getLoopTripCount(ScalarEvolution *SE, Loop *L) {
  const SCEV *LoopCount = SE->getBackedgeTakenCount(L);
  auto *LoopCountConstant = dyn_cast<SCEVConstant>(LoopCount);
  if (!LoopCountConstant)
    return 0;
  return LoopCountConstant->getValue()->getSExtValue();
}
