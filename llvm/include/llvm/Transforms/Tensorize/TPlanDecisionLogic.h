#ifndef LLVM_TRANSFORMS_TENSORIZE_TPLANDECISIONLOGIC_H
#define LLVM_TRANSFORMS_TENSORIZE_TPLANDECISIONLOGIC_H

#include "TPattern.h"
#include "TPlan.h"
#include "TPlanSpace.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {

enum SearchAlgorithm {
  generic_algorithm,
  IntegerLinearProgramming,
  DynamicProgramming,
};

class TPlanDecisionLogic {
public:
  SearchAlgorithm SA;
  std::unique_ptr<TPlanSpace> TPSpace;

  Triple::ArchType target;
  /// Target information.
  const TargetTransformInfo &TTI;

  std::shared_ptr<TensorizePattern> pattern;
  ScalarEvolution *SE;
  size_t SubOptimalIdx = SIZE_MAX;

  TFTy MaxTF;

  bool setHeuristic;

  TPlanDecisionLogic(bool setHeuristic_, const TargetTransformInfo &TTI,
                     Triple::ArchType target_, ScalarEvolution *se,
                     std::shared_ptr<TensorizePattern> pattern_)
      : TPSpace(std::make_unique<TPlanSpace>(setHeuristic_, target_)), TTI(TTI),
        target(target_), SE(se), setHeuristic(setHeuristic_),
        pattern(std::move(pattern_)) {}

  const std::unique_ptr<TPlanSpace> &getTPSpace() const { return TPSpace; }
  Triple::ArchType getTarget() const { return target; }
  bool getHeuristicFlag() const { return setHeuristic; }

  unsigned getLoopTripCount(ScalarEvolution *SE, Loop *L);
  void setMaxTF(TensorizePattern *tp, TFTy *MaxTFMap) {
    // set MaxTF after analyzing the instructions.
    MaxTF = *MaxTFMap;

    // set MaxTF after analyzing the instructions based on specific tpattern.
    tp->MaxTF = MaxTFMap;
  };

  // find best TPlan based on pattern
  bool ISAAwareSearch(TPlanPtr *tplan);
  bool ApplyPattern(SmallVector<TPlanPtr, 4> &TPlans);
  bool checkSubOptimal(TPlanPtr *tplan);
  bool findBestTPlan(SmallVector<TPlanPtr, 4> &TPlans, bool DoSearch);
  TPlan &getBestPlanFor(ElementCount VF) const;
  TPlan &getBestPlan() const;

private:
};

} // namespace llvm
#endif // LLVM_TRANSFORMS_TENSORIZE_TPLANDECISIONLOGIC_H
