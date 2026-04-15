//===- LoopTensorize.h ------------------------------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_TENSORIZE_LOOPTENSORIZE_H
#define LLVM_TRANSFORMS_TENSORIZE_LOOPTENSORIZE_H

#include "TPlan.h"
#include "TPlanDecisionLogic.h"
#include "llvm/Analysis/LoopNestAnalysis.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include <functional>

namespace llvm {

class AssumptionCache;
class BlockFrequencyInfo;
class DemandedBits;
class DominatorTree;
class Function;
class Instruction;
class Loop;
class LoopAccessInfoManager;
class LoopInfo;
class OptimizationRemarkEmitter;
class ProfileSummaryInfo;
class ScalarEvolution;
class TargetLibraryInfo;
class TargetTransformInfo;

extern cl::opt<bool> EnableLoopInterleaving;
extern cl::opt<bool> EnableLoopTensorize;

/// A marker to determine if extra passes after loop vectorization should be
/// run.
struct ShouldRunExtraTensorizePasses
    : public AnalysisInfoMixin<ShouldRunExtraTensorizePasses> {
  static AnalysisKey Key;
  struct Result {
    bool invalidate(Function &F, const PreservedAnalyses &PA,
                    FunctionAnalysisManager::Invalidator &) {
      // Check whether the analysis has been explicitly invalidated. Otherwise,
      // it remains preserved.
      auto PAC = PA.getChecker<ShouldRunExtraTensorizePasses>();
      return !PAC.preservedWhenStateless();
    }
  };

  Result run(Function &F, FunctionAnalysisManager &FAM) { return Result(); }
};

struct LoopTensorizeOptions {
  friend class TPlan;
  /// If false, consider all loops for interleaving.
  /// If true, only loops that explicitly request interleaving are considered.
  bool InterleaveOnlyWhenForced;

  // If false, consider all loops for multi-pattern vectorization.
  // If true, only loops that explicitly request multi-pattern vectorization are
  // considered.
  bool TensorizeOnlyWhenForced;

  // If false, consider vectorization of all loops including divergent branch.
  // If true, only vectorizing the loops without divergent branch.
  bool SupportDivergentBrOnlyWhenForced;

  /// If false, using default (limited) patterns of multi-pattern vectorize.
  /// If true, only loops that explicitly request auto-tensorize.
  bool AutoTensorizeOnlyWhenForced;

  /// If false, consider TargetTransformInfo(TTI) specifically.
  /// If true, only loops that explicitly request ISA semantic informations are
  // considered.
  bool UseInstrSemanticInfoOnlyWhenForced;

  LoopTensorizeOptions()
      : InterleaveOnlyWhenForced(false), TensorizeOnlyWhenForced(false),
        SupportDivergentBrOnlyWhenForced(false),
        AutoTensorizeOnlyWhenForced(false),
        UseInstrSemanticInfoOnlyWhenForced(false) {}
  LoopTensorizeOptions(bool InterleaveOnlyWhenForced,
                       bool TensorizeOnlyWhenForced,
                       bool AutoTensorizeOnlyWhenForced,
                       bool UseInstrSemanticInfoOnlyWhenForced,
                       bool SupportDivergentBrOnlyWhenForced)
      : InterleaveOnlyWhenForced(InterleaveOnlyWhenForced),
        TensorizeOnlyWhenForced(TensorizeOnlyWhenForced),
        SupportDivergentBrOnlyWhenForced(SupportDivergentBrOnlyWhenForced),
        AutoTensorizeOnlyWhenForced(AutoTensorizeOnlyWhenForced),
        UseInstrSemanticInfoOnlyWhenForced(UseInstrSemanticInfoOnlyWhenForced) {
  }

  LoopTensorizeOptions &setInterleaveOnlyWhenForced(bool Value) {
    InterleaveOnlyWhenForced = Value;
    return *this;
  }

  LoopTensorizeOptions &setTensorizeOnlyWhenForced(bool Value) {
    TensorizeOnlyWhenForced = Value;
    return *this;
  }

  LoopTensorizeOptions &setAutoTensorizeOnlyWhenForced(bool Value) {
    AutoTensorizeOnlyWhenForced = Value;
    return *this;
  }

  LoopTensorizeOptions &setUseInstrSemanticInfoOnlyWhenForced(bool Value) {
    UseInstrSemanticInfoOnlyWhenForced = Value;
    return *this;
  }
  LoopTensorizeOptions &setSupportDivergentBrOnlyWhenForced(bool Value) {
    SupportDivergentBrOnlyWhenForced = Value;
    return *this;
  }
};

/// Storage for information about made changes.
struct LoopTensorizeResult {
  bool MadeAnyChange;
  bool MadeCFGChange;

  LoopTensorizeResult(bool MadeAnyChange, bool MadeCFGChange)
      : MadeAnyChange(MadeAnyChange), MadeCFGChange(MadeCFGChange) {}
};

/// The LoopMultiPatternVectorize Pass.
struct LoopTensorizePass : public PassInfoMixin<LoopTensorizePass> {
private:
  bool InterleaveOnlyWhenForced;

  bool TensorizeOnlyWhenForced;
  /// If false, using default (limited) patterns of multi-pattern vectorize.
  /// If true, only loops that explicitly request auto-tensorize.
  bool AutoTensorizeOnlyWhenForced;

  /// If false, consider TargetTransformInfo(TTI) specifically.
  /// If true, only loops that explicitly request ISA semantic informations are
  /// considered.
  bool UseInstrSemanticInfoOnlyWhenForced;

  bool SupportDivergentBrOnlyWhenForced;

public:
  LoopTensorizePass(LoopTensorizeOptions Opts = {});

  SmallVector<Loop *> NestedLoop;
  ScalarEvolution *SE;
  LoopInfo *LI;
  TargetTransformInfo *TTI;
  DominatorTree *DT;
  BlockFrequencyInfo *BFI;
  TargetLibraryInfo *TLI;
  DemandedBits *DB;
  AssumptionCache *AC;
  LoopAccessInfoManager *LAIs;
  OptimizationRemarkEmitter *ORE;
  ProfileSummaryInfo *PSI;

  // PreservedAnalyses run(LoopNest &LN,
  //                       LoopAnalysisManager &LAM,
  //                       LoopStandardAnalysisResults &AR,
  //                       LPMUpdater &U);
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  void printPipeline(raw_ostream &OS,
                     function_ref<StringRef(StringRef)> MapClassName2PassName);

  // Shim for old PM.
  LoopTensorizeResult runImpl(Function &F, ScalarEvolution &SE_, LoopInfo &LI_,
                              TargetTransformInfo &TTI_, DominatorTree &DT_,
                              BlockFrequencyInfo *BFI_, TargetLibraryInfo *TLI_,
                              DemandedBits &DB_, AssumptionCache &AC_,
                              LoopAccessInfoManager &LAIs_,
                              OptimizationRemarkEmitter &ORE_,
                              ProfileSummaryInfo *PSI_);

  inline std::shared_ptr<TensorizePattern> choosePattern();
  bool processLoop(SmallVector<Loop *> NestedLoops);
};

/// Reports a vectorization failure: print \p DebugMsg for debugging
/// purposes along with the corresponding optimization remark \p RemarkName.
/// If \p I is passed, it is an instruction that prevents vectorization.
/// Otherwise, the loop \p TheLoop is used for the location of the remark.
void reportTensorizationFailure(const StringRef DebugMsg,
                                const StringRef OREMsg, const StringRef ORETag,
                                OptimizationRemarkEmitter *ORE, Loop *TheLoop,
                                Instruction *I = nullptr);

} // end namespace llvm

#endif // LLVM_TRANSFORMS_TENSORIZE_LOOPTENSORIZE_H
