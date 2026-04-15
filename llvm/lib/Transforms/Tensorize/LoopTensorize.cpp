#include "llvm/Transforms/Tensorize/LoopTensorize.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/DemandedBits.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Transforms/Utils/FixIrreducible.h"
// #include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueHandle.h"
// #include "llvm/IR/VectorBuilder.h"
#include "llvm/Analysis/LoopNestAnalysis.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Tensorize/LoopTensorizationLegality.h"
#include "llvm/Transforms/Tensorize/LoopTensorizeCostModel.h"
#include "llvm/Transforms/Tensorize/TPRecipeBuilder.h"
#include "llvm/Transforms/Tensorize/TPlan.h"
#include "llvm/Transforms/Tensorize/TPlanAnalysis.h"
#include "llvm/Transforms/Tensorize/TPlanner.h"
#include "llvm/Transforms/Tensorize/TensorizeCommon.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/InjectTLIMappings.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/LoopVersioning.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/Transforms/Utils/SizeOpts.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

using namespace llvm;

#define LV_NAME "loop-tensorize"
#define DEBUG_TYPE LV_NAME

#ifndef NDEBUG
const char VerboseDebug[] = DEBUG_TYPE "-verbose";
#endif

STATISTIC(LoopsTensorize, "Number of loops for LoopTensorize");
STATISTIC(LoopsAnalyzed, "Number of loops analyzed for vectorization");
STATISTIC(LoopsEpilogueVectorized, "Number of epilogues vectorized");

cl::list<unsigned> TensorizationFactorList(
    "force-tensor-widths", cl::Hidden,
    cl::desc("Sets the tensorization dimensions (row,col,inner). Format: "
             "-force-tensor-widths=0,0,0"),
    cl::CommaSeparated);

cl::opt<bool> UseTensorType("use-tensor-type", cl::init(false), cl::Hidden,
                            cl::desc("Use-tensor-type"));

cl::opt<bool> EnableHeuristicSearch("enable-heuristic-search", cl::init(false),
                                    cl::Hidden,
                                    cl::desc("Enable-heuristic-search"));

cl::opt<bool> EnableGemmPattern("enable-gemm-pattern", cl::init(false),
                                cl::Hidden, cl::desc("Enable-gemm-pattern"));
cl::opt<bool> EnableConvPattern("enable-conv-pattern", cl::init(false),
                                cl::Hidden, cl::desc("Enable-conv-pattern"));

cl::opt<bool> EnableElementWisePattern("enable-element-wise-pattern",
                                       cl::init(false), cl::Hidden,
                                       cl::desc("Enable-element-wise-pattern"));

cl::opt<bool> EnableMultiPatternVectorization(
    "enable-multi-pattern-vectorization", cl::init(true), cl::Hidden,
    cl::desc("Enable-multi-pattern-vectorization"));

cl::opt<bool> EnableAutoTensorize(
    "enable-auto-tensorization", cl::init(false), cl::Hidden,
    cl::desc("Enable-auto-tensorization for LoopTensorize"));

cl::opt<bool> EnableInstrSemanticInfo(
    "enable-using-instruction-semantic", cl::init(false), cl::Hidden,
    cl::desc("Enable-auto-tensorization for LoopTensorize"));

cl::opt<bool>
    llvm::EnableLoopTensorize("tensorize-loops", cl::init(true), cl::Hidden,
                              cl::desc("Run the Loop tensorization passes"));

cl::opt<bool> EnableSupportDivergentBr(
    "support-divergent-br", cl::init(true), cl::Hidden,
    cl::desc("Skip to check divergent branch and using vp or select instead"));
static cl::opt<unsigned> TensorizeMemoryCheckThreshold(
    "tensorize-memory-check-threshold", cl::init(128), cl::Hidden,
    cl::desc("The maximum allowed number of runtime memory checks"));
static cl::opt<bool> LoopTensorizeWithBlockFrequency(
    "loop-tensorize-with-block-frequency", cl::init(true), cl::Hidden,
    cl::desc("Enable the use of the block frequency analysis to access PGO "
             "heuristics minimizing code growth in cold regions and being more "
             "aggressive in hot regions."));
namespace llvm {
cl::opt<bool> EnableTPlanNativePath(
    "enable-tplan-native-path", cl::init(true), cl::Hidden,
    cl::desc("Enable TPlan-native vectorization path with "
             "support for outer loop vectorization."));
}
// Option prefer-predicate-over-epilogue indicates that an epilogue is
// undesired, that predication is preferred, and this lists all options. I.e.,
// the vectorizer will try to fold the tail-loop (epilogue) into the vector body
// and predicate the instructions accordingly. If tail-folding fails, there are
// different fallback strategies depending on these values:
namespace PreferPredicateTy {
enum Option {
  ScalarEpilogue = 0,
  PredicateElseScalarEpilogue,
  PredicateOrDontVectorize
};
} // namespace PreferPredicateTy

static cl::opt<PreferPredicateTy::Option> MPPreferPredicateOverEpilogue(
    "mp-prefer-predicate-over-epilogue",
    cl::init(PreferPredicateTy::ScalarEpilogue), cl::Hidden,
    cl::desc("Tail-folding and predication preferences over creating a scalar "
             "epilogue loop."),
    cl::values(
        clEnumValN(PreferPredicateTy::ScalarEpilogue, "scalar-epilogue",
                   "Don't tail-predicate loops, create scalar epilogue"),
        clEnumValN(PreferPredicateTy::PredicateElseScalarEpilogue,
                   "predicate-else-scalar-epilogue",
                   "prefer tail-folding, create scalar epilogue if tail "
                   "folding fails."),
        clEnumValN(PreferPredicateTy::PredicateOrDontVectorize,
                   "predicate-dont-vectorize",
                   "prefers tail-folding, don't attempt vectorization if "
                   "tail-folding fails.")));

// Likelyhood of bypassing the vectorized loop because assumptions about SCEV
// variables not overflowing do not hold. See `emitSCEVChecks`.
static constexpr uint32_t SCEVCheckBypassWeights[] = {1, 127};
// Likelyhood of bypassing the vectorized loop because pointers overlap. See
// `emitMemRuntimeChecks`.
static constexpr uint32_t MemCheckBypassWeights[] = {1, 127};

/// Write a \p DebugMsg about vectorization to the debug output stream. If \p I
/// is passed, the message relates to that particular instruction.
#ifndef NDEBUG
static void debugTensorizationMessage(const StringRef Prefix,
                                      const StringRef DebugMsg,
                                      Instruction *I) {
  dbgs() << "LT: " << Prefix << DebugMsg;
  if (I != nullptr)
    dbgs() << " " << *I;
  else
    dbgs() << '.';
  dbgs() << '\n';
}
#endif

namespace {
// Forward declare GeneratedRTChecks.
class GeneratedRTChecks;

using SCEV2ValueTy = DenseMap<const SCEV *, Value *>;

unsigned getLoopTripCount(ScalarEvolution *SE, Loop *L) {
  const SCEV *LoopCount = SE->getBackedgeTakenCount(L);
  auto *LoopCountConstant = dyn_cast<SCEVConstant>(LoopCount);
  if (!LoopCountConstant)
    return 0;
  return LoopCountConstant->getValue()->getSExtValue();
}

void adaptForTargetPre(Function &F, bool UseTensorType) {
  auto ArchType = Triple(F.getParent()->getTargetTriple()).getArch();
  auto *EntryBB = &F.front();
  IRBuilder<> Builder(EntryBB->getTerminator());
  LLVMContext &Ctx = F.getContext();
  auto ConstInt = [&Ctx](unsigned N, int64_t Val) {
    return ConstantInt::get(Type::getIntNTy(Ctx, N), Val);
  };
  auto CI64 = [ConstInt](int64_t Val) { return ConstInt(64, Val); };
  const int64_t SRAMGranularity = 128;

  if (ArchType == Triple::ArchType::gaia) {
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *GEP = dyn_cast<GEPOperator>(&I)) {
          if (MDNode *MD = I.getMetadata("addr")) {
            auto &Operand = MD->getOperand(0);
            if (auto *ConstMeta = dyn_cast<ConstantAsMetadata>(Operand)) {
              auto *GEPPtr = GEP->getPointerOperand();
              auto *PtrTy = cast<PointerType>(GEPPtr->getType());
              unsigned AddrSpace = PtrTy->getAddressSpace();
              auto *ConstMetaVal = ConstMeta->getValue();
              auto PtrAddr = ConstMetaVal->getUniqueInteger().getSExtValue();

              auto *ConstI =
                  CI64(/*IsSRAM*/ AddrSpace == 1 ? PtrAddr * SRAMGranularity
                                                 : PtrAddr);
              auto *Addr = Builder.CreateIntToPtr(ConstI, GEPPtr->getType());
              if (GEPPtr != Addr)
                GEPPtr->replaceAllUsesWith(Addr);
            }
          }
        }
      }
    }
  }
}

} // namespace

namespace llvm {

AnalysisKey ShouldRunExtraTensorizePasses::Key;

namespace {} // namespace

void reportTensorizationFailure(const StringRef DebugMsg,
                                const StringRef OREMsg, const StringRef ORETag,
                                OptimizationRemarkEmitter *ORE, Loop *TheLoop,
                                Instruction *I) {
  LLVM_DEBUG(debugTensorizationMessage("Not tensorizing: ", DebugMsg, I));

  // TODO(yuxin.an):
  // LoopVectorizeHints Hints(TheLoop, true /* doesn't matter */, *ORE);
  // ORE->emit(
  //     createLVAnalysis(Hints.vectorizeAnalysisPassName(), ORETag, TheLoop, I)
  //     << "loop not vectorized: " << OREMsg);
}

} // namespace llvm

LoopTensorizePass::LoopTensorizePass(LoopTensorizeOptions Opts)
    : InterleaveOnlyWhenForced(Opts.InterleaveOnlyWhenForced ||
                               !EnableLoopInterleaving),
      TensorizeOnlyWhenForced(Opts.TensorizeOnlyWhenForced ||
                              !EnableLoopTensorize),
      AutoTensorizeOnlyWhenForced(Opts.AutoTensorizeOnlyWhenForced ||
                                  !EnableAutoTensorize),
      UseInstrSemanticInfoOnlyWhenForced(
          Opts.UseInstrSemanticInfoOnlyWhenForced || !EnableInstrSemanticInfo),
      SupportDivergentBrOnlyWhenForced(Opts.SupportDivergentBrOnlyWhenForced ||
                                       !EnableSupportDivergentBr) {}

PreservedAnalyses LoopTensorizePass::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  auto &LI = AM.getResult<LoopAnalysis>(F);
  // There are no loops in the function. Return before computing other expensive
  // analyses.
  if (LI.empty())
    return PreservedAnalyses::all();
  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  auto &TTI = AM.getResult<TargetIRAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  auto &AC = AM.getResult<AssumptionAnalysis>(F);
  auto &DB = AM.getResult<DemandedBitsAnalysis>(F);
  auto &ORE = AM.getResult<OptimizationRemarkEmitterAnalysis>(F);

  LoopAccessInfoManager &LAIs = AM.getResult<LoopAccessAnalysis>(F);
  auto &MAMProxy = AM.getResult<ModuleAnalysisManagerFunctionProxy>(F);
  ProfileSummaryInfo *PSI =
      MAMProxy.getCachedResult<ProfileSummaryAnalysis>(*F.getParent());
  BlockFrequencyInfo *BFI = nullptr;
  if (PSI && PSI->hasProfileSummary())
    BFI = &AM.getResult<BlockFrequencyAnalysis>(F);
  LoopTensorizeResult Result =
      runImpl(F, SE, LI, TTI, DT, BFI, &TLI, DB, AC, LAIs, ORE, PSI);

  if (!Result.MadeAnyChange)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;

  if (isAssignmentTrackingEnabled(*F.getParent())) {
    for (auto &BB : F)
      RemoveRedundantDbgInstrs(&BB);
  }

  PA.preserve<LoopAnalysis>();
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<ScalarEvolutionAnalysis>();
  PA.preserve<LoopAccessAnalysis>();

  if (Result.MadeCFGChange) {
    // Making CFG changes likely means a loop got vectorized. Indicate that
    // extra simplification passes should be run.
    // TODO: MadeCFGChanges is not a prefect proxy. Extra passes should only
    // be run if runtime checks have been added.
    // AM.getResult<ShouldRunExtraTensorizePasses>(F);
    // PA.preserve<ShouldRunExtraTensorizePasses>();
  } else {
    PA.preserveSet<CFGAnalyses>();
  }
  return PA;
}

static void getNestedLoops(Loop *CurL, SmallVector<Loop *> &LVec) {
  if (!CurL)
    return;
  LVec.push_back(CurL);
  getNestedLoops(CurL->getParentLoop(), LVec);
}

static void collectSupportedLoops(Loop &L, LoopInfo *LI,
                                  OptimizationRemarkEmitter *ORE,
                                  SmallVectorImpl<SmallVector<Loop *>> &V) {
  // Collect inner loops and outer loops without irreducible control flow.
  // we collect the outermost loop of every loop nest.

  if (L.isInnermost()) {
    // YYG::REMOVE
    errs() << "L.isInnermost() w/ \n";
    L.dump();
    
    LLVM_DEBUG(dbgs() << "LT: Collecting outer/inner loops .. \n");
    LoopBlocksRPO RPOT(&L);
    RPOT.perform(LI);
    if (!containsIrreducibleCFG<const BasicBlock *>(RPOT, *LI)) {
      LLVM_DEBUG(dbgs() << "\n LT: is not containsIrreducibleCFG '"
                        << L.getHeader()->getParent()->getName() << "' from "
                        << L.getLocStr() << "\n");

      SmallVector<Loop *> LVec;
      getNestedLoops(&L, LVec);

      V.push_back(LVec);
      // TODO: Collect inner loops inside marked outer loops in case
      // vectorization fails for the outer loop. Do not invoke
      // 'containsIrreducibleCFG' again for inner loops when the outer loop is
      // already known to be reducible. We can use an inherited attribute for
      // that.
      return;
    }
  }
  for (Loop *InnerL : L)
    collectSupportedLoops(*InnerL, LI, ORE, V);
}

LoopTensorizeResult LoopTensorizePass::runImpl(
    Function &F, ScalarEvolution &SE_, LoopInfo &LI_, TargetTransformInfo &TTI_,
    DominatorTree &DT_, BlockFrequencyInfo *BFI_, TargetLibraryInfo *TLI_,
    DemandedBits &DB_, AssumptionCache &AC_, LoopAccessInfoManager &LAIs_,
    OptimizationRemarkEmitter &ORE_, ProfileSummaryInfo *PSI_) {
  SE = &SE_;
  LI = &LI_;
  TTI = &TTI_;
  DT = &DT_;
  BFI = BFI_;
  TLI = TLI_;
  AC = &AC_;
  LAIs = &LAIs_;
  DB = &DB_;
  ORE = &ORE_;
  PSI = PSI_;

  // Don't attempt if
  // 1. the target claims to have no vector registers, and
  // 2. interleaving won't help ILP.
  //
  // The second condition is necessary because, even if the target has no
  // vector registers, loop vectorization may still enable scalar
  // interleaving.
  if (!TTI->getNumberOfRegisters(TTI->getRegisterClassForType(true)) &&
      TTI->getMaxInterleaveFactor(ElementCount::getFixed(1)) < 2)
    return LoopTensorizeResult(false, false);

  bool Changed = false, CFGChanged = false;

  // The vectorizer requires loops to be in simplified form.
  // Since simplification may add new inner loops, it has to run before the
  // legality and profitability checks. This means running the loop vectorizer
  // will simplify all loops, regardless of whether anything end up being
  // vectorized.
  for (const auto &L : *LI) {
    // YYG:REMOVE
    errs() << "before simplifyLoop\n";
    L->dump(); // Nested-loop 

    Changed |= CFGChanged |=
        simplifyLoop(L, DT, LI, SE, AC, nullptr, false /* PreserveLCSSA */);
  }
  adaptForTargetPre(F, UseTensorType);

  // Build up a worklist of inner-loops to vectorize. This is necessary as
  // the act of vectorizing or partially unrolling a loop creates new loops
  // and can invalidate iterators across the loops.
  SmallVector<SmallVector<Loop *>, 8> Worklist;

  for (Loop *L : *LI)
    collectSupportedLoops(*L, LI, ORE, Worklist);

  LoopsAnalyzed += Worklist.size();

  // Now walk the identified inner loops.
  while (!Worklist.empty()) {
    SmallVector<Loop *> NestedLoops = Worklist.pop_back_val();

    for (Loop *CurL : NestedLoops) {
      // For the loop we actually process, form LCSSA to simplify the transform.
      Changed |= formLCSSARecursively(*CurL, *DT, LI, SE);
      errs() << "*CurL: " << *CurL << "\n";
    }
    // YYG : REMOVE
    // For our matrix-muptlication.mlir input, below L is full nested-loop, not
    // single sub-loop.
    // below is Copy?
    this->NestedLoop = NestedLoops;

    // YYG: REMOVE
    // We expect NestedLoops is a single NestedLoop.
    Changed |= CFGChanged |= processLoop(NestedLoops);

    if (Changed) {
      LAIs->clear();

#ifndef NDEBUG
      if (VerifySCEV)
        SE->verify();
#endif
    }
  }

  // Process each loop nest in the function.
  return LoopTensorizeResult(Changed, CFGChanged);
}

namespace llvm {} // end of namespace llvm

namespace {} // namespace

namespace llvm {

/// Return a value for Step multiplied by VF.
Value *createStepForTF(IRBuilderBase &B, Type *Ty, ElementCount TF,
                       int64_t Step) {
  assert(Ty->isIntegerTy() && "Expected an integer step");
  return B.CreateElementCount(Ty, TF.multiplyCoefficientBy(Step));
}

MapVector<Loop *, SCEV *> createTripCountSCEV(Type *IdxTy, ScalarEvolution &SE,
                                              SmallVector<Loop *> Loops) {
  // YYG::REMOVE
  errs() << "[createTripCountSCEV] \n";

  MapVector<Loop *, SCEV *> Res;
  for (auto *CurL : Loops) {
    PredicatedScalarEvolution PSE(SE, *CurL);
    const SCEV *BackedgeTakenCount = PSE.getBackedgeTakenCount();
    assert(!isa<SCEVCouldNotCompute>(BackedgeTakenCount) &&
           "Invalid loop count");
    // YYG::REMOVE
    errs() << "CurL: \n";
    CurL->dump();

    SCEV *TCSCEV = const_cast<SCEV *>(
        SE.getTripCountFromExitCount(BackedgeTakenCount, IdxTy, CurL));
    errs() << "TCSCEV: " << *TCSCEV << "\n";

    Res.insert({CurL, TCSCEV});
  }
  return Res;
}

} // namespace llvm

// Determine how to lower the scalar epilogue, which depends on 1) optimising
// for minimum code-size, 2) predicate compiler options, 3) loop hints forcing
// predication, and 4) a TTI hook that analyses whether the loop is suitable
// for predication.
static ScalarEpilogueLowering getScalarEpilogueLowering(
    Function *F, Loop *L, LoopTensorizeHints &Hints, ProfileSummaryInfo *PSI,
    BlockFrequencyInfo *BFI, TargetTransformInfo *TTI, TargetLibraryInfo *TLI,
    LoopTensorizationLegality &LVL, InterleavedAccessInfo *IAI) {
  // TODO(yuxin.an)

  // 1) OptSize takes precedence over all other options, i.e. if this is
  // set, don't look at hints or options, and don't request a scalar
  // epilogue. (For PGSO, as shouldOptimizeForSize isn't currently
  // accessible from LoopAccessInfo (due to code dependency and not being
  // able to reliably get PSI/BFI from a loop analysis under NPM), we cannot
  // suppress the collection of strides in LoopAccessInfo::analyzeLoop() and
  // vectorize without versioning when the vectorization is forced, unlike
  // hasOptSize. So revert back to the old way and vectorize with versioning
  // when forced. See D81345.) if (F->hasOptSize() ||
  // (llvm::shouldOptimizeForSize(L->getHeader(), PSI, BFI,
  //                                                     PGSOQueryType::IRPass)
  //                                                     &&
  //                         Hints.getForce() !=
  //                         LoopVectorizeHints::FK_Enabled))
  //   return CM_ScalarEpilogueNotAllowedOptSize;

  // // 2) If set, obey the directives
  // if (MPPreferPredicateOverEpilogue.getNumOccurrences()) {
  //   switch (MPPreferPredicateOverEpilogue) {
  //   case PreferPredicateTy::ScalarEpilogue:
  //     return CM_ScalarEpilogueAllowed;
  //   case PreferPredicateTy::PredicateElseScalarEpilogue:
  //     return CM_ScalarEpilogueNotNeededUsePredicate;
  //   case PreferPredicateTy::PredicateOrDontVectorize:
  //     return CM_ScalarEpilogueNotAllowedUsePredicate;
  //   };
  // }

  // // 3) If set, obey the hints
  // switch (Hints.getPredicate()) {
  // case LoopVectorizeHints::FK_Enabled:
  //   return CM_ScalarEpilogueNotNeededUsePredicate;
  // case LoopVectorizeHints::FK_Disabled:
  //   return CM_ScalarEpilogueAllowed;
  // };

  // // 4) if the TTI hook indicates this is profitable, request
  // predication. TailFoldingInfo TFI(TLI, &LVL, IAI); if
  // (TTI->preferPredicateOverEpilogue(&TFI))
  //   return CM_ScalarEpilogueNotNeededUsePredicate;

  return CM_ScalarEpilogueAllowed;
}

inline std::shared_ptr<TensorizePattern> LoopTensorizePass::choosePattern() {
  if (EnableGemmPattern)
    return std::make_shared<GEMMPattern>(this->NestedLoop);
  if (EnableElementWisePattern)
    return std::make_shared<ElementWiseTensorizePattern>(this->NestedLoop);
  if (EnableConvPattern)
    return std::make_shared<ConvolutionTensorizePattern>(this->NestedLoop);
  return std::make_shared<TargetAutoPattern>(this->NestedLoop);
}

bool LoopTensorizePass::processLoop(SmallVector<Loop *> NestedLoops) {

  // YYG. TODO: NestedLoops인가 NestedLoop 인가?
  auto Pattern = choosePattern();
  // Function containing loop
  Function *F = NestedLoops.front()->getHeader()->getParent();
  LoopTensorizeHints Hints(NestedLoops, InterleaveOnlyWhenForced, *ORE, TTI);
  LLVM_DEBUG(
      dbgs() << "LT: Loop hints:"
             << " force="
             << (Hints.getForce() == LoopTensorizeHints::FK_Disabled
                     ? "disabled"
                     : (Hints.getForce() == LoopTensorizeHints::FK_Enabled
                            ? "enabled"
                            : "?"))
             << " width=" << Hints.getWidth()
             << " interleave=" << Hints.getInterleave() << "\n");

  MapVector<Loop *, PredicatedScalarEvolution *> Loop2PSE;
  for (Loop *CurL : NestedLoops) {
    auto *PSE = new PredicatedScalarEvolution(*SE, *CurL);
    Loop2PSE.insert({CurL, PSE});
  }

  // Check if it is legal to vectorize the loop.
  LoopTensorizationRequirements Requirements;
  LoopTensorizationLegality LTL(NestedLoops, Loop2PSE, DT, TTI, TLI, F, *LAIs,
                                LI, ORE, &Requirements, &Hints, DB, AC, BFI,
                                PSI, Pattern);

  if (!LTL.canTensorize()) {
    LLVM_DEBUG(dbgs() << "LT: Not vectorizing: Cannot prove legality.\n");
    // Hints.emitRemarkWithHints();
    return false;
  }

  // YYG:REMOVE
  LLVM_DEBUG(dbgs() << "after canTensorize()\n");
  MapVector<Loop *, InterleavedAccessInfo *> Loop2IAI;

  for (Loop *CurL : NestedLoops) {
    // YYG:REMOVE
    LLVM_DEBUG(dbgs() << "CurL: \n");
    CurL->dump();
    auto *IAI = new InterleavedAccessInfo(*Loop2PSE[CurL], CurL, DT, LI,
                                          LTL.getLAI(CurL));
    Loop2IAI.insert({CurL, IAI});
  }

  ScalarEpilogueLowering SEL =
      CM_ScalarEpilogueAllowed; // TODO(yuxin.an): getScalarEpilogueLowering

  // Use the cost model.
  LoopTensorizeCostModel CM(SEL, NestedLoops, Loop2PSE, LI, &LTL, *TTI, TLI, DB,
                            AC, ORE, F, &Hints, Loop2IAI);

  // Use the planner for TPlan.
  Triple::ArchType ArchType =
      Triple(F->getParent()->getTargetTriple()).getArch();
  TPlanDecisionLogic DL =
      TPlanDecisionLogic(EnableHeuristicSearch, *TTI, ArchType, SE, Pattern);

  LoopTensorizePlanner LTP(NestedLoops, LI, DT, TLI, *TTI, &LTL, CM, DL,
                           Loop2IAI, Loop2PSE, SE, Hints, ORE);

  // Set Tensorization Factor for nested loop
  TFTy UserTFMap;
  TUFTy UserTICMap;
  CM.collectValuesToIgnore();
  CM.collectElementTypesForWidening();
  // Treat below IC as a hint for inner-most loop
  unsigned UserIC = Hints.getInterleave();

  // If user defines TensorizationFactor for all NestedLoops
  bool ScalableTensorizationAllowed = CM.isScalableVectorizationAllowed();
  std::vector<unsigned> TensorizationFactors = TensorizationFactorList;

  // FIXME(yg0412.yun) need to remove target-specific below code
  if (ArchType == Triple::ArchType::gaia) {
    for (Loop *L : Pattern->Info.Loops) {
      UserTFMap.insert({L, ElementCount::getFixed(getLoopTripCount(SE, L))});
    }
  } else {
    if (TensorizationFactors.size()) {
      if (TensorizationFactors.size() != NestedLoops.size()) {
        llvm::errs() << "ERROR: Number of tensor factors ("
                     << TensorizationFactors.size()
                     << ") does not match the number of loops ("
                     << NestedLoops.size() << ").\n";

        llvm::report_fatal_error("Tensor factor count mismatch");
      }

      for (size_t i = 0; i < NestedLoops.size(); ++i) {
        Loop *L = NestedLoops[i];
        unsigned rawVal = TensorizationFactors[i];

        ElementCount EC = ScalableTensorizationAllowed
                              ? ElementCount::getScalable(rawVal)
                              : ElementCount::getFixed(rawVal);

        UserTFMap[L] = EC;
        // YYG:REMOVE
        // remove fixed unsigned value;
        UserTICMap[L] = 2;
      }
    } else {
      // Treat below VF as a hint for inner-most loop
      ElementCount HintVF = Hints.getWidth();
      for (size_t i = 0; i < NestedLoops.size(); ++i) {
        Loop *L = NestedLoops[i];
        ElementCount EC = ScalableTensorizationAllowed
                              ? ElementCount::getScalable(2)
                              : ElementCount::getFixed(2);
        // FIXME(yg0412.yun) CM.computeMaxVF is not working need to be fixed
        FixedScalableVFPair MaxFactors = CM.computeMaxVF(EC, UserIC, L);
        if (!MaxFactors) // Cases that should not to be vectorized nor
                         // interleaved.
          return false;

        ElementCount MaxUserVF = ScalableTensorizationAllowed
                                     ? MaxFactors.ScalableVF
                                     : MaxFactors.FixedVF;
        bool UserVFIsLegal =
            ElementCount::isKnownLE(ElementCount::getFixed(1), MaxUserVF);
        if (UserVFIsLegal) {
          assert(isPowerOf2_32(ElementCount::getFixed(1).getKnownMinValue()) &&
                 "VF needs to be a power of two");
        }
        UserTFMap[L] = MaxUserVF;
        // YYG:REMOVE
        // remove fixed unsigned value;
        UserTICMap[L] = 2;
      }
    }
  }
  LTP.setMaxTF(Pattern.get(), &UserTFMap);
  // Get user vectorization factor. This is only working for inner-most loop
  // vectorization.

  // TODO (yuxin.an):
  // 1. Confirm the data structure of Width and Interleave, currently use
  // SmallVector<Ty>
  // 2. Analyze and confirm Interleave for tensorize, Analyze and confirm
  // Interleave for tensorize, currently assuming it is all 1.
  // 3. The current implementation first considers matrix-matrix multiplication,
  // corresponding to three nested loops.

  // std::vector<unsigned> TensorizationFactors = TensorizationFactorList;

  // assert(TensorizationFactors.size() == 3 && "Expected three Tensorization
  // Factors.");

  TUFTy UserTIC;
  for (size_t i = 0; i < (Pattern->Info.Loops.size()); i++) {
    UserTIC.insert({Pattern->Info.Loops[i], 1});
  }

  // *----------------------1. Build initial TPlan-----------------------*//
  // If User specify the GEMM-pattern then, createNestedLoopTPlan() will apply
  // GEMM-pattern, If not, TargetAuto Decision Logic will handle
  // createNestedLoopTPlan();
  bool successTocreate = LTP.createNestedLoopTPlan(UseTensorType);

  GeneratedRTChecks Checks(*Loop2PSE.begin()->second->getSE(), DT, LI, TTI,
                           F->getDataLayout(), false);
  // *---------------------------------------------------------------*//

  // *---------------2. Is this TPlan optimal for Target HW?------------*//
  // 1. choose single TPlan among the TPlnas which is made from
  // LTP.createNEstedLoopTPlan()
  // 2. if DoSearch, ISA-aware search will start otherwise, it returns 1's TPlan
  // as best TPlan
  bool successTosearch =
      DL.findBestTPlan(LTP.TPlans, /* DoSearch=*/!LTP.SuccessToApplyPattern);

  // *---------------------------------------------------------------*//
  
  // YYG:REMOVE
  errs() << "end of ToSearch\n";


  // *---3. Make real llvm instruction from TPRecipes (of BestPlan)--*//
  TPlanPtr BestTPlan;
  if (successTosearch) {
    assert(DL.SubOptimalIdx != SIZE_MAX && "no sub‑optimal plan");
    bool TensorizeLoop = true;
    {
      using namespace ore;
      // !FIXME(yuxin.an)
      TPlan *BestPlan = LTP.TPlans[DL.SubOptimalIdx].get();

      TFTy MinProfitableTripCount = UserTFMap;

      LoopTensorizer LT(NestedLoops, Loop2PSE, LI, DT, TLI, TTI, AC, ORE,
                        UserTFMap, UserTFMap, UserTIC, &LTL, &CM, BFI, PSI,
                        Checks, Pattern, ArchType);

      // YYG:REMOVE
      errs() << "successTosearch! \n";

      LTP.executePlan(UserTFMap, UserTIC, *BestPlan, LT, DT, UseTensorType,
                      false);
      
      // YYG:REMOVE
      errs() << "end of executePlan! \n";

    }
  }
  // *---------------------------------------------------------------*//

  // Verity Function after TPlan
  assert(!verifyFunction(*F, &dbgs()));
  return true;
}

void LoopTensorizePass::printPipeline(
    raw_ostream &OS, function_ref<StringRef(StringRef)> MapClassName2PassName) {
  static_cast<PassInfoMixin<LoopTensorizePass> *>(this)->printPipeline(
      OS, MapClassName2PassName);

  OS << '<';
  OS << (InterleaveOnlyWhenForced ? "" : "no-") << "interleave-forced-only;";
  OS << (TensorizeOnlyWhenForced ? "" : "no-") << "tensorize-forced-only;";
  OS << '>';
}
