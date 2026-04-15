#include "llvm/Transforms/Tensorize/LoopTensorizationLegality.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Tensorize/LoopTensorize.h"
#include "llvm/Transforms/Utils/SizeOpts.h"
#include <cassert>

using namespace llvm;
using namespace PatternMatch;

#define LT_NAME "loop-tensorize"
#define DEBUG_TYPE LT_NAME

static cl::opt<bool>
LTAllowStridedPointerIVs("lt-strided-pointer-ivs", cl::init(false), cl::Hidden,
                       cl::desc("Enable recognition of non-constant strided "
                                "pointer induction variables."));

static cl::opt<bool> EnableIfConversionForT(
    "enable-if-conversion-for-t", cl::init(true), cl::Hidden,
    cl::desc("Enable if-conversion during tensorization."));

static cl::opt<LoopTensorizeHints::ScalableForceKind>
    ForceScalableTensorization(
        "scalable-tensorization", cl::init(LoopTensorizeHints::SK_Unspecified),
        cl::Hidden,
        cl::desc("Control whether the compiler can use scalable vectors to "
                 "vectorize a loop"),
        cl::values(
            clEnumValN(LoopTensorizeHints::SK_FixedWidthOnly, "off",
                       "Scalable tensorization is disabled."),
            clEnumValN(
                LoopTensorizeHints::SK_PreferScalable, "preferred",
                "Scalable tensorization is available and favored when the "
                "cost is inconclusive."),
            clEnumValN(
                LoopTensorizeHints::SK_PreferScalable, "on",
                "Scalable tensorization is available and favored when the "
                "cost is inconclusive.")));

namespace llvm {
cl::opt<bool>
    TPHintsAllowReordering("tp-hints-allow-reordering", cl::init(true),
                           cl::Hidden,
                           cl::desc("Allow enabling loop hints to reorder "
                                    "FP operations during vectorization."));
}

namespace llvm {

LoopTensorizeHints::LoopTensorizeHints(SmallVector<Loop *> NestedLoop_,
                                       bool InterleaveOnlyWhenForced,
                                       OptimizationRemarkEmitter &ORE,
                                       const TargetTransformInfo *TTI)
    : Width("vectorize.width", VectorizerParams::VectorizationFactor, HK_WIDTH),
      Interleave("interleave.count", InterleaveOnlyWhenForced, HK_INTERLEAVE),
      Force("vectorize.enable", FK_Undefined, HK_FORCE),
      IsTensorized("IsTensorized", 0, HK_ISVECTORIZED),
      Predicate("vectorize.predicate.enable", FK_Undefined, HK_PREDICATE),
      Scalable("vectorize.scalable.enable", SK_Unspecified, HK_SCALABLE),
      NestedLoop(NestedLoop_), ORE(ORE) {
  // Populate values with existing loop metadata.
  getHintsFromMetadata();

  // force-vector-interleave overrides DisableInterleaving.
  if (VectorizerParams::isInterleaveForced())
    Interleave.Value = VectorizerParams::VectorizationInterleave;

  // If the metadata doesn't explicitly specify whether to enable scalable
  // tensorization, then decide based on the following criteria (increasing
  // level of priority):
  //  - Target default
  //  - Metadata width
  //  - Force option (always overrides)
  if ((LoopTensorizeHints::ScalableForceKind)Scalable.Value == SK_Unspecified) {
    if (TTI)
      Scalable.Value = TTI->enableScalableVectorization() ? SK_PreferScalable
                                                          : SK_FixedWidthOnly;

    if (Width.Value)
      // If the width is set, but the metadata says nothing about the scalable
      // property, then assume it concerns only a fixed-width UserVF.
      // If width is not set, the flag takes precedence.
      Scalable.Value = SK_FixedWidthOnly;
  }

  // If the flag is set to force any use of scalable vectors, override the loop
  // hints.
  if (ForceScalableTensorization.getValue() !=
      LoopTensorizeHints::SK_Unspecified)
    Scalable.Value = ForceScalableTensorization.getValue();

  // Scalable vectorization is disabled if no preference is specified.
  if ((LoopTensorizeHints::ScalableForceKind)Scalable.Value == SK_Unspecified)
    Scalable.Value = SK_FixedWidthOnly;

  if (IsTensorized.Value != 1)
    // If the vectorization width and interleaving count are both 1 then
    // consider the loop to have been already vectorized because there's
    // nothing more that we can do.
    IsTensorized.Value =
        getWidth() == ElementCount::getFixed(1) && getInterleave() == 1;
  LLVM_DEBUG(if (InterleaveOnlyWhenForced && getInterleave() == 1) dbgs()
             << "LT: Interleaving disabled by the pass manager\n");
}

static void extractHintsFromLoopID(Loop *L, SmallVectorImpl<LoopHint> &Hints) {
  MDNode *LoopID = L->getLoopID();

  assert(LoopID->getNumOperands() > 0 && "requires at least one operand");
  assert(LoopID->getOperand(0) == LoopID && "invalid loop id");

  for (const MDOperand &MDO : llvm::drop_begin(LoopID->operands())) {
    const MDString *S = nullptr;
    SmallVector<Metadata *, 4> Args;

    if (const MDNode *MD = dyn_cast<MDNode>(MDO)) {
      if (!MD || MD->getNumOperands() == 0)
        continue;
      S = dyn_cast<MDString>(MD->getOperand(0));
      for (unsigned i = 1, ie = MD->getNumOperands(); i < ie; ++i)
        Args.push_back(MD->getOperand(i));
    } else {
      S = dyn_cast<MDString>(MDO);
      assert(Args.empty() && "too many arguments for MDString");
    }

    if (!S)
      continue;

    if (Args.size() == 1) {
      LoopHint H;
      H.Name = S->getString().str();
      H.Arg = Args[0];
      Hints.push_back(std::move(H));
    }
  }
}

void LoopTensorizeHints::getHintsFromMetadata() {
  for (Loop *L : NestedLoop) {
    if (!L)
      continue;

    // Yugyoung: if loop is lowered by Kernel MLIR, there's no metadata for
    // current version So, I removed it.
    MDNode *ID = L->getLoopID();
    if (!ID)
      continue;
    // {
    //   errs() << "ERROR: Loop " << L->getHeader()->getName()
    //               << " has no LoopID!\n";
    //   report_fatal_error("Missing LoopID for a loop");
    // }
    // assert(ID->getNumOperands() > 0 && "requires at least one operand");
    // assert(ID->getOperand(0) == ID && "invalid loop id");

    SmallVector<LoopHint, 4> PerLoopHints;
    extractHintsFromLoopID(L, PerLoopHints);
    if (!PerLoopHints.empty())
      HintsMap.insert({L, std::move(PerLoopHints)});
  }
}

static IntegerType *getInductionIntegerTy(const DataLayout &DL, Type *Ty) {
  assert(Ty->isIntOrPtrTy() && "Expected integer or pointer type");

  if (Ty->isPointerTy())
    return DL.getIntPtrType(Ty->getContext(), Ty->getPointerAddressSpace());

  // It is possible that char's or short's overflow when we ask for the loop's
  // trip count, work around this by changing the type size.
  if (Ty->getScalarSizeInBits() < 32)
    return Type::getInt32Ty(Ty->getContext());

  return cast<IntegerType>(Ty);
}

static IntegerType *getWiderInductionTy(const DataLayout &DL, Type *Ty0,
                                        Type *Ty1) {
  IntegerType *TyA = getInductionIntegerTy(DL, Ty0);
  IntegerType *TyB = getInductionIntegerTy(DL, Ty1);
  return TyA->getScalarSizeInBits() > TyB->getScalarSizeInBits() ? TyA : TyB;
}

void LoopTensorizationLegality::addInductionPhi(
    PHINode *Phi, const InductionDescriptor &ID,
    SmallPtrSetImpl<Value *> &AllowedExit, Loop *L) {
  Inductions[Phi] = ID;

  // In case this induction also comes with casts that we know we can ignore
  // in the vectorized loop body, record them here. All casts could be recorded
  // here for ignoring, but suffices to record only the first (as it is the
  // only one that may bw used outside the cast sequence).
  ArrayRef<Instruction *> Casts = ID.getCastInsts();
  if (!Casts.empty())
    InductionCastsToIgnore.insert(*Casts.begin());

  Type *PhiTy = Phi->getType();
  const DataLayout &DL = Phi->getDataLayout();

  assert((PhiTy->isIntOrPtrTy() || PhiTy->isFloatingPointTy()) &&
         "Expected int, ptr, or FP induction phi type");

  // Get the widest type.
  if (PhiTy->isIntOrPtrTy()) {
    if (!WidestIndTy)
      WidestIndTy = getInductionIntegerTy(DL, PhiTy);
    else
      WidestIndTy = getWiderInductionTy(DL, PhiTy, WidestIndTy);
  }

  // Int inductions are special because we only allow one IV.
  if (ID.getKind() == InductionDescriptor::IK_IntInduction &&
      ID.getConstIntStepValue() && ID.getConstIntStepValue()->isOne() &&
      isa<Constant>(ID.getStartValue()) &&
      cast<Constant>(ID.getStartValue())->isNullValue()) {

    // Use the phi node with the widest type as induction. Use the last
    // one if there are multiple (no good reason for doing this other
    // than it is expedient). We've checked that it begins at zero and
    // steps by one, so this is a canonical induction variable.
    if (!PrimaryInduction || PhiTy == WidestIndTy)
      PrimaryInduction = Phi;
  }

  // Both the PHI node itself, and the "post-increment" value feeding
  // back into the PHI node may have external users.
  // We can allow those uses, except if the SCEVs we have for them rely
  // on predicates that only hold within the loop, since allowing the exit
  // currently means re-using this SCEV outside the loop (see PR33706 for more
  // details).
  if (Loop2PSE[L]->getPredicate().isAlwaysTrue()) {
    AllowedExit.insert(Phi);
    AllowedExit.insert(Phi->getIncomingValueForBlock(L->getLoopLatch()));
  }

  LLVM_DEBUG(dbgs() << "LV: Found an induction variable.\n");
}

/// Check that the instruction has outside loop users and is not an
/// identified reduction variable.
static bool hasOutsideLoopUser(const Loop *TheLoop, Instruction *Inst,
                               SmallPtrSetImpl<Value *> &AllowedExit) {
  // Reductions, Inductions and non-header phis are allowed to have exit users. All
  // other instructions must not have external users.
  if (!AllowedExit.count(Inst))
    // Check that all of the users of the loop are inside the BB.
    for (User *U : Inst->users()) {
      Instruction *UI = cast<Instruction>(U);
      // This user may be a reduction exit value.
      if (!TheLoop->contains(UI)) {
        LLVM_DEBUG(dbgs() << "LV: Found an outside user for : " << *UI << '\n');
        return true;
      }
    }
  return false;
}

/// Checks if a function is scalarizable according to the TLI, in
/// the sense that it should be vectorized and then expanded in
/// multiple scalar calls. This is represented in the
/// TLI via mappings that do not specify a vector name, as in the
/// following example:
///
///    const VecDesc VecIntrinsics[] = {
///      {"llvm.phx.abs.i32", "", 4}
///    };
static bool isTLIScalarize(const TargetLibraryInfo &TLI, const CallInst &CI) {
  const StringRef ScalarName = CI.getCalledFunction()->getName();
  bool Scalarize = TLI.isFunctionVectorizable(ScalarName);
  // Check that all known VFs are not associated to a vector
  // function, i.e. the vector name is emty.
  if (Scalarize) {
    ElementCount WidestFixedVF, WidestScalableVF;
    TLI.getWidestVF(ScalarName, WidestFixedVF, WidestScalableVF);
    for (ElementCount VF = ElementCount::getFixed(2);
         ElementCount::isKnownLE(VF, WidestFixedVF); VF *= 2)
      Scalarize &= !TLI.isFunctionVectorizable(ScalarName, VF);
    for (ElementCount VF = ElementCount::getScalable(1);
         ElementCount::isKnownLE(VF, WidestScalableVF); VF *= 2)
      Scalarize &= !TLI.isFunctionVectorizable(ScalarName, VF);
    assert((WidestScalableVF.isZero() || !Scalarize) &&
           "Caller may decide to scalarize a variant using a scalable VF");
  }
  return Scalarize;
}

bool LoopTensorizationLegality::canTensorizeInstrs() {
  // For each block in the loop.
  // From Inner-most Loops
  for (Loop* Loop : Loops) {
    // YYG:REMOVE
    errs() << "canTEnsorizeInstrs() !!!\n";
    Loop->dump();

    for (BasicBlock *BB : Loop->blocks()) {
      // Scan the instructions in the block and look for hazards.
      for (Instruction &I : *BB) {
        if (auto *Phi = dyn_cast<PHINode>(&I)) {
          // YYG:REMOVE
          errs() << "[canTensorizeInstrs()] Phi: " << *Phi << "\n";

          Type *PhiTy = Phi->getType();
          // Check that this PHI type is allowed.
          if (!PhiTy->isIntegerTy() && !PhiTy->isFloatingPointTy() &&
              !PhiTy->isPointerTy()) {
            reportTensorizationFailure("Found a non-int non-pointer PHI",
                                      "loop control flow is not understood by vectorizer",
                                      "CFGNotUnderstood", ORE, Loop);
            return false;
          }

          
          BasicBlock *Header = Loop->getHeader();
          // YYG:REMOVE
          errs() << "[canTensorize]Header: " << *Header << "\n";

          // If this PHINode is not in the header block, then we know that we
          // can convert it to select during if-conversion. No need to check if
          // the PHIs in this block are induction or reduction variables.
          if (BB != Header) {
            // Non-header phi nodes that have outside uses can be vectorized. Add
            // them to the list of allowed exits.
            // Unsafe cyclic dependencies with header phis are identified during
            // legalization for reduction, induction and fixed order
            // recurrences.
            
            // YYG:REMOVE
            errs() << "BB != Header \n";

            AllowedExit.insert(&I);
            continue;
          }

          // We only allow if-converted PHIs with exactly two incoming values.
          if (Phi->getNumIncomingValues() != 2) {
            reportTensorizationFailure("Found an invalid PHI",
                "loop control flow is not understood by vectorizer",
                "CFGNotUnderstood", ORE, Loop, Phi);
            return false;
          }

          RecurrenceDescriptor RedDes;
          if (RecurrenceDescriptor::isReductionPHI(Phi, Loop, RedDes, DB, AC,
                                                  DT, Loop2PSE[Loop]->getSE())) {
            // YYG:REMOVE
            errs() << "[canTen] isReductionPHI  \n";
            errs() << "[canTensorize] Phi: " << *Phi << "\n";
            Requirements->addExactFPMathInst(RedDes.getExactFPMathInst());
            AllowedExit.insert(RedDes.getLoopExitInstr());
            Reductions[Phi] = RedDes;
            continue;
          }

          // We prevent matching non-constant strided pointer IVS to preserve
          // historical vectorizer behavior after a generalization of the
          // IVDescriptor code.  The intent is to remove this check, but we
          // have to fix issues around code quality for such Loop first.
          auto isDisallowedStridedPointerInduction =
            [](const InductionDescriptor &ID) {
            if (LTAllowStridedPointerIVs)
              return false;
            return ID.getKind() == InductionDescriptor::IK_PtrInduction &&
              ID.getConstIntStepValue() == nullptr;
          };

          // TODO: Instead of recording the AllowedExit, it would be good to
          // record the complementary set: NotAllowedExit. These include (but may
          // not be limited to):
          // 1. Reduction phis as they represent the one-before-last value, which
          // is not available when vectorized
          // 2. Induction phis and increment when SCEV predicates cannot be used
          // outside the loop - see addInductionPhi
          // 3. Non-Phis with outside uses when SCEV predicates cannot be used
          // outside the loop - see call to hasOutsideLoopUser in the non-phi
          // handling below
          // 4. FixedOrderRecurrence phis that can possibly be handled by
          // extraction.
          // By recording these, we can then reason about ways to vectorize each
          // of these NotAllowedExit.
          InductionDescriptor ID;
          if (InductionDescriptor::isInductionPHI(Phi, Loop, Loop2PSE[Loop]->getSE(), ID) &&
              !isDisallowedStridedPointerInduction(ID)) {
            // YYG:REMOVE
            errs() << "canTEnsorize::isInductionPHI(Phi)\n";
            errs() << "[canTensorize] Phi: " << *Phi << "\n";
            addInductionPhi(Phi, ID, AllowedExit, Loop);
            Requirements->addExactFPMathInst(ID.getExactFPMathInst());
            continue;
          }

          if (RecurrenceDescriptor::isFixedOrderRecurrence(Phi, Loop, DT)) {
            // YYG:REMOVE
            errs() << "canTEnsorize::isFixedOrderRecurrence(Phi)\n";
            AllowedExit.insert(Phi);
            FixedOrderRecurrences.insert(Phi);
            continue;
          }

          // As a last resort, coerce the PHI to a AddRec expression
          // and re-try classifying it a an induction PHI.
          if (InductionDescriptor::isInductionPHI(Phi, Loop, *Loop2PSE[Loop], ID, true) &&
              !isDisallowedStridedPointerInduction(ID)) {
            // YYG:REMOVE
            errs() << "canTensorize::isInductionPHI with true!\n";
            addInductionPhi(Phi, ID, AllowedExit, Loop);
            continue;
          }

          reportTensorizationFailure("Found an unidentified PHI",
              "value that could not be identified as "
              "reduction is used outside the loop",
              "NonReductionValueUsedOutsideLoop", ORE, Loop, Phi);
          return false;
        } // end of PHI handling

        // We handle calls that:
        //   * Are debug info intrinsics.
        //   * Have a mapping to an IR intrinsic.
        //   * Have a vector version available.
        auto *CI = dyn_cast<CallInst>(&I);

        if (CI && !getVectorIntrinsicIDForCall(CI, TLI) &&
            !isa<DbgInfoIntrinsic>(CI) &&
            !(CI->getCalledFunction() && TLI &&
              (!VFDatabase::getMappings(*CI).empty() ||
              isTLIScalarize(*TLI, *CI)))) {
          // If the call is a recognized math libary call, it is likely that
          // we can vectorize it given loosened floating-point constraints.
          LibFunc Func;
          bool IsMathLibCall =
              TLI && CI->getCalledFunction() &&
              CI->getType()->isFloatingPointTy() &&
              TLI->getLibFunc(CI->getCalledFunction()->getName(), Func) &&
              TLI->hasOptimizedCodeGen(Func);

          if (IsMathLibCall) {
            // TODO: Ideally, we should not use clang-specific language here,
            // but it's hard to provide meaningful yet generic advice.
            // Also, should this be guarded by allowExtraAnalysis() and/or be part
            // of the returned info from isFunctionVectorizable()?
            reportTensorizationFailure(
                "Found a non-intrinsic callsite",
                "library call cannot be vectorized. "
                "Try compiling with -fno-math-errno, -ffast-math, "
                "or similar flags",
                "CantVectorizeLibcall", ORE, Loop, CI);
          } else {
            reportTensorizationFailure("Found a non-intrinsic callsite",
                                      "call instruction cannot be vectorized",
                                      "CantVectorizeLibcall", ORE, Loop, CI);
          }
          return false;
        }

        // Some intrinsics have scalar arguments and should be same in order for
        // them to be vectorized (i.e. loop invariant).
        if (CI) {
          auto *SE = Loop2PSE[Loop]->getSE();
          Intrinsic::ID IntrinID = getVectorIntrinsicIDForCall(CI, TLI);
          for (unsigned i = 0, e = CI->arg_size(); i != e; ++i)
            if (isVectorIntrinsicWithScalarOpAtArg(IntrinID, i)) {
              if (!SE->isLoopInvariant(Loop2PSE[Loop]->getSCEV(CI->getOperand(i)), Loop)) {
                reportTensorizationFailure("Found unvectorizable intrinsic",
                    "intrinsic instruction cannot be vectorized",
                    "CantVectorizeIntrinsic", ORE, Loop, CI);
                return false;
              }
            }
        }

        // If we found a vectorized variant of a function, note that so LV can
        // make better decisions about maximum VF.
        if (CI && !VFDatabase::getMappings(*CI).empty())
          VecCallVariantsFound = true;

        // Check that the instruction return type is vectorizable.
        // Also, we can't vectorize extractelement instructions.
        if ((!VectorType::isValidElementType(I.getType()) &&
            !I.getType()->isVoidTy()) ||
            isa<ExtractElementInst>(I)) {
          reportTensorizationFailure("Found unvectorizable type",
              "instruction return type cannot be vectorized",
              "CantVectorizeInstructionReturnType", ORE, Loop, &I);
          return false;
        }

        // Check that the stored type is vectorizable.
        if (auto *ST = dyn_cast<StoreInst>(&I)) {
          Type *T = ST->getValueOperand()->getType();
          if (!VectorType::isValidElementType(T)) {
            reportTensorizationFailure("Store instruction cannot be vectorized",
                                      "store instruction cannot be vectorized",
                                      "CantVectorizeStore", ORE, Loop, ST);
            return false;
          }

          // For nontemporal stores, check that a nontemporal vector version is
          // supported on the target.
          if (ST->getMetadata(LLVMContext::MD_nontemporal)) {
            // Arbitrarily try a vector of 2 elements.
            auto *VecTy = FixedVectorType::get(T, /*NumElts=*/2);
            assert(VecTy && "did not find vectorized version of stored type");
            if (!TTI->isLegalNTStore(VecTy, ST->getAlign())) {
              reportTensorizationFailure(
                  "nontemporal store instruction cannot be vectorized",
                  "nontemporal store instruction cannot be vectorized",
                  "CantVectorizeNontemporalStore", ORE, Loop, ST);
              return false;
            }
          }

        } else if (auto *LD = dyn_cast<LoadInst>(&I)) {
          if (LD->getMetadata(LLVMContext::MD_nontemporal)) {
            // For nontemporal loads, check that a nontemporal vector version is
            // supported on the target (arbitrarily try a vector of 2 elements).
            auto *VecTy = FixedVectorType::get(I.getType(), /*NumElts=*/2);
            assert(VecTy && "did not find vectorized version of load type");
            if (!TTI->isLegalNTLoad(VecTy, LD->getAlign())) {
              reportTensorizationFailure(
                  "nontemporal load instruction cannot be vectorized",
                  "nontemporal load instruction cannot be vectorized",
                  "CantVectorizeNontemporalLoad", ORE, Loop, LD);
              return false;
            }
          }

          // FP instructions can allow unsafe algebra, thus vectorizable by
          // non-IEEE-754 compliant SIMD units.
          // This applies to floating-point math operations and calls, not memory
          // operations, shuffles, or casts, as they don't change precision or
          // semantics.
        } else if (I.getType()->isFloatingPointTy() && (CI || I.isBinaryOp()) &&
                  !I.isFast()) {
          LLVM_DEBUG(dbgs() << "LV: Found FP op with unsafe algebra.\n");
          Hints->setPotentiallyUnsafe();
        }

        // Reduction instructions are allowed to have exit users.
        // All other instructions must not have external users.
        if (hasOutsideLoopUser(Loop, &I, AllowedExit)) {
          // We can safely vectorize Loop where instructions within the loop are
          // used outside the loop only if the SCEV predicates within the loop is
          // same as outside the loop. Allowing the exit means reusing the SCEV
          // outside the loop.
          if (Loop2PSE[Loop]->getPredicate().isAlwaysTrue()) {
            AllowedExit.insert(&I);
            continue;
          }
          reportTensorizationFailure("Value cannot be used outside the loop",
                                    "value cannot be used outside the loop",
                                    "ValueUsedOutsideLoop", ORE, Loop, &I);
          return false;
        }
      } // next instr.
    }
  }

  if (!PrimaryInduction) {
    if (Inductions.empty()) {
      reportTensorizationFailure("Did not find one integer induction var",
          "loop induction variable could not be identified",
          "NoInductionVariable", ORE, Loops[TotalLoopDegree]);
      return false;
    } else if (!WidestIndTy) {
      reportTensorizationFailure("Did not find one integer induction var",
          "integer loop induction variable could not be identified",
          "NoIntegerInductionVariable", ORE, Loops[TotalLoopDegree]);
      return false;
    } else {
      LLVM_DEBUG(dbgs() << "LV: Did not find one integer induction var.\n");
    }
  }

  // Now we know the widest induction type, check if our found induction
  // is the same size. If it's not, unset it here and InnerLoopVectorizer
  // will create another.
  if (PrimaryInduction && WidestIndTy != PrimaryInduction->getType())
    PrimaryInduction = nullptr;

  return true;
}

bool LoopTensorizationLegality::canTensorizeMemory() {

  // TODO
  return true;
}

bool LoopTensorizationLegality::isInductionPhi(const Value *V) {
  Value *In0 = const_cast<Value *>(V);
  PHINode *PN = dyn_cast_or_null<PHINode>(In0);
  //yyg:remove
  errs() << "PN: " << *PN << "\n";
  if (!PN)
    return false;
  
  // In Auto-T, the inductionPhi is firstly enroll inside of this function, 
  // Thus, when it comes to first, we need to addInductionPhi on Inductions.
  auto *D = new InductionDescriptor();
  Loop *L = Pattern->Info.PHI2Loop[PN];
  // YYG::REMOVE
  errs() << "L: \n";
  L->dump();
  
  if (D->isInductionPHI(PN, L, *Loop2PSE[L], *D)) {
    Inductions[PN] = *D;
  }
  // else, it isn't loop-PHI Node
  errs() << "Inductions.count(PN): " << Inductions.count(PN) << "\n";
  return Inductions.count(PN);
}

bool LoopTensorizationLegality::isCastedInductionVariable(
    const Value *V) const {
  auto *Inst = dyn_cast<Instruction>(V);
  return (Inst && InductionCastsToIgnore.count(Inst));
}

bool LoopTensorizationLegality::isInductionVariable(const Value *V) {
  return isInductionPhi(V) || isCastedInductionVariable(V);
}

bool LoopTensorizationLegality::isFixedOrderRecurrence(
    const PHINode *Phi) const {
  return FixedOrderRecurrences.count(Phi);
}

const InductionDescriptor *
LoopTensorizationLegality::getPointerInductionDescriptor(PHINode *Phi) {
  // YYG:REMOVE
  errs() << "[getPointerInductionDescriptor]\n";
  if (!isInductionPhi(Phi))
    return nullptr;
  // YYG:REMOVE
  errs() << "isInductionPhi(Phi)\n";
  auto &ID = getInductionVars().find(Phi)->second;
  if (ID.getKind() == InductionDescriptor::IK_PtrInduction)
    return &ID;
  return nullptr;
}


const InductionDescriptor *
LoopTensorizationLegality::getIntOrFpInductionDescriptor(PHINode *Phi) {
  // YYG:REMOVE
  errs() << "[getIntOrFpInductionDescriptor]\n";

  if (!isInductionPhi(Phi))
    return nullptr;
  auto &ID = getInductionVars().find(Phi)->second;
  if (ID.getKind() == InductionDescriptor::IK_IntInduction ||
      ID.getKind() == InductionDescriptor::IK_FpInduction)
      return &ID;
  return nullptr;

  // auto *D = new InductionDescriptor();
  // Loop *L = Pattern->Info.PHI2Loop[Phi];

  // D->isInductionPHI(Phi, L, *Loop2PSE[L], *D);

  // Inductions[Phi] = *D;

  LLVM_DEBUG(
      dbgs()
      << "[Warning] Please handle "
         "`LoopTensorizationLegality::getIntOrFpInductionDescriptor` \n");

  //return D;
}

bool LoopTensorizationLegality::blockNeedsPredication(BasicBlock *BB, Loop *L) const {
  return LoopAccessInfo::blockNeedsPredication(BB, L, DT);
}

bool LoopTensorizationLegality::blockCanBePredicated(
    BasicBlock *BB, SmallPtrSetImpl<Value *> &SafePtrs,
    SmallPtrSetImpl<const Instruction *> &MaskedOp) const {
  for (Instruction &I : *BB) {
    // We can predicate blocks with calls to assume, as long as we drop them in
    // case we flatten the CFG via predication.
    if (match(&I, m_Intrinsic<Intrinsic::assume>())) {
      MaskedOp.insert(&I);
      continue;
    }

    // Do not let llvm.experimental.noalias.scope.decl block the vectorization.
    // TODO: there might be cases that it should block the vectorization. Let's
    // ignore those for now.
    if (isa<NoAliasScopeDeclInst>(&I))
      continue;

    // We can allow masked calls if there's at least one vector variant, even
    // if we end up scalarizing due to the cost model calculations.
    // TODO: Allow other calls if they have appropriate attributes... readonly
    // and argmemonly?
    if (CallInst *CI = dyn_cast<CallInst>(&I)) {
      llvm_unreachable("");
      // if (VFDatabase::hasMaskedVariant(*CI)) {
      //   MaskedOp.insert(CI);
      //   continue;
      // }
    }

    // Loads are handled via masking (or speculated if safe to do so.)
    if (auto *LI = dyn_cast<LoadInst>(&I)) {
      if (!SafePtrs.count(LI->getPointerOperand()))
        MaskedOp.insert(LI);
      continue;
    }

    // Predicated store requires some form of masking:
    // 1) masked store HW instruction,
    // 2) emulation via load-blend-store (only if safe and legal to do so,
    //    be aware on the race conditions), or
    // 3) element-by-element predicate check and scalar store.
    if (auto *SI = dyn_cast<StoreInst>(&I)) {
      MaskedOp.insert(SI);
      continue;
    }

    if (I.mayReadFromMemory() || I.mayWriteToMemory() || I.mayThrow())
      return false;
  }

  return true;
}

bool LoopTensorizationLegality::canTensorizeWithIfConvert() {
  // TODO(yuxin.an): Confirm whether the outer loop needs to be verified
  if (!EnableIfConversionForT) {
    reportTensorizationFailure("If-conversion is disabled",
                               "if-conversion is disabled",
                               "IfConversionDisabled", ORE, Loops.front());
    return false;
  }

  for (Loop *Loop : Loops) {
    // YYG:REMOVE
    errs() << "Loop \n";
    Loop->dump();


    // from inner-most loop
    // FIXME(yg0412.yun) inner-most loop has single block with itself as header/exiting/latch node
    // assert(Loop->getNumBlocks() > 1 &&
    //     "Single block loops are tensorizable");

    // A list of pointers which are known to be dereferenceable within scope of
    // the loop body for each iteration of the loop which executes.  That is,
    // the memory pointed to can be dereferenced (with the access size implied by
    // the value's type) unconditionally within the loop header without
    // introducing a new fault.
    SmallPtrSet<Value *, 8> SafePointers;

    // Collect safe addresses.
    for (BasicBlock *BB : Loop->blocks()) {
      // YYG:REMOVE
      errs() << "BB: " << *BB << "\n";

      if (!blockNeedsPredication(BB, Loop)) {
        for (Instruction &I : *BB)
          if (auto *Ptr = getLoadStorePointerOperand(&I))
            SafePointers.insert(Ptr);
        continue;
      }

      // For a block which requires predication, a address may be safe to access
      // in the loop w/o predication if we can prove dereferenceability facts
      // sufficient to ensure it'll never fault within the loop. For the moment,
      // we restrict this to loads; stores are more complicated due to
      // concurrency restrictions.
      // TODO(yuxin.an): Confirm
      ScalarEvolution &SE = *Loop2PSE.front().second->getSE();
      for (Instruction &I : *BB) {
        LoadInst *LI = dyn_cast<LoadInst>(&I);
        if (LI && !LI->getType()->isVectorTy() && !mustSuppressSpeculation(*LI) &&
            isDereferenceableAndAlignedInLoop(LI, Loop, SE, *DT, AC))
          SafePointers.insert(LI->getPointerOperand());
      }
    }

    // Collect the blocks that need predication.
    for (BasicBlock *BB : Loop->blocks()) {
      // We don't support switch statements inside loop.
      if (!isa<BranchInst>(BB->getTerminator())) {
        reportTensorizationFailure("Loop contains a switch statement",
                                  "loop contains a switch statement",
                                  "LoopContainsSwitch", ORE, Loop,
                                  BB->getTerminator());
        return false;
      }

      // We must be able to predicate all blocks that need to be predicated.
      if (blockNeedsPredication(BB, Loop) &&
          !blockCanBePredicated(BB, SafePointers, MaskedOp)) {
        reportTensorizationFailure(
            "Control flow cannot be substituted for a select",
            "control flow cannot be substituted for a select", "NoCFGForSelect",
            ORE, Loop, BB->getTerminator());
        return false;
      }
    }
  }
  // We can if-convert this loop.
  return true;
}

// Helper function to canVectorizeLoopNestCFG.
bool LoopTensorizationLegality::canTensorizeLoopCFG(Loop *Lp) {
  // TODO: ORE should be improved to show more accurate information when an
  // outer loop can't be vectorized because a nested loop is not understood or
  // legal. Something like: "outer_loop_location: loop not vectorized:
  // (inner_loop_location) loop control flow is not understood by vectorizer".

  // Store the result and return it at the end instead of exiting early, in case
  // allowExtraAnalysis is used to report multiple reasons for not vectorizing.
  bool Result = true;
  bool DoExtraAnalysis = ORE->allowExtraAnalysis(DEBUG_TYPE);
  // YYG:REMOVE
  errs() << "canTEnsorizeLoopCFG\n";

  // We must have a loop in canonical form. Loops with indirectbr in them cannot
  // be canonicalized.
  if (!Lp->getLoopPreheader()) {
    reportTensorizationFailure(
        "Loop doesn't have a legal pre-header",
        "loop control flow is not understood by vectorizer", "CFGNotUnderstood",
        ORE, Loops.front());
    if (DoExtraAnalysis)
      Result = false;
    else
      return false;
  }

  // We must have a single backedge.
  if (Lp->getNumBackEdges() != 1) {
    reportTensorizationFailure(
        "The loop must have a single backedge",
        "loop control flow is not understood by vectorizer", "CFGNotUnderstood",
        ORE, Loops.front());
    if (DoExtraAnalysis)
      Result = false;
    else
      return false;
  }

  // yyg:remove
  errs() << "Result: " << Result << "\n";

  return Result;
}

bool LoopTensorizeHints::allowReordering() const {
  // Allow the vectorizer to change the order of operations if enabling
  // loop hints are provided
  ElementCount EC = getWidth();
  return TPHintsAllowReordering &&
         (getForce() == LoopTensorizeHints::FK_Enabled ||
          EC.getKnownMinValue() > 1);
}

bool LoopTensorizationLegality::canTensorizeLoopNestCFG() {
  // Store the result and return it at the end instead of exiting early, in case
  // allowExtraAnalysis is used to report multiple reasons for not vectorizing.
  // YYG:REMOVE
  errs() << "canTensorizeLoopNestCFG\n";
  bool Result = true;
  bool DoExtraAnalysis = ORE->allowExtraAnalysis(DEBUG_TYPE);

  for (Loop *L : Loops) {
    if (!canTensorizeLoopCFG(L)) {
      if (DoExtraAnalysis)
        Result = false;
      else
        return false;
    }
  }

  return Result;
}

bool LoopTensorizationLegality::isInvariant(Value *V, Loop *Curl) const {
  auto LAI = getLAI(Curl);
  return LAI->isInvariant(V);
}

bool LoopTensorizationLegality::isInvariantAddressOfReduction(
    Value *V, Loop *L) {
  // YYG:REMOVE
  errs() << "isInvariantAddressOfReduction\n";
  L->dump();
  return any_of(getReductionVars(), [&](auto &Reduction) -> bool {
    const RecurrenceDescriptor &RdxDesc = Reduction.second;
    if (!RdxDesc.IntermediateStore)
      return false;

    PredicatedScalarEvolution *PSE = Loop2PSE[L];
    ScalarEvolution *SE = PSE->getSE();
    Value *InvariantAddress = RdxDesc.IntermediateStore->getPointerOperand();
    return V == InvariantAddress ||
           SE->getSCEV(V) == SE->getSCEV(InvariantAddress);
  });
}

bool LoopTensorizationLegality::canFoldTailByMasking() const {

  LLVM_DEBUG(dbgs() << "LT: checking if tail can be folded by masking.\n");

  SmallPtrSet<const Value *, 8> ReductionLiveOuts;
  // TheLoop is inner-most loop for ELV
  Loop *TheLoop = Loops.back();
  for (const auto &Reduction : getReductionVars())
    ReductionLiveOuts.insert(Reduction.second.getLoopExitInstr());

  // TODO: handle non-reduction outside users when tail is folded by masking.
  for (auto *AE : AllowedExit) {
    // Check that all users of allowed exit values are inside the loop or
    // are the live-out of a reduction.
    if (ReductionLiveOuts.count(AE))
      continue;
    for (User *U : AE->users()) {
      Instruction *UI = cast<Instruction>(U);
      if (TheLoop->contains(UI))
        continue;
      LLVM_DEBUG(
          dbgs()
          << "LT: Cannot fold tail by masking, loop has an outside user for "
          << *UI << "\n");
      return false;
    }
  }

  for (const auto &Entry : getInductionVars()) {
    PHINode *OrigPhi = Entry.first;
    for (User *U : OrigPhi->users()) {
      auto *UI = cast<Instruction>(U);
      if (!TheLoop->contains(UI)) {
        LLVM_DEBUG(dbgs() << "LT: Cannot fold tail by masking, loop IV has an "
                             "outside user for "
                          << *UI << "\n");
        return false;
      }
    }
  }

  // The list of pointers that we can safely read and write to remains empty.
  SmallPtrSet<Value *, 8> SafePointers;

  // Check all blocks for predication, including those that ordinarily do not
  // need predication such as the header block.
  SmallPtrSet<const Instruction *, 8> TmpMaskedOp;
  for (BasicBlock *BB : TheLoop->blocks()) {
    if (!blockCanBePredicated(BB, SafePointers, TmpMaskedOp)) {
      LLVM_DEBUG(dbgs() << "LT: Cannot fold tail by masking.\n");
      return false;
    }
  }

  LLVM_DEBUG(dbgs() << "LT: can fold tail by masking.\n");

  return true;
}

bool LoopTensorizationLegality::canTensorize() {
  // Store the result and return it at the end instead of exiting early, in case
  // allowExtraAnalysis is used to report multiple reasons for not vectorizing.
  bool Result = true;

  bool DoExtraAnalysis = ORE->allowExtraAnalysis(DEBUG_TYPE);
  // Check whether the loop-related control flow in the loop nest is expected by
  // vectorizer.
  if (!canTensorizeLoopNestCFG()) {
    if (DoExtraAnalysis)
      Result = false;
    else
      return false;
  }
  // YYG::REMOVE
  errs() << "LT: Found nested loops on outer-most: \n";
  Loops[TotalLoopDegree]->dump();


  // We need to have a loop header.
  LLVM_DEBUG(dbgs() << "LT: Found nested loops: "
                    << Loops[TotalLoopDegree]->getHeader()->getName() << '\n');

  //assert(Loops.front()->isInnermost() && "Inner loop expected.");
  //YYG:REMOVE
  errs() << "Loops.front(): \n";
  Loops.front()->dump();

  unsigned NumBlocks = Loops[TotalLoopDegree]->getNumBlocks();
  // YYG:REMOVE
  errs() << "NumBlcks: " << NumBlocks << "\n";
  if (NumBlocks != 1 && !canTensorizeWithIfConvert()) {
    LLVM_DEBUG(dbgs() << "LT: Can't if-convert the loop.\n");
    if (DoExtraAnalysis)
      Result = false;
    else
      return false;
  }

  // Check if we can tensorize the instructions and CFG in this loop.
  if (!canTensorizeInstrs()) { // TODO(yuxin.an): Implement the logic
    LLVM_DEBUG(dbgs() << "LT: Can't tensorize the instructions or CFG\n");
    if (DoExtraAnalysis)
      Result = false;
    else
      return false;
  }

  // Go over each instruction and look at memory deps.
  if (!canTensorizeMemory()) { // TODO(yuxin.an): Implement the logic
    LLVM_DEBUG(dbgs() << "LT: Can't vectorize due to memory conflicts\n");
    if (DoExtraAnalysis)
      Result = false;
    else
      return false;
  }

  // TODO(yuxin.an)
  for (auto Elem : Loop2PSE) {
    if (isa<SCEVCouldNotCompute>(Elem.second->getBackedgeTakenCount())) {
      reportTensorizationFailure(
          "could not determine number of loop iterations",
          "could not determine number of loop iterations",
          "CantComputeNumberOfIterations", ORE, Elem.first);
      if (DoExtraAnalysis)
        Result = false;
      else
        return false;
    }
  }

  LLVM_DEBUG(
      dbgs() << "[Info] `LoopTensorizationLegality::canTensorize()` end\n");

  return Result;
}

void LoopTensorizationLegality::prepareToFoldTailByMasking(Loop *CurL) {
  // The list of pointers that we can safely read and write to remains empty.
  SmallPtrSet<Value *, 8> SafePointers;

  // Mark all blocks for predication, including those that ordinarily do not
  // need predication such as the header block.
  for (BasicBlock *BB : CurL->blocks()) {
    [[maybe_unused]] bool R = blockCanBePredicated(BB, SafePointers, MaskedOp);
    assert(R && "Must be able to predicate block when tail-folding.");
  }
}

} // namespace llvm
