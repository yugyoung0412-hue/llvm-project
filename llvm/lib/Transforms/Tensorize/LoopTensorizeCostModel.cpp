#include "llvm/Transforms/Tensorize/LoopTensorizeCostModel.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Transforms/Tensorize/TPlan.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Tensorize/LoopTensorizationLegality.h"

using namespace llvm;

#define DEBUG_TYPE "loop-tensorize-cost-model"

static cl::opt<bool> TPForceTargetSupportsScalableVectors(
    "tp-force-target-supports-scalable-vectors", cl::init(false), cl::Hidden,
    cl::desc(
        "Pretend that scalable vectors are supported, even if the target does "
        "not support them. This flag should only be used for testing."));

static cl::opt<bool>
    TPPreferInLoopReductions("tp-prefer-inloop-reductions", cl::init(false),
                             cl::Hidden,
                             cl::desc("Prefer in-loop vector reductions, "
                                      "overriding the targets preference."));

static cl::opt<cl::boolOrDefault> LTForceSafeDivisor(
    "lt-force-widen-divrem-via-safe-divisor", cl::Hidden,
    cl::desc(
        "Override cost based safe divisor widening for div/rem instructions"));

/// An interleave-group may need masking if it resides in a block that needs
/// predication, or in order to mask away gaps.
static cl::opt<bool> TPEnableMaskedInterleavedMemAccesses(
    "tp-enable-masked-interleaved-mem-accesses", cl::init(false), cl::Hidden,
    cl::desc("Enable vectorization on masked interleaved memory accesses in a "
             "loop"));

static cl::opt<TailFoldingStyle> TPForceTailFoldingStyle(
    "tp-force-tail-folding-style", cl::desc("Force the tail folding style"),
    cl::init(TailFoldingStyle::None),
    cl::values(
        clEnumValN(TailFoldingStyle::None, "none", "Disable tail folding"),
        clEnumValN(
            TailFoldingStyle::Data, "data",
            "Create lane mask for data only, using active.lane.mask intrinsic"),
        clEnumValN(TailFoldingStyle::DataWithoutLaneMask,
                   "data-without-lane-mask",
                   "Create lane mask with compare/stepvector"),
        clEnumValN(TailFoldingStyle::DataAndControlFlow, "data-and-control",
                   "Create lane mask using active.lane.mask intrinsic, and use "
                   "it for both data and control flow"),
        clEnumValN(TailFoldingStyle::DataAndControlFlowWithoutRuntimeCheck,
                   "data-and-control-without-rt-check",
                   "Similar to data-and-control, but remove the runtime check"),
        clEnumValN(TailFoldingStyle::DataWithEVL, "data-with-evl",
                   "Use predicated EVL instructions for tail folding. If EVL "
                   "is unsupported, fallback to data-without-lane-mask.")));

void LoopTensorizeCostModel::collectValuesToIgnore() {
  // Ignore ephemeral values.

  for (size_t i = 0; i < Loops.size(); ++i) {
    // YYG:REMOVE
    errs() << "[collectValuesToIgnore] Loops are starting from outer-most loop?\n";
    Loops[i]->dump();

    CodeMetrics::collectEphemeralValues(Loops[i], AC, ValuesToIgnore);

    SmallVector<Value *, 4> DeadInterleavePointerOps;
    for (BasicBlock *BB : Loops[i]->blocks())
      for (Instruction &I : *BB) {
        // Find all stores to invariant variables. Since they are going to sink
        // outside the loop we do not need calculate cost for them.
        StoreInst *SI;
        if ((SI = dyn_cast<StoreInst>(&I)) &&
            Legal->isInvariantAddressOfReduction(SI->getPointerOperand(), Loops[i]))
          ValuesToIgnore.insert(&I);

        // For interleave groups, we only create a pointer for the start of the
        // interleave group. Queue up addresses of group members except the
        // insert position for further processing.
        if (isAccessInterleaved(&I, Loops[i])) {
          auto *Group = getInterleavedAccessGroup(&I, Loops[i]);
          if (Group->getInsertPos() == &I)
            continue;
          Value *PointerOp = getLoadStorePointerOperand(&I);
          DeadInterleavePointerOps.push_back(PointerOp);
        }
      }

    // Mark ops feeding interleave group members as free, if they are only used
    // by other dead computations.
    for (unsigned I = 0; I != DeadInterleavePointerOps.size(); ++I) {
      auto *Op = dyn_cast<Instruction>(DeadInterleavePointerOps[I]);
      if (!Op || !Loops[i]->contains(Op) ||
          any_of(Op->users(), [this, i](User *U) {
            Instruction *UI = cast<Instruction>(U);
            return !VecValuesToIgnore.contains(U) &&
                   (!isAccessInterleaved(UI, Loops[i]) ||
                    getInterleavedAccessGroup(UI, Loops[i])->getInsertPos() ==
                        UI);
          }))
        continue;
      VecValuesToIgnore.insert(Op);
      DeadInterleavePointerOps.append(Op->op_begin(), Op->op_end());
    }

    // Ignore type-promoting instructions we identified during reduction
    // detection.
    for (const auto &Reduction : Legal->getReductionVars()) {
      const RecurrenceDescriptor &RedDes = Reduction.second;
      const SmallPtrSetImpl<Instruction *> &Casts = RedDes.getCastInsts();
      VecValuesToIgnore.insert(Casts.begin(), Casts.end());
    }
    // Ignore type-casting instructions we identified during induction
    // detection.
    for (const auto &Induction : Legal->getInductionVars()) {
      const InductionDescriptor &IndDes = Induction.second;
      ArrayRef<Instruction *> Casts = IndDes.getCastInsts();
      VecValuesToIgnore.insert(Casts.begin(), Casts.end());
    }
  }
}

void LoopTensorizeCostModel::collectElementTypesForWidening() {
  ElementTypesInLoop.clear();
  // For each block of NestedLoop.
  for (size_t i = 0; i < Loops.size(); ++i) {
    for (BasicBlock *BB : Loops[i]->blocks()) {
      // For each instruction in the loop.
      for (Instruction &I : BB->instructionsWithoutDebug()) {
        Type *T = I.getType();

        // Skip ignored values.
        if (ValuesToIgnore.count(&I))
          continue;

        // Only examine Loads, Stores and PHINodes.
        if (!isa<LoadInst>(I) && !isa<StoreInst>(I) && !isa<PHINode>(I))
          continue;

        // Examine PHI nodes that are reduction variables. Update the type to
        // account for the recurrence type.
        if (auto *PN = dyn_cast<PHINode>(&I)) {
          if (!Legal->isReductionVariable(PN))
            continue;
          const RecurrenceDescriptor &RdxDesc =
              Legal->getReductionVars().find(PN)->second;
          if (TPPreferInLoopReductions || useOrderedReductions(RdxDesc) ||
              TTI.preferInLoopReduction(RdxDesc.getRecurrenceKind(),
                                        RdxDesc.getRecurrenceType()))
            continue;
          T = RdxDesc.getRecurrenceType();
        }

        // Examine the stored values.
        if (auto *ST = dyn_cast<StoreInst>(&I))
          T = ST->getValueOperand()->getType();

        assert(T->isSized() &&
               "Expected the load/store/recurrence type to be sized");

        ElementTypesInLoop.insert(T);
      }
    }
  }
}

std::pair<unsigned, unsigned>
LoopTensorizeCostModel::getSmallestAndWidestTypes() {
  unsigned MinWidth = -1U;
  unsigned MaxWidth = 8;
  const DataLayout &DL = TheFunction->getDataLayout();
  // For in-loop reductions, no element types are added to ElementTypesInLoop
  // if there are no loads/stores in the loop. In this case, check through the
  // reduction variables to determine the maximum width.
  errs() << "getDataLayout \n";
  if (ElementTypesInLoop.empty() && !Legal->getReductionVars().empty()) {
    // Reset MaxWidth so that we can find the smallest type used by recurrences
    // in the loop.
    MaxWidth = -1U;
    for (const auto &PhiDescriptorPair : Legal->getReductionVars()) {
      const RecurrenceDescriptor &RdxDesc = PhiDescriptorPair.second;
      // When finding the min width used by the recurrence we need to account
      // for casts on the input operands of the recurrence.
      MaxWidth = std::min<unsigned>(
          MaxWidth, std::min<unsigned>(
                        RdxDesc.getMinWidthCastToRecurrenceTypeInBits(),
                        RdxDesc.getRecurrenceType()->getScalarSizeInBits()));
    }
  } else {
    for (Type *T : ElementTypesInLoop) {
      MinWidth = std::min<unsigned>(
          MinWidth, DL.getTypeSizeInBits(T->getScalarType()).getFixedValue());
      MaxWidth = std::max<unsigned>(
          MaxWidth, DL.getTypeSizeInBits(T->getScalarType()).getFixedValue());
    }
  }
  return {MinWidth, MaxWidth};
}

bool LoopTensorizeCostModel::isScalarWithPredication(
    Instruction *I, ElementCount VF, Loop *L) const {
  if (!isPredicatedInst(I, L))
    return false;

  // Do we have a non-scalar lowering for this predicated
  // instruction? No - it is scalar with predication.
  switch(I->getOpcode()) {
  default:
    return true;
  case Instruction::Call:
    if (VF.isScalar())
      return true;
    return CallWideningDecisions.at(std::make_pair(cast<CallInst>(I), VF))
               .Kind == CM_Scalarize;
  case Instruction::Load:
  case Instruction::Store: {
    auto *Ptr = getLoadStorePointerOperand(I);
    auto *Ty = getLoadStoreType(I);
    Type *VTy = Ty;
    if (VF.isVector())
      VTy = VectorType::get(Ty, VF);
    const Align Alignment = getLoadStoreAlignment(I);
    return isa<LoadInst>(I) ? !(isLegalMaskedLoad(Ty, Ptr, Alignment) ||
                                TTI.isLegalMaskedGather(VTy, Alignment))
                            : !(isLegalMaskedStore(Ty, Ptr, Alignment) ||
                                TTI.isLegalMaskedScatter(VTy, Alignment));
  }
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::SRem:
  case Instruction::URem: {
    // We have the option to use the safe-divisor idiom to avoid predication.
    // The cost based decision here will always select safe-divisor for
    // scalable vectors as scalarization isn't legal.
    const auto [ScalarCost, SafeDivisorCost] = getDivRemSpeculationCost(I, VF, L);
    return isDivRemScalarWithPredication(ScalarCost, SafeDivisorCost, LTForceSafeDivisor);
  }
  }
}

static Type *MaybeVectorizeType(Type *Elt, ElementCount VF) {
  if (VF.isScalar() || (!Elt->isIntOrPtrTy() && !Elt->isFloatingPointTy()))
    return Elt;
  return VectorType::get(Elt, VF);
}

bool LoopTensorizeCostModel::isPredicatedInst(Instruction *I, Loop *L) const {
  if (!blockNeedsPredicationForAnyReason(I->getParent(), L))
    return false;

  // Can we prove this instruction is safe to unconditionally execute?
  // If not, we must use some form of predication.
  switch(I->getOpcode()) {
  default:
    return false;
  case Instruction::Load:
  case Instruction::Store: {
    if (!Legal->isMaskRequired(I))
      return false;
    // When we know the load's address is loop invariant and the instruction
    // in the original scalar loop was unconditionally executed then we
    // don't need to mark it as a predicated instruction. Tail folding may
    // introduce additional predication, but we're guaranteed to always have
    // at least one active lane.  We call Legal->blockNeedsPredication here
    // because it doesn't query tail-folding.  For stores, we need to prove
    // both speculation safety (which follows from the same argument as loads),
    // but also must prove the value being stored is correct.  The easiest
    // form of the later is to require that all values stored are the same.
    if (Legal->isInvariant(getLoadStorePointerOperand(I), L) &&
        (isa<LoadInst>(I) ||
         (isa<StoreInst>(I) &&
          L->isLoopInvariant(cast<StoreInst>(I)->getValueOperand()))) &&
        !Legal->blockNeedsPredication(I->getParent(), L))
      return false;
    return true;
  }
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::SRem:
  case Instruction::URem:
    // TODO: We can use the loop-preheader as context point here and get
    // context sensitive reasoning
    return !isSafeToSpeculativelyExecute(I);
  case Instruction::Call:
    return Legal->isMaskRequired(I);
  }
}

InstructionCost LoopTensorizeCostModel::getScalarizationOverhead(
    Instruction *I, ElementCount VF, TTI::TargetCostKind CostKind) const {

  // There is no mechanism yet to create a scalable scalarization loop,
  // so this is currently Invalid.
  if (VF.isScalable())
    return InstructionCost::getInvalid();

  if (VF.isScalar())
    return 0;

  InstructionCost Cost = 0;
  Type *RetTy = toVectorTy(I->getType(), VF);
  if (!RetTy->isVoidTy() &&
      (!isa<LoadInst>(I) || !TTI.supportsEfficientVectorElementLoadStore()))
    Cost += TTI.getScalarizationOverhead(
        cast<VectorType>(RetTy), APInt::getAllOnes(VF.getKnownMinValue()),
        /*Insert*/ true,
        /*Extract*/ false, CostKind);

  // Some targets keep addresses scalar.
  if (isa<LoadInst>(I) && !TTI.prefersVectorizedAddressing())
    return Cost;

  // Some targets support efficient element stores.
  if (isa<StoreInst>(I) && TTI.supportsEfficientVectorElementLoadStore())
    return Cost;

  // Collect operands to consider.
  CallInst *CI = dyn_cast<CallInst>(I);
  Instruction::op_range Ops = CI ? CI->args() : I->operands();

  // Skip operands that do not require extraction/scalarization and do not incur
  // any overhead.
  SmallVector<Type *> Tys;
  for (auto *V : filterExtractingOperands(Ops, VF))
    Tys.push_back(MaybeVectorizeType(V->getType(), VF));
  return Cost + TTI.getOperandsScalarizationOverhead(Tys, CostKind);
}

std::pair<InstructionCost, InstructionCost>
LoopTensorizeCostModel::getDivRemSpeculationCost(Instruction *I,
                                                    ElementCount VF, Loop *L) const {
  assert(I->getOpcode() == Instruction::UDiv ||
         I->getOpcode() == Instruction::SDiv ||
         I->getOpcode() == Instruction::SRem ||
         I->getOpcode() == Instruction::URem);
  assert(!isSafeToSpeculativelyExecute(I));

  const TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;

  // Scalarization isn't legal for scalable vector types
  InstructionCost ScalarizationCost = InstructionCost::getInvalid();
  if (!VF.isScalable()) {
    // Get the scalarization cost and scale this amount by the probability of
    // executing the predicated block. If the instruction is not predicated,
    // we fall through to the next case.
    ScalarizationCost = 0;

    // These instructions have a non-void type, so account for the phi nodes
    // that we will create. This cost is likely to be zero. The phi node
    // cost, if any, should be scaled by the block probability because it
    // models a copy at the end of each predicated block.
    ScalarizationCost += VF.getKnownMinValue() *
      TTI.getCFInstrCost(Instruction::PHI, CostKind);

    // The cost of the non-predicated instruction.
    ScalarizationCost += VF.getKnownMinValue() *
      TTI.getArithmeticInstrCost(I->getOpcode(), I->getType(), CostKind);

    // The cost of insertelement and extractelement instructions needed for
    // scalarization.
    ScalarizationCost += getScalarizationOverhead(I, VF, CostKind);

    // Scale the cost by the probability of executing the predicated blocks.
    // This assumes the predicated block for each vector lane is equally
    // likely.
    ScalarizationCost = ScalarizationCost / getReciprocalPredBlockProb();
  }
  InstructionCost SafeDivisorCost = 0;

  auto *VecTy = toVectorTy(I->getType(), VF);

  // The cost of the select guard to ensure all lanes are well defined
  // after we speculate above any internal control flow.
  SafeDivisorCost += TTI.getCmpSelInstrCost(
    Instruction::Select, VecTy,
    toVectorTy(Type::getInt1Ty(I->getContext()), VF),
    CmpInst::BAD_ICMP_PREDICATE, CostKind);

  // Certain instructions can be cheaper to vectorize if they have a constant
  // second vector operand. One example of this are shifts on x86.
  Value *Op2 = I->getOperand(1);
  auto Op2Info = TTI.getOperandInfo(Op2);
  if (Op2Info.Kind == TargetTransformInfo::OK_AnyValue &&
      Legal->isInvariant(Op2, L))
    Op2Info.Kind = TargetTransformInfo::OK_UniformValue;

  SmallVector<const Value *, 4> Operands(I->operand_values());
  SafeDivisorCost += TTI.getArithmeticInstrCost(
    I->getOpcode(), VecTy, CostKind,
    {TargetTransformInfo::OK_AnyValue, TargetTransformInfo::OP_None},
    Op2Info, Operands, I);
  return {ScalarizationCost, SafeDivisorCost};
}

FixedScalableVFPair LoopTensorizeCostModel::computeFeasibleMaxVF(
    unsigned MaxTripCount, ElementCount UserTF, bool FoldTailByMasking,
    Loop *CurL) {

  MinBWs = computeMinimumValueSizes(CurL->getBlocks(), *DB, &TTI);

  unsigned SmallestType, WidestType;
  std::tie(SmallestType, WidestType) = getSmallestAndWidestTypes();

  // Loop *TheLoop = LTP.ExclusiveLoops[LoopDepth].getLoop(&LI);

  // Get the maximum safe dependence distance in bits computed by LAA.
  // It is computed by MaxVF * sizeOf(type) * 8, where type is taken from
  // the memory accesses that is most restrictive (involved in the smallest
  // dependence distance).

  // unsigned MaxSafeElements =
  //     llvm::bit_floor(Legal->getMaxSafeVectorWidthInBits(CurL) / WidestType);
  // TODO. yg0412.yun, we need to set it by getMaxSafeVectorWidthInBits(CurL);
  unsigned MaxSafeElements = 20;

  auto MaxSafeFixedVF = ElementCount::getFixed(MaxSafeElements);
  auto MaxSafeScalableVF = getMaxLegalScalableVF(MaxSafeElements, CurL);

  LLVM_DEBUG(dbgs() << "LV: The max safe fixed VF is: " << MaxSafeFixedVF
                    << ".\n");
  LLVM_DEBUG(dbgs() << "LV: The max safe scalable VF is: " << MaxSafeScalableVF
                    << ".\n");

  // First analyze the UserTF, fall back if the UserTF should be ignored.
  if (UserTF) {
    auto MaxSafeUserTF =
        UserTF.isScalable() ? MaxSafeScalableVF : MaxSafeFixedVF;

    if (ElementCount::isKnownLE(UserTF, MaxSafeUserTF)) {
      // If `VF=vscale x N` is safe, then so is `VF=N`
      if (UserTF.isScalable())
        return FixedScalableVFPair(
            ElementCount::getFixed(UserTF.getKnownMinValue()), UserTF);
      else
        return UserTF;
    }

    assert(ElementCount::isKnownGT(UserTF, MaxSafeUserTF));

    // Only clamp if the UserTF is not scalable. If the UserTF is scalable, it
    // is better to ignore the hint and let the compiler choose a suitable VF.
    if (!UserTF.isScalable()) {
      LLVM_DEBUG(dbgs() << "LT: User VF=" << UserTF
                        << " is unsafe, clamping to max safe VF="
                        << MaxSafeFixedVF << ".\n");
      ORE->emit([&]() {
        return OptimizationRemarkAnalysis(DEBUG_TYPE, "VectorizationFactor",
                                          CurL->getStartLoc(),
                                          CurL->getHeader())
               << "User-specified vectorization factor "
               << ore::NV("UserVectorizationFactor", UserTF)
               << " is unsafe, clamping to maximum safe vectorization factor "
               << ore::NV("VectorizationFactor", MaxSafeFixedVF);
      });
      return MaxSafeFixedVF;
    }

    if (!TTI.supportsScalableVectors() &&
        !TPForceTargetSupportsScalableVectors) {
      LLVM_DEBUG(dbgs() << "LT: User VF=" << UserTF
                        << " is ignored because scalable vectors are not "
                           "available.\n");
      ORE->emit([&]() {
        return OptimizationRemarkAnalysis(DEBUG_TYPE, "VectorizationFactor",
                                          CurL->getStartLoc(),
                                          CurL->getHeader())
               << "User-specified vectorization factor "
               << ore::NV("UserVectorizationFactor", UserTF)
               << " is ignored because the target does not support scalable "
                  "vectors. The compiler will pick a more suitable value.";
      });
    } else {
      LLVM_DEBUG(dbgs() << "LT: User VF=" << UserTF
                        << " is unsafe. Ignoring scalable UserTF.\n");
      ORE->emit([&]() {
        return OptimizationRemarkAnalysis(DEBUG_TYPE, "VectorizationFactor",
                                          CurL->getStartLoc(),
                                          CurL->getHeader())
               << "User-specified vectorization factor "
               << ore::NV("UserVectorizationFactor", UserTF)
               << " is unsafe. Ignoring the hint to let the compiler pick a "
                  "more suitable value.";
      });
    }
  }

  LLVM_DEBUG(dbgs() << "LT: The Smallest and Widest types: " << SmallestType
                    << " / " << WidestType << " bits.\n");

  // FixedScalableVFPair Result(ElementCount::getFixed(1),
  //                            ElementCount::getScalable(0));
  // if (auto MaxVF =
  //         getMaximizedVFForTarget(MaxTripCount, SmallestType, WidestType,
  //                                 MaxSafeFixedVF, FoldTailByMasking))
  //   Result.FixedVF = MaxVF;

  // if (auto MaxVF =
  //         getMaximizedVFForTarget(MaxTripCount, SmallestType, WidestType,
  //                                 MaxSafeScalableVF, FoldTailByMasking))
  //   if (MaxVF.isScalable()) {
  //     Result.ScalableVF = MaxVF;
  //     LLVM_DEBUG(dbgs() << "LV: Found feasible scalable VF = " << MaxVF
  //                       << "\n");
  //   }

  // return Result;
}

// Return whether we allow using masked interleave-groups (for dealing with
// strided loads/stores that reside in predicated blocks, or for dealing
// with gaps).
bool LoopTensorizeCostModel::useMaskedInterleavedAccesses(
    const TargetTransformInfo &TTI) {
  // If an override option has been passed in for interleaved accesses, use it.
  if (TPEnableMaskedInterleavedMemAccesses.getNumOccurrences() > 0)
    return TPEnableMaskedInterleavedMemAccesses;

  return TTI.enableMaskedInterleavedAccessVectorization();
}

void LoopTensorizeCostModel::setTailFoldingStyles(bool IsScalableVF,
                                                  unsigned UserIC) {
  assert(!ChosenTailFoldingStyle && "Tail folding must not be selected yet.");
  if (!Legal->canFoldTailByMasking()) {
    ChosenTailFoldingStyle =
        std::make_pair(TailFoldingStyle::None, TailFoldingStyle::None);
    return;
  }

  if (!TPForceTailFoldingStyle.getNumOccurrences()) {
    ChosenTailFoldingStyle = std::make_pair(
        TTI.getPreferredTailFoldingStyle(/*IVUpdateMayOverflow=*/true),
        TTI.getPreferredTailFoldingStyle(/*IVUpdateMayOverflow=*/false));
    return;
  }

  // Set styles when forced.
  ChosenTailFoldingStyle = std::make_pair(TPForceTailFoldingStyle.getValue(),
                                          TPForceTailFoldingStyle.getValue());
  if (TPForceTailFoldingStyle != TailFoldingStyle::DataWithEVL)
    return;
  // Override forced styles if needed.
  // FIXME: use actual opcode/data type for analysis here.
  // FIXME: Investigate opportunity for fixed vector factor.
  bool EVLIsLegal = IsScalableVF && UserIC <= 1 &&
                    TTI.hasActiveVectorLength() &&
                    Legal->isSafeForAnyVectorWidth(Loops.back());
  if (!EVLIsLegal) {
    // If for some reason EVL mode is unsupported, fallback to
    // DataWithoutLaneMask to try to vectorize the loop with folded tail
    // in a generic way.
    ChosenTailFoldingStyle =
        std::make_pair(TailFoldingStyle::DataWithoutLaneMask,
                       TailFoldingStyle::DataWithoutLaneMask);
    LLVM_DEBUG(
        dbgs() << "LV: Preference for VP intrinsics indicated. Will "
                  "not try to generate VP Intrinsics "
               << (UserIC > 1
                       ? "since interleave count specified is greater than 1.\n"
                       : "due to non-interleaving reasons.\n"));
  }
}

std::optional<unsigned>
LoopTensorizeCostModel::getMaxVScale(const Function &F,
                                     const TargetTransformInfo &TTI) {
  if (std::optional<unsigned> MaxVScale = TTI.getMaxVScale())
    return MaxVScale;

  if (F.hasFnAttribute(Attribute::VScaleRange))
    return F.getFnAttribute(Attribute::VScaleRange).getVScaleRangeMax();

  return std::nullopt;
}

FixedScalableVFPair LoopTensorizeCostModel::computeMaxVF(ElementCount UserTF,
                                                         unsigned UserIC,
                                                         Loop *CurL) {
  if (CurL->isInnermost() && TTI.hasBranchDivergence()) {
    if (Legal->getRuntimePointerChecking(CurL)->Need) {
      // TODO: It may by useful to do since it's still likely to be dynamically
      // uniform if the target can skip.
      reportTensorizationFailure(
          "Not inserting runtime ptr check for divergent target",
          "runtime pointer checks needed. Not enabled for divergent target",
          "CantVersionLoopWithDivergentTarget", ORE, CurL);
      return FixedScalableVFPair::getNone();
    }
  }
  Loop *L = CurL;

  PredicatedScalarEvolution *PSE = Loop2PSE[L];
  unsigned TC = PSE->getSE()->getSmallConstantTripCount(CurL);
  // YYG::REMOVE
  // llvm::errs() << "TC : " << TC << "\n";
  unsigned MaxTC = PSE->getSE()->getSmallConstantMaxTripCount(CurL);
  // YYG::REMOVE
  // llvm::errs() << "MaxTC : " << MaxTC << "\n";
  LLVM_DEBUG(dbgs() << "LT: Found trip count: " << TC << '\n');
  if (TC == 1) {
    reportTensorizationFailure(
        "Single iteration (non) loop",
        "loop trip count is one, irrelevant for vectorization",
        "SingleIterationLoop", ORE, CurL);
    return FixedScalableVFPair::getNone();
  }

  switch (ScalarEpilogueStatus) {
  case CM_ScalarEpilogueAllowed:
    return computeFeasibleMaxVF(MaxTC, UserTF, false, CurL);
  case CM_ScalarEpilogueNotAllowedUsePredicate:
    [[fallthrough]];
  case CM_ScalarEpilogueNotNeededUsePredicate:
    LLVM_DEBUG(
        dbgs() << "LT: vector predicate hint/switch found.\n"
               << "LT: Not allowing scalar epilogue, creating predicated "
               << "vector loop.\n");
    break;
  case CM_ScalarEpilogueNotAllowedLowTripLoop:
    // fallthrough as a special case of OptForSize
  case CM_ScalarEpilogueNotAllowedOptSize:
    if (ScalarEpilogueStatus == CM_ScalarEpilogueNotAllowedOptSize)
      LLVM_DEBUG(
          dbgs() << "LT: Not allowing scalar epilogue due to -Os/-Oz.\n");
    else
      LLVM_DEBUG(dbgs() << "LT: Not allowing scalar epilogue due to low trip "
                        << "count.\n");

    // // Bail if runtime checks are required, which are not good when
    // optimising
    // // for size.
    // if (runtimeChecksRequired())
    //   return FixedScalableVFPair::getNone();

    break;
  }

  // The only loops we can vectorize without a scalar epilogue, are loops with
  // a bottom-test and a single exiting block. We'd have to handle the fact
  // that not every instruction executes on the last iteration.  This will
  // require a lane mask which varies through the vector loop body.  (TODO)
  if (CurL->getExitingBlock() != CurL->getLoopLatch()) {
    // If there was a tail-folding hint/switch, but we can't fold the tail by
    // masking, fallback to a vectorization with a scalar epilogue.
    if (ScalarEpilogueStatus == CM_ScalarEpilogueNotNeededUsePredicate) {
      LLVM_DEBUG(dbgs() << "LT: Cannot fold tail by masking: vectorize with a "
                           "scalar epilogue instead.\n");
      ScalarEpilogueStatus = CM_ScalarEpilogueAllowed;
      return computeFeasibleMaxVF(MaxTC, UserTF, false, CurL);
    }
    return FixedScalableVFPair::getNone();
  }

  // Now try the tail folding

  // Invalidate interleave groups that require an epilogue if we can't mask
  // the interleave-group.
  if (!useMaskedInterleavedAccesses(TTI)) {
    assert(WideningDecisions.empty() && Uniforms.empty() && Scalars.empty() &&
           "No decisions should have been taken at this point");
    // Note: There is no need to invalidate any cost modeling decisions here, as
    // non where taken so far.
    Loop2IAI[L]->invalidateGroupsRequiringScalarEpilogue();
  }

  FixedScalableVFPair MaxFactors =
      computeFeasibleMaxVF(MaxTC, UserTF, true, CurL);

  // Avoid tail folding if the trip count is known to be a multiple of any VF
  // we choose.
  std::optional<unsigned> MaxPowerOf2RuntimeVF =
      MaxFactors.FixedVF.getFixedValue();
  if (MaxFactors.ScalableVF) {
    std::optional<unsigned> MaxVScale = getMaxVScale(*TheFunction, TTI);
    if (MaxVScale && TTI.isVScaleKnownToBeAPowerOfTwo()) {
      MaxPowerOf2RuntimeVF = std::max<unsigned>(
          *MaxPowerOf2RuntimeVF,
          *MaxVScale * MaxFactors.ScalableVF.getKnownMinValue());
    } else
      MaxPowerOf2RuntimeVF = std::nullopt; // Stick with tail-folding for now.
  }

  if (MaxPowerOf2RuntimeVF && *MaxPowerOf2RuntimeVF > 0) {
    assert((UserTF.isNonZero() || isPowerOf2_32(*MaxPowerOf2RuntimeVF)) &&
           "MaxFixedVF must be a power of 2");
    unsigned MaxVFtimesIC =
        UserIC ? *MaxPowerOf2RuntimeVF * UserIC : *MaxPowerOf2RuntimeVF;

    Loop *L = CurL;
    PredicatedScalarEvolution *PSE = Loop2PSE[L];

    ScalarEvolution *SE = PSE->getSE();
    const SCEV *BackedgeTakenCount = PSE->getBackedgeTakenCount();
    const SCEV *ExitCount = SE->getAddExpr(
        BackedgeTakenCount, SE->getOne(BackedgeTakenCount->getType()));
    const SCEV *Rem = SE->getURemExpr(
        SE->applyLoopGuards(ExitCount, CurL),
        SE->getConstant(BackedgeTakenCount->getType(), MaxVFtimesIC));
    if (Rem->isZero()) {
      // Accept MaxFixedVF if we do not have a tail.
      LLVM_DEBUG(dbgs() << "LT: No tail will remain for any chosen VF.\n");
      return MaxFactors;
    }
  }

  // If we don't know the precise trip count, or if the trip count that we
  // found modulo the vectorization factor is not zero, try to fold the tail
  // by masking.
  // FIXME: look for a smaller MaxVF that does divide TC rather than masking.
  setTailFoldingStyles(MaxFactors.ScalableVF.isScalable(), UserIC);
  if (foldTailByMasking()) {
    if (getTailFoldingStyle() == TailFoldingStyle::DataWithEVL) {
      LLVM_DEBUG(
          dbgs()
          << "LT: tail is folded with EVL, forcing unroll factor to be 1. Will "
             "try to generate VP Intrinsics with scalable vector "
             "factors only.\n");
      // Tail folded loop using VP intrinsics restricts the VF to be scalable
      // for now.
      // TODO: extend it for fixed vectors, if required.
      assert(MaxFactors.ScalableVF.isScalable() &&
             "Expected scalable vector factor.");

      MaxFactors.FixedVF = ElementCount::getFixed(1);
    }
    return MaxFactors;
  }

  // If there was a tail-folding hint/switch, but we can't fold the tail by
  // masking, fallback to a vectorization with a scalar epilogue.
  if (ScalarEpilogueStatus == CM_ScalarEpilogueNotNeededUsePredicate) {
    LLVM_DEBUG(dbgs() << "LT: Cannot fold tail by masking: vectorize with a "
                         "scalar epilogue instead.\n");
    ScalarEpilogueStatus = CM_ScalarEpilogueAllowed;
    return MaxFactors;
  }

  if (ScalarEpilogueStatus == CM_ScalarEpilogueNotAllowedUsePredicate) {
    LLVM_DEBUG(dbgs() << "LT: Can't fold tail by masking: don't vectorize\n");
    return FixedScalableVFPair::getNone();
  }

  if (TC == 0) {
    reportTensorizationFailure(
        "Unable to calculate the loop count due to complex control flow",
        "unable to calculate the loop count due to complex control flow",
        "UnknownLoopCountComplexCFG", ORE, CurL);
    return FixedScalableVFPair::getNone();
  }

  reportTensorizationFailure(
      "Cannot optimize for size and vectorize at the same time.",
      "cannot optimize for size and vectorize at the same time. "
      "Enable vectorization of this loop with '#pragma clang loop "
      "vectorize(enable)' when compiling with -Os/-Oz",
      "NoTailLoopWithOptForSize", ORE, CurL);
  return FixedScalableVFPair::getNone();
}

/// Write a \p DebugMsg about tensorization to the debug output stream. If \p I
/// is passed, the message relates to that particular instruction.
#ifndef NDEBUG
static void debugTensorizationMessage(const StringRef Prefix,
                                      const StringRef DebugMsg,
                                      Instruction *I) {
  dbgs() << "LV: " << Prefix << DebugMsg;
  if (I != nullptr)
    dbgs() << " " << *I;
  else
    dbgs() << '.';
  dbgs() << '\n';
}
#endif

/// Reports an informative message: print \p Msg for debugging purposes as well
/// as an optimization remark. Uses either \p I as location of the remark, or
/// otherwise \p TheLoop.
static void reportTensorizationInfo(const StringRef Msg, const StringRef ORETag,
                                    OptimizationRemarkEmitter *ORE,
                                    Loop *TheLoop, Instruction *I = nullptr) {
  LLVM_DEBUG(debugTensorizationMessage("", Msg, I));
  // LoopTensorizeHints Hints(TheLoop, true /* doesn't matter */, *ORE);
  // ORE->emit(
  //     createLVAnalysis(Hints.vectorizeAnalysisPassName(), ORETag, TheLoop, I)
  //     << Msg);
}

ElementCount
LoopTensorizeCostModel::getMaxLegalScalableVF(unsigned MaxSafeElements,
                                              Loop *CurL) {
  if (!isScalableVectorizationAllowed())
    return ElementCount::getScalable(0);

  auto MaxScalableVF = ElementCount::getScalable(
      std::numeric_limits<ElementCount::ScalarTy>::max());
  if (Legal->isSafeForAnyVectorWidth(CurL))
    return MaxScalableVF;

  std::optional<unsigned> MaxVScale = getMaxVScale(*TheFunction, TTI);
  // Limit MaxScalableVF by the maximum safe dependence distance.
  MaxScalableVF = ElementCount::getScalable(MaxSafeElements / *MaxVScale);

  if (!MaxScalableVF)
    reportTensorizationInfo(
        "Max legal vector width too small, scalable Tensorization "
        "unfeasible.",
        "ScalableVFUnfeasible", ORE, CurL);

  return MaxScalableVF;
}

bool LoopTensorizeCostModel::isScalableVectorizationAllowed() {
  if (IsScalableVectorizationAllowed)
    return *IsScalableVectorizationAllowed;

  IsScalableVectorizationAllowed = false;
  if (!TTI.supportsScalableVectors() && !TPForceTargetSupportsScalableVectors)
    return false;

  if (Hints->isScalableVectorizationDisabled()) {
    reportTensorizationInfo("Scalable vectorization is explicitly disabled",
                            "ScalableVectorizationDisabled", ORE, Loops[0]);
    return false;
  }

  LLVM_DEBUG(dbgs() << "LV: Scalable vectorization is available\n");

  auto MaxScalableVF = ElementCount::getScalable(
      std::numeric_limits<ElementCount::ScalarTy>::max());

  // Test that the loop-vectorizer can legalize all operations for this MaxVF.
  // FIXME: While for scalable vectors this is currently sufficient, this should
  // be replaced by a more detailed mechanism that filters out specific VFs,
  // instead of invalidating vectorization for a whole set of VFs based on the
  // MaxVF.

  // Disable scalable vectorization if the loop contains unsupported reductions.
  if (!canVectorizeReductions(MaxScalableVF)) {
    reportTensorizationInfo(
        "Scalable vectorization not supported for the reduction "
        "operations found in this loop.",
        "ScalableVFUnfeasible", ORE, Loops[0]);
    return false;
  }

  // Disable scalable vectorization if the loop contains any instructions
  // with element types not supported for scalable vectors.
  if (any_of(ElementTypesInLoop, [&](Type *Ty) {
        return !Ty->isVoidTy() &&
               !this->TTI.isElementTypeLegalForScalableVector(Ty);
      })) {
    reportTensorizationInfo("Scalable vectorization is not supported "
                            "for all element types found in this loop.",
                            "ScalableVFUnfeasible", ORE, Loops[0]);
    return false;
  }

  if (!Legal->isSafeForAnyVectorWidth(Loops[0]) &&
      !getMaxVScale(*TheFunction, TTI)) {
    reportTensorizationInfo("The target does not provide maximum vscale value "
                            "for safe distance analysis.",
                            "ScalableVFUnfeasible", ORE, Loops[0]);
    return false;
  }

  IsScalableVectorizationAllowed = true;
  return true;
}
