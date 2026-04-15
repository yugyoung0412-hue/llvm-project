#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/MatrixBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Tensorize/TPlan.h"
#include "llvm/Transforms/Tensorize/TPlanValue.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include <cassert>

using namespace llvm;

using VectorParts = SmallVector<Value *, 2>;

#define LV_NAME "loop-tensorize"
#define DEBUG_TYPE LV_NAME

// TODO(yuxin.an)
bool TPRecipeBase::mayWriteToMemory() const {
  switch (getTPDefID()) {
  case TPWidenStoreEVLSC:
  case TPWidenStoreSC:
    return true;
  case TPReplicateSC:
    return cast<Instruction>(getTPSingleValue()->getUnderlyingValue())
        ->mayWriteToMemory();
  case TPWidenCallSC:
    return !cast<TPWidenCallRecipe>(this)
                ->getCalledScalarFunction()
                ->onlyReadsMemory();
  case TPBranchOnMaskSC:
  case TPScalarIVStepsSC:
  case TPPredInstPHISC:
    return false;
  case TPBlendSC:
  case TPReductionEVLSC:
  case TPReductionSC:
  case TPWidenCanonicalIVSC:
  case TPWidenCastSC:
  case TPWidenGEPSC:
  case TPWidenIntOrFpInductionSC:
  case TPWidenLoadEVLSC:
  case TPWidenLoadSC:
  case TPWidenPHISC:
  case TPWidenSC:
  case TPWidenSelectSC: {
    const Instruction *I =
        dyn_cast_or_null<Instruction>(getTPSingleValue()->getUnderlyingValue());
    (void)I;
    assert((!I || !I->mayWriteToMemory()) &&
           "underlying instruction may write to memory");
    return false;
  }
  default:
    return true;
  }
}

// TODO(yuxin.an)
bool TPRecipeBase::mayReadFromMemory() const {
  switch (getTPDefID()) {
  case TPWidenLoadEVLSC:
  case TPWidenLoadSC:
    return true;
  case TPReplicateSC:
    return cast<Instruction>(getTPSingleValue()->getUnderlyingValue())
        ->mayReadFromMemory();
  case TPWidenCallSC:
    return !cast<TPWidenCallRecipe>(this)
                ->getCalledScalarFunction()
                ->onlyWritesMemory();
  case TPBranchOnMaskSC:
  case TPPredInstPHISC:
  case TPScalarIVStepsSC:
  case TPWidenStoreEVLSC:
  case TPWidenStoreSC:
    return false;
  case TPBlendSC:
  case TPReductionEVLSC:
  case TPReductionSC:
  case TPWidenCanonicalIVSC:
  case TPWidenCastSC:
  case TPWidenGEPSC:
  case TPWidenIntOrFpInductionSC:
  case TPWidenPHISC:
  case TPWidenSC:
  case TPWidenSelectSC: {
    const Instruction *I =
        dyn_cast_or_null<Instruction>(getTPSingleValue()->getUnderlyingValue());
    (void)I;
    assert((!I || !I->mayReadFromMemory()) &&
           "underlying instruction may read from memory");
    return false;
  }
  default:
    return true;
  }
}

// TODO(yuxin.an)
bool TPRecipeBase::mayHaveSideEffects() const {
  switch (getTPDefID()) {
  case TPDerivedIVSC:
  case TPPredInstPHISC:
  case TPScalarCastSC:
    return false;
  case TPInstructionSC:
    switch (cast<TPInstruction>(this)->getOpcode()) {
    case Instruction::Or:
    case Instruction::ICmp:
    case Instruction::Select:
    case TPInstruction::Not:
    case TPInstruction::CalculateTripCountMinusVF:
    case TPInstruction::CanonicalIVIncrementForPart:
    case TPInstruction::ExtractFromEnd:
    case TPInstruction::FirstOrderRecurrenceSplice:
    case TPInstruction::LogicalAnd:
    case TPInstruction::PtrAdd:
      return false;
    default:
      return true;
    }
  case TPWidenCallSC: {
    Function *Fn = cast<TPWidenCallRecipe>(this)->getCalledScalarFunction();
    return mayWriteToMemory() || !Fn->doesNotThrow() || !Fn->willReturn();
  }
  case TPBlendSC:
  case TPReductionEVLSC:
  case TPReductionSC:
  case TPScalarIVStepsSC:
  case TPWidenCanonicalIVSC:
  case TPWidenCastSC:
  case TPWidenGEPSC:
  case TPWidenIntOrFpInductionSC:
  case TPWidenPHISC:
  case TPWidenPointerInductionSC:
  case TPWidenSC:
  case TPWidenSelectSC: {
    const Instruction *I =
        dyn_cast_or_null<Instruction>(getTPSingleValue()->getUnderlyingValue());
    (void)I;
    assert((!I || !I->mayHaveSideEffects()) &&
           "underlying instruction has side-effects");
    return false;
  }
  case TPInterleaveSC:
    return mayWriteToMemory();
  case TPWidenLoadEVLSC:
  case TPWidenLoadSC:
  case TPWidenStoreEVLSC:
  case TPWidenStoreSC:
    assert(
        cast<TPWidenMemoryRecipe>(this)->getIngredient().mayHaveSideEffects() ==
            mayWriteToMemory() &&
        "mayHaveSideffects result for ingredient differs from this "
        "implementation");
    return mayWriteToMemory();
  case TPReplicateSC: {
    auto *R = cast<TPReplicateRecipe>(this);
    return R->getUnderlyingInstr()->mayHaveSideEffects();
  }
  default:
    return true;
  }
}

void TPLiveOut::fixPhi(TPlan &Plan, TPTransformState &State) {
  llvm_unreachable("");
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void TPLiveOut::print(raw_ostream &O, TPSlotTracker &SlotTracker) const {
  O << "Live-out ";
  getPhi()->printAsOperand(O);
  O << " = ";
  getOperand(0)->printAsOperand(O, SlotTracker);
  O << "\n";
}
#endif

void TPRecipeBase::insertBefore(TPRecipeBase *InsertPos) {
  assert(!Parent && "Recipe already in some VPBasicBlock");
  assert(InsertPos->getParent() &&
         "Insertion position not in any VPBasicBlock");
  InsertPos->getParent()->insert(this, InsertPos->getIterator());
}

void TPRecipeBase::insertBefore(TPBasicBlock &BB,
                                iplist<TPRecipeBase>::iterator I) {
  assert(!Parent && "Recipe already in some VPBasicBlock");
  assert(I == BB.end() || I->getParent() == &BB);
  BB.insert(this, I);
}

void TPRecipeBase::insertAfter(TPRecipeBase *InsertPos) {
  assert(!Parent && "Recipe already in some VPBasicBlock");
  assert(InsertPos->getParent() &&
         "Insertion position not in any VPBasicBlock");
  InsertPos->getParent()->insert(this, std::next(InsertPos->getIterator()));
}

void TPRecipeBase::removeFromParent() {
  assert(getParent() && "Recipe not in any VPBasicBlock");
  getParent()->getRecipeList().remove(getIterator());
  Parent = nullptr;
}

iplist<TPRecipeBase>::iterator TPRecipeBase::eraseFromParent() {
  assert(getParent() && "Recipe not in any VPBasicBlock");
  return getParent()->getRecipeList().erase(getIterator());
}

void TPRecipeBase::moveAfter(TPRecipeBase *InsertPos) {
  removeFromParent();
  insertAfter(InsertPos);
}

void TPRecipeBase::moveBefore(TPBasicBlock &BB,
                              iplist<TPRecipeBase>::iterator I) {
  removeFromParent();
  insertBefore(BB, I);
}

InstructionCost TPRecipeBase::cost(ElementCount VF, TPCostContext &Ctx) {
  // TODO(yuxin.an)
  llvm_unreachable("");
}

InstructionCost TPRecipeBase::computeCost(ElementCount VF,
                                          TPCostContext &Ctx) const {
  // TODO(yuxin.an)
  llvm_unreachable("");
}

FastMathFlags TPRecipeWithIRFlags::getFastMathFlags() const {
  assert(OpType == OperationType::FPMathOp &&
         "recipe doesn't have fast math flags");
  FastMathFlags Res;
  Res.setAllowReassoc(FMFs.AllowReassoc);
  Res.setNoNaNs(FMFs.NoNaNs);
  Res.setNoInfs(FMFs.NoInfs);
  Res.setNoSignedZeros(FMFs.NoSignedZeros);
  Res.setAllowReciprocal(FMFs.AllowReciprocal);
  Res.setAllowContract(FMFs.AllowContract);
  Res.setApproxFunc(FMFs.ApproxFunc);
  return Res;
}

TPInstruction::TPInstruction(unsigned Opcode, CmpInst::Predicate Pred,
                             TPValue *A, TPValue *B, DebugLoc DL,
                             const Twine &Name)
    : TPRecipeWithIRFlags(TPDef::TPInstructionSC, ArrayRef<TPValue *>({A, B}),
                          Pred, DL),
      Opcode(Opcode), Name(Name.str()) {
  assert(Opcode == Instruction::ICmp &&
         "only ICmp predicates supported at the moment");
}

TPInstruction::TPInstruction(unsigned Opcode,
                             std::initializer_list<TPValue *> Operands,
                             FastMathFlags FMFs, DebugLoc DL, const Twine &Name)
    : TPRecipeWithIRFlags(TPDef::TPInstructionSC, Operands, FMFs, DL),
      Opcode(Opcode), Name(Name.str()) {
  // Make sure the VPInstruction is a floating-point operation.
  assert(isFPMathOp() && "this op can't take fast-math flags");
}

bool TPInstruction::doesGeneratePerAllLanes() const {
  return Opcode == TPInstruction::PtrAdd && !tputils::onlyFirstLaneUsed(this);
}

bool TPInstruction::canGenerateScalarForFirstLane() const {
  if (Instruction::isBinaryOp(getOpcode()))
    return true;
  if (isSingleScalar() || isTensorToScalar())
    return true;
  switch (Opcode) {
  case Instruction::ICmp:
  case TPInstruction::BranchOnCond:
  case TPInstruction::BranchOnCount:
  case TPInstruction::CalculateTripCountMinusVF:
  case TPInstruction::CanonicalIVIncrementForPart:
  case TPInstruction::PtrAdd:
  case TPInstruction::ExplicitTensorLength:
    return true;
  default:
    return false;
  }
}

Value *TPInstruction::generatePerLane(TPTransformState &State,
                                      const TPIteration &Lane) {
  llvm_unreachable("");
}

Value *TPInstruction::generatePerPart(TPTransformState &State, unsigned Part) {
  IRBuilderBase &Builder = State.Builder;

  if (Instruction::isBinaryOp(getOpcode())) {
    bool OnlyFirstLaneUsed = tputils::onlyFirstLaneUsed(this);
    Value *A = State.get(getOperand(0), Part, OnlyFirstLaneUsed);
    Value *B = State.get(getOperand(1), Part, OnlyFirstLaneUsed);
    auto *Res =
        Builder.CreateBinOp((Instruction::BinaryOps)getOpcode(), A, B, Name);
    if (auto *I = dyn_cast<Instruction>(Res))
      setFlags(I);
    return Res;
  }

  switch (getOpcode()) {
  case TPInstruction::Not: {
    Value *A = State.get(getOperand(0), Part);
    return Builder.CreateNot(A, Name);
  }
  case Instruction::ICmp: {
    bool OnlyFirstLaneUsed = tputils::onlyFirstLaneUsed(this);
    Value *A = State.get(getOperand(0), Part, OnlyFirstLaneUsed);
    Value *B = State.get(getOperand(1), Part, OnlyFirstLaneUsed);
    return Builder.CreateCmp(getPredicate(), A, B, Name);
  }
  case Instruction::Select: {
    Value *Cond = State.get(getOperand(0), Part);
    Value *Op1 = State.get(getOperand(1), Part);
    Value *Op2 = State.get(getOperand(2), Part);
    return Builder.CreateSelect(Cond, Op1, Op2, Name);
  }
  case TPInstruction::ActiveLaneMask: {
    llvm_unreachable("");
  }
  case TPInstruction::FirstOrderRecurrenceSplice: {
    // Generate code to combine the previous and current values in vector v3.
    //
    //   vector.ph:
    //     v_init = vector(..., ..., ..., a[-1])
    //     br vector.body
    //
    //   vector.body
    //     i = phi [0, vector.ph], [i+4, vector.body]
    //     v1 = phi [v_init, vector.ph], [v2, vector.body]
    //     v2 = a[i, i+1, i+2, i+3];
    //     v3 = vector(v1(3), v2(0, 1, 2))

    // For the first part, use the recurrence phi (v1), otherwise v2.
    auto *V1 = State.get(getOperand(0), 0);
    Value *PartMinus1 = Part == 0 ? V1 : State.get(getOperand(1), Part - 1);
    if (!PartMinus1->getType()->isVectorTy())
      return PartMinus1;
    Value *V2 = State.get(getOperand(1), Part);
    return Builder.CreateVectorSplice(PartMinus1, V2, -1, Name);
  }
  case TPInstruction::CalculateTripCountMinusVF: {
    llvm_unreachable("");
  }
  case TPInstruction::ExplicitTensorLength: {
    llvm_unreachable("");
  }
  case TPInstruction::CanonicalIVIncrementForPart: {
    llvm_unreachable("");
  }
  case TPInstruction::BranchOnCond: {
    llvm_unreachable("");
  }
  case TPInstruction::BranchOnCount: {
    if (Part != 0)
      return nullptr;
    // First create the compare.
    Value *IV = State.get(getOperand(0), Part, /*IsScalar*/ true);
    Value *TC = State.get(getOperand(1), Part, /*IsScalar*/ true);
    Value *Cond = Builder.CreateICmpEQ(IV, TC);

    // Now create the branch.
    auto *Plan = getParent()->getPlan();
    TPRegionBlock *TopRegion = Plan->getTensorLoopRegion();
    TPBasicBlock *Header = TopRegion->getEntry()->getEntryBasicBlock();

    // Replace the temporary unreachable terminator with a new conditional
    // branch, hooking it up to backward destination (the header) now and to the
    // forward destination (the exit/middle block) later when it is created.
    // Note that CreateCondBr expects a valid BB as first argument, so we need
    // to set it to nullptr later.
    BranchInst *CondBr = Builder.CreateCondBr(Cond, Builder.GetInsertBlock(),
                                              State.CFG.TPBB2IRBB[Header]);
    CondBr->setSuccessor(0, nullptr);
    Builder.GetInsertBlock()->getTerminator()->eraseFromParent();
    return CondBr;
  }
  case TPInstruction::ComputeReductionResult: {
    // TODO(yuxin.an)
    llvm_unreachable("");
  }
  case TPInstruction::ExtractFromEnd: {
    llvm_unreachable("");
  }
  case TPInstruction::LogicalAnd: {
    Value *A = State.get(getOperand(0), Part);
    Value *B = State.get(getOperand(1), Part);
    return Builder.CreateLogicalAnd(A, B, Name);
  }
  case TPInstruction::PtrAdd: {
    assert(tputils::onlyFirstLaneUsed(this) &&
           "can only generate first lane for PtrAdd");
    Value *Ptr = State.get(getOperand(0), Part, /* IsScalar */ true);
    Value *Addend = State.get(getOperand(1), Part, /* IsScalar */ true);
    return Builder.CreatePtrAdd(Ptr, Addend, Name);
  }
  case TPInstruction::ResumePhi: {
    if (Part != 0)
      return State.get(this, 0, /*IsScalar*/ true);
    Value *IncomingFromVPlanPred =
        State.get(getOperand(0), Part, /* IsScalar */ true);
    Value *IncomingFromOtherPreds =
        State.get(getOperand(1), Part, /* IsScalar */ true);
    auto *NewPhi =
        Builder.CreatePHI(IncomingFromOtherPreds->getType(), 2, Name);
    BasicBlock *VPlanPred =
        State.CFG
            .TPBB2IRBB[cast<TPBasicBlock>(getParent()->getSinglePredecessor())];
    NewPhi->addIncoming(IncomingFromVPlanPred, VPlanPred);
    for (auto *OtherPred : predecessors(Builder.GetInsertBlock())) {
      assert(OtherPred != VPlanPred &&
             "VPlan predecessors should not be connected yet");
      NewPhi->addIncoming(IncomingFromOtherPreds, OtherPred);
    }
    return NewPhi;
  }

  default:
    llvm_unreachable("Unsupported opcode for instruction");
  }
}

bool TPInstruction::isTensorToScalar() const {
  return getOpcode() == TPInstruction::ExtractFromEnd ||
         getOpcode() == TPInstruction::ComputeReductionResult;
}

bool TPInstruction::isSingleScalar() const {
  return getOpcode() == TPInstruction::ResumePhi;
}

#if !defined(NDEBUG)
bool TPInstruction::isFPMathOp() const {
  // Inspired by FPMathOperator::classof. Notable differences are that we don't
  // support Call, PHI and Select opcodes here yet.
  return Opcode == Instruction::FAdd || Opcode == Instruction::FMul ||
         Opcode == Instruction::FNeg || Opcode == Instruction::FSub ||
         Opcode == Instruction::FDiv || Opcode == Instruction::FRem ||
         Opcode == Instruction::FCmp || Opcode == Instruction::Select;
}
#endif

void TPInstruction::execute(TPTransformState &State) {

  auto GetOperands = [&]() {
    SmallVector<Value *> Vals;
    for (auto *Elem : operands()) {
      Value *Val = State.TPValue2Value.count(Elem) ? State.TPValue2Value[Elem]
                                                   : Elem->getUnderlyingValue();
      assert(Val && "");
      Vals.push_back(Val);
    }
    return Vals;
  };

  SmallVector<Value *> Vals = GetOperands();

  Value *Res = nullptr;

  IRBuilder<> Builder(State.CurBB->getTerminator());
  IRBuilderBase::FastMathFlagGuard FMFGuard(Builder);
  assert((hasFastMathFlags() == isFPMathOp() ||
          getOpcode() == Instruction::Select) &&
         "Recipe not a FPMathOp but has fast-math flags?");
  if (hasFastMathFlags())
    Builder.setFastMathFlags(getFastMathFlags());

  if (Instruction::isBinaryOp(getOpcode())) {
    Res = Builder.CreateBinOp((Instruction::BinaryOps)Opcode, Vals[0], Vals[1],
                              Name);
    if (auto *I = dyn_cast<Instruction>(Res))
      setFlags(I);
    State.TPValue2Value[this] = Res;

    if (getOpcode() == Instruction::Add)
      State.IdxAddMap[State.CurBB] = Res;
  } else {
    switch (getOpcode()) {
    case TPInstruction::Not: {
      Res = Builder.CreateNot(Vals[0]);
      State.TPValue2Value[this] = Res;
      break;
    }
    case Instruction::ICmp: {
      Res = Builder.CreateCmp(getPredicate(), Vals[0], Vals[1], Name);
      State.TPValue2Value[this] = Res;
      break;
    }
    case Instruction::Select: {
      Res = Builder.CreateSelect(Vals[0], Vals[1], Vals[2], Name);
      State.TPValue2Value[this] = Res;
      break;
    }
    case TPInstruction::BranchOnCond: {
      BranchInst *CondBr =
          Builder.CreateCondBr(Vals[0], Builder.GetInsertBlock(), nullptr);
      CondBr->setSuccessor(0, nullptr);
      Builder.GetInsertBlock()->getTerminator()->eraseFromParent();
      if (getParent()->isExiting()) {
        llvm_unreachable("");
      }
      Res = CondBr;
      State.TPValue2Value[this] = Res;
      break;
    }
    case TPInstruction::BranchOnCount: {
      Value *Cond = Builder.CreateICmpEQ(Vals[0], Vals[1]);

      auto *Successor0 =
          State.TPBB2BB[cast<TPBasicBlock>(getParent()->getSuccessors()[0])];
      auto *Successor1 =
          State.TPBB2BB[cast<TPBasicBlock>(getParent()->getSuccessors()[1])];

      BranchInst *CondBr = Builder.CreateCondBr(Cond, Successor0, Successor1);
      Builder.GetInsertBlock()->getTerminator()->eraseFromParent();
      Res = CondBr;
      State.TPValue2Value[this] = Res;
      break;
    }
    default:
      dump();
      llvm_unreachable("");
    };
  }

  assert(Res && "TPInstruction execute error.");
}

bool TPInstruction::onlyFirstLaneUsed(const TPValue *Op) const {
  assert(is_contained(operands(), Op) && "Op must be an operand of the recipe");
  if (Instruction::isBinaryOp(getOpcode()))
    return tputils::onlyFirstLaneUsed(this);

  switch (getOpcode()) {
  default:
    return false;
  case Instruction::ICmp:
  case TPInstruction::PtrAdd:
    // TODO: Cover additional opcodes.
    return tputils::onlyFirstLaneUsed(this);
  case TPInstruction::ActiveLaneMask:
  case TPInstruction::ExplicitTensorLength:
  case TPInstruction::CalculateTripCountMinusVF:
  case TPInstruction::CanonicalIVIncrementForPart:
  case TPInstruction::BranchOnCount:
  case TPInstruction::BranchOnCond:
  case TPInstruction::ResumePhi:
    return true;
  };
  llvm_unreachable("switch should return");
}

bool TPInstruction::onlyFirstPartUsed(const TPValue *Op) const {
  assert(is_contained(operands(), Op) && "Op must be an operand of the recipe");
  if (Instruction::isBinaryOp(getOpcode()))
    return tputils::onlyFirstPartUsed(this);

  switch (getOpcode()) {
  default:
    return false;
  case Instruction::ICmp:
  case Instruction::Select:
    return tputils::onlyFirstPartUsed(this);
  case TPInstruction::BranchOnCount:
  case TPInstruction::BranchOnCond:
  case TPInstruction::CanonicalIVIncrementForPart:
    return true;
  };
  llvm_unreachable("switch should return");
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void TPInstruction::dump() const {
  TPSlotTracker SlotTracker(getParent()->getPlan());
  print(dbgs(), "", SlotTracker);
}

void TPInstruction::print(raw_ostream &O, const Twine &Indent,
                          TPSlotTracker &SlotTracker) const {
  O << Indent << "EMIT ";

  if (hasResult()) {
    printAsOperand(O, SlotTracker);
    O << " = ";
  }

  switch (getOpcode()) {
  case TPInstruction::Not:
    O << "not";
    break;
  case TPInstruction::SLPLoad:
    O << "combined load";
    break;
  case TPInstruction::SLPStore:
    O << "combined store";
    break;
  case TPInstruction::ActiveLaneMask:
    O << "active lane mask";
    break;
  case TPInstruction::ResumePhi:
    O << "resume-phi";
    break;
  case TPInstruction::ExplicitTensorLength:
    O << "EXPLICIT-TENSOR-LENGTH";
    break;
  case TPInstruction::FirstOrderRecurrenceSplice:
    O << "first-order splice";
    break;
  case TPInstruction::BranchOnCond:
    O << "branch-on-cond";
    break;
  case TPInstruction::CalculateTripCountMinusVF:
    O << "TC > VF ? TC - VF : 0";
    break;
  case TPInstruction::CanonicalIVIncrementForPart:
    O << "VF * Part +";
    break;
  case TPInstruction::BranchOnCount:
    O << "branch-on-count";
    break;
  case TPInstruction::ExtractFromEnd:
    O << "extract-from-end";
    break;
  case TPInstruction::ComputeReductionResult:
    O << "compute-reduction-result";
    break;
  case TPInstruction::LogicalAnd:
    O << "logical-and";
    break;
  case TPInstruction::PtrAdd:
    O << "ptradd";
    break;
  default:
    O << Instruction::getOpcodeName(getOpcode());
  }

  printFlags(O);
  printOperands(O, SlotTracker);

  if (auto DL = getDebugLoc()) {
    O << ", !dbg ";
    DL.print(O);
  }
}
#endif

void TPWidenCallRecipe::execute(TPTransformState &State) {
  // TODO(yuxin.an)
  llvm_unreachable("");
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void TPWidenCallRecipe::print(raw_ostream &O, const Twine &Indent,
                              TPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-CALL ";

  Function *CalledFn = getCalledScalarFunction();
  if (CalledFn->getReturnType()->isVoidTy())
    O << "void ";
  else {
    printAsOperand(O, SlotTracker);
    O << " = ";
  }

  O << "call @" << CalledFn->getName() << "(";
  interleaveComma(arg_operands(), O, [&O, &SlotTracker](TPValue *Op) {
    Op->printAsOperand(O, SlotTracker);
  });
  O << ")";

  if (VectorIntrinsicID)
    O << " (using vector intrinsic)";
  else {
    O << " (using library function";
    if (Variant->hasName())
      O << ": " << Variant->getName();
    O << ")";
  }
}
#endif

void TPMatrixCallRecipe::execute(TPTransformState &State) {
  IRBuilder<> Builder(State.CurBB->getTerminator());
  MatrixBuilder MBuilder(Builder);
  Type *ElemTy = State.Plan->getPattern()->Info.ElementTy;

  Value *Res = nullptr;

  auto GetOperands = [&]() {
    SmallVector<Value *> Vals;
    for (auto *Elem : operands()) {
      Value *Val = State.TPValue2Value.count(Elem) ? State.TPValue2Value[Elem]
                                                   : Elem->getUnderlyingValue();
      assert(Val && "");
      Vals.push_back(Val);
    }
    return Vals;
  };

  auto GetSExtVal = [](Value *Val) {
    return cast<ConstantInt>(Val)->getSExtValue();
  };

  SmallVector<Value *> Vals = GetOperands();

  switch (MatrixIntrinsicID) {
  case Intrinsic::matrix_column_major_load:
    Res = MBuilder.CreateColumnMajorLoad(ElemTy, Vals[0], Align(), Vals[1],
                                         true, GetSExtVal(Vals[2]),
                                         GetSExtVal(Vals[3]));
    State.TPValue2Value[this] = Res;
    break;
  case Intrinsic::matrix_column_major_load_addr_space_ext:
    Res = MBuilder.CreateColumnMajorLoadAddrSpaceExt(
        ElemTy, Vals[0], Align(), Vals[1], true, GetSExtVal(Vals[2]),
        GetSExtVal(Vals[3]));
    State.TPValue2Value[this] = Res;
    break;
  case Intrinsic::matrix_column_major_store:
    MBuilder.CreateColumnMajorStore(Vals[0], Vals[1], Align(), Vals[2], true,
                                    GetSExtVal(Vals[3]), GetSExtVal(Vals[4]));
    break;
  case Intrinsic::matrix_column_major_store_addr_space_ext:
    MBuilder.CreateColumnMajorStoreAddrSpaceExt(
        Vals[0], Vals[1], Align(), Vals[2], true, GetSExtVal(Vals[3]),
        GetSExtVal(Vals[4]));
    break;
  case Intrinsic::matrix_multiply:
    Res =
        MBuilder.CreateMatrixMultiply(Vals[0], Vals[1], GetSExtVal(Vals[2]),
                                      GetSExtVal(Vals[3]), GetSExtVal(Vals[4]));
    State.TPValue2Value[this] = Res;
    break;
  case Intrinsic::matrix_transpose:
    Res = MBuilder.CreateMatrixTranspose(Vals[0], GetSExtVal(Vals[1]),
                                         GetSExtVal(Vals[2]));
    State.TPValue2Value[this] = Res;
    break;
  case Intrinsic::tensor_new_load:
    Res = MBuilder.CreateTensorLoad(ElemTy, Vals[0], Align(), true,
                                    GetSExtVal(Vals[1]), GetSExtVal(Vals[2]),
                                    GetSExtVal(Vals[3]), GetSExtVal(Vals[4]));
    State.TPValue2Value[this] = Res;
    break;
  case Intrinsic::tensor_new_store:
    MBuilder.CreateTensorStore(ElemTy, Vals[0], Vals[1], Align(), Vals[2], true);
    break;
  case Intrinsic::tensor_convolution_2d:
    Res = MBuilder.CreateTensorConvolution2D(
        Vals[0], Vals[1], cast<ConstantInt>(Vals[2]),
        cast<ConstantInt>(Vals[3]), cast<ConstantInt>(Vals[4]),
        cast<ConstantInt>(Vals[5]), cast<ConstantInt>(Vals[6]),
        cast<ConstantInt>(Vals[7]), cast<ConstantInt>(Vals[8]),
        cast<ConstantInt>(Vals[9]), cast<ConstantInt>(Vals[10]),
        cast<ConstantInt>(Vals[11]), cast<ConstantInt>(Vals[12]),
        Intrinsic::tensor_convolution_2d);
    State.TPValue2Value[this] = Res;
    break;
  case Intrinsic::tensor_multiply:
    Res = MBuilder.CreateTensorMultiply(
        ElemTy, Vals[0], Vals[1], GetSExtVal(Vals[2]), GetSExtVal(Vals[3]),
        GetSExtVal(Vals[4]));
    State.TPValue2Value[this] = Res;
    break;
  case Intrinsic::tensor_add:
    Res = MBuilder.CreateTensorElementWise(
        ElemTy, Vals[0], Vals[1], GetSExtVal(Vals[2]), GetSExtVal(Vals[3]),
        Intrinsic::tensor_add);
    State.TPValue2Value[this] = Res;
    break;
  case Intrinsic::tensor_sub:
    Res = MBuilder.CreateTensorElementWise(
        ElemTy, Vals[0], Vals[1], GetSExtVal(Vals[2]), GetSExtVal(Vals[3]),
        Intrinsic::tensor_sub);
    State.TPValue2Value[this] = Res;
    break;
  case Intrinsic::tensor_mul:
    Res = MBuilder.CreateTensorElementWise(
        ElemTy, Vals[0], Vals[1], GetSExtVal(Vals[2]), GetSExtVal(Vals[3]),
        Intrinsic::tensor_mul);
    State.TPValue2Value[this] = Res;
    break;
  case Intrinsic::tensor_maximum:
    Res = MBuilder.CreateTensorElementWise(
        ElemTy, Vals[0], Vals[1], GetSExtVal(Vals[2]), GetSExtVal(Vals[3]),
        Intrinsic::tensor_maximum);
    State.TPValue2Value[this] = Res;
    break;
  case Intrinsic::tensor_sqrt:
    Res = MBuilder.CreateTensorUnaryOperation(
        ElemTy, Vals[0], GetSExtVal(Vals[1]), GetSExtVal(Vals[2]),
        Intrinsic::tensor_sqrt);
    State.TPValue2Value[this] = Res;
    break;
  case Intrinsic::tensor_abs:
    Res = MBuilder.CreateTensorUnaryOperation(
        ElemTy, Vals[0], GetSExtVal(Vals[1]), GetSExtVal(Vals[2]),
        Intrinsic::tensor_abs);
    State.TPValue2Value[this] = Res;
    break;
  default:
    llvm_unreachable("");
  }

  // State.CurBB->getParent()->dump();
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void TPMatrixCallRecipe::print(raw_ostream &O, const Twine &Indent,
                               TPSlotTracker &SlotTracker) const {
  O << Indent << "MATRIX-CALL ";

  DenseMap<Intrinsic::ID, StringRef> FunctionName{
      {Intrinsic::matrix_transpose, "llvm.matrix.transpose"},
      {Intrinsic::matrix_multiply, "llvm.matrix.multiply"},
      {Intrinsic::matrix_column_major_load, "llvm.matrix.column.major.load"},
      {Intrinsic::matrix_column_major_load_addr_space_ext,
       "llvm.matrix.column.major.load.addr.space.ext"},
      {Intrinsic::matrix_column_major_store, "llvm.matrix.column.major.store"},
      {Intrinsic::tensor_new_load, "llvm.tensor.new.load"},
      {Intrinsic::tensor_new_store, "llvm.tensor.new.store"},
      {Intrinsic::tensor_convolution_2d, "llvm.tensor.convolution2D"},
      {Intrinsic::tensor_multiply, "llvm.tensor.multiply"},
  };

  if (MatrixIntrinsicID != Intrinsic::matrix_column_major_store) {
    printAsOperand(O, SlotTracker);
    O << " = ";
  }

  // Function *CalledFn = getCalledScalarFunction();
  // if (CalledFn->getReturnType()->isVoidTy())
  //   O << "void ";
  // else {
  //   printAsOperand(O, SlotTracker);
  //   O << " = ";
  // }

  O << "call @" << FunctionName[MatrixIntrinsicID] << "(";
  interleaveComma(arg_operands(), O, [&O, &SlotTracker](TPValue *Op) {
    Op->printAsOperand(O, SlotTracker);
  });
  O << ")";

  // if (MatrixIntrinsicID)
  //   O << " (using vector intrinsic)";
  // else {
  //   O << " (using library function";
  //   if (Variant->hasName())
  //     O << ": " << Variant->getName();
  //   O << ")";
  // }
}
#endif

void TPWidenSelectRecipe::execute(TPTransformState &State) {
  // TODO(yuxin.an)
  llvm_unreachable("");
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void TPWidenSelectRecipe::print(raw_ostream &O, const Twine &Indent,
                                TPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-SELECT ";
  printAsOperand(O, SlotTracker);
  O << " = select ";
  getOperand(0)->printAsOperand(O, SlotTracker);
  O << ", ";
  getOperand(1)->printAsOperand(O, SlotTracker);
  O << ", ";
  getOperand(2)->printAsOperand(O, SlotTracker);
  O << (isInvariantCond() ? " (condition is loop invariant)" : "");
}
#endif

TPRecipeWithIRFlags::FastMathFlagsTy::FastMathFlagsTy(
    const FastMathFlags &FMF) {
  AllowReassoc = FMF.allowReassoc();
  NoNaNs = FMF.noNaNs();
  NoInfs = FMF.noInfs();
  NoSignedZeros = FMF.noSignedZeros();
  AllowReciprocal = FMF.allowReciprocal();
  AllowContract = FMF.allowContract();
  ApproxFunc = FMF.approxFunc();
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void TPRecipeWithIRFlags::printFlags(raw_ostream &O) const {
  switch (OpType) {
  case OperationType::Cmp:
    O << " " << CmpInst::getPredicateName(getPredicate());
    break;
  case OperationType::DisjointOp:
    if (DisjointFlags.IsDisjoint)
      O << " disjoint";
    break;
  case OperationType::PossiblyExactOp:
    if (ExactFlags.IsExact)
      O << " exact";
    break;
  case OperationType::OverflowingBinOp:
    if (WrapFlags.HasNUW)
      O << " nuw";
    if (WrapFlags.HasNSW)
      O << " nsw";
    break;
  case OperationType::FPMathOp:
    getFastMathFlags().print(O);
    break;
  case OperationType::GEPOp:
    if (GEPFlags.IsInBounds)
      O << " inbounds";
    break;
  case OperationType::NonNegOp:
    if (NonNegFlags.NonNeg)
      O << " nneg";
    break;
  case OperationType::Other:
    break;
  }
  if (getNumOperands() > 0)
    O << " ";
}
#endif

void TPWidenRecipe::execute(TPTransformState &State) {
  IRBuilder<> Builder(State.CurBB->getTerminator());
  
  auto GetOperandValue = [&](TPValue *TPV) -> Value* {
    if (State.TPValue2Value.count(TPV))
      return State.TPValue2Value[TPV];
    if (TPV->isLiveIn())
      return TPV->getLiveInIRValue();
    if (TPV->getUnderlyingValue())
      return TPV->getUnderlyingValue();
    return nullptr;
  };
  
  Value *LHS = GetOperandValue(getOperand(0));
  Value *RHS = getNumOperands() > 1 ? GetOperandValue(getOperand(1)) : nullptr;
  
  TensorOpKind Kind = getTensorOpKind();
  Value *Result = nullptr;
  
  switch (Kind) {
  case TensorOpKind::Scalar:
  case TensorOpKind::ElementWise:
  case TensorOpKind::BroadcastBinary:
    if (Instruction::isBinaryOp(Opcode))
      Result = Builder.CreateBinOp(static_cast<Instruction::BinaryOps>(Opcode),
                                    LHS, RHS);
    else if (Instruction::isCast(Opcode))
      Result = Builder.CreateCast(static_cast<Instruction::CastOps>(Opcode),
                                   LHS, getUnderlyingInstr()->getType());
    break;
    
  case TensorOpKind::Contraction: {
    MatrixBuilder MBuilder(Builder);
    unsigned M = State.Plan->getPFForDim(1);
    unsigned K = State.Plan->getPFForDim(0);
    unsigned N = State.Plan->getPFForDim(2);
    
    if (M > 0 && K > 0 && N > 0 && LHS && RHS) {
      Result = MBuilder.CreateMatrixMultiply(LHS, RHS, M, K, N);
    }
    break;
  }
  
  case TensorOpKind::OuterProduct: {
    if (LHS && RHS) {
      Result = Builder.CreateMul(LHS, RHS, "outer.product");
    }
    break;
  }
  
  case TensorOpKind::PlainReduction:
    break;
  }
  
  if (Result)
    State.TPValue2Value[this] = Result;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void TPWidenRecipe::print(raw_ostream &O, const Twine &Indent,
                          TPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN ";
  printAsOperand(O, SlotTracker);
  O << " = " << Instruction::getOpcodeName(Opcode);
  printFlags(O);
  printOperands(O, SlotTracker);
}
#endif

void TPWidenCastRecipe::execute(TPTransformState &State) {
  IRBuilder<> Builder(State.CurBB->getTerminator());
  
  Value *Op = State.TPValue2Value.count(getOperand(0))
                  ? State.TPValue2Value[getOperand(0)]
                  : getOperand(0)->getLiveInIRValue();
  
  Type *DestTy = getResultType();
  
  if (DimSet.any()) {
    unsigned TotalElements = 1;
    for (int D = DimSet.find_first(); D >= 0; D = DimSet.find_next(D)) {
      TotalElements *= State.Plan->getPFForDim(static_cast<unsigned>(D));
    }
    if (TotalElements > 1 && !DestTy->isVectorTy()) {
      DestTy = VectorType::get(DestTy, TotalElements, false);
    }
  }
  
  Value *Result = Builder.CreateCast(Opcode, Op, DestTy, "cast.wide");
  State.TPValue2Value[this] = Result;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void TPWidenCastRecipe::print(raw_ostream &O, const Twine &Indent,
                              TPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-CAST ";
  printAsOperand(O, SlotTracker);
  O << " = " << Instruction::getOpcodeName(Opcode) << " ";
  printFlags(O);
  printOperands(O, SlotTracker);
  O << " to " << *getResultType();
}
#endif

/// A helper function that returns an integer or floating-point constant with
/// value C.
static Constant *getSignedIntOrFpConstant(Type *Ty, int64_t C) {
  return Ty->isIntegerTy() ? ConstantInt::getSigned(Ty, C)
                           : ConstantFP::get(Ty, C);
}

void TPWidenIntOrFpInductionRecipe::execute(TPTransformState &State) {
  int Dim = getDimIndex();
  unsigned PF = (Dim >= 0) ? State.Plan->getPFForDim(static_cast<unsigned>(Dim)) : 1;
  
  IRBuilder<> Builder(State.CurBB->getTerminator());
  
  Value *Start = getStartValue()->getLiveInIRValue();
  Type *ScalarTy = getScalarType();
  
  PHINode *IVPhi = PHINode::Create(ScalarTy, 2, "induction");
  IVPhi->insertBefore(State.CurBB->getTerminator());
  
  BasicBlock *Preheader = State.CurBB->getSinglePredecessor();
  IVPhi->addIncoming(Start, Preheader ? Preheader : State.TBS.TPH);
  
  Value *Step = getStepValue()->getLiveInIRValue();
  Value *StepWithPF = Step;
  
  if (PF > 1) {
    Value *PFVal = ConstantInt::get(ScalarTy, PF);
    StepWithPF = Builder.CreateMul(Step, PFVal, "step.pf");
  }
  
  Value *NextIV = Builder.CreateAdd(IVPhi, StepWithPF, "induction.next");
  
  State.TPValue2Value[this] = IVPhi;
  State.BackedgeValues[IVPhi] = NextIV;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void TPWidenIntOrFpInductionRecipe::print(raw_ostream &O, const Twine &Indent,
                                          TPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-INDUCTION";
  if (getTruncInst()) {
    O << "\\l\"";
    O << " +\n" << Indent << "\"  " << TPlanIngredient(IV) << "\\l\"";
    O << " +\n" << Indent << "\"  ";
    getTPValue(0)->printAsOperand(O, SlotTracker);
  } else
    O << " " << TPlanIngredient(IV);

  O << ", ";
  getStepValue()->printAsOperand(O, SlotTracker);
}
#endif

bool TPWidenIntOrFpInductionRecipe::isCanonical() const {
  // The step may be defined by a recipe in the preheader (e.g. if it requires
  // SCEV expansion), but for the canonical induction the step is required to be
  // 1, which is represented as live-in.
  if (getStepValue()->getDefiningRecipe())
    return false;
  auto *StepC = dyn_cast<ConstantInt>(getStepValue()->getLiveInIRValue());
  auto *StartC = dyn_cast<ConstantInt>(getStartValue()->getLiveInIRValue());
  auto *CanIV = cast<TPCanonicalIVPHIRecipe>(&*getParent()->begin());
  return StartC && StartC->isZero() && StepC && StepC->isOne() &&
         getScalarType() == CanIV->getScalarType();
}

void TPDerivedIVRecipe::execute(TPTransformState &State) {
  // TODO(yuxin.an)
  llvm_unreachable("");
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void TPDerivedIVRecipe::print(raw_ostream &O, const Twine &Indent,
                              TPSlotTracker &SlotTracker) const {
  O << Indent;
  printAsOperand(O, SlotTracker);
  O << Indent << "= DERIVED-IV ";
  getStartValue()->printAsOperand(O, SlotTracker);
  O << " + ";
  getOperand(1)->printAsOperand(O, SlotTracker);
  O << " * ";
  getStepValue()->printAsOperand(O, SlotTracker);
}
#endif

void TPScalarIVStepsRecipe::execute(TPTransformState &State) {
  // llvm_unreachable("");
  State.TPValue2Value[this] = State.TPValue2Value[IV];
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void TPScalarIVStepsRecipe::print(raw_ostream &O, const Twine &Indent,
                                  TPSlotTracker &SlotTracker) const {
  O << Indent;
  printAsOperand(O, SlotTracker);
  O << " = SCALAR-STEPS ";
  printOperands(O, SlotTracker);
}
#endif

void TPWidenGEPRecipe::execute(TPTransformState &State) {
  IRBuilder<> Builder(State.CurBB->getTerminator());
  
  GetElementPtrInst *GEP = cast<GetElementPtrInst>(getUnderlyingInstr());
  Type *SourceTy = GEP->getSourceElementType();
  
  Value *BasePtr = State.TPValue2Value.count(getOperand(0))
                       ? State.TPValue2Value[getOperand(0)]
                       : getOperand(0)->getLiveInIRValue();
  
  SmallVector<Value *> Indices;
  for (unsigned I = 1; I < getNumOperands(); ++I) {
    TPValue *Op = getOperand(I);
    Value *Idx = State.TPValue2Value.count(Op)
                     ? State.TPValue2Value[Op]
                     : Op->getLiveInIRValue();
    Indices.push_back(Idx);
  }
  
  Value *Result = Builder.CreateGEP(SourceTy, BasePtr, Indices, "gep.wide");
  
  if (isInBounds())
    if (auto *GEPInst = dyn_cast<GetElementPtrInst>(Result))
      GEPInst->setIsInBounds(true);
  
  State.TPValue2Value[this] = Result;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void TPWidenGEPRecipe::print(raw_ostream &O, const Twine &Indent,
                             TPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-GEP ";
  O << (isPointerLoopInvariant() ? "Inv" : "Var");
  for (size_t I = 0; I < getNumOperands() - 1; ++I)
    O << "[" << (isIndexLoopInvariant(I) ? "Inv" : "Var") << "]";

  O << " ";
  printAsOperand(O, SlotTracker);
  O << " = getelementptr";
  printFlags(O);
  printOperands(O, SlotTracker);
}
#endif

void TPVectorPointerRecipe::execute(TPTransformState &State) {
  // TODO(yuxin.an)
  llvm_unreachable("");
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void TPVectorPointerRecipe::print(raw_ostream &O, const Twine &Indent,
                                  TPSlotTracker &SlotTracker) const {
  O << Indent;
  printAsOperand(O, SlotTracker);
  O << " = vector-pointer ";
  if (IsReverse)
    O << "(reverse) ";

  printOperands(O, SlotTracker);
}
#endif

void TPBlendRecipe::execute(TPTransformState &State) { llvm_unreachable(""); }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void TPBlendRecipe::print(raw_ostream &O, const Twine &Indent,
                          TPSlotTracker &SlotTracker) const {
  O << Indent << "BLEND ";
  printAsOperand(O, SlotTracker);
  O << " =";
  if (getNumIncomingValues() == 1) {
    // Not a User of any mask: not really blending, this is a
    // single-predecessor phi.
    O << " ";
    getIncomingValue(0)->printAsOperand(O, SlotTracker);
  } else {
    for (unsigned I = 0, E = getNumIncomingValues(); I < E; ++I) {
      O << " ";
      getIncomingValue(I)->printAsOperand(O, SlotTracker);
      if (I == 0)
        continue;
      O << "/";
      getMask(I)->printAsOperand(O, SlotTracker);
    }
  }
}
#endif

void TPNewInstrRecipe::execute(TPTransformState &State) {
  IRBuilder<> Builder(State.CurBB->getTerminator());

  auto GetOperands = [&]() {
    SmallVector<Value *> Vals;
    for (auto *Elem : operands()) {
      Value *Val = State.TPValue2Value.count(Elem) ? State.TPValue2Value[Elem]
                                                   : Elem->getUnderlyingValue();
      assert(Val && "");
      Vals.push_back(Val);
    }
    return Vals;
  };

  Value *Res = nullptr;

  SmallVector<Value *> Vals = GetOperands();

  if (Instruction::isBinaryOp(OpCode)) {
    Res = Builder.CreateBinOp((Instruction::BinaryOps)OpCode, Vals[0], Vals[1]);
  } else if (OpCode == Instruction::GetElementPtr) {
    Res = Builder.CreateGEP(State.Plan->getPattern()->Info.ElementTy, Vals[0],
                            {Vals[1]});
  } else {
    llvm_unreachable("");
  }

  State.TPValue2Value[this] = Res;
  // State.CurBB->getParent()->dump();
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void TPNewInstrRecipe::print(raw_ostream &O, const Twine &Indent,
                             TPSlotTracker &SlotTracker) const {
  O << Indent << "NEW ";

  DenseMap<unsigned, StringRef> FunctionName{
      {Instruction::Add, "add"},
      {Instruction::Mul, "mul"},
      {Instruction::GetElementPtr, "getelementptr"}};

  printAsOperand(O, SlotTracker);
  O << " = " << FunctionName[OpCode] << " ";
  printOperands(O, SlotTracker);

  // if (!getUnderlyingInstr()->getType()->isVoidTy()) {
  //   printAsOperand(O, SlotTracker);
  //   O << " = ";
  // }
  // if (auto *CB = dyn_cast<CallBase>(getUnderlyingInstr())) {
  //   O << "call";
  //   O << "@" << CB->getCalledFunction()->getName() << "(";
  //   interleaveComma(make_range(op_begin(), op_begin() + (getNumOperands() -
  //   1)),
  //                   O, [&O, &SlotTracker](TPValue *Op) {
  //                     Op->printAsOperand(O, SlotTracker);
  //                   });
  //   O << ")";
  // } else {
  //   O << Instruction::getOpcodeName(getUnderlyingInstr()->getOpcode());
  //   O << " ";
  //   printOperands(O, SlotTracker);
  // }
}
#endif

void TPReplicateRecipe::execute(TPTransformState &State) {
  // TODO(yuxin.an)
  llvm_unreachable("");
}

bool TPReplicateRecipe::shouldPack() const {
  // Find if the recipe is used by a widened recipe via an intervening
  // VPPredInstPHIRecipe. In this case, also pack the scalar values in a vector.
  return any_of(users(), [](const TPUser *U) {
    if (auto *PredR = dyn_cast<TPPredInstPHIRecipe>(U))
      return any_of(PredR->users(), [PredR](const TPUser *U) {
        return !U->usesScalars(PredR);
      });
    return false;
  });
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void TPReplicateRecipe::print(raw_ostream &O, const Twine &Indent,
                              TPSlotTracker &SlotTracker) const {
  O << Indent << (IsUniform ? "CLONE " : "REPLICATE ");

  if (!getUnderlyingInstr()->getType()->isVoidTy()) {
    printAsOperand(O, SlotTracker);
    O << " = ";
  }
  if (auto *CB = dyn_cast<CallBase>(getUnderlyingInstr())) {
    O << "call";
    printFlags(O);
    O << "@" << CB->getCalledFunction()->getName() << "(";
    interleaveComma(make_range(op_begin(), op_begin() + (getNumOperands() - 1)),
                    O, [&O, &SlotTracker](TPValue *Op) {
                      Op->printAsOperand(O, SlotTracker);
                    });
    O << ")";
  } else {
    O << Instruction::getOpcodeName(getUnderlyingInstr()->getOpcode());
    printFlags(O);
    printOperands(O, SlotTracker);
  }

  if (shouldPack())
    O << " (S->V)";
}
#endif

void TPScalarCastRecipe ::execute(TPTransformState &State) {
  // TODO(yuxin.an)
  llvm_unreachable("");
}
