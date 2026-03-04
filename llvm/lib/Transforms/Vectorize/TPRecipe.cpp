//===- TPRecipe.cpp - TPlan recipe implementations ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TPRecipe.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Transforms/Vectorize/TPlan.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// TPValue
//===----------------------------------------------------------------------===//

void TPValue::removeUser(TPUser *U) {
  auto It = llvm::find(Users, U);
  if (It != Users.end())
    Users.erase(It);
}

void TPValue::replaceAllUsesWith(TPValue *New) {
  // Iterate by index — setOperand modifies Users in place.
  for (unsigned I = 0, E = Users.size(); I < E;) {
    TPUser *U = Users[I];
    // Find which operand slot(s) this user holds us in.
    bool Advanced = false;
    for (unsigned J = 0; J < U->getNumOperands(); ++J) {
      if (U->getOperand(J) == this) {
        U->setOperand(J, New); // removes us from Users, adds New
        // After setOperand, Users[I] may have changed — don't advance.
        Advanced = true;
        break;
      }
    }
    if (!Advanced)
      ++I;
  }
}

//===----------------------------------------------------------------------===//
// TPUser
//===----------------------------------------------------------------------===//

void TPUser::dropAllOperands() {
  for (TPValue *V : Operands)
    if (V)
      V->removeUser(this);
  Operands.clear();
}

TPUser::~TPUser() { dropAllOperands(); }

//===----------------------------------------------------------------------===//
// TPRecipeBase
//===----------------------------------------------------------------------===//

TPRecipeBase::~TPRecipeBase() {
  // dropAllOperands() is called by TPUser destructor.
}

//===----------------------------------------------------------------------===//
// Concrete recipe execute() and print() stubs
// (Full implementations provided for key recipes; others are stubs.)
//===----------------------------------------------------------------------===//

void TPHeaderPHIRecipe::execute(TPTransformState &State) {
  // Emit an undef of the appropriate vector type as a placeholder.
  // Real PHI wiring is done by TPlanLowering.
  LLVMContext &Ctx = getType()->getContext();
  unsigned PF = getParallelFactor();
  Type *VTy = PF > 1
                  ? cast<Type>(VectorType::get(getType(), PF, /*Scalable=*/false))
                  : getType();
  Value *Undef = UndefValue::get(VTy);
  State.setValue(this, Undef);
}

void TPHeaderPHIRecipe::print(raw_ostream &O, TPSlotTracker &) const {
  O << "header-phi dim=" << DimIndex;
}

void TPScalarHeaderPHIRecipe::execute(TPTransformState &State) {}
void TPScalarHeaderPHIRecipe::print(raw_ostream &O, TPSlotTracker &) const {
  O << "scalar-header-phi";
}

void TPReductionPHIRecipe::execute(TPTransformState &State) {}
void TPReductionPHIRecipe::print(raw_ostream &O, TPSlotTracker &) const {
  O << "reduction-phi";
}

void TPWidenLoadRecipe::execute(TPTransformState &State) {
  Value *PtrIR = State.getValue(getPointerOperand());
  if (!PtrIR)
    return;
  Type *ElemTy = getType();
  unsigned PF = getParallelFactor();
  Type *LoadTy = PF > 1
                     ? cast<Type>(VectorType::get(ElemTy, PF, /*Scalable=*/false))
                     : ElemTy;
  Value *Load = State.Builder.CreateAlignedLoad(LoadTy, PtrIR, Alignment);
  State.setValue(this, Load);
}

void TPWidenLoadRecipe::print(raw_ostream &O, TPSlotTracker &) const {
  O << "widen-load align=" << Alignment.value();
}

void TPWidenStoreRecipe::execute(TPTransformState &State) {
  Value *PtrIR = State.getValue(getPointerOperand());
  Value *ValIR = State.getValue(getValueOperand());
  if (!PtrIR || !ValIR)
    return;
  State.Builder.CreateAlignedStore(ValIR, PtrIR, Alignment);
}

void TPWidenStoreRecipe::print(raw_ostream &O, TPSlotTracker &) const {
  O << "widen-store align=" << Alignment.value();
}

void TPTensorLoadRecipe::execute(TPTransformState &State) {}
void TPTensorLoadRecipe::print(raw_ostream &O, TPSlotTracker &) const {
  O << "tensor-load " << M << "x" << N;
}

void TPTensorStoreRecipe::execute(TPTransformState &State) {}
void TPTensorStoreRecipe::print(raw_ostream &O, TPSlotTracker &) const {
  O << "tensor-store";
}

void TPWidenBinaryOpRecipe::execute(TPTransformState &State) {}
void TPWidenBinaryOpRecipe::print(raw_ostream &O, TPSlotTracker &) const {
  O << "widen-binop opcode=" << Opcode;
}

void TPWidenUnaryOpRecipe::execute(TPTransformState &State) {}
void TPWidenUnaryOpRecipe::print(raw_ostream &O, TPSlotTracker &) const {
  O << "widen-unary opcode=" << Opcode;
}

void TPWidenCastRecipe::execute(TPTransformState &State) {}
void TPWidenCastRecipe::print(raw_ostream &O, TPSlotTracker &) const {
  O << "widen-cast opcode=" << Opcode;
}

void TPWidenCmpRecipe::execute(TPTransformState &State) {}
void TPWidenCmpRecipe::print(raw_ostream &O, TPSlotTracker &) const {
  O << "widen-cmp pred=" << static_cast<int>(Pred);
}

void TPWidenSelectRecipe::execute(TPTransformState &State) {}
void TPWidenSelectRecipe::print(raw_ostream &O, TPSlotTracker &) const {
  O << "widen-select";
}

void TPWidenIntrinsicRecipe::execute(TPTransformState &State) {}
void TPWidenIntrinsicRecipe::print(raw_ostream &O, TPSlotTracker &) const {
  O << "widen-intrinsic id=" << IID;
}

void TPMatMulRecipe::execute(TPTransformState &State) {
  Value *AIR = State.getValue(getOperand(0));
  Value *BIR = State.getValue(getOperand(1));
  if (!AIR || !BIR)
    return;
  // Use llvm.matrix.multiply intrinsic.
  Module *Mod = State.Builder.GetInsertBlock()->getModule();
  Type *ResTy = getType();
  Function *MatMulFn = Intrinsic::getOrInsertDeclaration(
      Mod, Intrinsic::matrix_multiply,
      {ResTy, AIR->getType(), BIR->getType()});
  Value *MVal = State.Builder.getInt32(M);
  Value *KVal = State.Builder.getInt32(K);
  Value *NVal = State.Builder.getInt32(N);
  Value *Result = State.Builder.CreateCall(MatMulFn, {AIR, BIR, MVal, KVal, NVal});
  State.setValue(this, Result);
}

void TPMatMulRecipe::print(raw_ostream &O, TPSlotTracker &) const {
  O << "matmul " << M << "x" << K << "x" << N;
}

void TPConvRecipe::execute(TPTransformState &State) {}
void TPConvRecipe::print(raw_ostream &O, TPSlotTracker &) const {
  O << "conv " << OH << "x" << OW << "x" << OC;
}

void TPOuterProductRecipe::execute(TPTransformState &State) {}
void TPOuterProductRecipe::print(raw_ostream &O, TPSlotTracker &) const {
  O << "outer-product";
}

void TPReductionRecipe::execute(TPTransformState &State) {
  Value *PartialIR = State.getValue(getOperand(0));
  if (!PartialIR)
    return;
  Intrinsic::ID RedIID;
  switch (RK) {
  case RecurKind::FAdd:
    RedIID = Intrinsic::vector_reduce_fadd;
    break;
  case RecurKind::Add:
    RedIID = Intrinsic::vector_reduce_add;
    break;
  default:
    RedIID = Intrinsic::vector_reduce_add;
    break;
  }
  Module *Mod = State.Builder.GetInsertBlock()->getModule();
  Function *ReduceFn = Intrinsic::getOrInsertDeclaration(
      Mod, RedIID, {PartialIR->getType()});
  Value *Result;
  if (RK == RecurKind::FAdd) {
    Value *Start = ConstantFP::get(getType(), 0.0);
    Result = State.Builder.CreateCall(ReduceFn, {Start, PartialIR});
  } else {
    Result = State.Builder.CreateCall(ReduceFn, {PartialIR});
  }
  State.setValue(this, Result);
}

void TPReductionRecipe::print(raw_ostream &O, TPSlotTracker &) const {
  O << "reduction";
}

void TPReplicateRecipe::execute(TPTransformState &State) {}
void TPReplicateRecipe::print(raw_ostream &O, TPSlotTracker &) const {
  O << "replicate";
}

void TPBranchOnCountRecipe::execute(TPTransformState &State) {
  Value *IVIR = State.getValue(getOperand(0));
  Value *TCIR = State.getValue(getOperand(1));
  if (!IVIR || !TCIR)
    return;
  // Emit a compare; the actual branch target is wired by the lowering pass.
  State.Builder.CreateICmpSLT(IVIR, TCIR);
}

void TPBranchOnCountRecipe::print(raw_ostream &O, TPSlotTracker &) const {
  O << "branch-on-count";
}

void TPBranchOnCondRecipe::execute(TPTransformState &State) {}
void TPBranchOnCondRecipe::print(raw_ostream &O, TPSlotTracker &) const {
  O << "branch-on-cond";
}

void TPScalarIVStepsRecipe::execute(TPTransformState &State) {}
void TPScalarIVStepsRecipe::print(raw_ostream &O, TPSlotTracker &) const {
  O << "scalar-iv-steps dim=" << Dim;
}

void TPExpandSCEVRecipe::execute(TPTransformState &State) {}
void TPExpandSCEVRecipe::print(raw_ostream &O, TPSlotTracker &) const {
  O << "expand-scev";
}

void TPVectorEndPointerRecipe::execute(TPTransformState &State) {}
void TPVectorEndPointerRecipe::print(raw_ostream &O, TPSlotTracker &) const {
  O << "vector-end-pointer dim=" << Dim;
}
