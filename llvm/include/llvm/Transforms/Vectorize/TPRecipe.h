//===- TPRecipe.h - TPlan recipe definitions ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines the TPRecipeBase hierarchy: value/user def-use chain, recipe kinds,
/// and all concrete recipe types for the TPlan CFG IR used in LoopTensorize.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_VECTORIZE_TPRECIPE_H
#define LLVM_TRANSFORMS_VECTORIZE_TPRECIPE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

namespace llvm {

// Forward declarations
class TPUser;
class TPBasicBlock;
struct TPTransformState;
class TPSlotTracker;
class TPlan;

//===----------------------------------------------------------------------===//
// TPValue — a typed value produced by a recipe, carrying a parallel factor.
//===----------------------------------------------------------------------===//
class TPValue {
  Type *ValType;
  unsigned PF = 1;
  SmallVector<TPUser *, 2> Users;

public:
  explicit TPValue(Type *Ty) : ValType(Ty) {}
  virtual ~TPValue() = default;

  Type *getType() const { return ValType; }

  unsigned getParallelFactor() const { return PF; }
  void setParallelFactor(unsigned F) { PF = F; }

  void addUser(TPUser *U) { Users.push_back(U); }
  void removeUser(TPUser *U);

  unsigned getNumUsers() const { return Users.size(); }
  ArrayRef<TPUser *> users() const { return Users; }

  /// Replace all uses of this value with \p New.
  void replaceAllUsesWith(TPValue *New);
};

//===----------------------------------------------------------------------===//
// TPUser — holds an ordered list of TPValue operands.
//===----------------------------------------------------------------------===//
class TPUser {
  SmallVector<TPValue *, 4> Operands;

public:
  TPUser() = default;
  virtual ~TPUser();

  void addOperand(TPValue *V) {
    Operands.push_back(V);
    if (V)
      V->addUser(this);
  }

  void setOperand(unsigned I, TPValue *New) {
    assert(I < Operands.size() && "Operand index out of range");
    if (Operands[I])
      Operands[I]->removeUser(this);
    Operands[I] = New;
    if (New)
      New->addUser(this);
  }

  TPValue *getOperand(unsigned I) const {
    assert(I < Operands.size() && "Operand index out of range");
    return Operands[I];
  }

  unsigned getNumOperands() const { return Operands.size(); }
  ArrayRef<TPValue *> operands() const { return Operands; }

protected:
  void dropAllOperands();
};

//===----------------------------------------------------------------------===//
// RecipeKind enum
//===----------------------------------------------------------------------===//
enum class RecipeKind {
  HeaderPHI,
  ScalarHeaderPHI,
  ReductionPHI,
  WidenLoad,
  WidenStore,
  TensorLoad,
  TensorStore,
  WidenBinaryOp,
  WidenUnaryOp,
  WidenCast,
  WidenSelect,
  WidenCmp,
  WidenIntrinsic,
  MatMul,
  Conv,
  OuterProduct,
  Reduction,
  Replicate,
  BranchOnCount,
  BranchOnCond,
  ScalarIVSteps,
  ExpandSCEV,
  VectorEndPointer,
};

//===----------------------------------------------------------------------===//
// TPRecipeBase — abstract base for all recipes; lives in an iplist.
//===----------------------------------------------------------------------===//
class TPRecipeBase : public ilist_node<TPRecipeBase>, public TPUser {
  RecipeKind Kind;
  TPBasicBlock *Parent = nullptr;
  SmallVector<TPValue *> DefinedValues;
  friend class TPBasicBlock;

public:
  explicit TPRecipeBase(RecipeKind K) : Kind(K) {}
  virtual ~TPRecipeBase();

  RecipeKind getKind() const { return Kind; }
  TPBasicBlock *getParent() const { return Parent; }

  unsigned getNumDefinedValues() const { return DefinedValues.size(); }
  TPValue *getDefinedValue(unsigned I) const { return DefinedValues[I]; }

  virtual void execute(TPTransformState &State) = 0;
  virtual void print(raw_ostream &O, TPSlotTracker &) const = 0;

protected:
  void addDefinedValue(TPValue *V) { DefinedValues.push_back(V); }
};

//===----------------------------------------------------------------------===//
// TPSingleDefRecipe — recipe that defines exactly one value (itself).
//===----------------------------------------------------------------------===//
class TPSingleDefRecipe : public TPRecipeBase, public TPValue {
public:
  TPSingleDefRecipe(RecipeKind K, Type *ResultTy)
      : TPRecipeBase(K), TPValue(ResultTy) {
    addDefinedValue(this);
  }
};

//===----------------------------------------------------------------------===//
// Concrete recipe declarations
//===----------------------------------------------------------------------===//

//--- PHI / Induction ---

/// One induction variable per loop dimension, carrying DimIndex + TileSize.
class TPHeaderPHIRecipe : public TPSingleDefRecipe {
  unsigned DimIndex;
  unsigned TileSize = 0;

public:
  TPHeaderPHIRecipe(unsigned DimIdx, unsigned PF, TPValue *Start,
                    TPValue *Step, Type *I64Ty)
      : TPSingleDefRecipe(RecipeKind::HeaderPHI, I64Ty), DimIndex(DimIdx) {
    setParallelFactor(PF);
    addOperand(Start);
    addOperand(Step);
  }

  unsigned getDimIndex() const { return DimIndex; }
  void setTileSize(unsigned S) { TileSize = S; }
  unsigned getTileSize() const { return TileSize; }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &O, TPSlotTracker &) const override;
};

/// Scalar (non-vectorized) induction variable.
class TPScalarHeaderPHIRecipe : public TPSingleDefRecipe {
public:
  TPScalarHeaderPHIRecipe(TPValue *Start, TPValue *Step, Type *Ty)
      : TPSingleDefRecipe(RecipeKind::ScalarHeaderPHI, Ty) {
    addOperand(Start);
    addOperand(Step);
  }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &O, TPSlotTracker &) const override;
};

/// PHI for a reduction accumulator.
class TPReductionPHIRecipe : public TPSingleDefRecipe {
  RecurKind RK;

public:
  TPReductionPHIRecipe(RecurKind RK, TPValue *Identity, Type *AccumTy)
      : TPSingleDefRecipe(RecipeKind::ReductionPHI, AccumTy), RK(RK) {
    addOperand(Identity);
  }

  RecurKind getRecurKind() const { return RK; }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &O, TPSlotTracker &) const override;
};

//--- Memory ---

class TPWidenLoadRecipe : public TPSingleDefRecipe {
  Align Alignment;

public:
  TPWidenLoadRecipe(TPValue *Ptr, Align A, Type *ElemTy)
      : TPSingleDefRecipe(RecipeKind::WidenLoad, ElemTy), Alignment(A) {
    addOperand(Ptr);
  }

  Align getAlign() const { return Alignment; }
  TPValue *getPointerOperand() const { return getOperand(0); }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &O, TPSlotTracker &) const override;
};

class TPWidenStoreRecipe : public TPRecipeBase {
  Align Alignment;

public:
  TPWidenStoreRecipe(TPValue *Ptr, TPValue *Val, Align A)
      : TPRecipeBase(RecipeKind::WidenStore), Alignment(A) {
    addOperand(Ptr);
    addOperand(Val);
  }

  Align getAlign() const { return Alignment; }
  TPValue *getPointerOperand() const { return getOperand(0); }
  TPValue *getValueOperand() const { return getOperand(1); }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &O, TPSlotTracker &) const override;
};

class TPTensorLoadRecipe : public TPSingleDefRecipe {
  unsigned M, N;

public:
  TPTensorLoadRecipe(TPValue *Ptr, unsigned M, unsigned N, Type *ElemTy)
      : TPSingleDefRecipe(RecipeKind::TensorLoad, ElemTy), M(M), N(N) {
    addOperand(Ptr);
  }

  unsigned getRows() const { return M; }
  unsigned getCols() const { return N; }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &O, TPSlotTracker &) const override;
};

class TPTensorStoreRecipe : public TPRecipeBase {
public:
  TPTensorStoreRecipe(TPValue *Ptr, TPValue *Val)
      : TPRecipeBase(RecipeKind::TensorStore) {
    addOperand(Ptr);
    addOperand(Val);
  }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &O, TPSlotTracker &) const override;
};

//--- Arithmetic (all TPSingleDefRecipe) ---

class TPWidenBinaryOpRecipe : public TPSingleDefRecipe {
  unsigned Opcode;

public:
  TPWidenBinaryOpRecipe(unsigned Opcode, TPValue *LHS, TPValue *RHS, Type *Ty)
      : TPSingleDefRecipe(RecipeKind::WidenBinaryOp, Ty), Opcode(Opcode) {
    addOperand(LHS);
    addOperand(RHS);
  }

  unsigned getOpcode() const { return Opcode; }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &O, TPSlotTracker &) const override;
};

class TPWidenUnaryOpRecipe : public TPSingleDefRecipe {
  unsigned Opcode;

public:
  TPWidenUnaryOpRecipe(unsigned Opcode, TPValue *Operand, Type *Ty)
      : TPSingleDefRecipe(RecipeKind::WidenUnaryOp, Ty), Opcode(Opcode) {
    addOperand(Operand);
  }

  unsigned getOpcode() const { return Opcode; }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &O, TPSlotTracker &) const override;
};

class TPWidenCastRecipe : public TPSingleDefRecipe {
  unsigned Opcode;

public:
  TPWidenCastRecipe(unsigned Opcode, TPValue *Src, Type *DestTy)
      : TPSingleDefRecipe(RecipeKind::WidenCast, DestTy), Opcode(Opcode) {
    addOperand(Src);
  }

  unsigned getOpcode() const { return Opcode; }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &O, TPSlotTracker &) const override;
};

class TPWidenCmpRecipe : public TPSingleDefRecipe {
  CmpInst::Predicate Pred;

public:
  TPWidenCmpRecipe(CmpInst::Predicate P, TPValue *LHS, TPValue *RHS, Type *Ty)
      : TPSingleDefRecipe(RecipeKind::WidenCmp, Ty), Pred(P) {
    addOperand(LHS);
    addOperand(RHS);
  }

  CmpInst::Predicate getPredicate() const { return Pred; }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &O, TPSlotTracker &) const override;
};

class TPWidenSelectRecipe : public TPSingleDefRecipe {
public:
  TPWidenSelectRecipe(TPValue *Cond, TPValue *T, TPValue *F, Type *Ty)
      : TPSingleDefRecipe(RecipeKind::WidenSelect, Ty) {
    addOperand(Cond);
    addOperand(T);
    addOperand(F);
  }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &O, TPSlotTracker &) const override;
};

class TPWidenIntrinsicRecipe : public TPSingleDefRecipe {
  Intrinsic::ID IID;

public:
  TPWidenIntrinsicRecipe(Intrinsic::ID IID, ArrayRef<TPValue *> Args, Type *Ty)
      : TPSingleDefRecipe(RecipeKind::WidenIntrinsic, Ty), IID(IID) {
    for (TPValue *A : Args)
      addOperand(A);
  }

  Intrinsic::ID getIntrinsicID() const { return IID; }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &O, TPSlotTracker &) const override;
};

//--- Tensor ---

class TPMatMulRecipe : public TPSingleDefRecipe {
  unsigned M, K, N;
  Intrinsic::ID IID;

public:
  /// \p Accum may be null (no accumulation).
  TPMatMulRecipe(unsigned M, unsigned K, unsigned N, Intrinsic::ID IID,
                 TPValue *A, TPValue *B, TPValue *Accum, Type *ResultTy)
      : TPSingleDefRecipe(RecipeKind::MatMul, ResultTy), M(M), K(K), N(N),
        IID(IID) {
    addOperand(A);
    addOperand(B);
    addOperand(Accum); // may be null
  }

  unsigned getM() const { return M; }
  unsigned getK() const { return K; }
  unsigned getN() const { return N; }
  Intrinsic::ID getIntrinsicID() const { return IID; }
  bool hasAccum() const { return getOperand(2) != nullptr; }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &O, TPSlotTracker &) const override;
};

class TPConvRecipe : public TPSingleDefRecipe {
  unsigned OH, OW, OC;

public:
  TPConvRecipe(unsigned OH, unsigned OW, unsigned OC, TPValue *Input,
               TPValue *Filter, Type *ResultTy)
      : TPSingleDefRecipe(RecipeKind::Conv, ResultTy), OH(OH), OW(OW),
        OC(OC) {
    addOperand(Input);
    addOperand(Filter);
  }

  unsigned getOH() const { return OH; }
  unsigned getOW() const { return OW; }
  unsigned getOC() const { return OC; }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &O, TPSlotTracker &) const override;
};

class TPOuterProductRecipe : public TPSingleDefRecipe {
public:
  TPOuterProductRecipe(TPValue *V1, TPValue *V2, Type *ResultTy)
      : TPSingleDefRecipe(RecipeKind::OuterProduct, ResultTy) {
    addOperand(V1);
    addOperand(V2);
  }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &O, TPSlotTracker &) const override;
};

//--- Reduction ---

class TPReductionRecipe : public TPSingleDefRecipe {
  RecurKind RK;

public:
  TPReductionRecipe(RecurKind RK, TPValue *PartialAccum, Type *ScalarTy)
      : TPSingleDefRecipe(RecipeKind::Reduction, ScalarTy), RK(RK) {
    addOperand(PartialAccum);
  }

  RecurKind getRecurKind() const { return RK; }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &O, TPSlotTracker &) const override;
};

//--- Scalar / CFG ---

class TPReplicateRecipe : public TPSingleDefRecipe {
  Instruction *UI;

public:
  explicit TPReplicateRecipe(Instruction *UI)
      : TPSingleDefRecipe(RecipeKind::Replicate, UI->getType()), UI(UI) {}

  Instruction *getInstruction() const { return UI; }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &O, TPSlotTracker &) const override;
};

class TPBranchOnCountRecipe : public TPRecipeBase {
public:
  TPBranchOnCountRecipe(TPValue *IV, TPValue *TripCount)
      : TPRecipeBase(RecipeKind::BranchOnCount) {
    addOperand(IV);
    addOperand(TripCount);
  }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &O, TPSlotTracker &) const override;
};

class TPBranchOnCondRecipe : public TPRecipeBase {
public:
  explicit TPBranchOnCondRecipe(TPValue *Cond)
      : TPRecipeBase(RecipeKind::BranchOnCond) {
    addOperand(Cond);
  }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &O, TPSlotTracker &) const override;
};

class TPScalarIVStepsRecipe : public TPSingleDefRecipe {
  unsigned Dim;

public:
  TPScalarIVStepsRecipe(unsigned Dim, Type *VecI64Ty)
      : TPSingleDefRecipe(RecipeKind::ScalarIVSteps, VecI64Ty), Dim(Dim) {}

  unsigned getDim() const { return Dim; }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &O, TPSlotTracker &) const override;
};

class TPExpandSCEVRecipe : public TPSingleDefRecipe {
  const SCEV *Expr;
  ScalarEvolution &SE;

public:
  TPExpandSCEVRecipe(const SCEV *Expr, ScalarEvolution &SE, Type *Ty)
      : TPSingleDefRecipe(RecipeKind::ExpandSCEV, Ty), Expr(Expr), SE(SE) {}

  const SCEV *getSCEV() const { return Expr; }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &O, TPSlotTracker &) const override;
};

class TPVectorEndPointerRecipe : public TPSingleDefRecipe {
  unsigned Dim;

public:
  TPVectorEndPointerRecipe(TPValue *BasePtr, unsigned Dim, Type *PtrTy)
      : TPSingleDefRecipe(RecipeKind::VectorEndPointer, PtrTy), Dim(Dim) {
    addOperand(BasePtr);
  }

  unsigned getDim() const { return Dim; }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &O, TPSlotTracker &) const override;
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_TPRECIPE_H
