//===- TPlanLowering.cpp - Lower TPlan to LLVM IR -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Implements execute() for every TPlan recipe kind and the
/// TPlanLowering_lower() entry point.
///
/// Pipeline inside TPlanLowering_lower():
///   1. TPlanWidener_widen()           — propagate DimSets via union BFS
///   2. TPRecipePatternMatcher_match() — classify every recipe
///   3. Execute recipes in program order (depth-first region walk)
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TPlan.h"
#include "llvm/Transforms/Vectorize/TPRecipeMatcher.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "tplan-lower"

//===----------------------------------------------------------------------===//
// Intrinsic declaration helpers
//===----------------------------------------------------------------------===//

/// Maps an LLVM element type to the suffix used in tensor intrinsic names.
/// Returns "" for unsupported types — callers must fall back to scalar.
static StringRef getTypeSuffix(Type *Ty) {
  if (Ty->isHalfTy())         return "f16";
  if (Ty->isFloatTy())        return "f32";
  if (Ty->isDoubleTy())       return "f64";
  if (Ty->isIntegerTy(1))     return "i1";
  if (Ty->isIntegerTy(8))     return "i8";
  if (Ty->isIntegerTy(16))    return "i16";
  if (Ty->isIntegerTy(32))    return "i32";
  if (Ty->isIntegerTy(64))    return "i64";
  return "";
}

/// Maps a BinaryOperator or CmpInst to the opcode string used in tensor
/// intrinsic names. Returns "" for unsupported instructions.
static std::string getOpcodeStr(const Instruction *I) {
  if (const auto *BO = dyn_cast<BinaryOperator>(I)) {
    switch (BO->getOpcode()) {
    case Instruction::FAdd: return "fadd";
    case Instruction::FSub: return "fsub";
    case Instruction::FMul: return "fmul";
    case Instruction::FDiv: return "fdiv";
    case Instruction::FRem: return "frem";
    case Instruction::Add:  return "add";
    case Instruction::Sub:  return "sub";
    case Instruction::Mul:  return "mul";
    case Instruction::SDiv: return "sdiv";
    case Instruction::UDiv: return "udiv";
    case Instruction::SRem: return "srem";
    case Instruction::URem: return "urem";
    case Instruction::And:  return "and";
    case Instruction::Or:   return "or";
    case Instruction::Xor:  return "xor";
    case Instruction::Shl:  return "shl";
    case Instruction::LShr: return "lshr";
    case Instruction::AShr: return "ashr";
    default: return "";
    }
  }
  if (const auto *CI = dyn_cast<FCmpInst>(I)) {
    switch (CI->getPredicate()) {
    case CmpInst::FCMP_OEQ: return "fcmp_oeq";
    case CmpInst::FCMP_OLT: return "fcmp_olt";
    case CmpInst::FCMP_OLE: return "fcmp_ole";
    case CmpInst::FCMP_OGT: return "fcmp_ogt";
    case CmpInst::FCMP_OGE: return "fcmp_oge";
    case CmpInst::FCMP_ONE: return "fcmp_one";
    case CmpInst::FCMP_ORD: return "fcmp_ord";
    case CmpInst::FCMP_UEQ: return "fcmp_ueq";
    case CmpInst::FCMP_ULT: return "fcmp_ult";
    case CmpInst::FCMP_ULE: return "fcmp_ule";
    case CmpInst::FCMP_UGT: return "fcmp_ugt";
    case CmpInst::FCMP_UGE: return "fcmp_uge";
    case CmpInst::FCMP_UNE: return "fcmp_une";
    case CmpInst::FCMP_UNO: return "fcmp_uno";
    // FCMP_TRUE / FCMP_FALSE are tautologies — no tensor intrinsic defined.
    default: return "";
    }
  }
  if (const auto *CI = dyn_cast<ICmpInst>(I)) {
    switch (CI->getPredicate()) {
    case CmpInst::ICMP_EQ:  return "icmp_eq";
    case CmpInst::ICMP_NE:  return "icmp_ne";
    case CmpInst::ICMP_SLT: return "icmp_slt";
    case CmpInst::ICMP_SLE: return "icmp_sle";
    case CmpInst::ICMP_SGT: return "icmp_sgt";
    case CmpInst::ICMP_SGE: return "icmp_sge";
    case CmpInst::ICMP_ULT: return "icmp_ult";
    case CmpInst::ICMP_ULE: return "icmp_ule";
    case CmpInst::ICMP_UGT: return "icmp_ugt";
    case CmpInst::ICMP_UGE: return "icmp_uge";
    default: return "";
    }
  }
  return "";
}

/// Returns (creating if needed) @llvm.tensor.matmul.<type>.
/// Signature: void(ptr C, i64 M, i64 N, i64 ldc,
///                 ptr A, i64 M, i64 K, i64 lda,
///                 ptr B, i64 K, i64 N, i64 ldb)
static FunctionCallee getTensorMatmulFn(Module &M, Type *ElemTy) {
  LLVMContext &Ctx = M.getContext();
  StringRef TypeSuffix = getTypeSuffix(ElemTy);
  assert(!TypeSuffix.empty() && "unsupported element type for matmul");
  std::string Name = "llvm.tensor.matmul.";
  Name += TypeSuffix;
  Type *PtrTy = PointerType::getUnqual(Ctx);
  Type *I64Ty = Type::getInt64Ty(Ctx);
  FunctionType *FT = FunctionType::get(
      Type::getVoidTy(Ctx),
      {PtrTy, I64Ty, I64Ty, I64Ty,
       PtrTy, I64Ty, I64Ty, I64Ty,
       PtrTy, I64Ty, I64Ty, I64Ty},
      /*isVarArg=*/false);
  return M.getOrInsertFunction(Name, FT);
}

/// Returns (creating if needed) @llvm.tensor.contract.<Ra>d.<Rb>d.<Rc>d.<type>.
/// RankC = |(A.DimSet | B.DimSet) - {ContractDim}| (number of output dims).
/// Signature:
///   void(ptr C, i64×RankC C_strides,
///        ptr A, i64×RankC A_strides, i64 A_contract_stride,
///        ptr B, i64×RankC B_strides, i64 B_contract_stride,
///        i64 K,
///        i64×RankC output_dims)
/// A/B strides are in output-dim order (OutputDimSet iteration order).
/// stride=0 means the operand does not span that output dim (broadcast).
static FunctionCallee getTensorContractFn(Module &M, unsigned RankA,
                                           unsigned RankB, unsigned RankC,
                                           Type *ElemTy) {
  LLVMContext &Ctx = M.getContext();
  StringRef TypeSuffix = getTypeSuffix(ElemTy);
  assert(!TypeSuffix.empty() && "unsupported element type for contract");
  std::string Name = (Twine("llvm.tensor.contract.") + Twine(RankA) + "d." +
                      Twine(RankB) + "d." + Twine(RankC) + "d." +
                      TypeSuffix).str();
  Type *PtrTy = PointerType::getUnqual(Ctx);
  Type *I64Ty = Type::getInt64Ty(Ctx);
  SmallVector<Type *> Params;
  Params.push_back(PtrTy);                                          // C
  for (unsigned i = 0; i < RankC; ++i) Params.push_back(I64Ty);   // C strides
  Params.push_back(PtrTy);                                          // A
  for (unsigned i = 0; i < RankC; ++i) Params.push_back(I64Ty);   // A strides
  Params.push_back(I64Ty);                                          // A contract stride
  Params.push_back(PtrTy);                                          // B
  for (unsigned i = 0; i < RankC; ++i) Params.push_back(I64Ty);   // B strides
  Params.push_back(I64Ty);                                          // B contract stride
  Params.push_back(I64Ty);                                          // K
  for (unsigned i = 0; i < RankC; ++i) Params.push_back(I64Ty);   // output dims
  FunctionType *FT = FunctionType::get(Type::getVoidTy(Ctx), Params,
                                        /*isVarArg=*/false);
  return M.getOrInsertFunction(Name, FT);
}


/// Returns (creating if needed) @llvm.tensor.reduce.<op>.<rank_in>d.<type>.
/// Signature: void(ptr Acc, i64×rank_in Acc_strides,
///                 ptr A,   i64×rank_in A_strides,
///                 i64×rank_in dims)
/// Reduction dims are encoded as Acc_stride=0.
static FunctionCallee getTensorReduceFn(Module &M, StringRef OpName,
                                        unsigned RankIn, Type *ElemTy) {
  StringRef TypeSuffix = getTypeSuffix(ElemTy);
  assert(!TypeSuffix.empty() && "unsupported element type");
  std::string Name = "llvm.tensor.reduce.";
  Name += OpName.str();
  Name += "." + std::to_string(RankIn) + "d.";
  Name += TypeSuffix;
  LLVMContext &Ctx = M.getContext();
  Type *PtrTy = PointerType::getUnqual(Ctx);
  Type *I64Ty = Type::getInt64Ty(Ctx);
  SmallVector<Type *> Params;
  // Acc: ptr + rank_in strides
  Params.push_back(PtrTy);
  for (unsigned R = 0; R < RankIn; ++R) Params.push_back(I64Ty);
  // A: ptr + rank_in strides
  Params.push_back(PtrTy);
  for (unsigned R = 0; R < RankIn; ++R) Params.push_back(I64Ty);
  // dims
  for (unsigned R = 0; R < RankIn; ++R) Params.push_back(I64Ty);
  FunctionType *FT = FunctionType::get(Type::getVoidTy(Ctx), Params, false);
  return M.getOrInsertFunction(Name, FT);
}

/// Returns (creating if needed) @llvm.tensor.binary.<op>.<Ra>d.<Rb>d.<Rc>d.<type>.
/// Signature:
///   void(ptr C, i64×RankC C_strides,
///        ptr A, i64×RankC A_strides,   ; 0 if A ∌ that dim (broadcast)
///        ptr B, i64×RankC B_strides,   ; 0 if B ∌ that dim (broadcast)
///        i64×RankC output_dims)
/// RankC = |(A.DimSet ∪ B.DimSet)|  (no contraction dim removed).
static FunctionCallee getTensorBinaryFn(Module &M, StringRef Op,
                                         unsigned RankA, unsigned RankB,
                                         unsigned RankC, Type *ElemTy) {
  LLVMContext &Ctx = M.getContext();
  StringRef TypeSuffix = getTypeSuffix(ElemTy);
  assert(!TypeSuffix.empty() && "unsupported element type for binary");
  std::string Name = (Twine("llvm.tensor.binary.") + Op + "." +
                      Twine(RankA) + "d." + Twine(RankB) + "d." +
                      Twine(RankC) + "d." + TypeSuffix).str();
  Type *PtrTy = PointerType::getUnqual(Ctx);
  Type *I64Ty = Type::getInt64Ty(Ctx);
  SmallVector<Type *> Params;
  Params.push_back(PtrTy);                                          // C
  for (unsigned i = 0; i < RankC; ++i) Params.push_back(I64Ty);   // C strides
  Params.push_back(PtrTy);                                          // A
  for (unsigned i = 0; i < RankC; ++i) Params.push_back(I64Ty);   // A strides
  Params.push_back(PtrTy);                                          // B
  for (unsigned i = 0; i < RankC; ++i) Params.push_back(I64Ty);   // B strides
  for (unsigned i = 0; i < RankC; ++i) Params.push_back(I64Ty);   // output dims
  FunctionType *FT = FunctionType::get(Type::getVoidTy(Ctx), Params,
                                        /*isVarArg=*/false);
  return M.getOrInsertFunction(Name, FT);
}

//===----------------------------------------------------------------------===//
// Stage-print helpers (unconditional; not gated by LLVM_DEBUG)
//===----------------------------------------------------------------------===//

static StringRef kindToStr(TensorOpKind K) {
  switch (K) {
  case TensorOpKind::Scalar:          return "Scalar";
  case TensorOpKind::ElementWise:     return "ElementWise";
  case TensorOpKind::BroadcastBinary: return "BroadcastBinary";
  case TensorOpKind::BinaryOp:        return "BinaryOp";
  case TensorOpKind::OuterProduct:    return "OuterProduct";
  case TensorOpKind::Contraction:     return "Contraction";
  case TensorOpKind::PlainReduction:  return "PlainReduction";
  }
  return "Unknown";
}

static void collectAllBBs(TPBlockBase *Start,
                            SmallVectorImpl<TPBasicBlock *> &Out,
                            SmallPtrSet<TPBlockBase *, 32> &Visited) {
  if (!Start || !Visited.insert(Start).second)
    return;
  if (auto *BB = dyn_cast<TPBasicBlock>(Start))
    Out.push_back(BB);
  if (auto *R = dyn_cast<TPRegionBlock>(Start))
    if (R->getEntry())
      collectAllBBs(R->getEntry(), Out, Visited);
  for (TPBlockBase *Succ : Start->getSuccessors())
    collectAllBBs(Succ, Out, Visited);
}

static void printClassificationSummary(const TPlan &Plan,
                                        const RecipeClassMap &CM,
                                        raw_ostream &OS) {
  OS << "Recipe classifications (non-Scalar):\n";
  SmallVector<TPBasicBlock *, 32> AllBBs;
  SmallPtrSet<TPBlockBase *, 32> Visited;
  if (Plan.getEntry())
    collectAllBBs(const_cast<TPBlockBase *>(Plan.getEntry()), AllBBs, Visited);
  bool Any = false;
  for (TPBasicBlock *BB : AllBBs) {
    for (const TPRecipeBase &R : *BB) {
      auto It = CM.find(&R);
      if (It == CM.end())
        continue;
      const auto &C = It->second;
      if (C.Kind == TensorOpKind::Scalar)
        continue;
      OS << "  [" << BB->getName() << "] " << kindToStr(C.Kind);
      if (C.ContractDim >= 0)
        OS << " (contractDim=" << C.ContractDim << ")";
      OS << "\n";
      Any = true;
    }
  }
  if (!Any)
    OS << "  (none)\n";
}

//===----------------------------------------------------------------------===//
// Helper: emit @llvm.tensor.contract for a Contraction reduction
//===----------------------------------------------------------------------===//

static Value *emitContraction(const TPRecipeBase *FusedMul,
                               const TPRecipeBase *ReductionUpdate,
                               TPTransformState &State) {
  if (!FusedMul || FusedMul->operands().size() < 2)
    return nullptr;

  auto *LHSDR = dyn_cast<TPSingleDefRecipe>(FusedMul->getOperand(0));
  auto *RHSDR = dyn_cast<TPSingleDefRecipe>(FusedMul->getOperand(1));
  if (!LHSDR || !RHSDR)
    return nullptr;

  unsigned RankA = LHSDR->DimSet.count();
  unsigned RankB = RHSDR->DimSet.count();
  if (RankA < 1 || RankA > 4 || RankB < 1 || RankB > 4) {
    LLVM_DEBUG(dbgs() << "TPlanLowering: Contraction rank out of [1,4]\n");
    return nullptr;
  }

  auto *LHSLoad = dyn_cast<TPWidenLoadRecipe>(LHSDR);
  auto *RHSLoad = dyn_cast<TPWidenLoadRecipe>(RHSDR);
  if (!LHSLoad || !RHSLoad)
    return nullptr;

  auto *LHSPtrDR = dyn_cast<TPSingleDefRecipe>(LHSLoad->getOperand(0));
  auto *RHSPtrDR = dyn_cast<TPSingleDefRecipe>(RHSLoad->getOperand(0));
  if (!LHSPtrDR || !RHSPtrDR)
    return nullptr;

  Value *LHSPtr = State.getValue(LHSPtrDR);
  Value *RHSPtr = State.getValue(RHSPtrDR);
  if (!LHSPtr || !RHSPtr)
    return nullptr;

  Type *ElemTy = LHSLoad->getInstruction()->getType()->getScalarType();
  StringRef TypeSuffix = getTypeSuffix(ElemTy);
  if (TypeSuffix.empty())
    return nullptr;

  int ContractDim = State.getContractDim(ReductionUpdate);
  if (ContractDim < 0 ||
      !LHSDR->DimSet.test(static_cast<unsigned>(ContractDim)) ||
      !RHSDR->DimSet.test(static_cast<unsigned>(ContractDim))) {
    LLVM_DEBUG(dbgs() << "TPlanLowering: Contraction dim not in DimSets\n");
    return nullptr;
  }

  // Build OutputDimSet = (A.DimSet | B.DimSet) - {ContractDim}.
  unsigned NBits = std::max({static_cast<unsigned>(LHSDR->DimSet.size()),
                              static_cast<unsigned>(RHSDR->DimSet.size()),
                              static_cast<unsigned>(ContractDim + 1)});
  SmallBitVector LHSBits = LHSDR->DimSet, RHSBits = RHSDR->DimSet;
  LHSBits.resize(NBits);
  RHSBits.resize(NBits);
  SmallBitVector OutputDimSet = LHSBits;
  OutputDimSet |= RHSBits;
  OutputDimSet.reset(static_cast<unsigned>(ContractDim));

  unsigned RankC = OutputDimSet.count();
  if (RankC > 4) {
    LLVM_DEBUG(dbgs() << "TPlanLowering: Contraction output rank out of [0,4]\n");
    return nullptr;
  }

  // Locate C accumulator pointer (primary: recipe users; fallback: IR users).
  // For RankC=0 (dot product), CPtr is the scalar accumulator address,
  // not a tensor base. The store-lookup logic below handles both cases.
  Value *CPtr = nullptr;
  TPWidenStoreRecipe *CStoreRecipe = nullptr;
  if (auto *DefVal = ReductionUpdate->getDefinedValue()) {
    for (TPUser *U : DefVal->users()) {
      auto *RB = dyn_cast<TPRecipeBase>(U);
      if (!RB)
        continue;
      if (auto *SR = dyn_cast<TPWidenStoreRecipe>(RB)) {
        CStoreRecipe = SR;
        if (auto *PD = dyn_cast<TPSingleDefRecipe>(SR->getOperand(0)))
          CPtr = State.getValue(PD);
        break;
      }
    }
  }
  if (!CPtr) {
    if (auto *WR = dyn_cast<TPWidenRecipe>(ReductionUpdate)) {
      for (User *U : WR->getInstruction()->users()) {
        if (auto *SI = dyn_cast<StoreInst>(U)) {
          Value *P = SI->getPointerOperand();
          CPtr = isa<GetElementPtrInst>(P)
                     ? cast<GetElementPtrInst>(P)->getPointerOperand()
                     : P;
          break;
        }
      }
    }
  }
  if (!CPtr) {
    LLVM_DEBUG(dbgs() << "TPlanLowering: Contraction cannot find C pointer\n");
    return nullptr;
  }

  IRBuilder<> &B = State.Builder;
  auto I64 = [&](uint64_t V) -> Value * { return B.getInt64(V); };
  auto expandStride = [&](const SCEV *S, unsigned Dim) -> Value * {
    if (State.Expander && State.Expander->isSafeToExpand(S))
      return State.Expander->expandCodeFor(S, B.getInt64Ty(),
                                            &*B.GetInsertPoint());
    return I64(State.Plan.getDenseStrideForDim(Dim));
  };
  // Returns stride for output dim Dim in operand DR; 0 if DR doesn't span it.
  auto getOperandStride = [&](const TPSingleDefRecipe *DR,
                               unsigned Dim) -> Value * {
    if (Dim >= DR->DimSet.size() || !DR->DimSet.test(Dim))
      return I64(0);
    return expandStride(DR->getMemStride(Dim, State.Plan, *State.SE), Dim);
  };

  // Build stride/dim vectors in output-dim order (OutputDimSet iteration order).
  SmallVector<Value *> CStrides, AStrides, BStrides, OutDims;
  for (int D = OutputDimSet.find_first(); D >= 0;
       D = OutputDimSet.find_next(D)) {
    unsigned UD = static_cast<unsigned>(D);
    if (CStoreRecipe && State.SE)
      CStrides.push_back(expandStride(
          CStoreRecipe->getMemStride(UD, State.Plan, *State.SE), UD));
    else
      CStrides.push_back(I64(State.Plan.getDenseStrideForDim(UD)));
    AStrides.push_back(getOperandStride(LHSDR, UD));
    BStrides.push_back(getOperandStride(RHSDR, UD));
    OutDims.push_back(I64(State.Plan.getPFForDim(UD)));
  }

  // Contraction dim strides and K size.
  unsigned ContUD = static_cast<unsigned>(ContractDim);
  Value *AContractStride =
      expandStride(LHSDR->getMemStride(ContUD, State.Plan, *State.SE), ContUD);
  Value *BContractStride =
      expandStride(RHSDR->getMemStride(ContUD, State.Plan, *State.SE), ContUD);
  Value *K = I64(State.Plan.getPFForDim(ContUD));

  Module *Mod = B.GetInsertBlock()->getModule();
  FunctionCallee ContractFn =
      getTensorContractFn(*Mod, RankA, RankB, RankC, ElemTy);

  SmallVector<Value *> Args;
  Args.push_back(CPtr);
  Args.append(CStrides.begin(), CStrides.end());
  Args.push_back(LHSPtr);
  Args.append(AStrides.begin(), AStrides.end());
  Args.push_back(AContractStride);
  Args.push_back(RHSPtr);
  Args.append(BStrides.begin(), BStrides.end());
  Args.push_back(BContractStride);
  Args.push_back(K);
  Args.append(OutDims.begin(), OutDims.end());

  return B.CreateCall(ContractFn, Args);
}

//===----------------------------------------------------------------------===//
// Helper: emit @llvm.tensor.binary for a BinaryOp recipe
//===----------------------------------------------------------------------===//

/// Returns true if the tensor.binary intrinsic was emitted; false triggers
/// scalar-clone fallback in the caller.
static bool emitBinaryOp(const TPWidenRecipe *WR,
                          TPTransformState &State) {
  auto *ADR = dyn_cast<TPSingleDefRecipe>(WR->getOperand(0));
  auto *BDR = dyn_cast<TPSingleDefRecipe>(WR->getOperand(1));
  if (!ADR || !BDR) return false;

  unsigned RankA = ADR->DimSet.count();
  unsigned RankB = BDR->DimSet.count();
  if (RankA < 1 || RankA > 4 || RankB < 1 || RankB > 4) {
    LLVM_DEBUG(dbgs() << "TPlanLowering: BinaryOp rank out of [1,4]\n");
    return false;
  }

  Instruction *Inst = WR->getInstruction();

  std::string OpName = getOpcodeStr(Inst);
  if (OpName.empty()) return false;

  // For CmpInst the result type is i1 — use operand type for the suffix.
  Type *ElemTy = Inst->getType()->getScalarType();
  if (isa<CmpInst>(Inst) && Inst->getNumOperands() >= 1)
    ElemTy = Inst->getOperand(0)->getType()->getScalarType();
  if (getTypeSuffix(ElemTy).empty()) return false;

  auto *ALoad = dyn_cast<TPWidenLoadRecipe>(ADR);
  auto *BLoad = dyn_cast<TPWidenLoadRecipe>(BDR);
  if (!ALoad || !BLoad) return false;

  auto *APtrDR = dyn_cast<TPSingleDefRecipe>(ALoad->getOperand(0));
  auto *BPtrDR = dyn_cast<TPSingleDefRecipe>(BLoad->getOperand(0));
  if (!APtrDR || !BPtrDR) return false;

  // Use the GEP's IR base pointer (tensor base), not the indexed GEP result.
  // Strides encode all per-dim offsets; the intrinsic expects a flat base ptr.
  auto getBasePtr = [&](const TPSingleDefRecipe *PtrDR) -> Value * {
    if (auto *GR = dyn_cast<TPWidenGEPRecipe>(PtrDR)) {
      Value *Base = cast<GetElementPtrInst>(GR->getInstruction())->getPointerOperand();
      // Check if this base ptr was itself emitted (i.e., is a cloned instruction).
      auto It = State.EmittedMap.find(Base);
      return It != State.EmittedMap.end() ? It->second : Base;
    }
    // Not a GEP recipe — fall back to the emitted value.
    return State.getValue(PtrDR);
  };

  Value *APtr = getBasePtr(APtrDR);
  Value *BPtr = getBasePtr(BPtrDR);
  if (!APtr || !BPtr) return false;

  // Build OutputDimSet = A.DimSet ∪ B.DimSet (no contraction dim removed).
  unsigned NBits = std::max(ADR->DimSet.size(), BDR->DimSet.size());
  SmallBitVector ABits = ADR->DimSet, BBits = BDR->DimSet;
  ABits.resize(NBits); BBits.resize(NBits);
  SmallBitVector OutputDimSet = ABits;
  OutputDimSet |= BBits;
  unsigned RankC = OutputDimSet.count();
  if (RankC < 1 || RankC > 4) {
    LLVM_DEBUG(dbgs() << "TPlanLowering: BinaryOp output rank out of [1,4]\n");
    return false;
  }

  // Locate C pointer from the store recipe that uses this recipe's result.
  Value *CPtr = nullptr;
  TPWidenStoreRecipe *CStoreRecipe = nullptr;
  if (auto *DefVal = WR->getDefinedValue()) {
    for (TPUser *U : DefVal->users()) {
      auto *RB = dyn_cast<TPRecipeBase>(U);
      if (!RB) continue;
      if (auto *SR = dyn_cast<TPWidenStoreRecipe>(RB)) {
        CStoreRecipe = SR;
        if (auto *PD = dyn_cast<TPSingleDefRecipe>(SR->getOperand(0)))
          CPtr = State.getValue(PD);
        break;
      }
    }
  }
  // Fallback: look through IR users of the original instruction for a store.
  if (!CPtr) {
    for (User *U : WR->getInstruction()->users()) {
      if (auto *SI = dyn_cast<StoreInst>(U)) {
        Value *P = SI->getPointerOperand();
        CPtr = isa<GetElementPtrInst>(P)
                   ? cast<GetElementPtrInst>(P)->getPointerOperand()
                   : P;
        break;
      }
    }
  }
  if (!CPtr) {
    LLVM_DEBUG(dbgs() << "TPlanLowering: emitBinaryOp: CPtr not found\n");
    return false;
  }

  IRBuilder<> &B = State.Builder;
  auto I64 = [&](uint64_t V) -> Value * { return B.getInt64(V); };
  auto expandStride = [&](const SCEV *S, unsigned Dim) -> Value * {
    if (State.Expander && State.Expander->isSafeToExpand(S))
      return State.Expander->expandCodeFor(S, B.getInt64Ty(),
                                            &*B.GetInsertPoint());
    return I64(State.Plan.getDenseStrideForDim(Dim));
  };
  // Returns stride for output dim D in operand DR; 0 if DR doesn't span it.
  auto getOperandStride = [&](const TPSingleDefRecipe *DR,
                               unsigned D) -> Value * {
    if (D >= DR->DimSet.size() || !DR->DimSet.test(D))
      return I64(0);
    return expandStride(DR->getMemStride(D, State.Plan, *State.SE), D);
  };

  // Collect output dims in outer-to-inner order (highest dim index first).
  SmallVector<unsigned> OrderedDims;
  for (int D = OutputDimSet.find_last(); D >= 0;
       D = OutputDimSet.find_prev(static_cast<unsigned>(D))) {
    OrderedDims.push_back(static_cast<unsigned>(D));
  }

  // Build stride/dim vectors in outer-to-inner order.
  SmallVector<Value *> CStrides, AStrides, BStrides, OutDims;
  for (unsigned UD : OrderedDims) {
    if (CStoreRecipe && State.SE)
      CStrides.push_back(expandStride(
          CStoreRecipe->getMemStride(UD, State.Plan, *State.SE), UD));
    else
      CStrides.push_back(I64(State.Plan.getDenseStrideForDim(UD)));
    AStrides.push_back(getOperandStride(ADR, UD));
    BStrides.push_back(getOperandStride(BDR, UD));
    OutDims.push_back(I64(State.Plan.getPFForDim(UD)));
  }

  Module *Mod = B.GetInsertBlock()->getModule();
  FunctionCallee BinFn =
      getTensorBinaryFn(*Mod, StringRef(OpName), RankA, RankB, RankC, ElemTy);

  SmallVector<Value *> Args;
  Args.push_back(CPtr);
  Args.append(CStrides.begin(), CStrides.end());
  Args.push_back(APtr);
  Args.append(AStrides.begin(), AStrides.end());
  Args.push_back(BPtr);
  Args.append(BStrides.begin(), BStrides.end());
  Args.append(OutDims.begin(), OutDims.end());
  B.CreateCall(BinFn, Args);
  return true;
}

//===----------------------------------------------------------------------===//
// execute() implementations per recipe kind
//===----------------------------------------------------------------------===//

void TPCanonicalIVRecipe::execute(TPTransformState &) const {
  // Canonical IV is handled by the loop structure — no direct IR emission.
}

void TPCanonicalIVIncrRecipe::execute(TPTransformState &) const {
  // Same as above — no direct IR emission.
}

void TPCanonicalIVExitCmpRecipe::execute(TPTransformState &) const {
  // Same as above — no direct IR emission.
}

void TPWidenIntOrFpInductionRecipe::execute(TPTransformState &State) const {
  // IV values are loop PHIs already present in IR; register them in ValueMap.
  State.setValue(this, IVPhi);
}

void TPWidenPointerInductionRecipe::execute(TPTransformState &State) const {
  State.setValue(this, IVPhi);
}

void TPReductionPHIRecipe::execute(TPTransformState &State) const {
  State.setValue(this, RedPhi);
}

void TPFirstOrderRecurrencePHIRecipe::execute(TPTransformState &) const {
  // TODO: implement when lowering supports this recipe
}
void TPActiveLaneMaskPHIRecipe::execute(TPTransformState &) const {
  // TODO: implement when lowering supports this recipe
}
void TPEVLBasedIVPHIRecipe::execute(TPTransformState &) const {
  // TODO: implement when lowering supports this recipe
}
void TPWidenPHIRecipe::execute(TPTransformState &) const {
  // TODO: implement when lowering supports this recipe
}
void TPPredInstPHIRecipe::execute(TPTransformState &) const {
  // TODO: implement when lowering supports this recipe
}
void TPPhi::execute(TPTransformState &) const {
  // TODO: implement when lowering supports this recipe
}

void TPWidenCastRecipe::execute(TPTransformState &State) const {
  auto *SrcDR = dyn_cast<TPSingleDefRecipe>(getOperand(0));
  Value *Src = SrcDR ? State.getValue(SrcDR) : nullptr;
  if (!Src) return;
  auto *Clone = CastInst->clone();
  State.remapClone(Clone);
  Value *Result = State.Builder.Insert(Clone);
  applyFlags(*cast<Instruction>(Result));
  State.EmittedMap[CastInst] = Result;
  State.setValue(this, Result);
}

void TPWidenGEPRecipe::execute(TPTransformState &State) const {
  auto *Clone = GEPInst->clone();
  State.remapClone(Clone);
  Value *Result = State.Builder.Insert(Clone);
  applyFlags(*cast<Instruction>(Result));
  State.EmittedMap[GEPInst] = Result;
  State.setValue(this, Result);
}

void TPWidenLoadRecipe::execute(TPTransformState &State) const {
  auto *Clone = LoadInst->clone();
  State.remapClone(Clone);
  Value *Result = State.Builder.Insert(Clone);
  State.EmittedMap[LoadInst] = Result;
  State.setValue(this, Result);
}

void TPWidenStoreRecipe::execute(TPTransformState &State) const {
  auto *Clone = StoreInst->clone();
  State.remapClone(Clone);
  State.Builder.Insert(Clone);
  State.EmittedMap[StoreInst] = Clone;
}

void TPWidenRecipe::execute(TPTransformState &State) const {
  TensorOpKind Kind = State.getKind(this);

  switch (Kind) {
  case TensorOpKind::Contraction: {
    // For the fused mul (fmul): deferred to its reduction consumer — no-op.
    // For the reduction update (fadd): emit @llvm.matrix.multiply.
    TPRecipeBase *FusedMul = State.getFusedMulRecipe(this);
    if (FusedMul) {
      Value *Result = emitContraction(FusedMul, this, State);
      if (Result)
        State.setValue(this, Result);
    }
    // else: this is the fmul itself (FusedMulRecipe==nullptr here) — no-op.
    return;
  }

  case TensorOpKind::BinaryOp: {
    if (emitBinaryOp(this, State)) return;
    // Scalar fallback.
    auto *Clone = Inst->clone();
    State.remapClone(Clone);
    Value *Result = State.Builder.Insert(Clone);
    applyFlags(*cast<Instruction>(Result));
    State.EmittedMap[Inst] = Result;
    State.setValue(this, Result);
    return;
  }

  // LEGACY — unreachable after classifyBinaryOp() now returns BinaryOp.
  case TensorOpKind::ElementWise:
    llvm_unreachable("ElementWise/BroadcastBinary are unreachable — "
                     "classifyBinaryOp() returns BinaryOp");
    return;

  case TensorOpKind::Scalar: {
    auto *Clone = Inst->clone();
    State.remapClone(Clone);
    Value *Result = State.Builder.Insert(Clone);
    applyFlags(*cast<Instruction>(Result));
    State.EmittedMap[Inst] = Result;
    State.setValue(this, Result);
    return;
  }

  // LEGACY — unreachable after classifyBinaryOp() now returns BinaryOp.
  case TensorOpKind::BroadcastBinary:
    llvm_unreachable("ElementWise/BroadcastBinary are unreachable — "
                     "classifyBinaryOp() returns BinaryOp");
    return;

  case TensorOpKind::OuterProduct: {
    auto tryVectorize = [&]() -> bool {
      auto *LHSDR = dyn_cast<TPSingleDefRecipe>(getOperand(0));
      auto *RHSDR = dyn_cast<TPSingleDefRecipe>(getOperand(1));
      if (!LHSDR || !RHSDR) return false;

      auto *LHSLoad = dyn_cast<TPWidenLoadRecipe>(LHSDR);
      auto *RHSLoad = dyn_cast<TPWidenLoadRecipe>(RHSDR);
      if (!LHSLoad || !RHSLoad) return false;
      auto *LHSPtrDR = dyn_cast<TPSingleDefRecipe>(LHSLoad->getOperand(0));
      auto *RHSPtrDR = dyn_cast<TPSingleDefRecipe>(RHSLoad->getOperand(0));
      if (!LHSPtrDR || !RHSPtrDR) return false;
      Value *LHSPtr = State.getValue(LHSPtrDR);
      Value *RHSPtr = State.getValue(RHSPtrDR);
      if (!LHSPtr || !RHSPtr) return false;

      SmallVector<unsigned> LHSShape = getTPValueShape(*LHSDR, State.Plan);
      SmallVector<unsigned> RHSShape = getTPValueShape(*RHSDR, State.Plan);
      if (LHSShape.empty() || RHSShape.empty()) return false;

      uint64_t M = 1, N = 1;
      for (unsigned D : LHSShape) M *= D;
      for (unsigned D : RHSShape) N *= D;

      // Store operand 0 is pointer, operand 1 is value.
      // Primary: recipe-level user walk. Fallback: IR-level store users
      // (same pattern as emitContraction, for stores in outer-loop latches).
      Value *CPtr = nullptr;
      if (auto *DefVal = this->getDefinedValue()) {
        for (TPUser *U : DefVal->users()) {
          auto *RB = dyn_cast<TPRecipeBase>(U);
          if (!RB) continue;
          if (auto *SR = dyn_cast<TPWidenStoreRecipe>(RB)) {
            auto *PtrDR = dyn_cast<TPSingleDefRecipe>(SR->getOperand(0));
            if (PtrDR) CPtr = State.getValue(PtrDR);
            break;
          }
        }
      }
      if (!CPtr) {
        for (User *U : LHSLoad->getInstruction()->users()) {
          if (auto *SI = dyn_cast<StoreInst>(U)) {
            Value *Ptr = SI->getPointerOperand();
            if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr))
              CPtr = GEP->getPointerOperand();
            else
              CPtr = Ptr;
            break;
          }
        }
      }
      if (!CPtr) return false;

      Type *ElemTy = LHSLoad->getInstruction()->getType()->getScalarType();
      if (!ElemTy->isFloatTy() && !ElemTy->isDoubleTy()) return false;
      Module *Mod  = State.Builder.GetInsertBlock()->getModule();
      auto MatmulFn = getTensorMatmulFn(*Mod, ElemTy);
      IRBuilder<> &B = State.Builder;
      auto I64 = [&](uint64_t V) -> Value * { return B.getInt64(V); };

      // OuterProduct = matmul(col-vec[M×1], row-vec[1×N]) → C[M×N]
      B.CreateCall(MatmulFn,
          {CPtr,    I64(M), I64(N), I64(N),    // C, M, N, ldc=N (dense)
           LHSPtr,  I64(M), I64(1), I64(1),    // A: M×1 col-vector, lda=1
           RHSPtr,  I64(1), I64(N), I64(N)});  // B: 1×N row-vector, ldb=N
      return true;
    };

    if (tryVectorize()) return;

    // Scalar fallback.
    LLVM_DEBUG(dbgs() << "TPlanLowering: OuterProduct vectorize failed, "
                         "falling back to scalar clone\n");
    auto *Clone = Inst->clone();
    State.remapClone(Clone);
    Value *Result = State.Builder.Insert(Clone);
    applyFlags(*cast<Instruction>(Result));
    State.EmittedMap[Inst] = Result;
    State.setValue(this, Result);
    return;
  }

  case TensorOpKind::PlainReduction: {
    auto tryReduce = [&]() -> bool {
      std::string OpName = getOpcodeStr(Inst);
      if (OpName.empty()) return false;

      Type *ElemTy = Inst->getType()->getScalarType();
      if (getTypeSuffix(ElemTy).empty()) return false;

      // Get the non-PHI input operand (the loaded tensor value).
      TPValue *Input = nullptr;
      for (TPValue *Op : operands()) {
        auto *RV = dyn_cast<TPRecipeValue>(Op);
        if (!RV || !isa<TPReductionPHIRecipe>(RV->getDefiningRecipe()))
          Input = Op;
      }
      if (!Input) return false;

      auto *InputDR = dyn_cast<TPSingleDefRecipe>(Input);
      if (!InputDR) return false;

      unsigned RankIn = InputDR->DimSet.count();
      if (RankIn < 1 || RankIn > 3) return false;

      auto *InputLoad = dyn_cast<TPWidenLoadRecipe>(InputDR);
      if (!InputLoad) return false;
      auto *APtrDR = dyn_cast<TPSingleDefRecipe>(InputLoad->getOperand(0));
      if (!APtrDR) return false;
      Value *APtr = State.getValue(APtrDR);
      if (!APtr) return false;

      // Get Acc pointer: allocate stack slot, initialise with PHI's preheader value.
      Value *AccPtr = nullptr;
      for (TPValue *Op : operands()) {
        auto *RV = dyn_cast<TPRecipeValue>(Op);
        if (!RV) continue;
        auto *RedPHI = dyn_cast<TPReductionPHIRecipe>(RV->getDefiningRecipe());
        if (!RedPHI) continue;
        PHINode *Phi = RedPHI->getReductionPhi();
        if (!Phi) return false;
        // Alloca in entry block.
        IRBuilder<> AllocaB(
            &Phi->getParent()->getParent()->getEntryBlock().front());
        AccPtr = AllocaB.CreateAlloca(ElemTy, nullptr, "reduce.acc");
        // Store initial value (preheader incoming).
        // Identify the preheader incoming by exclusion: the latch incoming
        // value IS the reduction-update instruction (Inst) itself; the
        // preheader incoming is the initial accumulator value.
        Value *InitVal = nullptr;
        for (unsigned Idx = 0; Idx < Phi->getNumIncomingValues(); ++Idx) {
          Value *InVal = Phi->getIncomingValue(Idx);
          // The latch feeds the reduction-update instruction back into the phi;
          // the preheader feeds the initial value (which is NOT Inst).
          // This correctly handles nested loops where the preheader block also
          // has the loop header as a successor (the forward entry edge).
          if (InVal == Inst)
            continue;
          InitVal = InVal;
          break;
        }
        if (!InitVal) return false;
        State.Builder.CreateStore(InitVal, AccPtr);
        break;
      }
      if (!AccPtr) return false;

      IRBuilder<> &B = State.Builder;
      Module *Mod = B.GetInsertBlock()->getModule();
      auto ReduceFn = getTensorReduceFn(*Mod, StringRef(OpName), RankIn, ElemTy);

      SmallBitVector ReductionDims = State.Plan.getReductionDims();
      ReductionDims.resize(RankIn);

      auto I64 = [&](uint64_t V) -> Value * { return B.getInt64(V); };
      auto expandStride = [&](const SCEV *S, unsigned Dim) -> Value * {
        if (State.Expander && State.Expander->isSafeToExpand(S))
          return State.Expander->expandCodeFor(S, B.getInt64Ty(),
                                               &*B.GetInsertPoint());
        return B.getInt64(State.Plan.getDenseStrideForDim(Dim));
      };

      SmallVector<Value *> Args;
      // Acc strides: reduction dims → 0, others → dense stride.
      Args.push_back(AccPtr);
      for (unsigned D = 0; D < RankIn; ++D) {
        if (D < ReductionDims.size() && ReductionDims.test(D))
          Args.push_back(I64(0));
        else
          Args.push_back(I64(State.Plan.getDenseStrideForDim(D)));
      }
      // A strides.
      Args.push_back(APtr);
      SmallVector<const SCEV *> AStrides =
          getTPValueStrides(*InputDR, State.Plan, *State.SE);
      int Didx = InputDR->DimSet.find_first();
      for (const SCEV *S : AStrides) {
        Args.push_back(expandStride(S, Didx >= 0 ? (unsigned)Didx : 0));
        if (Didx >= 0) Didx = InputDR->DimSet.find_next(Didx);
      }
      // Shape dims.
      SmallVector<unsigned> Shape = getTPValueShape(*InputDR, State.Plan);
      for (unsigned D : Shape) Args.push_back(I64(D));

      B.CreateCall(ReduceFn, Args);

      // Load result from Acc and register as this recipe's value.
      Value *Result = B.CreateLoad(ElemTy, AccPtr, "reduce.result");
      State.setValue(this, Result);
      return true;
    };

    if (tryReduce()) return;

    LLVM_DEBUG(dbgs() << "TPlanLowering: PlainReduction tryReduce failed, "
                         "scalar fallback\n");
    auto *Clone = Inst->clone();
    State.remapClone(Clone);
    Value *Result = State.Builder.Insert(Clone);
    applyFlags(*cast<Instruction>(Result));
    State.EmittedMap[Inst] = Result;
    State.setValue(this, Result);
    return;
  }
  }
}

//===----------------------------------------------------------------------===//
// Public entry point
//===----------------------------------------------------------------------===//

bool llvm::TPlanLowering_lower(TPlan &Plan, Function &F, LoopInfo &LI,
                                ScalarEvolution &SE, DominatorTree &DT) {
  // 1. Propagate DimSets via BFS.
  TPlanWidener_widen(Plan);
  LLVM_DEBUG({
    dbgs() << "\n=== Stage 2: After Widening (DimSets propagated) ===\n";
    Plan.print(dbgs());
  });

  // 2. Classify every recipe by DimSet patterns.
  RecipeClassMap CM;
  TPRecipePatternMatcher_match(Plan, CM, SE, LI);
  LLVM_DEBUG({
    dbgs() << "\n=== Stage 3: After Pattern Matching (recipe classifications) ===\n";
    Plan.print(dbgs());
    printClassificationSummary(Plan, CM, dbgs());
  });

  // 3. Lower: walk block CFG in construction order.
  IRBuilder<> Builder(F.getContext());
  if (!F.empty())
    Builder.SetInsertPoint(&F.getEntryBlock().front());

  TPTransformState State(Builder, Plan);
  State.ClassMap = &CM;
  SCEVExpander Expander(SE, "tplan.stride");
  State.SE = &SE;
  State.Expander = &Expander;

  if (Plan.getEntry()) {
    // Collect and execute all blocks in top-level construction order.
    // TPRegionBlock::execute() recurses into its interior.
    for (TPBlockBase *B : constructionOrder(Plan.getEntry()))
      B->execute(State);
  }
  return true;
}
