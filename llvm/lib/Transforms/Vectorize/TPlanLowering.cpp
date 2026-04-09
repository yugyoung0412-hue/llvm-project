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
#include "llvm/Transforms/Vectorize/TPlanAnalysis.h"
#include "llvm/Transforms/Vectorize/TPRecipeMatcher.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
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

/// Build a map from TPlan dimension index -> Loop* by scanning IV recipes.
/// Mirrors the same function in TPRecipeMatcher.cpp for use during lowering.
static DenseMap<unsigned, Loop *>
buildDimToLoopForLowering(TPlan &Plan, LoopInfo &LI) {
  DenseMap<unsigned, Loop *> DimToLoop;
  SmallVector<TPBlockBase *> Worklist;
  SmallPtrSet<TPBlockBase *, 32> Visited;
  if (Plan.getEntry())
    Worklist.push_back(const_cast<TPBlockBase *>(Plan.getEntry()));
  while (!Worklist.empty()) {
    auto *Blk = Worklist.pop_back_val();
    if (!Visited.insert(Blk).second)
      continue;
    if (auto *BB = dyn_cast<TPBasicBlock>(Blk))
      for (TPRecipeBase &R : *BB)
        if (auto *IV = dyn_cast<TPWidenInductionRecipe>(&R))
          if (Loop *L = LI.getLoopFor(IV->getIVPhi()->getParent()))
            DimToLoop[IV->getDimIndex()] = L;
    if (auto *Reg = dyn_cast<TPRegionBlock>(Blk))
      if (Reg->getEntry())
        Worklist.push_back(Reg->getEntry());
    for (TPBlockBase *S : Blk->getSuccessors())
      Worklist.push_back(S);
  }
  return DimToLoop;
}

//===----------------------------------------------------------------------===//
// Helper: emit @llvm.tensor.contract for a Contraction reduction
//===----------------------------------------------------------------------===//

/// State for one fixed-count tiling loop (Option B main loop).
/// Unlike TilingLoopInfo, tile size is always exactly PrimaryPF (no umin).
struct FixedCountLoopInfo {
  BasicBlock *LatchBB;     ///< IV += TileSize, br Header.
  BasicBlock *ExitBB;      ///< Fall-through after main_limit reached.
  PHINode    *IV;          ///< i64 induction: 0, PF, 2*PF, ...
  PHINode    *ExitIV;      ///< PHI in ExitBB: 0 (no-tiles) or end-of-last-tile.
};

/// Emits a fixed-tile counting loop for Option B dynamic tiling:
///   preheader: main_limit = (TC / TileSize) * TileSize
///              has_tiles  = TC > TileSize
///              if !has_tiles goto ExitBB
///   header:    IV = phi [0, pre], [IV+TileSize, latch]
///              if IV >= main_limit goto ExitBB else goto BodyBB
///   body:      [caller inserts tensor.contract with fixed TileSize]
///              [caller calls B.CreateBr(info.LatchBB)]
///   latch:     IV += TileSize; br header
///   exit:      first PHI holds main_limit (or 0 if no tiles)
///
/// On return, B's insertion point is in the body BB.
static FixedCountLoopInfo emitFixedCountingLoop(IRBuilder<> &B,
                                                 Value *TcVal,
                                                 unsigned TileSizeConst,
                                                 const Twine &Name) {
  Function    *F   = B.GetInsertBlock()->getParent();
  LLVMContext &Ctx = F->getContext();
  Type        *I64 = Type::getInt64Ty(Ctx);
  Value       *PF  = ConstantInt::get(I64, TileSizeConst);

  // Compute main_limit = (TC / PF) * PF  in the preheader.
  Value *MainTrips = B.CreateUDiv(TcVal, PF, Name + ".trips");
  Value *MainLimit = B.CreateMul(MainTrips, PF, Name + ".limit");

  // Guard: skip the loop entirely if TC < PF (TC==PF means exactly one tile).
  Value *HasTiles = B.CreateICmpUGE(TcVal, PF, Name + ".guard");

  BasicBlock *PreBB  = B.GetInsertBlock();
  BasicBlock *HdrBB  = BasicBlock::Create(Ctx, Name + ".header", F);
  BasicBlock *BodyBB = BasicBlock::Create(Ctx, Name + ".body",   F);
  BasicBlock *LchBB  = BasicBlock::Create(Ctx, Name + ".latch",  F);
  BasicBlock *ExitBB = BasicBlock::Create(Ctx, Name + ".exit",   F);

  // Preheader → Header (if has_tiles) or Exit (if TC <= PF).
  B.CreateCondBr(HasTiles, HdrBB, ExitBB);

  // Header: phi + exit-check against main_limit.
  B.SetInsertPoint(HdrBB);
  PHINode *IV = B.CreatePHI(I64, 2, Name + ".iv");
  IV->addIncoming(ConstantInt::get(I64, 0), PreBB);
  Value *Done = B.CreateICmpUGE(IV, MainLimit, Name + ".done");
  B.CreateCondBr(Done, ExitBB, BodyBB);

  // Latch: advance IV.
  IRBuilder<> LB(LchBB);
  Value *NextIV = LB.CreateAdd(IV, PF, Name + ".next");
  IV->addIncoming(NextIV, LchBB);
  LB.CreateBr(HdrBB);

  // ExitBB PHI: holds 0 (no-tiles path, from PreBB) or IV==MainLimit
  // (tiles-done path, from HdrBB when IV >= MainLimit check fires).
  // Note: LchBB does NOT branch to ExitBB — only PreBB and HdrBB do.
  B.SetInsertPoint(ExitBB);
  PHINode *ExitIV = B.CreatePHI(I64, 2, Name + ".exit.iv");
  ExitIV->addIncoming(ConstantInt::get(I64, 0), PreBB);
  ExitIV->addIncoming(IV, HdrBB); // IV == MainLimit when exiting header

  // Restore insert point to body.
  B.SetInsertPoint(BodyBB);

  return FixedCountLoopInfo{LchBB, ExitBB, IV, ExitIV};
}

/// Describes the structure of one tiling loop emitted by emitTilingLoop().
struct TilingLoopInfo {
  BasicBlock *LatchBB;    ///< Back-edge block: IV += TileSize, br Header.
  BasicBlock *ExitBB;     ///< Fall-through after loop completes.
  PHINode    *IV;         ///< i64 induction: 0, TileSize, 2*TileSize, ...
  Value      *ActualSize; ///< min(TileSize, TripCount - IV) for remainder.
};

/// Emits a counting loop around a tensor tile:
///   header: IV = phi [0, preheader], [IV+TileSize, latch]
///           if IV >= TripCount goto exit else goto body
///   body:   ActualSize = umin(TileSize, TripCount - IV)
///           [caller inserts code here, then calls B.CreateBr(info.LatchBB)]
///   latch:  IV.next = IV + TileSize; br header
///   exit:   [set as insertion point by loop-close reversal in caller]
///
/// On return, B's insertion point is in `body`. The caller MUST:
///   1. Insert body code (tensor.contract call, etc.)
///   2. Call B.CreateBr(info.LatchBB) to terminate the body
///   3. Call B.SetInsertPoint(info.ExitBB) to advance past the loop
/// Use the loop-close reversal pattern (innermost first) for nested loops.
static TilingLoopInfo emitTilingLoop(IRBuilder<> &B, Value *TripCount,
                                      Value *TileSize, const Twine &Name) {
  Function    *F   = B.GetInsertBlock()->getParent();
  LLVMContext &Ctx = F->getContext();
  Type        *I64 = Type::getInt64Ty(Ctx);

  BasicBlock *PreheaderBB = B.GetInsertBlock();
  BasicBlock *HeaderBB = BasicBlock::Create(Ctx, Name + ".header", F);
  BasicBlock *BodyBB   = BasicBlock::Create(Ctx, Name + ".body",   F);
  BasicBlock *LatchBB  = BasicBlock::Create(Ctx, Name + ".latch",  F);
  BasicBlock *ExitBB   = BasicBlock::Create(Ctx, Name + ".exit",   F);

  // Preheader → Header.
  B.CreateBr(HeaderBB);

  // Header: phi + exit-check.
  B.SetInsertPoint(HeaderBB);
  PHINode *IV = B.CreatePHI(I64, 2, Name + ".iv");
  IV->addIncoming(ConstantInt::get(I64, 0), PreheaderBB);
  Value *Done = B.CreateICmpUGE(IV, TripCount, Name + ".done");
  B.CreateCondBr(Done, ExitBB, BodyBB);

  // Body: compute remainder-safe tile size.
  B.SetInsertPoint(BodyBB);
  Value *Remaining  = B.CreateSub(TripCount, IV, Name + ".rem");
  Value *ActualSize = B.CreateIntrinsic(Intrinsic::umin, {I64},
                                         {TileSize, Remaining},
                                         nullptr, Name + ".actual");
  // Insertion point stays in BodyBB for the caller.

  // Latch: advance IV and loop back. Use a separate builder so the
  // caller's builder stays in BodyBB.
  IRBuilder<> LB(LatchBB);
  Value *NextIV = LB.CreateAdd(IV, TileSize, Name + ".next");
  IV->addIncoming(NextIV, LatchBB);
  LB.CreateBr(HeaderBB);

  return TilingLoopInfo{LatchBB, ExitBB, IV, ActualSize};
}

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

  // Find the outermost loop covering the GEMM dims (highest-numbered in DimSet).
  Loop *OutermostGEMMLoop = nullptr;
  for (int D = static_cast<int>(LHSDR->DimSet.size()) - 1; D >= 0; --D) {
    if (!LHSDR->DimSet.test(static_cast<unsigned>(D)))
      continue;
    auto It = State.DimToLoop.find(static_cast<unsigned>(D));
    if (It != State.DimToLoop.end()) {
      OutermostGEMMLoop = It->second;
      break;
    }
  }

  // Decompose A and B pointer chains into base + per-dim affine strides.
  PtrDecomposition ADecomp = decomposePtrForDims(
      cast<LoadInst>(LHSLoad->getInstruction())->getPointerOperand(),
      LHSDR->DimSet, State.DimToLoop, OutermostGEMMLoop, *State.SE);
  PtrDecomposition BDecomp = decomposePtrForDims(
      cast<LoadInst>(RHSLoad->getInstruction())->getPointerOperand(),
      RHSDR->DimSet, State.DimToLoop, OutermostGEMMLoop, *State.SE);

  Value *LHSPtr = ADecomp.Base;
  Value *RHSPtr = BDecomp.Base;
  if (!LHSPtr || !RHSPtr)
    return nullptr;

  // Ensure base pointers are in the default address space (intrinsics expect ptr).
  IRBuilder<> &PreB = State.Builder;
  auto ensureAS0 = [&](Value *P) -> Value * {
    if (P->getType()->getPointerAddressSpace() != 0)
      return PreB.CreateAddrSpaceCast(P, PointerType::get(P->getType()->getContext(), 0));
    return P;
  };
  LHSPtr = ensureAS0(LHSPtr);
  RHSPtr = ensureAS0(RHSPtr);

  // Recompute ranks from affine dims only (non-affine dims excluded from intrinsic).
  RankA = ADecomp.AffineDims.count();
  RankB = BDecomp.AffineDims.count();
  if (RankA < 1 || RankA > 4 || RankB < 1 || RankB > 4) {
    LLVM_DEBUG(dbgs() << "TPlanLowering: Contraction effective rank out of [1,4]\n");
    return nullptr;
  }

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
  // Non-affine dims (e.g. srem batch broadcasting) are excluded — the outer
  // scalar loops already handle them; their effect is baked into Base.
  unsigned NBits = std::max({static_cast<unsigned>(LHSDR->DimSet.size()),
                              static_cast<unsigned>(RHSDR->DimSet.size()),
                              static_cast<unsigned>(ContractDim + 1)});
  SmallBitVector LHSBits = LHSDR->DimSet, RHSBits = RHSDR->DimSet;
  LHSBits.resize(NBits);
  RHSBits.resize(NBits);
  SmallBitVector OutputDimSet = LHSBits;
  OutputDimSet |= RHSBits;
  OutputDimSet.reset(static_cast<unsigned>(ContractDim));
  // Exclude non-affine dims (handled by outer scalar loops).
  SmallBitVector NonAffine = ADecomp.NonAffineDims;
  NonAffine.resize(NBits);
  OutputDimSet &= ~NonAffine;

  unsigned RankC = OutputDimSet.count();
  if (RankC > 4) {
    LLVM_DEBUG(dbgs() << "TPlanLowering: Contraction output rank out of [0,4]\n");
    return nullptr;
  }

  // Locate the C accumulator store instruction.
  // Primary: trace through TPlan recipe users of the reduction update.
  // Fallback: walk IR users, following one level of PHI nodes (handles the
  //   fadd→phi→store pattern that arises when the outer loop writes back via
  //   a phi-selected accumulator).
  StoreInst *CStore = nullptr;
  TPWidenStoreRecipe *CStoreRecipe = nullptr;
  if (auto *DefVal = ReductionUpdate->getDefinedValue()) {
    for (TPUser *U : DefVal->users()) {
      auto *RB = dyn_cast<TPRecipeBase>(U);
      if (!RB)
        continue;
      if (auto *SR = dyn_cast<TPWidenStoreRecipe>(RB)) {
        CStoreRecipe = SR;
        CStore = cast<StoreInst>(SR->getInstruction());
        break;
      }
    }
  }
  if (!CStore) {
    if (auto *WR = dyn_cast<TPWidenRecipe>(ReductionUpdate)) {
      Instruction *UpdInst = WR->getInstruction();
      // Level 0: direct StoreInst users.
      for (User *U : UpdInst->users()) {
        if (auto *SI = dyn_cast<StoreInst>(U)) { CStore = SI; break; }
      }
      // Level 1: through a single PHI node (fadd → phi → store).
      if (!CStore) {
        for (User *U : UpdInst->users()) {
          if (auto *Phi = dyn_cast<PHINode>(U)) {
            for (User *PU : Phi->users()) {
              if (auto *SI = dyn_cast<StoreInst>(PU)) { CStore = SI; break; }
            }
            if (CStore) break;
          }
        }
      }
    }
  }
  if (!CStore) {
    LLVM_DEBUG(dbgs() << "TPlanLowering: Contraction cannot find C pointer\n");
    return nullptr;
  }

  // Decompose the C store pointer into base + per-dim affine strides.
  PtrDecomposition CDecomp = decomposePtrForDims(
      CStore->getPointerOperand(),
      OutputDimSet, State.DimToLoop, OutermostGEMMLoop, *State.SE);
  Value *CPtr = CDecomp.Base ? CDecomp.Base : CStore->getPointerOperand();

  IRBuilder<> &B = State.Builder;
  auto I64 = [&](uint64_t V) -> Value * { return B.getInt64(V); };
  auto expandStride = [&](const SCEV *S, unsigned Dim) -> Value * {
    if (State.Expander && State.Expander->isSafeToExpand(S))
      return State.Expander->expandCodeFor(S, B.getInt64Ty(),
                                            &*B.GetInsertPoint());
    return I64(State.Plan.getDenseStrideForDim(Dim));
  };
  auto getAStride = [&](unsigned Dim) -> Value * {
    auto It = ADecomp.Strides.find(Dim);
    if (It == ADecomp.Strides.end()) return I64(0);
    return expandStride(It->second, Dim);
  };
  auto getBStride = [&](unsigned Dim) -> Value * {
    auto It = BDecomp.Strides.find(Dim);
    if (It == BDecomp.Strides.end()) return I64(0);
    return expandStride(It->second, Dim);
  };

  auto getCStride = [&](unsigned Dim) -> Value * {
    // Prefer decomposed strides; fall back to CStoreRecipe SCEV; then dense.
    auto It = CDecomp.Strides.find(Dim);
    if (It != CDecomp.Strides.end())
      return expandStride(It->second, Dim);
    if (CStoreRecipe && State.SE)
      return expandStride(CStoreRecipe->getMemStride(Dim, State.Plan, *State.SE), Dim);
    return I64(State.Plan.getDenseStrideForDim(Dim));
  };

  // --- Tiling decision ---
  // Collect dimensions where real TripCount > PF (needs tiling loops).
  // getTCForDim() stores backedge-taken count (BTC); real TC = BTC+1.
  struct TileDimInfo {
    unsigned     Dim;     // DimIdx (innermost=0)
    unsigned     PF;      // tile size: PF for static; PrimaryK from TTI for dynamic
    const SCEV  *BTCSCEV; // Backedge-taken count SCEV for runtime expansion
    bool         IsDynamic = false; // true → TC is a runtime value
  };
  SmallVector<TileDimInfo, 4> TiledDims;

  auto checkDim = [&](unsigned D) {
    const SCEV *BTC = State.Plan.getTCForDim(D);
    if (!BTC)
      return; // Unknown TC — skip tiling for this dim.
    unsigned PF = State.Plan.getPFForDim(D);
    if (auto *C = dyn_cast<SCEVConstant>(BTC)) {
      uint64_t RealTC = C->getValue()->getZExtValue() + 1;
      if (RealTC > static_cast<uint64_t>(PF))
        TiledDims.push_back({D, PF, BTC, /*IsDynamic=*/false});
    } else {
      // Dynamic TC: only support dynamic tiling for the contraction dim (K).
      // Output dims (M, N) with dynamic TCs are not yet supported; skip them.
      if (D != static_cast<unsigned>(ContractDim))
        return;
      // Query TTI for primary tile size; fall back to CLI PF.
      unsigned DynPF = 0;
      if (State.TTI) {
        auto TileInfo = State.TTI->getTensorContractTileInfo(
            ElemTy, RankA, RankB, RankC);
        if (TileInfo && TileInfo->PrimaryK > 0)
          DynPF = TileInfo->PrimaryK;
      }
      if (DynPF == 0)
        DynPF = PF; // CLI override / default PF as fallback
      if (DynPF == 0)
        return; // no usable PF, skip
      LLVM_DEBUG(dbgs() << "TPlanLowering: dynamic TC dim " << D
                        << " → tensor.body PrimaryPF=" << DynPF << "\n");
      TiledDims.push_back({D, DynPF, BTC, /*IsDynamic=*/true});
    }
  };

  for (int D = OutputDimSet.find_first(); D >= 0;
       D = OutputDimSet.find_next(D))
    checkDim(static_cast<unsigned>(D));
  checkDim(static_cast<unsigned>(ContractDim));

  bool NeedsTiling = !TiledDims.empty();

  // Separate static-TC dims from dynamic-TC dims.
  SmallVector<TileDimInfo, 4> StaticTiledDims, DynamicTiledDims;
  for (auto &TD : TiledDims) {
    if (TD.IsDynamic)
      DynamicTiledDims.push_back(TD);
    else
      StaticTiledDims.push_back(TD);
  }
  bool NeedsStaticTiling  = !StaticTiledDims.empty();
  bool NeedsDynamicTiling = !DynamicTiledDims.empty();

  // Common setup used by both paths.
  unsigned ContUD = static_cast<unsigned>(ContractDim);

  // Returns the real dimension size as an i64 constant when the trip count is
  // a compile-time constant; falls back to PF for dynamic/unknown TCs.
  auto getRealDim = [&](unsigned D) -> Value * {
    if (const SCEV *BTC = State.Plan.getTCForDim(D))
      if (const auto *C = dyn_cast<SCEVConstant>(BTC))
        return I64(C->getValue()->getZExtValue() + 1);
    return I64(State.Plan.getPFForDim(D));
  };
  Module *Mod = B.GetInsertBlock()->getModule();
  FunctionCallee ContractFn =
      getTensorContractFn(*Mod, RankA, RankB, RankC, ElemTy);

  if (!NeedsTiling) {
    // ----------------------------------------------------------------
    // Fast path: TC <= PF for all dims — single tensor.contract call.
    // ----------------------------------------------------------------
    SmallVector<Value *> CStrides, AStrides, BStrides, OutDims;
    for (int D = OutputDimSet.find_first(); D >= 0;
         D = OutputDimSet.find_next(D)) {
      unsigned UD = static_cast<unsigned>(D);
      CStrides.push_back(getCStride(UD));
      AStrides.push_back(getAStride(UD));
      BStrides.push_back(getBStride(UD));
      OutDims.push_back(getRealDim(UD));
    }
    Value *AContractStride = getAStride(ContUD);
    Value *BContractStride = getBStride(ContUD);
    Value *K = getRealDim(ContUD);

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

  // ----------------------------------------------------------------
  // Shared setup for both static and dynamic tiling paths.
  //
  // Pre-cache all stride Values needed in STEP C and STEP D while B's
  // insert point is still in the preheader. This prevents stride SCEV
  // expansion from landing inside a tiling loop body.
  struct DimStrideCache {
    Value *AStr;
    Value *BStr;
    Value *CStr;
  };
  // Cache strides for tiled dims (used in STEP C pointer offsets).
  SmallVector<DimStrideCache, 4> TiledStrides;
  for (auto &TD : TiledDims)
    TiledStrides.push_back({getAStride(TD.Dim), getBStride(TD.Dim),
                             getCStride(TD.Dim)});
  // Cache strides for output dims and contract dim (used in STEP D args).
  SmallVector<DimStrideCache, 4> OutputStrides; // parallel to OutputDimSet iteration
  for (int D = OutputDimSet.find_first(); D >= 0;
       D = OutputDimSet.find_next(D)) {
    unsigned UD = static_cast<unsigned>(D);
    OutputStrides.push_back({getAStride(UD), getBStride(UD), getCStride(UD)});
  }
  Value *CachedAContractStride = getAStride(ContUD);
  Value *CachedBContractStride = getBStride(ContUD);

  // TPIRBasicBlock::execute positions B at IRBB->getFirstNonPHIIt(), which is
  // BEFORE the original (scalar) loop instructions. Inserting the tiling
  // branch at the current point would strand those instructions after the
  // terminator. Reposition to just before the scalar terminator so that:
  //   (a) SCEV expansion and the tiling branch land at the END of the block,
  //       after all original instructions.
  //   (b) Erasing OrigTerm leaves the block with exactly one terminator.
  //
  // Save the entry block NOW (before any repositioning) — this IS the loop
  // header block whose PHIs carry the back-edge incoming value.
  //
  // OrigSuccessor: The latch block (tensor.latch0) has InsertionBB = the
  // inner loop's IR latch (k.loop), so it repositions B back into k.loop
  // after emitContraction returns. The outermost tiling-loop exit must be
  // explicitly terminated with a branch to OrigSuccessor (the scalar loop
  // exit, i.e., k.latch) so that the loop-exit path is CFG-complete.
  BasicBlock *LoopHeaderBB = B.GetInsertBlock(); // loop header before any move
  Instruction *OrigTerm = LoopHeaderBB->getTerminator();
  BasicBlock  *OrigSuccessor = nullptr;
  if (!OrigTerm) {
    LLVM_DEBUG(dbgs() << "TPlanLowering: emitContraction tiling: "
                         "no terminator in current block, skipping tiling\n");
    return nullptr;
  }
  B.SetInsertPoint(OrigTerm); // insert before the scalar loop exit branch
  if (OrigTerm->getNumSuccessors() > 0)
    OrigSuccessor = OrigTerm->getSuccessor(0);

  if (NeedsStaticTiling && !NeedsDynamicTiling) {
    // ----------------------------------------------------------------
    // Static tiling path: nested loops around a PF-sized tensor.contract.
    // ----------------------------------------------------------------

    // STEP A: Expand ALL TripCount SCEVs to Value* BEFORE creating any
    // loop BBs. This keeps expansion code in the current preheader BB,
    // not inside a loop body (which would cause dominance violations).
    SmallVector<Value *> TCValues; // parallel to StaticTiledDims
    for (auto &TD : StaticTiledDims) {
      Value *BTC = State.Expander->expandCodeFor(
          TD.BTCSCEV, B.getInt64Ty(), &*B.GetInsertPoint());
      TCValues.push_back(B.CreateAdd(BTC, B.getInt64(1), "tc.real"));
    }

    // STEP B: Emit tiling loops (outermost first = StaticTiledDims order).
    // Each call nests inside the previous loop's body.
    SmallVector<TilingLoopInfo, 4> LoopInfos;
    for (unsigned I = 0; I < StaticTiledDims.size(); ++I) {
      std::string Name = (Twine("tile.d") + Twine(StaticTiledDims[I].Dim)).str();
      Value *TileSize  = B.getInt64(StaticTiledDims[I].PF);
      LoopInfos.push_back(emitTilingLoop(B, TCValues[I], TileSize, Name));
      // B's insertion point is now in loop I's body BB.
    }

    // Erase the preheader's original terminator. emitTilingLoop() already
    // inserted a new branch (br tile.d*.header) in its place; leaving the
    // original terminator would make the preheader have two terminators.
    OrigTerm->eraseFromParent();

    // The original loop's back-edge has been replaced by the tiling branch.
    // Remove self-referencing incoming values from PHIs in the loop header.
    if (LoopHeaderBB) {
      for (PHINode &Phi : LoopHeaderBB->phis()) {
        int Idx = Phi.getBasicBlockIndex(LoopHeaderBB);
        if (Idx >= 0)
          Phi.removeIncomingValue(Idx, /*DeletePHIIfEmpty=*/false);
      }
    }

    // STEP C: Compute offset base pointers inside the innermost body.
    // For each tiled dim: tile_ptr = base_ptr + IV * stride (GEP in elements).
    Value *TiledCPtr = CPtr, *TiledAPtr = LHSPtr, *TiledBPtr = RHSPtr;
    DenseMap<unsigned, Value *> ActualSizes; // dim → min(PF, TC-IV)
    for (unsigned I = 0; I < StaticTiledDims.size(); ++I) {
      unsigned Dim     = StaticTiledDims[I].Dim;
      Value   *IV      = LoopInfos[I].IV;
      ActualSizes[Dim] = LoopInfos[I].ActualSize;

      auto OffsetPtr = [&](Value *Base, Value *Stride) -> Value * {
        Value *Off = B.CreateMul(IV, Stride, "tile.off");
        return B.CreateGEP(ElemTy, Base, Off, "tile.ptr");
      };
      auto NonZero = [](Value *V) -> bool {
        auto *CI = dyn_cast<ConstantInt>(V);
        return !CI || !CI->isZero();
      };
      Value *AStr = TiledStrides[I].AStr;
      Value *BStr = TiledStrides[I].BStr;
      Value *CStr = TiledStrides[I].CStr;
      if (NonZero(AStr)) TiledAPtr = OffsetPtr(TiledAPtr, AStr);
      if (NonZero(BStr)) TiledBPtr = OffsetPtr(TiledBPtr, BStr);
      if (NonZero(CStr)) TiledCPtr = OffsetPtr(TiledCPtr, CStr);
    }

    // STEP D: Build tile-sized tensor.contract arguments and call.
    SmallVector<Value *> Args;
    Args.push_back(TiledCPtr);
    for (unsigned SI = 0; SI < OutputStrides.size(); ++SI)
      Args.push_back(OutputStrides[SI].CStr);
    Args.push_back(TiledAPtr);
    for (unsigned SI = 0; SI < OutputStrides.size(); ++SI)
      Args.push_back(OutputStrides[SI].AStr);
    Args.push_back(CachedAContractStride);
    Args.push_back(TiledBPtr);
    for (unsigned SI = 0; SI < OutputStrides.size(); ++SI)
      Args.push_back(OutputStrides[SI].BStr);
    Args.push_back(CachedBContractStride);
    Args.push_back(ActualSizes.count(ContUD) ? ActualSizes[ContUD]
                                              : getRealDim(ContUD));
    for (int D = OutputDimSet.find_first(); D >= 0;
         D = OutputDimSet.find_next(D)) {
      unsigned UD = static_cast<unsigned>(D);
      Args.push_back(ActualSizes.count(UD) ? ActualSizes[UD] : getRealDim(UD));
    }
    Value *Call = B.CreateCall(ContractFn, Args);

    // STEP E: Close tiling loops in reverse (innermost first).
    for (int I = static_cast<int>(LoopInfos.size()) - 1; I >= 0; --I) {
      B.CreateBr(LoopInfos[I].LatchBB);
      B.SetInsertPoint(LoopInfos[I].ExitBB);
    }

    if (OrigSuccessor)
      B.CreateBr(OrigSuccessor);

    return Call;
  }

  if (NeedsDynamicTiling) {
    // ----------------------------------------------------------------
    // Dynamic tiling path (Option B): tensor.body fixed-tile loop +
    // optional epilogue tiers + scalar remainder block.
    // ----------------------------------------------------------------

    // Support exactly one dynamic dim (the contraction dim K) for now.
    if (DynamicTiledDims.size() != 1) {
      LLVM_DEBUG(dbgs() << "TPlanLowering: expected 1 dynamic dim, got "
                        << DynamicTiledDims.size() << "; skipping\n");
      return nullptr;
    }
    const TileDimInfo &DD = DynamicTiledDims[0];
    unsigned PrimaryPF = DD.PF;

    // Retrieve epilogue tier sizes from TTI (may be empty on generic targets).
    SmallVector<unsigned, 4> EpiPFs;
    if (State.TTI) {
      auto TileInfo = State.TTI->getTensorContractTileInfo(
          ElemTy, RankA, RankB, RankC);
      if (TileInfo)
        EpiPFs = SmallVector<unsigned, 4>(TileInfo->EpilogueKSizes);
    }

    // STEP A': Expand runtime TC for the dynamic dim.
    Value *BTCVal = State.Expander->expandCodeFor(
        DD.BTCSCEV, B.getInt64Ty(), &*B.GetInsertPoint());
    Value *TcVal = B.CreateAdd(BTCVal, B.getInt64(1), "k.tc");

    // STEP B': Erase OrigTerm and PHI self-edges (same as static path).
    // Note: B is already positioned before OrigTerm from the earlier
    // B.SetInsertPoint(OrigTerm) call. After erasing, we must reset B.
    OrigTerm->eraseFromParent();
    // Reset B to the end of the (now-unterminated) preheader block.
    B.SetInsertPoint(LoopHeaderBB, LoopHeaderBB->end());
    if (LoopHeaderBB) {
      for (PHINode &Phi : LoopHeaderBB->phis()) {
        int Idx = Phi.getBasicBlockIndex(LoopHeaderBB);
        if (Idx >= 0)
          Phi.removeIncomingValue(Idx, /*DeletePHIIfEmpty=*/false);
      }
    }

    // Emit the main tensor.body fixed-count loop.
    FixedCountLoopInfo MainLoop =
        emitFixedCountingLoop(B, TcVal, PrimaryPF, "tensor.body");

    // STEP C': Pointer offsets in the tensor.body body (B is in body BB).
    Value *AOffMain = B.CreateMul(MainLoop.IV, CachedAContractStride, "tensor.body.a.off");
    Value *AMainPtr = B.CreateGEP(ElemTy, LHSPtr, AOffMain, "tensor.body.a.ptr");
    Value *BOffMain = B.CreateMul(MainLoop.IV, CachedBContractStride, "tensor.body.b.off");
    Value *BMainPtr = B.CreateGEP(ElemTy, RHSPtr, BOffMain, "tensor.body.b.ptr");

    // STEP D': tensor.contract with fixed PrimaryPF as K dim.
    {
      SmallVector<Value *> Args;
      Args.push_back(CPtr);
      for (auto &S : OutputStrides) Args.push_back(S.CStr);
      Args.push_back(AMainPtr);
      for (auto &S : OutputStrides) Args.push_back(S.AStr);
      Args.push_back(CachedAContractStride);
      Args.push_back(BMainPtr);
      for (auto &S : OutputStrides) Args.push_back(S.BStr);
      Args.push_back(CachedBContractStride);
      Args.push_back(B.getInt64(PrimaryPF)); // K = fixed PrimaryPF
      for (int D = OutputDimSet.find_first(); D >= 0;
           D = OutputDimSet.find_next(D))
        Args.push_back(getRealDim(static_cast<unsigned>(D)));
      B.CreateCall(ContractFn, Args);
    }

    // Close main loop.
    B.CreateBr(MainLoop.LatchBB);
    B.SetInsertPoint(MainLoop.ExitBB);

    // Retrieve epi_start: ExitIV PHI holds main_limit (or 0 if no tiles ran).
    Value *EpiStart = MainLoop.ExitIV;

    // STEP E': Epilogue tensor tiers (one per EpiPFs entry, often empty).
    for (unsigned EpiPF : EpiPFs) {
      Value *RemVal = B.CreateSub(TcVal, EpiStart, "epi.rem");
      Value *HasEpi = B.CreateICmpUGE(RemVal, B.getInt64(EpiPF), "epi.guard");

      BasicBlock *EpiPHBB  = B.GetInsertBlock();
      BasicBlock *EpiHdrBB = BasicBlock::Create(
          B.getContext(),
          (Twine("tensor.epi.") + Twine(EpiPF) + ".header").str(),
          B.GetInsertBlock()->getParent());
      BasicBlock *EpiBodyBB = BasicBlock::Create(
          B.getContext(),
          (Twine("tensor.epi.") + Twine(EpiPF) + ".body").str(),
          B.GetInsertBlock()->getParent());
      BasicBlock *EpiLchBB = BasicBlock::Create(
          B.getContext(),
          (Twine("tensor.epi.") + Twine(EpiPF) + ".latch").str(),
          B.GetInsertBlock()->getParent());
      BasicBlock *EpiExitBB = BasicBlock::Create(
          B.getContext(),
          (Twine("tensor.epi.") + Twine(EpiPF) + ".exit").str(),
          B.GetInsertBlock()->getParent());

      B.CreateCondBr(HasEpi, EpiHdrBB, EpiExitBB);

      B.SetInsertPoint(EpiHdrBB);
      PHINode *EIV = B.CreatePHI(B.getInt64Ty(), 2,
                                  (Twine("tensor.epi.") + Twine(EpiPF) + ".iv").str());
      EIV->addIncoming(EpiStart, EpiPHBB);
      Value *EpiLimit = B.CreateAdd(
          EpiStart,
          B.CreateMul(B.CreateUDiv(RemVal, B.getInt64(EpiPF)),
                      B.getInt64(EpiPF)),
          (Twine("tensor.epi.") + Twine(EpiPF) + ".limit").str());
      Value *EDone = B.CreateICmpUGE(EIV, EpiLimit,
                                      (Twine("tensor.epi.") + Twine(EpiPF) + ".done").str());
      B.CreateCondBr(EDone, EpiExitBB, EpiBodyBB);

      B.SetInsertPoint(EpiBodyBB);
      Value *EAOff = B.CreateMul(EIV, CachedAContractStride, "epi.a.off");
      Value *EAPtr = B.CreateGEP(ElemTy, LHSPtr, EAOff, "epi.a.ptr");
      Value *EBOff = B.CreateMul(EIV, CachedBContractStride, "epi.b.off");
      Value *EBPtr = B.CreateGEP(ElemTy, RHSPtr, EBOff, "epi.b.ptr");
      {
        SmallVector<Value *> Args;
        Args.push_back(CPtr);
        for (auto &S : OutputStrides) Args.push_back(S.CStr);
        Args.push_back(EAPtr);
        for (auto &S : OutputStrides) Args.push_back(S.AStr);
        Args.push_back(CachedAContractStride);
        Args.push_back(EBPtr);
        for (auto &S : OutputStrides) Args.push_back(S.BStr);
        Args.push_back(CachedBContractStride);
        Args.push_back(B.getInt64(EpiPF));
        for (int D = OutputDimSet.find_first(); D >= 0;
             D = OutputDimSet.find_next(D))
          Args.push_back(getRealDim(static_cast<unsigned>(D)));
        B.CreateCall(ContractFn, Args);
      }
      B.CreateBr(EpiLchBB);

      IRBuilder<> EpiLB(EpiLchBB);
      Value *ENext = EpiLB.CreateAdd(EIV, B.getInt64(EpiPF),
                                      (Twine("tensor.epi.") + Twine(EpiPF) + ".next").str());
      EIV->addIncoming(ENext, EpiLchBB);
      EpiLB.CreateBr(EpiHdrBB);

      B.SetInsertPoint(EpiExitBB);
      PHINode *NewEpiStart = B.CreatePHI(B.getInt64Ty(), 2, "epi.start");
      NewEpiStart->addIncoming(EpiStart, EpiPHBB);
      NewEpiStart->addIncoming(EpiLimit, EpiHdrBB);
      EpiStart = NewEpiStart;
    }

    // STEP F': Scalar block for final remainder.
    Value *ScRem = B.CreateSub(TcVal, EpiStart, "scalar.rem");
    Value *HasSc = B.CreateICmpUGT(ScRem, B.getInt64(0), "scalar.guard");

    BasicBlock *ScPHBB   = B.GetInsertBlock();
    BasicBlock *ScBodyBB = BasicBlock::Create(
        B.getContext(), "scalar.block", B.GetInsertBlock()->getParent());
    BasicBlock *ScExitBB = BasicBlock::Create(
        B.getContext(), "scalar.block.exit", B.GetInsertBlock()->getParent());

    B.CreateCondBr(HasSc, ScBodyBB, ScExitBB);

    B.SetInsertPoint(ScBodyBB);
    PHINode *ScIV = B.CreatePHI(B.getInt64Ty(), 2, "scalar.iv");
    ScIV->addIncoming(EpiStart, ScPHBB);

    Value *CVal = B.CreateLoad(ElemTy, CPtr, "scalar.c");
    Value *SAOff = B.CreateMul(ScIV, CachedAContractStride, "scalar.a.off");
    Value *SAPtr = B.CreateGEP(ElemTy, LHSPtr, SAOff, "scalar.a.ptr");
    Value *SAVal = B.CreateLoad(ElemTy, SAPtr, "scalar.a");
    Value *SBOff = B.CreateMul(ScIV, CachedBContractStride, "scalar.b.off");
    Value *SBPtr = B.CreateGEP(ElemTy, RHSPtr, SBOff, "scalar.b.ptr");
    Value *SBVal = B.CreateLoad(ElemTy, SBPtr, "scalar.b");
    Value *SProd = B.CreateFMul(SAVal, SBVal, "scalar.mul");
    Value *SSum  = B.CreateFAdd(CVal, SProd, "scalar.sum");
    B.CreateStore(SSum, CPtr);

    Value *ScNext = B.CreateAdd(ScIV, B.getInt64(1), "scalar.next");
    ScIV->addIncoming(ScNext, ScBodyBB);
    Value *ScDone = B.CreateICmpUGE(ScNext, TcVal, "scalar.done");
    B.CreateCondBr(ScDone, ScExitBB, ScBodyBB);

    B.SetInsertPoint(ScExitBB);
    if (OrigSuccessor)
      B.CreateBr(OrigSuccessor);

    return nullptr; // C updated in-place; no SSA Value to return.
  }

  return nullptr; // unreachable
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
      // Guard against double-emission: depth-3 GEMMs cause the same IR
      // instruction to appear in two recipe objects (once in ir-bb<k.loop>
      // and once in the latch block). The second execution must reuse the
      // first emitContraction() result rather than re-emitting the call.
      Instruction *ReductionInst = this->getInstruction();
      if (State.EmittedContractions.count(ReductionInst)) {
        if (Value *Cached = State.ContractionResults.lookup(ReductionInst))
          State.setValue(this, Cached);
        return;
      }
      // Save the builder position. In the tiling path, emitContraction()
      // moves B into the outermost tiling-loop exit (tile.d*.exit) so that
      // the exit block gets a `br OrigSuccessor` terminator. Any recipes
      // that follow the fadd in this TPBasicBlock (k.next, k.done) must NOT
      // emit into that terminated block; restoring B keeps them in k.loop
      // where they become harmless dead code before `br tile.d*.header`.
      BasicBlock           *SaveBB = State.Builder.GetInsertBlock();
      BasicBlock::iterator  SavePt = State.Builder.GetInsertPoint();
      Value *Result = emitContraction(FusedMul, this, State);
      // Mark as emitted regardless of whether a Value was returned.
      // The dynamic tiling path returns nullptr (C updated in-place) but
      // still emits all necessary IR — we must prevent double-emission.
      State.EmittedContractions.insert(ReductionInst);
      // Restore insert point for subsequent recipes in this block.
      // In both static (Result != nullptr) and dynamic (Result == nullptr)
      // paths, subsequent recipes become harmless dead code in the original
      // scalar loop block (before the new tiling branch).
      State.Builder.SetInsertPoint(SaveBB, SavePt);
      if (Result) {
        State.setValue(this, Result);
        State.ContractionResults[ReductionInst] = Result;
      }
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
                                ScalarEvolution &SE, DominatorTree &DT,
                                const TargetTransformInfo *TTI) {
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
  State.TTI = TTI;
  // Build DimToLoop for use in decomposePtrForDims().
  State.DimToLoop = buildDimToLoopForLowering(Plan, LI);

  if (Plan.getEntry()) {
    // Collect and execute all blocks in top-level construction order.
    // TPRegionBlock::execute() recurses into its interior.
    for (TPBlockBase *B : constructionOrder(Plan.getEntry()))
      B->execute(State);
  }
  return true;
}
