//===- TensorCodeGen.cpp - Code generation for LoopTensorize --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TensorCodeGen.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/MatrixBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Vectorize/LoopNestAnalyzer.h"
#include "llvm/Transforms/Vectorize/TensorISAInfo.h"
#include "llvm/Transforms/Vectorize/TensorPatternClassifier.h"
#include "llvm/Transforms/Vectorize/TensorTransformSpace.h"

using namespace llvm;

static bool emitMatrixMultiply(unsigned M, unsigned K, unsigned N,
                                Type *ElemTy, Value *APtr, Value *BPtr,
                                Value *CPtr, BasicBlock *Preheader,
                                BasicBlock *ExitBlock, Function &F) {
  IRBuilder<> Builder(Preheader->getTerminator());
  MatrixBuilder MB(Builder);

  Value *StrideA = Builder.getInt64(M);
  Value *StrideB = Builder.getInt64(K);
  Value *StrideC = Builder.getInt64(M);

  Value *MatA = MB.CreateColumnMajorLoad(ElemTy, APtr, Align(4), StrideA,
                                         /*IsVolatile=*/false, M, K, "a");
  Value *MatB = MB.CreateColumnMajorLoad(ElemTy, BPtr, Align(4), StrideB,
                                         /*IsVolatile=*/false, K, N, "b");
  Value *MatC = MB.CreateMatrixMultiply(MatA, MatB, M, K, N, "matmul");
  MB.CreateColumnMajorStore(MatC, CPtr, Align(4), StrideC,
                            /*IsVolatile=*/false, M, N);

  // Replace the preheader terminator with a direct branch to the exit block.
  BranchInst *NewBr = BranchInst::Create(ExitBlock);
  ReplaceInstWithInst(Preheader->getTerminator(), NewBr);

  // Remove the now-unreachable loop blocks.
  EliminateUnreachableBlocks(F);
  return true;
}

bool llvm::applyPlan(const SearchState &Plan, const PatternHint &Hint,
                     ArrayRef<TensorOpDesc> SupportedOps, Function &F,
                     LoopInfo &LI, ScalarEvolution &SE, DominatorTree &DT) {
  // Only proceed if TensorRecognize was applied.
  bool HasTensorRecognize = false;
  for (const auto &T : Plan.Applied)
    if (T.Kind == TransformKind::TensorRecognize) {
      HasTensorRecognize = true;
      break;
    }
  if (!HasTensorRecognize)
    return false;

  // Only handle GEMM for now.
  if (Hint.Kind != PatternKind::GEMM)
    return false;

  // Need at least 3 loops (i, j, k).
  if (Plan.Current.Loops.size() < 3)
    return false;

  // Extract constant trip counts: IVs[0]=i(M), IVs[1]=j(N), IVs[2]=k(K).
  const auto *SCA = dyn_cast<SCEVConstant>(Plan.Current.IVs[0].TripCount);
  const auto *SCB = dyn_cast<SCEVConstant>(Plan.Current.IVs[1].TripCount);
  const auto *SCC = dyn_cast<SCEVConstant>(Plan.Current.IVs[2].TripCount);
  if (!SCA || !SCB || !SCC)
    return false;

  // getBackedgeTakenCount returns (trip_count - 1); add 1 to get actual count.
  unsigned M = static_cast<unsigned>(SCA->getValue()->getZExtValue()) + 1;
  unsigned N = static_cast<unsigned>(SCB->getValue()->getZExtValue()) + 1;
  unsigned K = static_cast<unsigned>(SCC->getValue()->getZExtValue()) + 1;

  // Only transform when all dimensions are multiples of 4 (alignment guard).
  if (M % 4 != 0 || N % 4 != 0 || K % 4 != 0)
    return false;

  // Identify base pointers: A = first Read, B = second Read, C = first Write.
  Value *APtr = nullptr, *BPtr = nullptr, *CPtr = nullptr;
  unsigned ReadCount = 0;
  for (const auto &Acc : Plan.Current.Accesses) {
    if (Acc.Kind == AccessKind::Read) {
      if (ReadCount == 0)
        APtr = Acc.BasePtr;
      else if (ReadCount == 1)
        BPtr = Acc.BasePtr;
      ++ReadCount;
    } else if ((Acc.Kind == AccessKind::Write ||
                Acc.Kind == AccessKind::ReadWrite) &&
               !CPtr) {
      CPtr = Acc.BasePtr;
    }
  }
  if (!APtr || !BPtr || !CPtr)
    return false;

  // Get the outermost loop's preheader and unique exit block.
  Loop *OuterLoop = Plan.Current.Loops[0];
  BasicBlock *Preheader = OuterLoop->getLoopPreheader();
  BasicBlock *ExitBlock = OuterLoop->getUniqueExitBlock();
  if (!Preheader || !ExitBlock)
    return false;

  // Bail if exit block has PHI nodes (would need fixing up).
  if (isa<PHINode>(&ExitBlock->front()))
    return false;

  // Determine element type from accesses or default to float.
  Type *ElemTy = nullptr;
  if (!Plan.Current.Accesses.empty())
    ElemTy = Plan.Current.Accesses[0].ElemType;
  if (!ElemTy)
    ElemTy = Type::getFloatTy(F.getContext());

  return emitMatrixMultiply(M, K, N, ElemTy, APtr, BPtr, CPtr, Preheader,
                            ExitBlock, F);
}
