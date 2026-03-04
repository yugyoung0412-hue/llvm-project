//===- TPlanLowering.cpp - Lower TPlan to LLVM IR -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implements TPlanLowering_lower(): converts a TPlan to LLVM IR by creating
/// four structural basic blocks (entry, vector header, vector latch, middle)
/// and executing each block's recipes through TPTransformState.
///
/// Current implementation creates structural BBs and dispatches execute();
/// full PHI wiring and branch targeting are left as stubs.
///
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TPlan.h"
#include "llvm/Transforms/Vectorize/TPRecipe.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"

using namespace llvm;

namespace {

class TPlanLoweringImpl {
  TPlan &Plan;
  Function &F;
  LoopInfo &LI;
  ScalarEvolution &SE;
  DominatorTree &DT;

public:
  TPlanLoweringImpl(TPlan &Plan, Function &F, LoopInfo &LI, ScalarEvolution &SE,
                    DominatorTree &DT)
      : Plan(Plan), F(F), LI(LI), SE(SE), DT(DT) {}

  bool lower() {
    LLVMContext &Ctx = F.getContext();

    // Create structural basic blocks.
    BasicBlock *EntryBB =
        BasicBlock::Create(Ctx, "tplan.entry", &F);
    BasicBlock *VecHeaderBB =
        BasicBlock::Create(Ctx, "tplan.vector.header", &F);
    BasicBlock *VecLatchBB =
        BasicBlock::Create(Ctx, "tplan.vector.latch", &F);
    BasicBlock *MiddleBB =
        BasicBlock::Create(Ctx, "tplan.middle", &F);

    IRBuilder<> Builder(EntryBB);
    TPTransformState State(Builder, LI, DT, SE, &Plan);

    // Lower the entry block recipes.
    if (!Plan.getEntry()->empty()) {
      Builder.SetInsertPoint(EntryBB);
      Plan.getEntry()->execute(State);
    }
    // Placeholder: unconditional branch from entry to vector header.
    Builder.SetInsertPoint(EntryBB);
    Builder.CreateBr(VecHeaderBB);

    // Lower vector body header recipes.
    Builder.SetInsertPoint(VecHeaderBB);
    if (Plan.getVectorBody()->getHeader())
      Plan.getVectorBody()->getHeader()->execute(State);
    // Placeholder: fall through to latch.
    if (!VecHeaderBB->getTerminator())
      Builder.CreateBr(VecLatchBB);

    // Lower vector body latch recipes.
    Builder.SetInsertPoint(VecLatchBB);
    if (Plan.getVectorBody()->getLatch())
      Plan.getVectorBody()->getLatch()->execute(State);
    // Placeholder: loop back to header (actual cond branch wired by recipes).
    if (!VecLatchBB->getTerminator())
      Builder.CreateBr(MiddleBB);

    // Lower middle block recipes.
    Builder.SetInsertPoint(MiddleBB);
    if (!Plan.getMiddleBlock()->empty())
      Plan.getMiddleBlock()->execute(State);
    // Placeholder: unreachable (real successor will be wired later).
    if (!MiddleBB->getTerminator())
      Builder.CreateUnreachable();

    return true;
  }
};

} // anonymous namespace

bool llvm::TPlanLowering_lower(TPlan &Plan, Function &F, LoopInfo &LI,
                                 ScalarEvolution &SE, DominatorTree &DT) {
  return TPlanLoweringImpl(Plan, F, LI, SE, DT).lower();
}
