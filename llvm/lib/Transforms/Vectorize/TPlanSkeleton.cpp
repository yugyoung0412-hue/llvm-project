//===- TPlanSkeleton.cpp - Tensorized loop skeleton -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Implementation of createTensorizedLoopSkeleton().
///
/// Pattern mirrors LoopVersioning::versionLoop() in LoopVersioning.cpp:
///   1. Clone the outermost loop with cloneLoopWithPreheader().
///   2. Fix up cloned instruction operands with remapInstructionsInBlocks().
///   3. Insert GuardBB between the preheader's predecessor and the preheader.
///   4. Emit: if (TC >=u PF) → tensor path; else → scalar clone.
///   5. Update DominatorTree.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TPlanSkeleton.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

#define DEBUG_TYPE "tplan-skeleton"

TensorizedLoopSkeleton llvm::createTensorizedLoopSkeleton(Loop *OutermostLoop,
                                                           Value *RuntimeTC,
                                                           unsigned PF,
                                                           LoopInfo &LI,
                                                           DominatorTree &DT,
                                                           ValueToValueMapTy &VMap) {
  TensorizedLoopSkeleton Skel;

  // ---- Precondition checks ------------------------------------------------

  BasicBlock *OrigPreheader = OutermostLoop->getLoopPreheader();
  if (!OrigPreheader) {
    LLVM_DEBUG(dbgs() << "TPlanSkeleton: loop has no unique preheader\n");
    return Skel;
  }

  BasicBlock *ExitBB = OutermostLoop->getExitBlock();
  if (!ExitBB) {
    LLVM_DEBUG(dbgs() << "TPlanSkeleton: loop has multiple exits; unsupported\n");
    return Skel;
  }

  // GuardBB will be inserted between OrigPred and OrigPreheader, so
  // OrigPreheader must have exactly one predecessor.
  BasicBlock *OrigPred = OrigPreheader->getSinglePredecessor();
  if (!OrigPred) {
    LLVM_DEBUG(dbgs() << "TPlanSkeleton: preheader has multiple predecessors\n");
    return Skel;
  }

  // ---- Step 1: Clone the loop as the scalar fallback ----------------------
  //
  // cloneLoopWithPreheader() inserts the clone before OrigPreheader and wires
  // its exit edges to the original exit block (ExitBB). LI and DT are updated
  // by the call. The clone is not yet reachable — we fix that in Step 3.

  SmallVector<BasicBlock *, 16> ClonedBlocks;
  Loop *ScalarLoop = cloneLoopWithPreheader(
      OrigPreheader, // Insert cloned blocks before this block.
      OrigPred,      // Dominator of the region being cloned into.
      OutermostLoop, VMap, ".scalar", &LI, &DT, ClonedBlocks);

  if (!ScalarLoop) {
    LLVM_DEBUG(dbgs() << "TPlanSkeleton: cloneLoopWithPreheader() failed\n");
    return Skel;
  }

  // Remap all cloned instruction operands to point to cloned values.
  remapInstructionsInBlocks(ClonedBlocks, VMap);

  // The clone's preheader is the VMap image of OrigPreheader.
  BasicBlock *ScalarPreheader = cast<BasicBlock>(VMap[OrigPreheader]);

  // ---- Step 2: Create GuardBB and wire it between OrigPred and OrigPreheader

  LLVMContext &Ctx = OrigPreheader->getContext();
  Function *F = OrigPreheader->getParent();

  // New empty block, placed in the function layout before OrigPreheader.
  BasicBlock *GuardBB =
      BasicBlock::Create(Ctx, "tensor.guard", F, OrigPreheader);

  // Redirect OrigPred's successor from OrigPreheader to GuardBB.
  Instruction *PredTerm = OrigPred->getTerminator();
  for (unsigned I = 0, E = PredTerm->getNumSuccessors(); I < E; ++I) {
    if (PredTerm->getSuccessor(I) == OrigPreheader) {
      PredTerm->setSuccessor(I, GuardBB);
      break;
    }
  }

  // PHI nodes in OrigPreheader: predecessor changed from OrigPred to GuardBB.
  for (PHINode &Phi : OrigPreheader->phis()) {
    int Idx = Phi.getBasicBlockIndex(OrigPred);
    if (Idx >= 0)
      Phi.setIncomingBlock(static_cast<unsigned>(Idx), GuardBB);
  }

  // PHI nodes in ScalarPreheader: cloned from OrigPreheader, same fixup.
  for (PHINode &Phi : ScalarPreheader->phis()) {
    int Idx = Phi.getBasicBlockIndex(OrigPred);
    if (Idx >= 0)
      Phi.setIncomingBlock(static_cast<unsigned>(Idx), GuardBB);
  }

  // Emit runtime guard: TC >=u PF → tensor path; else → scalar clone.
  {
    IRBuilder<> GB(GuardBB);
    Value *PFVal = ConstantInt::get(RuntimeTC->getType(), PF);
    Value *Cond = GB.CreateICmpUGE(RuntimeTC, PFVal, "tensor.profitable");
    GB.CreateCondBr(Cond, OrigPreheader, ScalarPreheader);
  }

  // ---- Step 3: Update DominatorTree ---------------------------------------

  DT.addNewBlock(GuardBB, OrigPred);
  DT.changeImmediateDominator(OrigPreheader, GuardBB);
  DT.changeImmediateDominator(ScalarPreheader, GuardBB);

  // ---- Populate result ----------------------------------------------------

  Skel.GuardBB = GuardBB;
  Skel.TensorPreheader = OrigPreheader;
  Skel.ScalarPreheader = ScalarPreheader;
  Skel.MergeBB = ExitBB;
  Skel.Valid = true;

  LLVM_DEBUG(dbgs() << "TPlanSkeleton: created successfully\n"
                    << "  GuardBB:    " << GuardBB->getName() << "\n"
                    << "  TensorPH:   " << OrigPreheader->getName() << "\n"
                    << "  ScalarPH:   " << ScalarPreheader->getName() << "\n"
                    << "  MergeBB:    " << ExitBB->getName() << "\n");
  return Skel;
}
