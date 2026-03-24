//===- LoopNestAnalyzer.cpp - Loop nest analysis for LoopTensorize --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/LoopNestAnalyzer.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopNestAnalysis.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

SmallVector<SmallVector<Loop *>> llvm::collectLoopNests(LoopInfo &LI) {
  SmallVector<SmallVector<Loop *>> Result;
  for (Loop *L : LI.getTopLevelLoops()) {
    SmallVector<Loop *> Nest;
    Loop *Cur = L;
    while (Cur) {
      Nest.push_back(Cur);
      Cur = Cur->getSubLoops().empty() ? nullptr : Cur->getSubLoops()[0];
    }
    Result.push_back(std::move(Nest));
  }
  return Result;
}

std::optional<LoopNestInfo>
llvm::analyzeLoopNest(ArrayRef<Loop *> Nest, ScalarEvolution &SE,
                      DependenceInfo &DI) {
  LoopNestInfo Info;
  Info.Depth = Nest.size();

  for (Loop *L : Nest) {
    if (!L->isLoopSimplifyForm())
      return std::nullopt;

    InductionDesc IV;
    IV.IndVar    = L->getInductionVariable(SE);
    IV.TripCount = SE.getBackedgeTakenCount(L);
    if (!IV.IndVar || isa<SCEVCouldNotCompute>(IV.TripCount))
      return std::nullopt;

    // Fix 1: Extract step from the SCEV AddRec expression rather than
    // hardcoding 1.
    const SCEV *IndVarSCEV = SE.getSCEV(IV.IndVar);
    if (auto *AddRec = dyn_cast<SCEVAddRecExpr>(IndVarSCEV))
      IV.Step = AddRec->getStepRecurrence(SE);
    else
      return std::nullopt; // Not a recognized induction variable

    Info.IVs.push_back(IV);
  }

  // Fix 2: Use LoopNest::arePerfectlyNested() instead of the fragile
  // block-count heuristic.
  Info.IsPerfectNest = true;
  if (Nest.size() > 1) {
    for (unsigned I = 0; I + 1 < Nest.size(); ++I) {
      if (!LoopNest::arePerfectlyNested(*Nest[I], *Nest[I + 1], SE)) {
        Info.IsPerfectNest = false;
        break;
      }
    }
  }

  // Collect memory accesses from innermost loop body
  Loop *Innermost = Nest.back();
  Info.IsAffine = true;
  for (BasicBlock *BB : Innermost->blocks()) {
    for (Instruction &I : *BB) {
      Value *Ptr = nullptr;
      AccessKind Kind;
      // Fix 4: Renamed inner variable from LI to Load to avoid shadowing the
      // outer LoopInfo LI parameter and to clarify intent.
      if (auto *Load = dyn_cast<LoadInst>(&I)) {
        Ptr  = Load->getPointerOperand();
        Kind = AccessKind::Read;
      } else if (auto *Store = dyn_cast<StoreInst>(&I)) {
        Ptr  = Store->getPointerOperand();
        Kind = AccessKind::Write;
      } else {
        continue;
      }

      MemAccess MA;
      MA.BasePtr  = Ptr->stripInBoundsOffsets();
      MA.Kind     = Kind;
      MA.ElemType = (Kind == AccessKind::Read)
                        ? cast<LoadInst>(&I)->getType()
                        : cast<StoreInst>(&I)->getValueOperand()->getType();

      const SCEV *S = SE.getSCEV(Ptr);
      if (isa<SCEVCouldNotCompute>(S))
        Info.IsAffine = false;
      else
        MA.IndexExprs.push_back(S);

      Info.Accesses.push_back(std::move(MA));
    }
  }

  Info.Loops = SmallVector<Loop *>(Nest);

  // Populate ReductionDims: a dim is a reduction dim if IndVar[d] does not
  // appear as an AddRec over Nest[d] in any write MemAccess IndexExpr.
  Info.ReductionDims.resize(Info.Depth, false);
  for (unsigned D = 0; D < Info.Depth; ++D) {
    bool AppearInStore = false;
    for (const MemAccess &MA : Info.Accesses) {
      if (MA.Kind == AccessKind::Read)
        continue;
      for (const SCEV *IdxExpr : MA.IndexExprs) {
        // Check if IdxExpr contains an AddRec over Nest[D].
        struct AddRecFinder {
          Loop *L;
          bool Found = false;
          bool follow(const SCEV *S) {
            if (auto *AR = dyn_cast<SCEVAddRecExpr>(S))
              if (AR->getLoop() == L) { Found = true; return false; }
            return !Found;
          }
          bool isDone() const { return Found; }
        } Finder{Nest[D]};
        SCEVTraversal<AddRecFinder> T(Finder);
        T.visitAll(IdxExpr);
        if (Finder.Found) { AppearInStore = true; break; }
      }
      if (AppearInStore) break;
    }
    if (!AppearInStore)
      Info.ReductionDims.set(D);
  }

  return Info;
}
