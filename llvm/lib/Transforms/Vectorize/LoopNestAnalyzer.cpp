//===- LoopNestAnalyzer.cpp - Loop nest analysis for LoopTensorize --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/LoopNestAnalyzer.h"
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
    InductionDesc IV;

    if (L->isLoopSimplifyForm()) {
      IV.IndVar    = L->getInductionVariable(SE);
      IV.TripCount = SE.getBackedgeTakenCount(L);
      if (isa<SCEVCouldNotCompute>(IV.TripCount))
        IV.TripCount = nullptr;
    }

    // If getInductionVariable failed (e.g., loop not in simplify form),
    // try to find a canonical IV manually: a PHI in the header that starts
    // at 0 and increments by 1.
    if (!IV.IndVar) {
      BasicBlock *Header = L->getHeader();
      BasicBlock *Latch  = L->getLoopLatch();
      for (PHINode &Phi : Header->phis()) {
        if (!Phi.getType()->isIntegerTy())
          continue;
        // Check for start value == 0 from outside the loop.
        Value *StartVal = nullptr;
        Value *LatchVal = nullptr;
        for (unsigned I = 0, E = Phi.getNumIncomingValues(); I < E; ++I) {
          BasicBlock *Pred = Phi.getIncomingBlock(I);
          if (Latch && Pred == Latch)
            LatchVal = Phi.getIncomingValue(I);
          else if (!L->contains(Pred))
            StartVal = Phi.getIncomingValue(I);
        }
        auto *StartConst = dyn_cast_or_null<ConstantInt>(StartVal);
        if (!StartConst || !StartConst->isZero())
          continue;
        // Check for latch value == phi + 1.
        auto *Inc = dyn_cast_or_null<BinaryOperator>(LatchVal);
        if (!Inc || Inc->getOpcode() != Instruction::Add)
          continue;
        bool IsCanonical = false;
        for (unsigned I = 0; I < 2; ++I) {
          if (Inc->getOperand(I) == &Phi)
            if (auto *C = dyn_cast<ConstantInt>(Inc->getOperand(1 - I)))
              if (C->isOne())
                IsCanonical = true;
        }
        if (IsCanonical) {
          IV.IndVar = &Phi;
          // Try to get trip count via SCEV even without simplify form.
          IV.TripCount = SE.getBackedgeTakenCount(L);
          if (isa<SCEVCouldNotCompute>(IV.TripCount))
            IV.TripCount = nullptr;
          break;
        }
      }
    }

    if (!IV.IndVar)
      return std::nullopt;

    // Extract step from the SCEV AddRec expression.
    const SCEV *IndVarSCEV = SE.getSCEV(IV.IndVar);
    if (auto *AddRec = dyn_cast<SCEVAddRecExpr>(IndVarSCEV))
      IV.Step = AddRec->getStepRecurrence(SE);
    else
      IV.Step = nullptr; // Step unknown but don't reject

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

  // Populate ReductionDims: dim d is a reduction dim if the loop at depth d
  // does not appear as an AddRec in any store access's SCEV expression.
  Info.ReductionDims.resize(Info.Depth, false);
  for (unsigned D = 0; D < Info.Depth; ++D) {
    bool AppearInStore = false;
    for (const MemAccess &MA : Info.Accesses) {
      if (MA.Kind == AccessKind::Read)
        continue;
      for (const SCEV *IdxExpr : MA.IndexExprs) {
        struct ContainsAddRec {
          Loop *L;
          bool Found = false;
          bool follow(const SCEV *S) {
            if (const auto *AR = dyn_cast<SCEVAddRecExpr>(S))
              if (AR->getLoop() == L) { Found = true; return false; }
            return !Found;
          }
          bool isDone() const { return Found; }
        } Checker{Nest[D]};
        SCEVTraversal<ContainsAddRec> T(Checker);
        T.visitAll(IdxExpr);
        if (Checker.Found) { AppearInStore = true; break; }
      }
      if (AppearInStore) break;
    }
    if (!AppearInStore)
      Info.ReductionDims.set(D);
  }

  return Info;
}
