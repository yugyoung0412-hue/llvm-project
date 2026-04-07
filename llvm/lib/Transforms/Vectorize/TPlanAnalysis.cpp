//===- TPlanAnalysis.cpp - Pointer decomposition for TPlan codegen --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TPlanAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

/// Strip a single layer of bitcast or addrspacecast, returning the operand.
/// Returns nullptr if \p V is not a cast instruction of these kinds.
static Value *stripPointerCast(Value *V) {
  if (auto *BC = dyn_cast<BitCastInst>(V))
    return BC->getOperand(0);
  if (auto *AC = dyn_cast<AddrSpaceCastInst>(V))
    return AC->getOperand(0);
  // Handle constant expressions (e.g. bitcast in globals).
  if (auto *CE = dyn_cast<ConstantExpr>(V)) {
    if (CE->getOpcode() == Instruction::BitCast ||
        CE->getOpcode() == Instruction::AddrSpaceCast)
      return CE->getOperand(0);
  }
  return nullptr;
}

/// Given a PHINode that is a pointer, return the incoming value that is
/// loop-invariant with respect to \p OutermostLoop.  Returns nullptr if no
/// such value exists or if more than one invariant incoming exists (ambiguous).
static Value *getLoopInvariantIncoming(PHINode *Phi, Loop *OutermostLoop,
                                        ScalarEvolution &SE) {
  if (!Phi || !OutermostLoop)
    return nullptr;
  Value *Candidate = nullptr;
  for (unsigned I = 0, E = Phi->getNumIncomingValues(); I < E; ++I) {
    Value *In = Phi->getIncomingValue(I);
    // A value is loop-invariant w.r.t. OutermostLoop if its SCEV does not
    // contain an AddRec for OutermostLoop or any loop nested inside it.
    const SCEV *S = SE.getSCEV(In);
    if (isa<SCEVCouldNotCompute>(S))
      continue;
    // Check whether S contains any AddRec whose loop is OutermostLoop or
    // a descendant of it.
    struct Checker {
      Loop *L;
      bool Found = false;
      bool follow(const SCEV *S) {
        if (const auto *AR = dyn_cast<SCEVAddRecExpr>(S))
          if (L->contains(AR->getLoop())) { Found = true; return false; }
        return !Found;
      }
      bool isDone() const { return Found; }
    } C{OutermostLoop};
    SCEVTraversal<Checker> T(C);
    T.visitAll(S);
    bool HasAddRec = C.Found;
    if (!HasAddRec) {
      if (Candidate)
        return nullptr; // More than one invariant incoming — ambiguous.
      Candidate = In;
    }
  }
  return Candidate;
}

PtrDecomposition llvm::decomposePtrForDims(
    Value *Ptr,
    const SmallBitVector &DimSet,
    const DenseMap<unsigned, Loop *> &DimToLoop,
    Loop *OutermostGEMMLoop,
    ScalarEvolution &SE) {

  PtrDecomposition Result;
  Result.NonAffineDims.resize(DimSet.size());
  Result.AffineDims.resize(DimSet.size());

  // Build reverse map: Loop* → dim index, for the dims we care about.
  DenseMap<const Loop *, unsigned> LoopToDim;
  for (int D = DimSet.find_first(); D >= 0; D = DimSet.find_next(D)) {
    unsigned UD = static_cast<unsigned>(D);
    auto It = DimToLoop.find(UD);
    if (It != DimToLoop.end())
      LoopToDim[It->second] = UD;
  }

  Value *Cur = Ptr;
  // Bound the walk to avoid pathological IR.
  unsigned MaxSteps = 64;

  while (Cur && MaxSteps-- > 0) {
    // 1. Transparent: strip bitcast / addrspacecast.
    if (Value *Stripped = stripPointerCast(Cur)) {
      Cur = Stripped;
      continue;
    }

    // 2. Transparent: follow loop-invariant PHI incoming.
    if (auto *Phi = dyn_cast<PHINode>(Cur)) {
      Value *Inv = getLoopInvariantIncoming(Phi, OutermostGEMMLoop, SE);
      if (Inv) {
        Cur = Inv;
        continue;
      }
      // No invariant incoming found — treat current ptr as base.
      break;
    }

    // 3. Single-index GEP: attempt stride extraction.
    auto *GEP = dyn_cast<GetElementPtrInst>(Cur);
    if (!GEP || GEP->getNumIndices() != 1) {
      // Multi-index GEP or non-GEP non-cast non-PHI — stop.
      break;
    }

    Value *Idx = GEP->getOperand(1);
    const SCEV *IdxSCEV = SE.getSCEV(Idx);

    if (isa<SCEVCouldNotCompute>(IdxSCEV)) {
      // Non-affine index (e.g. srem): this GEP result is the base.
      Result.Base = Cur;
      return Result;
    }

    // Walk all AddRec nodes in IdxSCEV (handles both nested chains and
    // SCEVAdd-of-AddRecs via a worklist).
    SmallVector<const SCEV *, 8> Worklist;
    Worklist.push_back(IdxSCEV);
    bool FoundAffine = false;
    while (!Worklist.empty()) {
      const SCEV *S = Worklist.pop_back_val();
      if (const auto *Add = dyn_cast<SCEVAddExpr>(S)) {
        for (const SCEV *Op : Add->operands())
          Worklist.push_back(Op);
        continue;
      }
      if (const auto *AR = dyn_cast<SCEVAddRecExpr>(S)) {
        auto It = LoopToDim.find(AR->getLoop());
        if (It != LoopToDim.end()) {
          unsigned D = It->second;
          if (!Result.Strides.count(D)) {
            Result.Strides[D] = AR->getStepRecurrence(SE);
            Result.AffineDims.set(D);
            FoundAffine = true;
          }
        }
        // Recurse into start to handle nested AddRecs.
        Worklist.push_back(AR->getStart());
      }
      // Scalar / unknown terms: ignore (they contribute to the base offset).
    }

    if (!FoundAffine) {
      // This GEP's index is loop-invariant at all GEMM dims —
      // could be an outer-scope offset. Continue walking upward.
    }

    Cur = GEP->getPointerOperand();
  }

  Result.Base = Cur;

  // Populate NonAffineDims = DimSet - AffineDims.
  unsigned N = DimSet.size();
  Result.NonAffineDims.resize(N);
  Result.AffineDims.resize(N);
  for (int D = DimSet.find_first(); D >= 0; D = DimSet.find_next(D)) {
    unsigned UD = static_cast<unsigned>(D);
    if (!Result.AffineDims.test(UD))
      Result.NonAffineDims.set(UD);
  }

  return Result;
}
