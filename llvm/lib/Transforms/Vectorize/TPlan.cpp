//===- TPlan.cpp - Tensor Plan IR for LoopTensorize -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TPlan.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// TPSlotTracker
//===----------------------------------------------------------------------===//

void TPSlotTracker::preAssignSynthetic(const TPSyntheticValue *V) {
  SlotMap.try_emplace(V, NextSlot++);
}

unsigned TPSlotTracker::getSlot(const TPValue *V) {
  auto [It, Inserted] = SlotMap.try_emplace(V, NextSlot);
  if (Inserted)
    ++NextSlot;
  return It->second;
}

void TPSyntheticValue::printAsOperand(raw_ostream &OS,
                                      TPSlotTracker &Tracker) const {
  OS << "tp<%" << Tracker.getSlot(this) << ">";
}

//===----------------------------------------------------------------------===//
// TPLiveIn::printAsOperand
//===----------------------------------------------------------------------===//

void TPLiveIn::printAsOperand(raw_ostream &OS, TPSlotTracker &) const {
  OS << "ir<";
  if (auto *CI = dyn_cast<ConstantInt>(IRVal)) {
    OS << CI->getSExtValue();
  } else if (auto *CF = dyn_cast<ConstantFP>(IRVal)) {
    SmallString<16> Buf;
    CF->getValueAPF().toString(Buf);
    OS << Buf;
  } else {
    IRVal->printAsOperand(OS, /*PrintType=*/false);
  }
  OS << ">";
}

//===----------------------------------------------------------------------===//
// TPDefVal::printAsOperand
//===----------------------------------------------------------------------===//

void TPDefVal::printAsOperand(raw_ostream &OS, TPSlotTracker &Tracker) const {
  OS << "tp<%" << Tracker.getSlot(this) << ">";
}

//===----------------------------------------------------------------------===//
// Recipe print() implementations
//===----------------------------------------------------------------------===//

static void printIndent(raw_ostream &OS, unsigned Indent) {
  for (unsigned I = 0; I < Indent; ++I)
    OS << "  ";
}

void TPWidenInductionRecipe::print(raw_ostream &OS, unsigned Indent,
                                   TPSlotTracker &Tracker) const {
  printIndent(OS, Indent);
  OS << "WIDEN-INDUCTION ";
  DefVal->printAsOperand(OS, Tracker);
  OS << " = phi ";
  Operands[0]->printAsOperand(OS, Tracker);
  OS << ", ";
  Operands[1]->printAsOperand(OS, Tracker);
  OS << "\n";
}

void TPReductionPHIRecipe::print(raw_ostream &OS, unsigned Indent,
                                 TPSlotTracker &Tracker) const {
  printIndent(OS, Indent);
  OS << "WIDEN-REDUCTION-PHI ";
  DefVal->printAsOperand(OS, Tracker);
  OS << " = phi ";
  Operands[0]->printAsOperand(OS, Tracker);
  OS << ", ";
  Operands[1]->printAsOperand(OS, Tracker);
  OS << "\n";
}

void TPWidenRecipe::print(raw_ostream &OS, unsigned Indent,
                          TPSlotTracker &Tracker) const {
  printIndent(OS, Indent);
  OS << "WIDEN ";
  DefVal->printAsOperand(OS, Tracker);
  OS << " = " << Inst->getOpcodeName();
  for (unsigned I = 0, E = Operands.size(); I < E; ++I) {
    OS << " ";
    Operands[I]->printAsOperand(OS, Tracker);
    if (I + 1 < E)
      OS << ",";
  }
  OS << "\n";
}

void TPWidenGEPRecipe::print(raw_ostream &OS, unsigned Indent,
                              TPSlotTracker &Tracker) const {
  printIndent(OS, Indent);
  OS << "WIDEN-GEP ";
  DefVal->printAsOperand(OS, Tracker);
  OS << " = getelementptr";
  for (unsigned I = 0, E = Operands.size(); I < E; ++I) {
    OS << " ";
    Operands[I]->printAsOperand(OS, Tracker);
    if (I + 1 < E)
      OS << ",";
  }
  OS << "\n";
}

void TPWidenLoadRecipe::print(raw_ostream &OS, unsigned Indent,
                               TPSlotTracker &Tracker) const {
  printIndent(OS, Indent);
  OS << "WIDEN ";
  DefVal->printAsOperand(OS, Tracker);
  OS << " = load ";
  Operands[0]->printAsOperand(OS, Tracker);
  OS << "\n";
}

void TPWidenStoreRecipe::print(raw_ostream &OS, unsigned Indent,
                                TPSlotTracker &Tracker) const {
  printIndent(OS, Indent);
  OS << "WIDEN store ";
  Operands[0]->printAsOperand(OS, Tracker);
  OS << ", ";
  Operands[1]->printAsOperand(OS, Tracker);
  OS << "\n";
}

void TPWidenCastRecipe::print(raw_ostream &OS, unsigned Indent,
                               TPSlotTracker &Tracker) const {
  printIndent(OS, Indent);
  OS << "WIDEN-CAST ";
  DefVal->printAsOperand(OS, Tracker);
  OS << " = " << CastInst->getOpcodeName() << " ";
  Operands[0]->printAsOperand(OS, Tracker);
  OS << "\n";
}

void TPCanonicalIVRecipe::print(raw_ostream &OS, unsigned Indent,
                                TPSlotTracker &Tracker) const {
  printIndent(OS, Indent);
  OS << "CANONICAL-INDUCTION ";
  DefVal->printAsOperand(OS, Tracker);
  OS << " = phi ";
  Operands[0]->printAsOperand(OS, Tracker);
  OS << ", ";
  Operands[1]->printAsOperand(OS, Tracker);
  OS << "\n";
}

void TPCanonicalIVIncrRecipe::print(raw_ostream &OS, unsigned Indent,
                                    TPSlotTracker &Tracker) const {
  printIndent(OS, Indent);
  OS << "CANONICAL-INDUCTION-INC ";
  DefVal->printAsOperand(OS, Tracker);
  OS << " = add ";
  Operands[0]->printAsOperand(OS, Tracker);
  OS << ", ";
  Operands[1]->printAsOperand(OS, Tracker);
  OS << "\n";
}

void TPCanonicalIVExitCmpRecipe::print(raw_ostream &OS, unsigned Indent,
                                       TPSlotTracker &Tracker) const {
  printIndent(OS, Indent);
  OS << "CANONICAL-INDUCTION-CMP ";
  DefVal->printAsOperand(OS, Tracker);
  OS << " = icmp ";
  Operands[0]->printAsOperand(OS, Tracker);
  OS << ", ";
  Operands[1]->printAsOperand(OS, Tracker);
  OS << "\n";
}

//===----------------------------------------------------------------------===//
// TPLoopRegion::print
//===----------------------------------------------------------------------===//

void TPLoopRegion::print(raw_ostream &OS, unsigned Indent,
                         TPSlotTracker &Tracker) const {
  printIndent(OS, Indent);
  OS << "<x1> loop[" << Level << "] (trip=";
  if (TripCount)
    TripCount->print(OS);
  else
    OS << "<unknown>";
  OS << ") {\n";

  for (const TPRecipeBase &R : Recipes)
    R.print(OS, Indent + 1, Tracker);

  if (Child)
    Child->print(OS, Indent + 1, Tracker);

  printIndent(OS, Indent);
  OS << "}\n";
}

//===----------------------------------------------------------------------===//
// TPlan::getOrCreateLiveIn / getTPValue
//===----------------------------------------------------------------------===//

TPLiveIn *TPlan::getOrCreateLiveIn(Value *V) {
  auto It = ValueMap.find(V);
  if (It != ValueMap.end())
    return static_cast<TPLiveIn *>(It->second);
  auto *LI = new TPLiveIn(V);
  LiveIns.emplace_back(LI);
  ValueMap[V] = LI;
  return LI;
}

TPValue *TPlan::getTPValue(Value *V) {
  auto It = ValueMap.find(V);
  if (It != ValueMap.end())
    return It->second;
  // Not in map — treat as live-in
  return getOrCreateLiveIn(V);
}

//===----------------------------------------------------------------------===//
// TPlan::buildInitial
//===----------------------------------------------------------------------===//

TPlan TPlan::buildInitial(const LoopNestInfo &Info) {
  TPlan P;
  if (!Info.Loops.empty()) {
    if (auto *F = Info.Loops[0]->getHeader()->getParent())
      P.FuncName = std::string(F->getName());
  }
  P.Depth = Info.Depth;

  // We'll build regions recursively. Track which loops are "above" current.
  ArrayRef<Loop *> AllLoops = Info.Loops;

  // Recursive lambda to build a region for loop at index Idx
  std::function<std::unique_ptr<TPLoopRegion>(unsigned)> BuildRegion =
      [&](unsigned Idx) -> std::unique_ptr<TPLoopRegion> {
    Loop *L = AllLoops[Idx];

    // Get trip count from InductionDesc
    const SCEV *TC = Info.IVs[Idx].TripCount;

    // Get the loop exit bound: latch branch condition ICmpInst RHS.
    Value *LatchBound = nullptr;
    if (BasicBlock *Latch = L->getLoopLatch()) {
      if (auto *BI = dyn_cast<BranchInst>(Latch->getTerminator())) {
        if (BI->isConditional()) {
          if (auto *Cmp = dyn_cast<ICmpInst>(BI->getCondition()))
            LatchBound = Cmp->getOperand(1);
        }
      }
    }
    TPValue *BoundTP = LatchBound ? P.getOrCreateLiveIn(LatchBound) : nullptr;

    auto Region = std::make_unique<TPLoopRegion>(Idx, L, TC);

    // Hoist InductionPhi declaration so it can be used for CanonIV creation.
    PHINode *InductionPhi = Info.IVs[Idx].IndVar;

    // Insert canonical IV phi as the first recipe (VPlan-style).
    // Use a zero live-in as start matching the IV phi's type; step is patched below.
    TPValue *ZeroTP = P.getOrCreateLiveIn(
        ConstantInt::get(InductionPhi->getType(), 0));
    auto *CanonIV = new TPCanonicalIVRecipe(ZeroTP, ZeroTP /*placeholder step*/);
    Region->appendRecipe(CanonIV);

    // Loops that are "outer" (index < Idx) — their IVs are live-ins to us
    // Loops that are at or inside (index >= Idx) — defined within this region

    // Set of basic blocks belonging to loops strictly inside this one
    SmallPtrSet<BasicBlock *, 16> InnerBlocks;
    for (unsigned J = Idx + 1; J < AllLoops.size(); ++J)
      for (BasicBlock *BB : AllLoops[J]->blocks())
        InnerBlocks.insert(BB);

    // Process header PHIs
    BasicBlock *Header = L->getHeader();

    for (PHINode &Phi : Header->phis()) {
      Value *PhiV = &Phi;

      // Check if this is the induction variable
      if (&Phi == InductionPhi) {
        // Find start value (from outside loop) and step value (from latch)
        Value *StartVal = nullptr;
        Value *StepVal = nullptr;
        BasicBlock *Latch = L->getLoopLatch();
        for (unsigned I = 0, E = Phi.getNumIncomingValues(); I < E; ++I) {
          BasicBlock *Pred = Phi.getIncomingBlock(I);
          if (Latch && Pred == Latch)
            StepVal = Phi.getIncomingValue(I);
          else if (!L->contains(Pred))
            StartVal = Phi.getIncomingValue(I);
        }
        TPValue *StartTP = StartVal ? P.getTPValue(StartVal) : P.getOrCreateLiveIn(
            ConstantInt::get(Phi.getType(), 0));
        // StepVal will be defined by a later recipe (the increment); use a
        // placeholder live-in for now. We'll fix it up after building body.
        // For simplicity, create the recipe with a forward-ref placeholder.
        // We'll handle step as a live-in initially and patch below.
        (void)StepVal; // handled in post-pass

        auto *R = new TPWidenInductionRecipe(
            &Phi, StartTP,
            StartTP /* placeholder; patched after body */, Idx);
        Region->appendRecipe(R);
        P.ValueMap[PhiV] = R->getDefinedValue();
        Region->setIV(R->getDefinedValue());
      } else {
        // Reduction PHI: start from outside, loop value from inside
        Value *InitVal = nullptr;
        Value *LoopVal = nullptr;
        for (unsigned I = 0, E = Phi.getNumIncomingValues(); I < E; ++I) {
          BasicBlock *Pred = Phi.getIncomingBlock(I);
          if (!L->contains(Pred))
            InitVal = Phi.getIncomingValue(I);
          else
            LoopVal = Phi.getIncomingValue(I);
        }
        TPValue *InitTP = InitVal ? P.getTPValue(InitVal) : P.getOrCreateLiveIn(
            ConstantInt::get(Phi.getType(), 0));
        TPValue *LoopTP = LoopVal ? P.getTPValue(LoopVal) : InitTP;

        auto *R = new TPReductionPHIRecipe(&Phi, InitTP, LoopTP);
        Region->appendRecipe(R);
        P.ValueMap[PhiV] = R->getDefinedValue();
      }
    }

    // Process blocks: include header (non-PHI instructions) and other body blocks,
    // but exclude blocks belonging to inner loops.
    // Helper lambda to emit instructions from a basic block.
    auto EmitBlock = [&](BasicBlock *BB) {
      for (Instruction &Inst : *BB) {
        if (isa<BranchInst>(&Inst) || isa<SwitchInst>(&Inst))
          continue;
        if (isa<PHINode>(&Inst))
          continue;

        Value *InstV = &Inst;

        if (auto *GEP = dyn_cast<GetElementPtrInst>(&Inst)) {
          SmallVector<TPValue *, 4> Ops;
          for (Value *Op : GEP->operands())
            Ops.push_back(P.getTPValue(Op));
          auto *R = new TPWidenGEPRecipe(&Inst, Ops);
          Region->appendRecipe(R);
          P.ValueMap[InstV] = R->getDefinedValue();
        } else if (auto *LI = dyn_cast<LoadInst>(&Inst)) {
          TPValue *PtrOp = P.getTPValue(LI->getPointerOperand());
          auto *R = new TPWidenLoadRecipe(&Inst, PtrOp);
          Region->appendRecipe(R);
          P.ValueMap[InstV] = R->getDefinedValue();
        } else if (auto *SI = dyn_cast<StoreInst>(&Inst)) {
          TPValue *PtrOp = P.getTPValue(SI->getPointerOperand());
          TPValue *ValOp = P.getTPValue(SI->getValueOperand());
          auto *R = new TPWidenStoreRecipe(&Inst, PtrOp, ValOp);
          Region->appendRecipe(R);
        } else if (isa<BitCastInst>(&Inst) || isa<SExtInst>(&Inst) ||
                   isa<ZExtInst>(&Inst)) {
          TPValue *SrcOp = P.getTPValue(Inst.getOperand(0));
          auto *R = new TPWidenCastRecipe(&Inst, SrcOp);
          Region->appendRecipe(R);
          P.ValueMap[InstV] = R->getDefinedValue();
        } else {
          SmallVector<TPValue *, 4> Ops;
          for (Value *Op : Inst.operands())
            Ops.push_back(P.getTPValue(Op));
          auto *R = new TPWidenRecipe(&Inst, Ops);
          Region->appendRecipe(R);
          P.ValueMap[InstV] = R->getDefinedValue();
        }
      }
    };

    // Emit non-PHI instructions from header first
    EmitBlock(Header);

    // Process body blocks (excluding inner loop blocks and header)
    for (BasicBlock *BB : L->blocks()) {
      if (BB == Header)
        continue;
      if (InnerBlocks.count(BB))
        continue;
      // Also skip if this BB belongs to a sub-loop
      bool IsInSubLoop = false;
      for (unsigned J = Idx + 1; J < AllLoops.size(); ++J) {
        if (AllLoops[J]->contains(BB)) {
          IsInSubLoop = true;
          break;
        }
      }
      if (IsInSubLoop)
        continue;

      EmitBlock(BB);
    }

    // Recurse into child region (next loop level)
    if (Idx + 1 < AllLoops.size()) {
      // Process inner loop's header and body into child region
      auto Child = BuildRegion(Idx + 1);
      Region->setChild(std::move(Child));
    }

    // Patch induction recipe: fix up the step operand now that body is built.
    // Find the increment instruction for this loop's IV.
    BasicBlock *Latch = L->getLoopLatch();
    if (Latch && InductionPhi) {
      Value *StepVal = nullptr;
      for (unsigned I = 0, E = InductionPhi->getNumIncomingValues(); I < E; ++I)
        if (InductionPhi->getIncomingBlock(I) == Latch)
          StepVal = InductionPhi->getIncomingValue(I);

      if (StepVal) {
        // Find the TPWidenInductionRecipe for this phi
        for (TPRecipeBase &R : Region->getRecipes()) {
          if (auto *WI = dyn_cast<TPWidenInductionRecipe>(&R)) {
            if (WI->getIVPhi() == InductionPhi) {
              // Patch operand[1] (step)
              TPValue *StepTP = P.getTPValue(StepVal);
              WI->setOperand(1, StepTP);
              StepTP->addUser(WI);
              break;
            }
          }
        }
      }
    }

    // Create canonical IV companion recipes.
    if (CanonIV->getDefinedValue() && BoundTP) {
      // Increment: canonical_iv + PF
      auto *IncrRecipe = new TPCanonicalIVIncrRecipe(
          CanonIV->getDefinedValue(), &P.PF);
      Region->appendRecipe(IncrRecipe);

      // Patch canonical IV step operand to point to the increment result.
      CanonIV->setOperand(1, IncrRecipe->getDefinedValue());
      IncrRecipe->getDefinedValue()->addUser(CanonIV);

      // Exit cmp: incremented_iv icmp bound
      auto *CmpRecipe = new TPCanonicalIVExitCmpRecipe(
          IncrRecipe->getDefinedValue(), BoundTP);
      Region->appendRecipe(CmpRecipe);
    }

    return Region;
  };

  if (!AllLoops.empty())
    P.RootRegion = BuildRegion(0);

  P.ReductionDims = Info.ReductionDims;
  return P;
}

//===----------------------------------------------------------------------===//
// TPlan::print
//===----------------------------------------------------------------------===//

void TPlan::print(raw_ostream &OS) const {
  OS << "TPlan '" << FuncName << "' (depth=" << Depth << ") {\n";

  // Pre-assign PF as tp<%0> before any lazy recipe-slot assignment.
  Tracker.reset();
  Tracker.preAssignSynthetic(&PF);

  // Print synthetic live-ins first (VPlan style).
  OS << "Live-in ";
  PF.printAsOperand(OS, Tracker);
  OS << " = PF\n";

  // Print IR-backed live-ins (unchanged, still ir<>).
  for (const auto &LI : LiveIns) {
    OS << "Live-in ";
    LI->printAsOperand(OS, Tracker);
    OS << "\n";
  }
  OS << "\n";

  if (RootRegion)
    RootRegion->print(OS, 0, Tracker);

  OS << "}\n";
}
