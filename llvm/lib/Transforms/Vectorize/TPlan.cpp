//===- TPlan.cpp - Tensor Plan IR for LoopTensorize -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TPlan.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <string>  // for std::to_string

#define DEBUG_TYPE "tplan"

using namespace llvm;

const SCEV *TPSingleDefRecipe::getMemStride(unsigned Dim, const TPlan &Plan,
                                             ScalarEvolution &SE) const {
  auto It = MemStrides.find(Dim);
  if (It != MemStrides.end())
    return It->second;
  return SE.getConstant(APInt(64, Plan.getDenseStrideForDim(Dim)));
}

const SCEV *TPWidenStoreRecipe::getMemStride(unsigned Dim, const TPlan &Plan,
                                              ScalarEvolution &SE) const {
  auto It = MemStrides.find(Dim);
  if (It != MemStrides.end())
    return It->second;
  return SE.getConstant(APInt(64, Plan.getDenseStrideForDim(Dim)));
}

//===----------------------------------------------------------------------===//
// TPSlotTracker
//===----------------------------------------------------------------------===//

void TPSlotTracker::preAssignSynthetic(const TPSymbolicValue *V) {
  SlotMap.try_emplace(V, NextSlot++);
}

unsigned TPSlotTracker::getSlot(const TPValue *V) {
  auto [It, Inserted] = SlotMap.try_emplace(V, NextSlot);
  if (Inserted)
    ++NextSlot;
  return It->second;
}

void TPSymbolicValue::printAsOperand(raw_ostream &OS,
                                      TPSlotTracker &Tracker) const {
  OS << "tp<%" << Tracker.getSlot(this) << ">";
}

//===----------------------------------------------------------------------===//
// TPIRValue::printAsOperand
//===----------------------------------------------------------------------===//

void TPIRValue::printAsOperand(raw_ostream &OS, TPSlotTracker &) const {
  Value *IRVal = getUnderlyingValue();
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
// TPConstantInt methods
//===----------------------------------------------------------------------===//

bool TPConstantInt::isOne()  const { return cast<ConstantInt>(getUnderlyingValue())->isOne(); }
bool TPConstantInt::isZero() const { return cast<ConstantInt>(getUnderlyingValue())->isZero(); }
const APInt &TPConstantInt::getAPInt() const {
  return cast<ConstantInt>(getUnderlyingValue())->getValue();
}
unsigned TPConstantInt::getBitWidth() const { return getAPInt().getBitWidth(); }
uint64_t TPConstantInt::getZExtValue() const { return getAPInt().getZExtValue(); }

//===----------------------------------------------------------------------===//
// TPValue utility methods
//===----------------------------------------------------------------------===//

void TPValue::replaceAllUsesWith(TPValue *New) {
  SmallVector<TPUser *, 4> Snapshot(Users.begin(), Users.end());
  for (TPUser *U : Snapshot)
    for (unsigned I = 0, E = U->getNumOperands(); I != E; ++I)
      if (U->getOperand(I) == this)
        U->setOperand(I, New);
}

void TPValue::replaceUsesWithIf(TPValue *New,
                                 function_ref<bool(TPUser &, unsigned)> Fn) {
  SmallVector<TPUser *, 4> Snapshot(Users.begin(), Users.end());
  for (TPUser *U : Snapshot)
    for (unsigned I = 0, E = U->getNumOperands(); I != E; ++I)
      if (U->getOperand(I) == this && Fn(*U, I))
        U->setOperand(I, New);
}

// Base implementation returns nullptr; TPRecipeValue overrides in TPlan.h.
TPRecipeBase *TPValue::getDefiningRecipe() { return nullptr; }
const TPRecipeBase *TPValue::getDefiningRecipe() const { return nullptr; }

//===----------------------------------------------------------------------===//
// TPPhiAccessors
//===----------------------------------------------------------------------===//

void TPPhiAccessors::printPhiOperands(raw_ostream &OS,
                                       TPSlotTracker &SlotTracker) const {
  for (unsigned I = 0, E = getNumIncoming(); I != E; ++I) {
    if (I > 0) OS << ", ";
    getIncomingValue(I)->printAsOperand(OS, SlotTracker);
  }
}

//===----------------------------------------------------------------------===//
// TPIRFlags
//===----------------------------------------------------------------------===//

TPIRFlags::TPIRFlags(Instruction &I) {
  if (auto *OBO = dyn_cast<OverflowingBinaryOperator>(&I)) {
    OpType = OperationType::OverflowingBinOp;
    OvflowFlags.HasNUW = OBO->hasNoUnsignedWrap();
    OvflowFlags.HasNSW = OBO->hasNoSignedWrap();
  } else if (isa<TruncInst>(&I)) {
    OpType = OperationType::Trunc;
  } else if (auto *CI = dyn_cast<ICmpInst>(&I)) {
    OpType = OperationType::Cmp;
    CmpPred = CI->getPredicate();
  } else if (auto *FCI = dyn_cast<FCmpInst>(&I)) {
    OpType = OperationType::FCmp;
    CmpPred = FCI->getPredicate();
    FMF = FCI->getFastMathFlags();
  } else if (auto *FPO = dyn_cast<FPMathOperator>(&I)) {
    OpType = OperationType::FPMathOp;
    FMF = FPO->getFastMathFlags();
  } else if (isa<GetElementPtrInst>(&I)) {
    OpType = OperationType::GEPOp;
  } else if (auto *PEO = dyn_cast<PossiblyExactOperator>(&I)) {
    OpType = OperationType::PossiblyExactOp;
    ExactFlags.IsExact = PEO->isExact();
  } else if (auto *PDO = dyn_cast<PossiblyDisjointInst>(&I)) {
    OpType = OperationType::DisjointOp;
    DisjointFlags.IsDisjoint = PDO->isDisjoint();
  }
}

void TPIRFlags::applyFlags(Instruction &I) const {
  switch (OpType) {
  case OperationType::OverflowingBinOp:
    I.setHasNoUnsignedWrap(OvflowFlags.HasNUW);
    I.setHasNoSignedWrap(OvflowFlags.HasNSW);
    break;
  case OperationType::FCmp:
  case OperationType::FPMathOp:
    I.setFastMathFlags(FMF);
    break;
  case OperationType::PossiblyExactOp:
    I.setIsExact(ExactFlags.IsExact);
    break;
  case OperationType::DisjointOp:
    cast<PossiblyDisjointInst>(&I)->setIsDisjoint(DisjointFlags.IsDisjoint);
    break;
  default:
    break;
  }
}

//===----------------------------------------------------------------------===//
// TPIRMetadata
//===----------------------------------------------------------------------===//

TPIRMetadata::TPIRMetadata(Instruction &I) {
  SmallVector<std::pair<unsigned, MDNode *>, 8> All;
  I.getAllMetadataOtherThanDebugLoc(All);
  for (auto &[Kind, Node] : All) {
    if (Kind == LLVMContext::MD_tbaa ||
        Kind == LLVMContext::MD_fpmath ||
        Kind == LLVMContext::MD_access_group)
      Metadata.push_back({Kind, Node});
  }
}

void TPIRMetadata::applyMetadata(Instruction &I) const {
  for (auto &[Kind, Node] : Metadata)
    I.setMetadata(Kind, Node);
}

void TPIRMetadata::setMetadata(unsigned Kind, MDNode *Node) {
  for (auto &P : Metadata) {
    if (P.first == Kind) { P.second = Node; return; }
  }
  Metadata.push_back({Kind, Node});
}

MDNode *TPIRMetadata::getMetadata(unsigned Kind) const {
  for (auto &P : Metadata)
    if (P.first == Kind) return P.second;
  return nullptr;
}

void TPIRMetadata::intersect(const TPIRMetadata &Other) {
  Metadata.erase(llvm::remove_if(Metadata, [&](auto &P) {
    return Other.getMetadata(P.first) == nullptr;
  }), Metadata.end());
}

//===----------------------------------------------------------------------===//
// TPRecipeBase insertion/movement helpers
//===----------------------------------------------------------------------===//

void TPRecipeBase::insertBefore(TPRecipeBase *InsertPos) {
  assert(InsertPos->Parent && "InsertPos has no parent block");
  Parent = InsertPos->Parent;
  Parent->getRecipeList().insert(InsertPos->getIterator(), this);
}

void TPRecipeBase::insertAfter(TPRecipeBase *InsertPos) {
  assert(InsertPos->Parent && "InsertPos has no parent block");
  Parent = InsertPos->Parent;
  auto It = std::next(InsertPos->getIterator());
  Parent->getRecipeList().insert(It, this);
}

void TPRecipeBase::removeFromParent() {
  assert(Parent && "Recipe not in any block");
  Parent->getRecipeList().remove(getIterator());
  Parent = nullptr;
}

iplist<TPRecipeBase>::iterator TPRecipeBase::eraseFromParent() {
  assert(Parent && "Recipe not in any block");
  return Parent->getRecipeList().erase(getIterator());
}

//===----------------------------------------------------------------------===//
// TPSingleDefRecipe::printAsOperand
//===----------------------------------------------------------------------===//

void TPSingleDefRecipe::printAsOperand(raw_ostream &OS,
                                        TPSlotTracker &Tracker) const {
  // For recipes that wrap IR instructions or PHIs, use the IR value name so
  // the TPlan printout is directly readable against the original IR.
  StringRef Name;
  if (auto *R = dyn_cast<TPWidenRecipe>(this))
    Name = R->getInstruction()->getName();
  else if (auto *R = dyn_cast<TPWidenInductionRecipe>(this))
    Name = R->getIVPhi()->getName();
  else if (auto *R = dyn_cast<TPWidenLoadRecipe>(this))
    Name = R->getInstruction()->getName();
  else if (auto *R = dyn_cast<TPWidenGEPRecipe>(this))
    Name = R->getInstruction()->getName();
  else if (auto *R = dyn_cast<TPWidenCastRecipe>(this))
    Name = R->getInstruction()->getName();
  else if (auto *R = dyn_cast<TPReductionPHIRecipe>(this))
    Name = R->getPhi()->getName();
  if (!Name.empty()) {
    OS << "ir<%" << Name << ">";
    return;
  }
  OS << "tp<%" << Tracker.getSlot(this) << ">";
}

//===----------------------------------------------------------------------===//
// Recipe print() implementations
//===----------------------------------------------------------------------===//

static void printIndent(raw_ostream &OS, unsigned Indent) {
  for (unsigned I = 0; I < Indent; ++I)
    OS << "  ";
}

/// DFS pre-order traversal starting from \p Start, following successors in
/// insertion order. Visited tracking prevents re-visiting LatchBB.
SmallVector<TPBlockBase *, 8>
llvm::constructionOrder(TPBlockBase *Start) {
  SmallVector<TPBlockBase *, 8> Order;
  SmallPtrSet<TPBlockBase *, 8> Visited;
  SmallVector<TPBlockBase *, 8> Stack;
  Stack.push_back(Start);
  while (!Stack.empty()) {
    TPBlockBase *B = Stack.pop_back_val();
    if (!Visited.insert(B).second)
      continue;
    Order.push_back(B);
    // Push successors reversed so successor[0] is visited first (LIFO).
    for (TPBlockBase *Succ : llvm::reverse(B->getSuccessors()))
      if (!Visited.count(Succ))
        Stack.push_back(Succ);
  }
  return Order;
}

/// DFS pre-order from \p Start, treating TPRegionBlock nodes as opaque leaves
/// (adds them to the result but does not descend into their internal CFG).
/// Used to traverse a single region's own blocks without crossing into nested regions.
static SmallVector<TPBlockBase *, 8>
intraRegionOrder(TPBlockBase *Start) {
  SmallVector<TPBlockBase *, 8> Order;
  SmallPtrSet<TPBlockBase *, 8> Visited;
  SmallVector<TPBlockBase *, 8> Stack;
  Stack.push_back(Start);
  while (!Stack.empty()) {
    TPBlockBase *B = Stack.pop_back_val();
    if (!Visited.insert(B).second)
      continue;
    Order.push_back(B);
    // Do NOT descend into nested regions — treat them as opaque leaves.
    if (isa<TPRegionBlock>(B))
      continue;
    for (TPBlockBase *Succ : llvm::reverse(B->getSuccessors()))
      if (!Visited.count(Succ))
        Stack.push_back(Succ);
  }
  return Order;
}

static void printBlockSuccessors(raw_ostream &OS, const Twine &Indent,
                                  const TPBlockBase *B) {
  OS << Indent;
  if (B->getSuccessors().empty()) {
    OS << "No successors\n";
    return;
  }
  OS << "Successor(s):";
  for (const TPBlockBase *S : B->getSuccessors())
    OS << " " << S->getName();
  OS << "\n";
}

void TPWidenIntOrFpInductionRecipe::print(raw_ostream &OS, unsigned Indent,
                                          TPSlotTracker &Tracker) const {
  printIndent(OS, Indent);
  OS << "WIDEN-INDUCTION ";
  printAsOperand(OS, Tracker);
  OS << " = phi ";
  Operands[0]->printAsOperand(OS, Tracker);
  OS << ", ";
  Operands[1]->printAsOperand(OS, Tracker);
  OS << "\n";
}

void TPWidenPointerInductionRecipe::print(raw_ostream &OS, unsigned Indent,
                                           TPSlotTracker &Tracker) const {
  printIndent(OS, Indent);
  OS << "WIDEN-POINTER-INDUCTION ";
  printAsOperand(OS, Tracker);
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
  printAsOperand(OS, Tracker);
  OS << " = phi ";
  Operands[0]->printAsOperand(OS, Tracker);
  OS << ", ";
  Operands[1]->printAsOperand(OS, Tracker);
  OS << "\n";
}

void TPFirstOrderRecurrencePHIRecipe::print(raw_ostream &OS, unsigned Indent,
                                             TPSlotTracker &Tracker) const {
  printIndent(OS, Indent);
  OS << "FIRST-ORDER-RECURRENCE-PHI ";
  printAsOperand(OS, Tracker); // TODO: print incoming values
  OS << "\n";
}

void TPActiveLaneMaskPHIRecipe::print(raw_ostream &OS, unsigned Indent,
                                       TPSlotTracker &Tracker) const {
  printIndent(OS, Indent);
  OS << "ACTIVE-LANE-MASK-PHI ";
  printAsOperand(OS, Tracker); // TODO: print incoming values
  OS << "\n";
}

void TPEVLBasedIVPHIRecipe::print(raw_ostream &OS, unsigned Indent,
                                   TPSlotTracker &Tracker) const {
  printIndent(OS, Indent);
  OS << "EVL-BASED-IV-PHI ";
  printAsOperand(OS, Tracker); // TODO: print incoming values
  OS << "\n";
}

void TPWidenPHIRecipe::print(raw_ostream &OS, unsigned Indent,
                              TPSlotTracker &Tracker) const {
  printIndent(OS, Indent);
  OS << "WIDEN-PHI ";
  printAsOperand(OS, Tracker); // TODO: print incoming values
  OS << "\n";
}

void TPPredInstPHIRecipe::print(raw_ostream &OS, unsigned Indent,
                                 TPSlotTracker &Tracker) const {
  printIndent(OS, Indent);
  OS << "PRED-PHI ";
  printAsOperand(OS, Tracker); // TODO: print incoming values
  OS << "\n";
}

void TPPhi::print(raw_ostream &OS, unsigned Indent,
                  TPSlotTracker &Tracker) const {
  printIndent(OS, Indent);
  OS << "PHI ";
  printAsOperand(OS, Tracker); // TODO: print incoming values
  OS << "\n";
}

void TPWidenRecipe::print(raw_ostream &OS, unsigned Indent,
                          TPSlotTracker &Tracker) const {
  printIndent(OS, Indent);
  OS << "WIDEN ";
  printAsOperand(OS, Tracker);
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
  printAsOperand(OS, Tracker);
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
  printAsOperand(OS, Tracker);
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
  printAsOperand(OS, Tracker);
  OS << " = " << CastInst->getOpcodeName() << " ";
  Operands[0]->printAsOperand(OS, Tracker);
  OS << "\n";
}

void TPCanonicalIVRecipe::print(raw_ostream &OS, unsigned Indent,
                                TPSlotTracker &Tracker) const {
  printIndent(OS, Indent);
  OS << "CANONICAL-INDUCTION ";
  printAsOperand(OS, Tracker);
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
  printAsOperand(OS, Tracker);
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
  printAsOperand(OS, Tracker);
  OS << " = icmp ";
  Operands[0]->printAsOperand(OS, Tracker);
  OS << ", ";
  Operands[1]->printAsOperand(OS, Tracker);
  OS << "\n";
}

//===----------------------------------------------------------------------===//
// TPBlockBase subclass print() implementations
//===----------------------------------------------------------------------===//

/// Append DimSet annotation for a TPSingleDefRecipe (only when non-empty).
/// Shown at Stage 2+ after TPlanWidener_widen() has populated DimSet fields.
static void printDimSetAnnotation(raw_ostream &OS, const TPRecipeBase &R,
                                   unsigned Indent) {
  const auto *SDR = dyn_cast<TPSingleDefRecipe>(&R);
  if (!SDR || SDR->DimSet.none())
    return;
  printIndent(OS, Indent);
  OS << "; DimSet={";
  bool First = true;
  for (int D = SDR->DimSet.find_first(); D >= 0; D = SDR->DimSet.find_next(D)) {
    if (!First)
      OS << ",";
    OS << D;
    First = false;
  }
  OS << "}\n";
}

void TPBasicBlock::print(raw_ostream &OS, const Twine &Indent,
                          TPSlotTracker &Tracker) const {
  OS << Indent << getName() << ":\n";
  // Recipes use unsigned Indent (existing API); compute depth from Twine length.
  // Invariant: each nesting level is exactly 2 spaces, so depth = len/2.
  unsigned RecipeDepth = Indent.str().size() / 2 + 1;
  for (const TPRecipeBase &R : Recipes) {
    R.print(OS, RecipeDepth, Tracker);
    printDimSetAnnotation(OS, R, RecipeDepth + 1);
  }
  printBlockSuccessors(OS, Indent, this);
  OS << "\n";
}

void TPIRBasicBlock::print(raw_ostream &OS, const Twine &Indent,
                            TPSlotTracker &Tracker) const {
  OS << Indent << getName() << ":\n";
  unsigned RecipeDepth = Indent.str().size() / 2 + 1;
  for (const TPRecipeBase &R : Recipes) {
    R.print(OS, RecipeDepth, Tracker);
    printDimSetAnnotation(OS, R, RecipeDepth + 1);
  }
  printBlockSuccessors(OS, Indent, this);
  OS << "\n";
}

void TPRegionBlock::print(raw_ostream &OS, const Twine &Indent,
                           TPSlotTracker &Tracker) const {
  OS << Indent << "<x1> " << getName() << ": {\n";
  if (Entry) {
    std::string InnerIndentStr = (Indent + "  ").str();
    for (TPBlockBase *B : intraRegionOrder(Entry)) {
      if (Inner && B == Inner)
        Inner->print(OS, InnerIndentStr, Tracker);
      else
        B->print(OS, InnerIndentStr, Tracker);
    }
  }
  OS << Indent << "}\n";
  printBlockSuccessors(OS, Indent, this);
  OS << "\n";
}

//===----------------------------------------------------------------------===//
// TPBlockBase subclass execute() stubs
//===----------------------------------------------------------------------===//

void TPBasicBlock::execute(TPTransformState &State) {
  // If an explicit IR BB was provided (e.g. a synthetic latch block that wraps
  // the loop's real latch), reposition the builder there so that cloned
  // instructions are inserted in the correct IR block.  Without this, the
  // builder stays at whatever position the preceding TPIRBasicBlock (the loop
  // header) left it, causing dominance violations when recipes reference
  // values defined in the latch (e.g. reduction-accumulator PHIs).
  if (InsertionBB)
    State.Builder.SetInsertPoint(&*InsertionBB->getFirstNonPHIIt());
  for (TPRecipeBase &R : Recipes)
    R.execute(State);
}

void TPIRBasicBlock::execute(TPTransformState &State) {
  // Recipes are inserted before the first non-phi instruction of IRBB.
  Instruction *InsertPt = &*IRBB->getFirstNonPHIIt();
  State.Builder.SetInsertPoint(InsertPt);
  for (TPRecipeBase &R : Recipes)
    R.execute(State);
}

void TPRegionBlock::execute(TPTransformState &State) {
  // If a TPlanTransformer installed a tiling override (e.g. TPTilingRegion),
  // first run Entry to position the builder at the K-loop entry block (e.g.
  // ir-bb<k.loop>), then delegate to the tiling override which takes KLoopBB
  // from Builder.GetInsertBlock(). Entry's recipes are IsSubsumed=true so no
  // IR is emitted — it's purely a builder-repositioning step.
  if (TilingOverride) {
    if (Entry)
      Entry->execute(State);
    TilingOverride->execute(State);
    return;
  }
  if (!Entry)
    return;
  for (TPBlockBase *B : intraRegionOrder(Entry)) {
    if (Inner && B == Inner)
      Inner->execute(State);
    else
      B->execute(State);
  }
}

void TPRegionBlock::printFlat(raw_ostream &OS, const Twine &Indent,
                               TPSlotTracker &Tracker) const {
  OS << Indent << "<x1> " << getName() << ": {\n";
  if (Entry) {
    std::string InnerIndentStr = (Indent + "  ").str();
    for (TPBlockBase *B : constructionOrder(Entry))
      B->print(OS, InnerIndentStr, Tracker);
  }
  OS << Indent << "}\n";
  printBlockSuccessors(OS, Indent, this);
  OS << "\n";
}

void TPRegionBlock::executeFlat(TPTransformState &State) {
  // Walk internal CFG in construction order (DFS pre-order from Entry).
  // The latch block's recipes (IV incr + cmp) have no successors within the
  // region, so execution order matches the def-use requirements.
  if (Entry)
    for (TPBlockBase *B : constructionOrder(Entry))
      B->execute(State);
}

//===----------------------------------------------------------------------===//
// TPlan::getOrCreateLiveIn / getTPValue
//===----------------------------------------------------------------------===//

TPIRValue *TPlan::getOrCreateLiveIn(Value *V) {
  auto It = ValueMap.find(V);
  if (It != ValueMap.end())
    return static_cast<TPIRValue *>(It->second);
  auto *LI = new TPIRValue(V);
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

  // DimPFs[D] is the per-dimension parallel-factor for DimIdx D.
  // After DimIdx reversal: DimPFs[0]=innermost loop PF, DimPFs[Depth-1]=outermost.
  for (unsigned D = 0; D < P.Depth; ++D)
    P.DimPFs.push_back(
        std::make_unique<TPSymbolicValue>("PF[" + std::to_string(D) + "]"));

  // We'll build regions recursively. Track which loops are "above" current.
  ArrayRef<Loop *> AllLoops = Info.Loops;

  // Helper lambda to emit instructions from a basic block into a target block.
  auto EmitBlock = [&](BasicBlock *BB, TPBasicBlock *Target) {
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
        Target->appendRecipe(R);
        P.ValueMap[InstV] = R;
      } else if (auto *LI = dyn_cast<LoadInst>(&Inst)) {
        TPValue *PtrOp = P.getTPValue(LI->getPointerOperand());
        auto *R = new TPWidenLoadRecipe(&Inst, PtrOp);
        Target->appendRecipe(R);
        P.ValueMap[InstV] = R;
      } else if (auto *SI = dyn_cast<StoreInst>(&Inst)) {
        TPValue *PtrOp = P.getTPValue(SI->getPointerOperand());
        TPValue *ValOp = P.getTPValue(SI->getValueOperand());
        auto *R = new TPWidenStoreRecipe(&Inst, PtrOp, ValOp);
        Target->appendRecipe(R);
      } else if (isa<BitCastInst>(&Inst) || isa<SExtInst>(&Inst) ||
                 isa<ZExtInst>(&Inst)) {
        TPValue *SrcOp = P.getTPValue(Inst.getOperand(0));
        auto *R = new TPWidenCastRecipe(&Inst, SrcOp);
        Target->appendRecipe(R);
        P.ValueMap[InstV] = R;
      } else {
        SmallVector<TPValue *, 4> Ops;
        for (Value *Op : Inst.operands())
          Ops.push_back(P.getTPValue(Op));
        auto *R = new TPWidenRecipe(&Inst, Ops);
        Target->appendRecipe(R);
        P.ValueMap[InstV] = R;
      }
    }
  };

  // Recursive lambda to build a TPRegionBlock for loop at index Idx.
  std::function<TPRegionBlock *(unsigned)> BuildRegion =
      [&](unsigned Idx) -> TPRegionBlock * {
    Loop *L = AllLoops[Idx];

    // Get trip count from InductionDesc
    const SCEV *TC = Info.IVs[Idx].TripCount;

    // Compute per-level naming suffix.
    unsigned Level  = P.Depth - 1 - Idx;
    unsigned DimIdx = Level; // innermost=0, outermost=Depth-1 (equals Level)
    if (TC)
      P.setDimTC(DimIdx, TC); // SE returns backedge-taken count (iterations-1).
    std::string LevelStr = std::to_string(Level);

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

    // Hoist InductionPhi declaration so it can be used for CanonIV creation.
    PHINode *InductionPhi = Info.IVs[Idx].IndVar;

    // Create the header and latch blocks for this loop level.
    auto *HeaderBB = P.createTPIRBasicBlock(L->getHeader());
    auto *LatchBB = P.createTPBasicBlock("tensor.latch" + LevelStr);

    // Insert canonical IV phi as the first recipe (VPlan-style).
    // Use a zero live-in as start matching the IV phi's type; step is patched below.
    TPValue *ZeroTP = P.getOrCreateLiveIn(
        ConstantInt::get(InductionPhi->getType(), 0));
    auto *CanonIV = new TPCanonicalIVRecipe(ZeroTP, ZeroTP /*placeholder step*/);
    HeaderBB->appendRecipe(CanonIV);

    // Process header PHIs into HeaderBB.
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

        TPWidenInductionRecipe *R;
        if (Phi.getType()->isPointerTy())
          R = new TPWidenPointerInductionRecipe(
              &Phi, StartTP,
              StartTP /* placeholder; patched after body */, DimIdx);
        else
          R = new TPWidenIntOrFpInductionRecipe(
              &Phi, StartTP,
              StartTP /* placeholder; patched after body */, DimIdx);
        HeaderBB->appendRecipe(R);
        P.ValueMap[PhiV] = R;
        // Note: Region->setIV(R) removed — IV tracking is via P.ValueMap
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
        HeaderBB->appendRecipe(R);
        P.ValueMap[&Phi] = R;
      }
    }

    // Emit non-PHI instructions from header into HeaderBB.
    EmitBlock(Header, HeaderBB);

    // Emit latch non-PHI instructions to LatchBB.
    // Also record the IR latch BB so that TPBasicBlock::execute() can reposition
    // the builder there, preventing dominance violations when latch recipes
    // (e.g. reduction-accumulator stores) reference PHIs defined in the latch.
    if (BasicBlock *Latch = L->getLoopLatch()) {
      EmitBlock(Latch, LatchBB);
      LatchBB->setInsertionBB(Latch);
    }

    if (Idx + 1 < AllLoops.size()) {
      // Non-innermost: create inner preheader + recurse.
      unsigned ChildLevel = Level - 1;
      std::string ChildStr = std::to_string(ChildLevel);
      auto *InnerPH = P.createTPBasicBlock("tensor.ph" + ChildStr);

      // Emit outer body blocks BEFORE recursing into the inner region so that
      // any values computed between the outer loop header and the inner loop
      // preheader (e.g. A's row GEP in ggml-style GEMM: %a.row = gep %A,
      // %i*K, in an i.body block between i.loop header and j.loop) are in
      // ValueMap when BuildRegion processes inner-loop GEP recipes.
      // Without this ordering those GEPs fall back to ir<> live-ins (DimSet={})
      // and BFS never propagates the outer-dim bit.
      //
      // Pin InnerPH to the outer loop header so that its clones are inserted
      // at a block that dominates all inner content.  This is required because
      // InnerPH is visited in intraRegionOrder AFTER tensor.latch (which
      // repositions the builder to the actual IR latch); without an explicit
      // InsertionBB the clones would land in the latch block, which does not
      // dominate the inner loop body, causing verifier failures.
      InnerPH->setInsertionBB(L->getHeader());
      Loop *InnerLoop = AllLoops[Idx + 1];
      for (BasicBlock *BB : L->blocks()) {
        if (BB == L->getHeader() || BB == L->getLoopLatch()) continue;
        if (InnerLoop->contains(BB)) continue;
        EmitBlock(BB, InnerPH);
      }

      auto *Child = BuildRegion(Idx + 1);
      auto *MiddleBB = P.createTPBasicBlock("middle.block" + ChildStr);
      BasicBlock *ExitBB = AllLoops[Idx + 1]->getExitBlock();
      auto *CleanupBB = ExitBB ? P.createTPIRBasicBlock(ExitBB) : nullptr;
      auto *ScalarPH = P.createTPBasicBlock("scalar.ph" + ChildStr);

      // Wire intra-region CFG.
      // Note: setParent() calls happen after wiring so connectBlocks asserts
      // parent == parent (nullptr == nullptr at this point) pass correctly.
      TPBlockUtils::connectBlocks(HeaderBB, LatchBB);
      TPBlockUtils::connectBlocks(HeaderBB, InnerPH);
      TPBlockUtils::connectBlocks(InnerPH, Child);
      TPBlockUtils::connectBlocks(Child, MiddleBB);
      if (CleanupBB) {
        TPBlockUtils::connectBlocks(MiddleBB, CleanupBB);
        TPBlockUtils::connectBlocks(MiddleBB, ScalarPH);
        TPBlockUtils::connectBlocks(CleanupBB, ScalarPH);
      } else {
        TPBlockUtils::connectBlocks(MiddleBB, ScalarPH);
      }
      TPBlockUtils::connectBlocks(ScalarPH, LatchBB);

      // Carve out region: setEntry/setExiting set parent on header/latch.
      auto *Region = P.createTPRegionBlock("tensor loop" + LevelStr);
      Region->setEntry(HeaderBB);   // sets HeaderBB->Parent = Region
      Region->setExiting(LatchBB);  // sets LatchBB->Parent = Region
      InnerPH->setParent(Region);
      Child->setParent(Region);
      MiddleBB->setParent(Region);
      if (CleanupBB) CleanupBB->setParent(Region);
      ScalarPH->setParent(Region);

      // Append canonical IV companion recipes to LatchBB.
      if (BoundTP) {
        auto *IncrRecipe = new TPCanonicalIVIncrRecipe(CanonIV, P.DimPFs[DimIdx].get());
        LatchBB->appendRecipe(IncrRecipe);
        CanonIV->setOperand(1, IncrRecipe);
        IncrRecipe->addUser(CanonIV);
        auto *CmpRecipe = new TPCanonicalIVExitCmpRecipe(IncrRecipe, BoundTP);
        LatchBB->appendRecipe(CmpRecipe);
      }

      // Patch widen-induction step operand now that body is fully built.
      BasicBlock *Latch = L->getLoopLatch();
      if (Latch && InductionPhi) {
        Value *StepVal = nullptr;
        for (unsigned I = 0, E = InductionPhi->getNumIncomingValues(); I < E; ++I)
          if (InductionPhi->getIncomingBlock(I) == Latch)
            StepVal = InductionPhi->getIncomingValue(I);

        if (StepVal) {
          for (TPRecipeBase &R : *HeaderBB) {
            if (auto *WI = dyn_cast<TPWidenInductionRecipe>(&R)) {
              if (WI->getIVPhi() == InductionPhi) {
                TPValue *StepTP = P.getTPValue(StepVal);
                WI->setOperand(1, StepTP);
                StepTP->addUser(WI);
                break;
              }
            }
          }
        }
      }

      // Populate named structural fields.
      Region->setHeaderForLoop(L, HeaderBB);
      Region->setLatchForLoop(L, LatchBB);
      // setMiddle/setScalar: for Idx>0 (intermediate levels) these are the correct
      // final values. For Idx==0 (outermost), the top-level wiring block will
      // overwrite these with the outermost MiddleBB/ScalarPH (safe — top-level
      // runs after this return).
      Region->setMiddle(MiddleBB);
      Region->setScalar(ScalarPH);
      Region->setInner(Child);

      // Register region — innermost pushes first (recursion unwinds inner → outer).
      P.Regions.push_back(Region);
      P.LoopIdx2TPRB[L] = Region;

      return Region;
    }

    // Innermost: create body block.
    auto *BodyBB = P.createTPBasicBlock("tensor.body.0");

    // Emit body blocks (not header, not latch) to BodyBB.
    for (BasicBlock *BB : L->blocks()) {
      if (BB == L->getHeader() || BB == L->getLoopLatch()) continue;
      EmitBlock(BB, BodyBB);
    }

    // Wire intra-region CFG.
    TPBlockUtils::connectBlocks(HeaderBB, LatchBB);
    TPBlockUtils::connectBlocks(HeaderBB, BodyBB);

    // Carve out region.
    auto *Region = P.createTPRegionBlock("tensor loop" + LevelStr);
    Region->setEntry(HeaderBB);
    Region->setExiting(LatchBB);
    BodyBB->setParent(Region);

    // Append canonical IV companion recipes to LatchBB.
    if (BoundTP) {
      auto *IncrRecipe = new TPCanonicalIVIncrRecipe(CanonIV, P.DimPFs[DimIdx].get());
      LatchBB->appendRecipe(IncrRecipe);
      CanonIV->setOperand(1, IncrRecipe);
      IncrRecipe->addUser(CanonIV);
      auto *CmpRecipe = new TPCanonicalIVExitCmpRecipe(IncrRecipe, BoundTP);
      LatchBB->appendRecipe(CmpRecipe);
    }

    // Patch widen-induction step operand now that body is fully built.
    BasicBlock *Latch = L->getLoopLatch();
    if (Latch && InductionPhi) {
      Value *StepVal = nullptr;
      for (unsigned I = 0, E = InductionPhi->getNumIncomingValues(); I < E; ++I)
        if (InductionPhi->getIncomingBlock(I) == Latch)
          StepVal = InductionPhi->getIncomingValue(I);

      if (StepVal) {
        for (TPRecipeBase &R : *HeaderBB) {
          if (auto *WI = dyn_cast<TPWidenInductionRecipe>(&R)) {
            if (WI->getIVPhi() == InductionPhi) {
              TPValue *StepTP = P.getTPValue(StepVal);
              WI->setOperand(1, StepTP);
              StepTP->addUser(WI);
              break;
            }
          }
        }
      }
    }

    // Populate named structural fields.
    Region->setHeaderForLoop(L, HeaderBB);
    Region->setLatchForLoop(L, LatchBB);
    // Middle, Scalar, Inner stay null for leaf region.

    P.Regions.push_back(Region);
    P.LoopIdx2TPRB[L] = Region;

    return Region;
  };

  if (!AllLoops.empty()) {
    unsigned OuterLevel = P.Depth - 1;
    std::string OuterStr = std::to_string(OuterLevel);

    auto *OuterPH  = P.createTPBasicBlock("tensor.ph" + OuterStr);
    auto *Outer    = BuildRegion(0);
    auto *MiddleBB = P.createTPBasicBlock("middle.block" + OuterStr);
    BasicBlock *ExitBB = AllLoops[0]->getExitBlock();
    auto *CleanupBB = ExitBB ? P.createTPIRBasicBlock(ExitBB) : nullptr;
    auto *ScalarPH  = P.createTPBasicBlock("scalar.ph" + OuterStr);

    // Top-level blocks all have parent = nullptr (top-level plan).
    TPBlockUtils::connectBlocks(OuterPH, Outer);
    TPBlockUtils::connectBlocks(Outer, MiddleBB);
    if (CleanupBB) {
      TPBlockUtils::connectBlocks(MiddleBB, CleanupBB);
      TPBlockUtils::connectBlocks(MiddleBB, ScalarPH);
      TPBlockUtils::connectBlocks(CleanupBB, ScalarPH);
    } else {
      TPBlockUtils::connectBlocks(MiddleBB, ScalarPH);
    }
    // ScalarPH: no successors at top level.

    P.setEntry(OuterPH);

    // Wire Middle/Scalar on the outermost region (created outside BuildRegion).
    // For depth-1 plans Outer is also the leaf region — Middle/Scalar must stay null
    // (leaf-region invariant). Only set them for multi-level nests.
    if (P.Depth > 1) {
      Outer->setMiddle(MiddleBB);
      Outer->setScalar(ScalarPH);
    }

    // tensor.preheader: reserved for SCEV expansions. Not connected to the CFG
    // in this commit — will be wired as predecessor of Entry in a future commit.
    // Do not use getPreheader() before it is wired.
    P.Preheader = P.createTPBasicBlock("tensor.preheader");
  }

  // Remap ReductionDims from outermost=0 (LoopNestAnalyzer convention) to
  // innermost=0 (TPlan DimIdx convention). Always test Info.ReductionDims —
  // never P.ReductionDims — to avoid reading back previously written bits.
  P.ReductionDims.resize(P.Depth);
  for (unsigned I = 0; I < P.Depth; ++I)
    if (I < Info.ReductionDims.size() && Info.ReductionDims.test(I))
      P.ReductionDims.set(P.Depth - 1 - I);
  return P;
}

//===----------------------------------------------------------------------===//
// TPlan::print
//===----------------------------------------------------------------------===//

void TPlan::print(raw_ostream &OS) const {
  OS << "TPlan '" << FuncName << "' (depth=" << Depth << ") {\n";

  // Pre-assign PF[d] slots in order (tp<%0>…tp<%D-1>) before any lazy
  // recipe-slot assignment, so they always have the lowest slot numbers.
  Tracker.reset();
  for (const auto &DP : DimPFs)
    Tracker.preAssignSynthetic(DP.get());

  // Print per-dim synthetic live-ins (PF[0], PF[1], …).
  for (const auto &DP : DimPFs) {
    OS << "Live-in ";
    DP->printAsOperand(OS, Tracker);
    OS << " = " << DP->getName() << "\n";
  }

  // Print IR-backed live-ins (unchanged, still ir<>).
  for (const auto &LI : LiveIns) {
    OS << "Live-in ";
    LI->printAsOperand(OS, Tracker);
    OS << "\n";
  }
  OS << "\n";

  // Walk the block CFG in DFS pre-order from the entry block.
  if (Entry) {
    for (TPBlockBase *B : constructionOrder(Entry))
      B->print(OS, "", Tracker);
  }

  OS << "}\n";
}

//===----------------------------------------------------------------------===//
// TPGuardBlock — execute(): emit runtime profitability guard + scalar clone
//===----------------------------------------------------------------------===//
void TPGuardBlock::execute(TPTransformState &State) {
  assert(State.LI && "TPGuardBlock::execute() requires LoopInfo in State");
  assert(State.DT && "TPGuardBlock::execute() requires DominatorTree in State");

  LoopInfo &LI = *State.LI;
  DominatorTree &DT = *State.DT;

  // ---- Precondition checks ------------------------------------------------

  BasicBlock *OrigPreheader = OutermostLoop->getLoopPreheader();
  assert(OrigPreheader && "OutermostLoop must have a unique preheader");

  assert(OutermostLoop->getExitBlock() &&
         "OutermostLoop must have a single exit block");

  BasicBlock *OrigPred = OrigPreheader->getSinglePredecessor();
  assert(OrigPred && "Loop preheader must have a single predecessor");

  // ---- Step 1: Clone the loop as the scalar fallback ----------------------
  //
  // cloneLoopWithPreheader() inserts the clone before OrigPreheader and wires
  // its exit edges to ExitBB. LI and DT are updated by the call.

  ValueToValueMapTy SkelVMap;
  SmallVector<BasicBlock *, 16> ClonedBlocks;
  [[maybe_unused]] Loop *ScalarLoop = cloneLoopWithPreheader(
      OrigPreheader, // Insert cloned blocks before this block.
      OrigPred,      // Dominator of the region being cloned into.
      OutermostLoop, SkelVMap, ".scalar", &LI, &DT, ClonedBlocks);
  assert(ScalarLoop && "cloneLoopWithPreheader() failed");

  // Remap all cloned instruction operands to point to cloned values.
  remapInstructionsInBlocks(ClonedBlocks, SkelVMap);

  // The clone's preheader is the VMap image of OrigPreheader.
  BasicBlock *ScalarPreheader = cast<BasicBlock>(SkelVMap[OrigPreheader]);

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
    Value *PFVal = ConstantInt::get(RuntimeTC->getType(), GuardPF);
    Value *Cond = GB.CreateICmpUGE(RuntimeTC, PFVal, "tensor.profitable");
    GB.CreateCondBr(Cond, OrigPreheader, ScalarPreheader);
  }

  // ---- Step 3: Update DominatorTree ---------------------------------------

  DT.addNewBlock(GuardBB, OrigPred);
  DT.changeImmediateDominator(OrigPreheader, GuardBB);
  DT.changeImmediateDominator(ScalarPreheader, GuardBB);

  LLVM_DEBUG(dbgs() << "TPGuardBlock: guard inserted\n"
                    << "  GuardBB:    " << GuardBB->getName() << "\n"
                    << "  TensorPH:   " << OrigPreheader->getName() << "\n"
                    << "  ScalarPH:   " << ScalarPreheader->getName() << "\n");

  // ---- Execute the tensor path (original TPlan subtree) -------------------

  TensorPath->execute(State);
}
void TPGuardBlock::print(raw_ostream &OS, const Twine &Indent,
                         TPSlotTracker &) const {
  OS << Indent << "TPGuardBlock (TC >= " << GuardPF << ")\n";
}

//===----------------------------------------------------------------------===//
// TPTilingRegion — implementations
//===----------------------------------------------------------------------===//
void TPTilingRegion::execute(TPTransformState &State) {
  IRBuilder<> &B = State.Builder;
  LLVMContext &Ctx = B.getContext();
  Function *F = B.GetInsertBlock()->getParent();

  // Current insert block is the K-loop body BB set by the parent TPRegionBlock.
  BasicBlock *KLoopBB = B.GetInsertBlock();

  // Find the non-self successor of the K-loop body (the BB after the loop).
  Instruction *OrigTerm = KLoopBB->getTerminator();
  BasicBlock *OrigSuccessor = nullptr;
  for (BasicBlock *Succ : successors(KLoopBB))
    if (Succ != KLoopBB) { OrigSuccessor = Succ; break; }

  // Remove K-loop self-edges from PHI nodes; the tiling loop drives iteration.
  for (PHINode &Phi : KLoopBB->phis()) {
    int SelfIdx = Phi.getBasicBlockIndex(KLoopBB);
    if (SelfIdx >= 0)
      Phi.removeIncomingValue(static_cast<unsigned>(SelfIdx),
                              /*DeletePHIIfEmpty=*/false);
  }

  // Erase the original K-loop terminator.
  OrigTerm->eraseFromParent();

  if (Mode == DimEmitMode::StaticTiled) {
    Value *TCVal = State.TilingTCVal;
    assert(TCVal && "TPTilingRegion: TilingTCVal not set for StaticTiled dim");

    // Use OrigKIVPhi's type for all arithmetic to remain type-agnostic.
    Type *IVTy = OrigKIVPhi->getType();

    // Normalize TCVal to IVTy (in case TilingTCVal was produced as i64).
    if (TCVal->getType() != IVTy)
      TCVal = B.CreateZExtOrTrunc(TCVal, IVTy, "tc.cast");

    Value *PFVal   = ConstantInt::get(IVTy, TilingPF);
    Value *Trips   = B.CreateUDiv(TCVal, PFVal, "tile.trips");
    Value *Limit   = B.CreateMul(Trips, PFVal, "tile.limit");

    BasicBlock *TileHeader = BasicBlock::Create(Ctx, "tile.header", F);
    BasicBlock *TileBody   = BasicBlock::Create(Ctx, "tile.body",   F);
    BasicBlock *TileLatch  = BasicBlock::Create(Ctx, "tile.latch",  F);
    BasicBlock *TileExit   = BasicBlock::Create(Ctx, "tile.exit",   F);

    // KLoopBB falls through to TileHeader.
    B.CreateBr(TileHeader);

    // TileHeader: PHI + bounds check.
    B.SetInsertPoint(TileHeader);
    PHINode *TileIV = B.CreatePHI(IVTy, 2, "tile.iv");
    TileIV->addIncoming(ConstantInt::get(IVTy, 0), KLoopBB);
    Value *Done = B.CreateICmpUGE(TileIV, Limit, "tile.done");
    B.CreateCondBr(Done, TileExit, TileBody);

    // Register TileIV so WIDEN-GEP recipes' remapClone() produces
    // tile-corner GEPs by substituting OrigKIVPhi → TileIV.
    State.EmittedMap[OrigKIVPhi] = TileIV;

    // TileBody: run all body recipes.
    // Iterate recipes directly instead of calling Body->execute(State):
    // Body is a TPIRBasicBlock whose execute() would reposition the builder
    // back to IRBB->getFirstNonPHIIt() (k.loop), undoing our SetInsertPoint.
    B.SetInsertPoint(TileBody);
    for (TPRecipeBase &R : *Body)
      R.execute(State);
    assert(B.GetInsertBlock() == TileBody &&
           "TPTilingRegion: body recipes left builder in unexpected block");
    B.CreateBr(TileLatch);

    // TileLatch: increment IV and loop back.
    B.SetInsertPoint(TileLatch);
    Value *TileNext = B.CreateAdd(TileIV, PFVal, "tile.next");
    TileIV->addIncoming(TileNext, TileLatch);
    B.CreateBr(TileHeader);

    // TileExit: continue to the original successor.
    // OrigSuccessor must be non-null: KLoopBB is a loop body block; its
    // non-self successor is always the outer latch/exit block.
    // PHI nodes in OrigSuccessor that list KLoopBB as predecessor must use
    // KLoopBB (not TileExit) because KLoopBB still directly precedes TileHeader,
    // which precedes TileExit — the CFG path from KLoopBB → TileHeader → TileExit
    // means KLoopBB dominates TileExit, so OrigSuccessor PHIs are still valid.
    assert(OrigSuccessor && "TPTilingRegion: KLoopBB must have a non-self successor");
    B.SetInsertPoint(TileExit);
    B.CreateBr(OrigSuccessor);

    // Clear the EmittedMap entry to avoid polluting other tiling regions.
    State.EmittedMap.erase(OrigKIVPhi);

    LLVM_DEBUG(dbgs() << "TPTilingRegion[static]: tile loop created, PF="
                      << TilingPF << "\n");
  } else {
    // ── Dynamic tiling path ─────────────────────────────────────────────────
    Value *TCVal = State.TilingTCVal;
    assert(TCVal && "TPTilingRegion: TilingTCVal not set for DynamicTiled dim");

    Type *IVTy = OrigKIVPhi->getType();
    Value *TCNorm = (TCVal->getType() != IVTy)
                       ? B.CreateZExtOrTrunc(TCVal, IVTy, "tc.norm")
                       : TCVal;
    Value *PFVal  = ConstantInt::get(IVTy, TilingPF);
    Value *Trips  = B.CreateUDiv(TCNorm, PFVal, "tensor.body.trips");
    Value *Limit  = B.CreateMul(Trips, PFVal, "tensor.body.limit");
    Value *Guard  = B.CreateICmpUGE(TCNorm, PFVal, "tensor.body.guard");

    BasicBlock *TBHeader = BasicBlock::Create(Ctx, "tensor.body.header", F);
    BasicBlock *TBBody   = BasicBlock::Create(Ctx, "tensor.body.body",   F);
    BasicBlock *TBLatch  = BasicBlock::Create(Ctx, "tensor.body.latch",  F);
    BasicBlock *TBExit   = BasicBlock::Create(Ctx, "tensor.body.exit",   F);

    // KLoopBB → (tensor.body.header if TC >= PF, else tensor.body.exit).
    B.CreateCondBr(Guard, TBHeader, TBExit);

    // tensor.body.header: IV + bounds check.
    B.SetInsertPoint(TBHeader);
    PHINode *TBIV = B.CreatePHI(IVTy, 2, "tensor.body.iv");
    TBIV->addIncoming(ConstantInt::get(IVTy, 0), KLoopBB);
    Value *TBDone = B.CreateICmpUGE(TBIV, Limit, "tensor.body.done");
    B.CreateCondBr(TBDone, TBExit, TBBody);

    // Register TBIV in EmittedMap so WIDEN-GEP recipes get tile-corner GEPs.
    State.EmittedMap[OrigKIVPhi] = TBIV;

    // tensor.body.body: run body recipes.
    B.SetInsertPoint(TBBody);
    Body->execute(State);
    assert(B.GetInsertBlock() == TBBody &&
           "TPTilingRegion: Body::execute() left builder in unexpected block");
    B.CreateBr(TBLatch);

    // tensor.body.latch: IV += PF.
    B.SetInsertPoint(TBLatch);
    Value *TBNext = B.CreateAdd(TBIV, PFVal, "tensor.body.next");
    TBIV->addIncoming(TBNext, TBLatch);
    B.CreateBr(TBHeader);

    // tensor.body.exit: ExitIV = PHI(0 from KLoopBB, TBIV from TBHeader).
    // (If guard was false, ExitIV = 0; otherwise ExitIV = Limit.)
    B.SetInsertPoint(TBExit);
    PHINode *ExitIV = B.CreatePHI(IVTy, 2, "tensor.body.exit.iv");
    ExitIV->addIncoming(ConstantInt::get(IVTy, 0), KLoopBB);
    ExitIV->addIncoming(TBIV, TBHeader);

    // scalar.block: iterate [ExitIV, TC) one element at a time.
    Value *ScRem  = B.CreateSub(TCNorm, ExitIV, "scalar.rem");
    Value *HasSc  = B.CreateICmpUGT(ScRem, ConstantInt::get(IVTy, 0),
                                    "scalar.guard");

    BasicBlock *ScPHBB  = TBExit;
    BasicBlock *ScBody  = BasicBlock::Create(Ctx, "scalar.block",      F);
    BasicBlock *ScExit  = BasicBlock::Create(Ctx, "scalar.block.exit", F);
    B.CreateCondBr(HasSc, ScBody, ScExit);

    // scalar.block body: ScIV iterates one element at a time.
    B.SetInsertPoint(ScBody);
    PHINode *ScIV = B.CreatePHI(IVTy, 2, "scalar.iv");
    ScIV->addIncoming(ExitIV, ScPHBB);

    // Register ScIV so ScalarEpilogue WIDEN-GEP emits scalar GEPs.
    State.EmittedMap[OrigKIVPhi] = ScIV;
    if (ScalarEpilogue)
      ScalarEpilogue->execute(State);
    // ScalarEpilogue::execute() may reposition the builder; restore to ScBody
    // before emitting the latch increment and back-branch.
    B.SetInsertPoint(ScBody);

    Value *ScNext = B.CreateAdd(ScIV, ConstantInt::get(IVTy, 1), "scalar.next");
    ScIV->addIncoming(ScNext, ScBody);
    Value *ScDone = B.CreateICmpUGE(ScNext, TCNorm, "scalar.done");
    B.CreateCondBr(ScDone, ScExit, ScBody);

    // scalar.block.exit: continue to original successor.
    assert(OrigSuccessor &&
           "TPTilingRegion: KLoopBB must have a non-self successor");
    B.SetInsertPoint(ScExit);
    B.CreateBr(OrigSuccessor);

    // Clear EmittedMap.
    State.EmittedMap.erase(OrigKIVPhi);

    LLVM_DEBUG(dbgs() << "TPTilingRegion[dynamic]: tensor.body + scalar.block created, PF="
                      << TilingPF << "\n");
  }
}
void TPTilingRegion::print(raw_ostream &OS, const Twine &Indent,
                           TPSlotTracker &) const {
  StringRef ModeStr = Mode == DimEmitMode::StaticTiled  ? "StaticTiled"
                    : Mode == DimEmitMode::DynamicTiled ? "DynamicTiled"
                                                        : "Inline";
  OS << Indent << "TPTilingRegion dim=" << TilingDim << " PF=" << TilingPF
     << " mode=" << ModeStr << "\n";
}
