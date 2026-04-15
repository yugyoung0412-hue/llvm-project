#include "llvm/Transforms/Tensorize/TPlan.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/GenericDomTreeConstruction.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Tensorize/TPlanCFG.h"
#include "llvm/Transforms/Tensorize/TPlanner.h"
#include "llvm/Transforms/Tensorize/TPlanDominatorTree.h"
#include "llvm/Transforms/Tensorize/TPlanPatternMatch.h"
#include "llvm/Transforms/Tensorize/TensorizeCommon.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopVersioning.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include <cassert>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;
using namespace llvm::TPlanPatternMatch;

#define DEBUG_TYPE "tplan"

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
raw_ostream &llvm::operator<<(raw_ostream &OS, const TPValue &V) {
  const TPInstruction *Instr = dyn_cast<TPInstruction>(&V);
  TPSlotTracker SlotTracker(
      (Instr && Instr->getParent()) ? Instr->getParent()->getPlan() : nullptr);
  V.print(OS, SlotTracker);
  return OS;
}
#endif

const SCEV *TPSingleDefRecipe::getMemStride(unsigned Dim, const TPlan &Plan,
                                             ScalarEvolution &SE) const {
  auto It = MemStrides.find(Dim);
  if (It != MemStrides.end())
    return It->second;
  return SE.getConstant(APInt(64, Plan.getDenseStrideForDim(Dim)));
}

const SCEV *TPWidenMemoryRecipe::getMemStride(unsigned Dim, const TPlan &Plan,
                                              ScalarEvolution &SE) const {
  auto It = MemStrides.find(Dim);
  if (It != MemStrides.end())
    return It->second;
  return SE.getConstant(APInt(64, Plan.getDenseStrideForDim(Dim)));
}

Value *TPLane::getAsRuntimeExpr(IRBuilderBase &Builder,
                                const ElementCount &VF) const {
  // TODO(yuxin.an)
  llvm_unreachable("");
}

TPValue::TPValue(const unsigned char SC, Value *UV, TPDef *Def)
    : SubclassID(SC), UnderlyingVal(UV), Def(Def) {
  if (Def)
    Def->addDefinedValue(this);
}

TPValue::~TPValue() {
  // TODO(maxim.o): Reenable for generic TPlan.
  // assert(Users.empty() && "trying to delete a VPValue with remaining users");
  if (Def)
    Def->removeDefinedValue(this);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void TPValue::print(raw_ostream &OS, TPSlotTracker &SlotTracker) const {
  if (const TPRecipeBase *R = dyn_cast_or_null<TPRecipeBase>(Def))
    R->print(OS, "", SlotTracker);
  else
    printAsOperand(OS, SlotTracker);
}

void TPValue::dump() const {
  const TPRecipeBase *Instr = dyn_cast_or_null<TPRecipeBase>(this->Def);
  TPSlotTracker SlotTracker(
      (Instr && Instr->getParent()) ? Instr->getParent()->getPlan() : nullptr);
  print(dbgs(), SlotTracker);
  dbgs() << "\n";
}

void TPDef::dump() const {
  const TPRecipeBase *Instr = dyn_cast_or_null<TPRecipeBase>(this);
  TPSlotTracker SlotTracker(
      (Instr && Instr->getParent()) ? Instr->getParent()->getPlan() : nullptr);
  print(dbgs(), "", SlotTracker);
  dbgs() << "\n";
}
#endif

TPRecipeBase *TPValue::getDefiningRecipe() {
  return cast_or_null<TPRecipeBase>(Def);
}

const TPRecipeBase *TPValue::getDefiningRecipe() const {
  return cast_or_null<TPRecipeBase>(Def);
}

// Get the top-most entry block of \p Start. This is the entry block of the
// containing VPlan. This function is templated to support both const and
// non-const blocks
template <typename T> static T *getPlanEntry(T *Start) {
  T *Next = Start;
  T *Current = Start;
  while ((Next = Next->getParent()))
    Current = Next;

  SmallSetVector<T *, 8> WorkList;
  WorkList.insert(Current);

  for (unsigned i = 0; i < WorkList.size(); i++) {
    T *Current = WorkList[i];
    if (Current->getNumPredecessors() == 0)
      return Current;
    auto &Predecessors = Current->getPredecessors();
    WorkList.insert(Predecessors.begin(), Predecessors.end());
  }

  llvm_unreachable("VPlan without any entry node without predecessors");
}

TPlan *TPBlockBase::getPlan() { return getPlanEntry(this)->Plan; }

const TPlan *TPBlockBase::getPlan() const { return getPlanEntry(this)->Plan; }

/// \return the VPBasicBlock that is the entry of Block, possibly indirectly.
const TPBasicBlock *TPBlockBase::getEntryBasicBlock() const {
  const TPBlockBase *Block = this;
  while (const TPRegionBlock *Region = dyn_cast<TPRegionBlock>(Block))
    Block = Region->getEntry();
  return cast<TPBasicBlock>(Block);
}

TPBasicBlock *TPBlockBase::getEntryBasicBlock() {
  TPBlockBase *Block = this;
  while (TPRegionBlock *Region = dyn_cast<TPRegionBlock>(Block))
    Block = Region->getEntry();
  return cast<TPBasicBlock>(Block);
}

void TPBlockBase::setPlan(TPlan *ParentPlan) {
  assert(
      (ParentPlan->getEntry() == this || ParentPlan->getPreheader() == this) &&
      "Can only set plan on its entry or preheader block.");
  Plan = ParentPlan;
}

/// \return the VPBasicBlock that is the exit of Block, possibly indirectly.
const TPBasicBlock *TPBlockBase::getExitingBasicBlock() const {
  const TPBlockBase *Block = this;
  while (const TPRegionBlock *Region = dyn_cast<TPRegionBlock>(Block))
    Block = Region->getExiting();
  return cast<TPBasicBlock>(Block);
}

TPBasicBlock *TPBlockBase::getExitingBasicBlock() {
  TPBlockBase *Block = this;
  while (TPRegionBlock *Region = dyn_cast<TPRegionBlock>(Block))
    Block = Region->getExiting();
  return cast<TPBasicBlock>(Block);
}

TPBlockBase *TPBlockBase::getEnclosingBlockWithSuccessors() {
  if (!Successors.empty() || !Parent)
    return this;
  assert(Parent->getExiting() == this &&
         "Block w/o successors not the exiting block of its parent.");
  return Parent->getEnclosingBlockWithSuccessors();
}

TPBlockBase *TPBlockBase::getEnclosingBlockWithPredecessors() {
  if (!Predecessors.empty() || !Parent)
    return this;
  assert(Parent->getEntry() == this &&
         "Block w/o predecessors not the entry of its parent.");
  return Parent->getEnclosingBlockWithPredecessors();
}

void TPBlockBase::deleteCFG(TPBlockBase *Entry) {
  for (TPBlockBase *Block : to_vector(tp_depth_first_shallow(Entry)))
    delete Block;
}

TPBasicBlock::iterator TPBasicBlock::getFirstNonPhi() {
  iterator It = begin();
  while (It != end() && It->isPhi())
    It++;
  return It;
}

TPValue *tputils::getOrCreateTPValueForSCEVExpr(TPlan &Plan, const SCEV *Expr,
                                                ScalarEvolution &SE) {
  if (auto *Expanded = Plan.getSCEVExpansion(Expr))
    return Expanded;
  TPValue *Expanded = nullptr;
  // YYG::REMOVE
  errs() << "[getOrCreateTPValueForSCEVExpr] Expr: " << *Expr << "\n";
  if (auto *E = dyn_cast<SCEVConstant>(Expr)) {
    // YYG::REMOVE
    errs() << "SCEVConst\n";
    Expanded = Plan.getOrAddLiveIn(E->getValue());
  }
  else if (auto *E = dyn_cast<SCEVUnknown>(Expr)) {
    // YYG::REMOVE
    errs() << "SCEVUnknown\n";
    Expanded = Plan.getOrAddLiveIn(E->getValue());
  } else {
    // YYG::REMOVE
    errs() << "TPExpandSCEVRecipe\n";
    Expanded = new TPExpandSCEVRecipe(Expr, SE);
    Plan.getPreheader()->appendRecipe(Expanded->getDefiningRecipe());
  }
  Plan.addSCEVExpansion(Expr, Expanded);
  return Expanded;
}

TPTransformState::TPTransformState(TFTy TF, TUFTy UF, LoopInfo *LI,
                                   DominatorTree *DT, IRBuilderBase &Builder,
                                   LoopTensorizer *LT, TPlan *Plan,
                                   LLVMContext &Ctx)
    : TF(TF), UF(UF), CFG(DT), LI(LI), DT(DT), Builder(Builder), LT(LT),
      Plan(Plan), LVer(nullptr),
      TypeAnalysis(/* TODO(maxim.o): fix */ nullptr, Ctx) {}
//      TypeAnalysis(Plan->getCanonicalIV()->getScalarType, Ctx) {}

Value *TPTransformState::get(TPValue *Def, const TPIteration &Instance,
                             Loop *L) {
  if (Def->isLiveIn())
    return Def->getLiveInIRValue();
  if (hasScalarValue(Def, Instance, L)) {
    return Data.PerPartScalars[Def][Instance.Part]
                              [Instance.Lane.mapToCacheIndex(TF[L])];
  }
  if (!Instance.Lane.isFirstLane() &&
      tputils::isUniformAfterTensorization(Def) &&
      hasScalarValue(Def, {Instance.Part, TPLane::getFirstLane()}, L)) {
    return Data.PerPartScalars[Def][Instance.Part][0];
  }
  llvm_unreachable("");
}

DenseMap<Loop *, Value *> TPTransformState::get(DenseMap<Loop *, TPValue *> Def,
                                                const TPIteration &Instance) {
  DenseMap<Loop *, Value *> Res;
  for (auto DefElem : Def) {
    if (DefElem.second->isLiveIn())
      Res.insert({DefElem.first, DefElem.second->getLiveInIRValue()});
    else
      llvm_unreachable("");
  }
  return Res;
}

Value *TPTransformState::get(TPValue *Def, unsigned Part, bool NeedsScalar) {
  llvm_unreachable("");
}

BasicBlock *TPTransformState::CFGState::getPreheaderBBFor(TPRecipeBase *R) {
  TPRegionBlock *LoopRegion = R->getParent()->getEnclosingLoopRegion();
  return TPBB2IRBB[LoopRegion->getPreheaderTPBB()];
}

void TPTransformState::addNewMetadata(Instruction *To,
                                      const Instruction *Orig) {
  // If the loop was versioned with memchecks, add the corresponding no-alias
  // metadata.
  if (LVer && (isa<LoadInst>(Orig) || isa<StoreInst>(Orig)))
    LVer->annotateInstWithNoAlias(To, Orig);
}

void TPTransformState::addMetadata(Value *To, Instruction *From) {
  // No source instruction to transfer metadata from?
  if (!From)
    return;

  if (Instruction *ToI = dyn_cast<Instruction>(To)) {
    propagateMetadata(ToI, From);
    addNewMetadata(ToI, From);
  }
}

void TPTransformState::setDebugLocFrom(DebugLoc DL) { llvm_unreachable(""); }

void TPTransformState::packScalarIntoTensorValue(TPValue *Def,
                                                 const TPIteration &Instance) {
  llvm_unreachable("");
}

BasicBlock *TPBasicBlock::createEmptyBasicBlock(TPTransformState::CFGState &CFG,
                                                TPTransformState *State) {
  llvm_unreachable("");
}

// ---------------------------------------------------------------------
// execute : 실제 벡터화 로직을 넣는다.
// 현재는 최소 동작만 하는 스텁(stub) 형태로 구현한다.
void TPWidenPointerInductionRecipe::execute(TPTransformState &State) {
  // 여기서는 간단히 디버그 메시지만 출력한다.
  LLVM_DEBUG(dbgs() << "TPWidenPointerInductionRecipe::execute called\n");
  // 실제 구현은 나중에 추가한다.
}

// ---------------------------------------------------------------------
// onlyScalarsGenerated : 스칼라만 생성되는 경우를 알려준다.
bool TPWidenPointerInductionRecipe::onlyScalarsGenerated(bool IsScalable) {
  // 클래스 생성자에서 넘겨 받은 플래그와 전달받은 인자를 조합한다.
  return IsScalarAfterVectorization || !IsScalable;
}

// ---------------------------------------------------------------------
// print : 디버그용 출력 (디버그 빌드에서만 컴파일됨)
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void TPWidenPointerInductionRecipe::print(raw_ostream &O,
                                          const Twine &Indent,
                                          TPSlotTracker &SlotTracker) const {
  llvm_unreachable("");
}
#endif

void TPIRBasicBlock::execute(TPTransformState *State) {}

void TPBasicBlock::execute(TPTransformState *State) {
  assert(State->TPBB2BB.count(this) && "TPBasicBlock::execute error");
  State->CurBB = State->TPBB2BB[this];
  executeRecipes(State, State->CurBB);
}

void TPBasicBlock::dropAllReferences(TPValue *NewValue) {
  for (TPRecipeBase &R : Recipes) {
    for (auto *Def : R.definedValues())
      Def->replaceAllUsesWith(NewValue);

    for (unsigned I = 0, E = R.getNumOperands(); I != E; I++)
      R.setOperand(I, NewValue);
  }
}

void TPBasicBlock::executeRecipes(TPTransformState *State, BasicBlock *BB) {
  LLVM_DEBUG(dbgs() << "LT: tensorizing TPBB:" << getName()
                    << " in BB:" << BB->getName() << '\n');

  State->CFG.TPBB2IRBB[this] = BB;
  State->CFG.PrevTPBB = this;

  for (TPRecipeBase &Recipe : Recipes)
    Recipe.execute(*State);

  LLVM_DEBUG(dbgs() << "LT: filled BB:" << *BB);
}

TPBasicBlock *TPBasicBlock::splitAt(iterator SplitAt) {
  // TODO(yuxin.an)
  llvm_unreachable("");
}

TPRegionBlock *TPBasicBlock::getEnclosingLoopRegion() {
  TPRegionBlock *P = getParent();
  if (P && P->isReplicator()) {
    P = P->getParent();
    assert(!cast<TPRegionBlock>(P)->isReplicator() &&
           "unexpected nested replicate regions");
  }
  return P;
}

static bool hasConditionalTerminator(const TPBasicBlock *TPBB) {
  if (TPBB->empty()) {
    assert(
        TPBB->getNumSuccessors() < 2 &&
        "block with multiple successors doesn't have a recipe as terminator");
    return false;
  }

  const TPRecipeBase *R = &TPBB->back();
  bool IsCondBranch = isa<TPBranchOnMaskRecipe>(R) ||
                      match(R, m_BranchOnCond(m_TPValue())) ||
                      match(R, m_BranchOnCount(m_TPValue(), m_TPValue()));
  (void)IsCondBranch;

  if (TPBB->getNumSuccessors() >= 2 ||
      (TPBB->isExiting() && !TPBB->getParent()->isReplicator())) {
    assert(IsCondBranch && "block with multiple successors not terminated by "
                           "conditional branch recipe");

    return true;
  }

  assert(
      !IsCondBranch &&
      "block with 0 or 1 successors terminated by conditional branch recipe");
  return false;
}

TPRecipeBase *TPBasicBlock::getTerminator() {
  if (hasConditionalTerminator(this))
    return &back();
  return nullptr;
}

const TPRecipeBase *TPBasicBlock::getTerminator() const {
  if (hasConditionalTerminator(this))
    return &back();
  return nullptr;
}

bool TPBasicBlock::isExiting() const {
  return getParent() && getParent()->getExitingBasicBlock() == this;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void TPBlockBase::printSuccessors(raw_ostream &O, const Twine &Indent) const {
  if (getSuccessors().empty()) {
    O << Indent << "No successors\n";
  } else {
    O << Indent << "Successor(s): ";
    ListSeparator LS;
    for (auto *Succ : getSuccessors())
      O << LS << Succ->getName();
    O << '\n';
  }
}

static StringRef kindToStr(const TensorOpKind K) {
  switch (K) {
    case TensorOpKind::Scalar:          return "Scalar";
    case TensorOpKind::ElementWise:     return "ElementWise";
    case TensorOpKind::BroadcastBinary: return "BroadcastBinary";
    case TensorOpKind::OuterProduct:    return "OuterProduct";
    case TensorOpKind::Contraction:     return "Contraction";
    case TensorOpKind::PlainReduction:  return "PlainReduction";
  }
}

static void printTensorOpKindAnnotation(raw_ostream &OS, const TPRecipeBase &R) {
  const TensorOpKind TK = R.getTensorOpKind();
  if (TK != TensorOpKind::Scalar)
    OS << "; [TensorOpKind::" << kindToStr(TK) << "];";
}

/// Append DimSet annotation for a TPSingleDefRecipe (only when non-empty).
/// Shown at Stage 2+ after TPlanWidener_widen() has populated DimSet fields.
static void printDimSetAnnotation(raw_ostream &OS, const TPRecipeBase &R) {
  // const auto *SDR = dyn_cast<TPSingleDefRecipe>(&R);
  // if (!SDR || SDR->DimSet.none())
  if (R.DimSet.none())
    return;
  OS << "; DimSet={";
  bool First = true;
  for (int D = R.DimSet.find_first(); D >= 0; D = R.DimSet.find_next(D)) {
    if (!First)
      OS << ",";
    OS << D;
    First = false;
  }
  OS << "}";
}

void TPBasicBlock::print(raw_ostream &O, const Twine &Indent,
                         TPSlotTracker &SlotTracker) const {
  O << Indent << getName() << ":\n";

  auto RecipeIndent = Indent + "  ";
  for (const TPRecipeBase &Recipe : *this) {
    Recipe.print(O, RecipeIndent, SlotTracker);
    printDimSetAnnotation(O, Recipe);
    printTensorOpKindAnnotation(O, Recipe);
    O << " [PF= " << Recipe.getPF() << "];";
    O << '\n';
  }

  printSuccessors(O, Indent);
}
#endif

static std::pair<TPBlockBase *, TPBlockBase *>
cloneFrom(TPBlockBase *Entry,
          DenseMap<TPBlockBase *, TPBlockBase *> &Old2NewTPBlocks);

// Clone the CFG for all nodes reachable from \p Entry, this includes cloning
// the blocks and their recipes. Operands of cloned recipes will NOT be updated.
// Remapping of operands must be done separately. Returns a pair with the new
// entry and exiting blocks of the cloned region. If \p Entry isn't part of a
// region, return nullptr for the exiting block.
static std::pair<TPBlockBase *, TPBlockBase *>
cloneFrom(TPBlockBase *Entry,
          DenseMap<TPBlockBase *, TPBlockBase *> &Old2NewTPBlocks) {
  TPBlockBase *Exiting = nullptr;
  bool InRegion = Entry->getParent();
  // First, clone blocks reachable from Entry.
  for (TPBlockBase *BB : tp_depth_first_shallow(Entry)) {
    TPBlockBase *NewBB = BB->clone();
    Old2NewTPBlocks[BB] = NewBB;
    if (InRegion && BB->getNumSuccessors() == 0) {
      assert(!Exiting && "Multiple exiting blocks?");
      Exiting = BB;
    }
  }
  assert((!InRegion || Exiting) && "regions must have a single exiting block");

  // Second, update the predecessors & successors of the cloned blocks.
  for (TPBlockBase *BB : tp_depth_first_shallow(Entry)) {
    TPBlockBase *NewBB = Old2NewTPBlocks[BB];
    SmallVector<TPBlockBase *> NewPreds;
    for (TPBlockBase *Pred : BB->getPredecessors()) {
      NewPreds.push_back(Old2NewTPBlocks[Pred]);
    }
    NewBB->setPredecessors(NewPreds);
    SmallVector<TPBlockBase *> NewSuccs;
    for (TPBlockBase *Succ : BB->successors()) {
      NewSuccs.push_back(Old2NewTPBlocks[Succ]);
    }
    NewBB->setSuccessors(NewSuccs);
  }

#if !defined(NDEBUG)
  // Verify that the order of predecessors and successors matches in the cloned
  // version.
  for (const auto &[OldBB, NewBB] :
       zip(tp_depth_first_shallow(Entry),
           tp_depth_first_shallow(Old2NewTPBlocks[Entry]))) {
    for (const auto &[OldPred, NewPred] :
         zip(OldBB->getPredecessors(), NewBB->getPredecessors()))
      assert(NewPred == Old2NewTPBlocks[OldPred] && "Different predecessors");

    for (const auto &[OldSucc, NewSucc] :
         zip(OldBB->successors(), NewBB->successors()))
      assert(NewSucc == Old2NewTPBlocks[OldSucc] && "Different successors");
  }
#endif

  return std::make_pair(Old2NewTPBlocks[Entry],
                        Exiting ? Old2NewTPBlocks[Exiting] : nullptr);
}

TPRegionBlock::TPRegionBlock(TPBlockBase *Entry, TPBlockBase *Exiting,
                             DenseMap<Loop *, TPBlockBase *> Loop2HeaderTPB,
                             DenseMap<Loop *, TPBlockBase *> Loop2LatchTPB,
                             const std::string &Name, bool IsReplicator)
    : TPBlockBase(TPRegionBlockSC, Name), Entry(Entry), Exiting(Exiting),
      Loop2HeaderTPB(Loop2HeaderTPB), Loop2LatchTPB(Loop2LatchTPB),
      IsReplicator(IsReplicator) {
  assert(Entry->getPredecessors().empty() && "Entry block has predecessors.");
  assert(Exiting->getSuccessors().empty() && "Exit block has successors.");
  for (auto Elem : Loop2HeaderTPB)
    HeaderTPB2Loop.insert({Elem.second, Elem.first});
  for (auto Elem : Loop2LatchTPB)
    LatchTPB2Loop.insert({Elem.second, Elem.first});

  for (TPBlockBase *Block : tp_depth_first_shallow(Entry))
    Block->setParent(this);
}

TPRegionBlock *TPRegionBlock::clone() {
  DenseMap<TPBlockBase *, TPBlockBase *> Old2NewTPBlocks;
  const auto &[NewEntry, NewExiting] = cloneFrom(getEntry(), Old2NewTPBlocks);

  DenseMap<Loop *, TPBlockBase *> NewLoop2HeaderTPB, NewLoop2LatchTPB;

  for (auto Elem : Loop2HeaderTPB)
    NewLoop2HeaderTPB.insert({Elem.first, Old2NewTPBlocks[Elem.second]});
  for (auto Elem : Loop2LatchTPB)
    NewLoop2LatchTPB.insert({Elem.first, Old2NewTPBlocks[Elem.second]});

  auto *NewRegion =
      new TPRegionBlock(NewEntry, NewExiting, NewLoop2HeaderTPB,
                        NewLoop2LatchTPB, getName(), isReplicator());
  for (TPBlockBase *Block : tp_depth_first_shallow(NewEntry))
    Block->setParent(NewRegion);
  return NewRegion;
}

void TPRegionBlock::dropAllReferences(TPValue *NewValue) {
  for (TPBlockBase *Block : tp_depth_first_shallow(Entry))
    // Drop all references in VPBasicBlocks and replace all uses with
    // DummyValue.
    Block->dropAllReferences(NewValue);
}

void TPRegionBlock::execute(TPTransformState *State) {
  State->TPBB2BB.insert({cast<TPBasicBlock>(Entry), State->TBS.TEntry});
  State->BB2TPBB.insert({State->TBS.TEntry, cast<TPBasicBlock>(Entry)});

  State->TPBB2BB.insert({cast<TPBasicBlock>(Exiting), State->TBS.TExiting});
  State->BB2TPBB.insert({State->TBS.TExiting, cast<TPBasicBlock>(Exiting)});

  ReversePostOrderTraversal<TPBlockShallowTraversalWrapper<TPBlockBase *>> RPOT(
      Entry);

  // TODO(yuxin.an)
  for (TPBlockBase *Block : RPOT) {
    LLVM_DEBUG(dbgs() << "LT: TPBlock in RPO " << Block->getName() << '\n');
    Block->execute(State);
  }
}

InstructionCost TPBasicBlock::cost(ElementCount VF, TPCostContext &Ctx) {
  InstructionCost Cost = 0;
  for (TPRecipeBase &R : Recipes)
    Cost += R.cost(VF, Ctx);
  return Cost;
}

InstructionCost TPRegionBlock::cost(ElementCount VF, TPCostContext &Ctx) {
  // TODO(yuxin.an)
  llvm_unreachable("");
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void TPRegionBlock::print(raw_ostream &O, const Twine &Indent,
                          TPSlotTracker &SlotTracker) const {
  O << Indent << (isReplicator() ? "<xVFxUF> " : "<x1> ") << getName() << ": {";
  auto NewIndent = Indent + "  ";
  for (auto *BlockBase : tp_depth_first_shallow(Entry)) {
    O << '\n';
    BlockBase->print(O, NewIndent, SlotTracker);
  }
  O << Indent << "}\n";

  printSuccessors(O, Indent);
}
#endif

TPlan::~TPlan() {
  for (auto &KV : LiveOuts)
    delete KV.second;
  LiveOuts.clear();

  if (Entry) {
    TPValue DummyValue;
    for (TPBlockBase *Block : tp_depth_first_shallow(Entry))
      Block->dropAllReferences(&DummyValue);

    TPBlockBase::deleteCFG(Entry);

    Preheader->dropAllReferences(&DummyValue);
    delete Preheader;
  }
  for (TPValue *TPV : TPLiveInsToFree)
    delete TPV;
  for (auto &Elem : BackedgeTakenCount)
    delete Elem.second;
  BackedgeTakenCount.clear();
}

/// Create nested-loop TPlan
TPlanPtr TPlan::createInitialTPlan(MapVector<Loop *, SCEV *> TripCount,
                                   ScalarEvolution &SE,
                                   bool RequiresScalarEpilogueCheck,
                                   bool TailFolded,
                                   std::shared_ptr<TensorizePattern> Pattern) {
  
  // From Inner-most loop, arrange the preheader and its' control-flow-based TPBB  
  TPIRBasicBlock *Entry = nullptr;
  TPBasicBlock   *TensorPreheader = nullptr;
  TPlanPtr        Plan = nullptr;
  TPRegionBlock  *PrevRegion = nullptr;   // 가장 안쪽(innermost) region
  TPBasicBlock   *PrevHeader = nullptr;
  BasicBlock   *PrevExit = nullptr;
  TPRegionBlock *CurRegion = nullptr;

  // ----- Loop list : outer(R) → inner 순서라 가정 --------------------
  for (auto [Idx, CurL] : enumerate(Pattern->Info.LoopsR)) {
    // Although we visit outer-most loop first but its Idx starts from 0.
    // We want Idx=0 for inner-most loop. Thus, use IdxR.
    auto IdxR = Pattern->Info.LoopsR.size() - Idx - 1;
    // YYG::REMOVE
    errs() << "{Idx, IdxR} " << Idx << "IdxR: " << IdxR << "\n";
    CurL->dump();

    Entry = new TPIRBasicBlock(CurL->getLoopPreheader());
    TensorPreheader = new TPBasicBlock("tensor.ph" + Twine(IdxR));

    if (!Plan) {
      Plan = std::make_unique<TPlan>(Entry, TensorPreheader, Pattern);
    }

    // Header / Latch
    BasicBlock *IRHeaderBlock = CurL->getHeader();
    auto *HeaderTPBB = new TPIRBasicBlock(IRHeaderBlock, "tensor.header" + Twine(IdxR));
    // TPBasicBlock *HeaderTPBB = new TPBasicBlock("tensor.header" + Twine(IdxR));
    TPBasicBlock *LatchTPBB  = new TPBasicBlock("tensor.latch" + Twine(IdxR));

    // Middle / ScalarPH
    TPBasicBlock *MiddleTPBB = new TPBasicBlock("middle.block" + Twine(IdxR));
    TPBasicBlock *ScalarPH   = new TPBasicBlock("scalar.ph" + Twine(IdxR));
    if (!RequiresScalarEpilogueCheck)
      TPBlockUtils::connectBlocks(MiddleTPBB, ScalarPH);

    BasicBlock *IRExitBlock = CurL->getUniqueExitBlock();
    auto *TPExitBlock = new TPIRBasicBlock(IRExitBlock);
    // The connection order corresponds to the operands of the conditional branch.
    TPBlockUtils::insertBlockAfter(TPExitBlock, MiddleTPBB);
    TPBlockUtils::connectBlocks(MiddleTPBB, ScalarPH);

    // TODO(yg0412.yun) : (ScalarPH 와 연결은 나중에 InnerRegion.Exit 와 연결한다)
    // → 여기서는 아직 연결 안 함
    // middle → scalar (scalar‑epilogue가 필요하면)
    if (!PrevRegion) {
      // ---------- innermost (no inner region) ----------
      CurRegion = new TPRegionBlock(HeaderTPBB, LatchTPBB,
                                   "tensor loop" + Twine(IdxR), false);
      // YYG:REMOVE
      errs() << "---------- create PrevRegion1 \n";
    } else {
      // Prev.header (N+1) -> Cur.Prev-header(N)
      TPBlockUtils::connectBlocks(PrevRegion->getEntry(), TensorPreheader);

      // ---------- outer region that *contains* PrevRegion ----------
      CurRegion = new TPRegionBlock(HeaderTPBB, LatchTPBB,
                                    PrevRegion,               // ← inner region
                                    "tensor loop" + Twine(IdxR), false);
      // ScalarPH (N) -> Latch (N+1)
      TPBlockUtils::connectBlocks(ScalarPH, PrevRegion->getExiting());
      // // CurHeader (N) -> CurLatch (N)
      // TPBlockUtils::connectBlocks(LatchTPBB, HeaderTPBB);
    }
    // HeaderTPBB -> LatchTPBB 
    TPBlockUtils::connectBlocks(HeaderTPBB, LatchTPBB);
    // CurRegion -> MiddleTPBB
    TPBlockUtils::insertBlockAfter(MiddleTPBB, CurRegion);
    // TensorPreheader -> Header (of CurRegion)
    TPBlockUtils::insertBlockAfter(CurRegion, TensorPreheader);

    // Loop ↔ Region 매핑
    Plan->LoopIdx2Loop.insert({IdxR, CurL});
    Plan->Loop2LoopIdx.insert({CurL, IdxR});
    Plan->LoopIdx2TPRB.insert({IdxR, CurRegion});
    Plan->LoopIdx2PreHeaderTPBB.insert({IdxR, TensorPreheader});
    Plan->PreHeaderTPBB2LoopIdx.insert({TensorPreheader, IdxR});
    Plan->LoopIdx2HeaderTPBB.insert({IdxR, HeaderTPBB});
    Plan->HeaderTPBB2LoopIdx.insert({HeaderTPBB, IdxR});
    Plan->LoopIdx2LatchTPBB.insert({IdxR, LatchTPBB});
    Plan->LatchTPBB2LoopIdx.insert({LatchTPBB, IdxR});
    Plan->LoopIdx2ExitingTPBB.insert({IdxR, TPExitBlock});
    Plan->ExitingTPBB2LoopIdx.insert({TPExitBlock, IdxR});
    
    if (PrevRegion)
      PrevRegion->setInner(CurRegion);
    
    PrevRegion = CurRegion;
    PrevHeader = HeaderTPBB;
    PrevExit = IRExitBlock;
    Plan->dump();
  }
  // ----- Loop list : inner → outer --------------------
  for (auto [Idx, CurL] : enumerate(Pattern->Info.Loops)) {
    // Set the original trip-count
    Plan->TensorTripCount[CurL] =
        tputils::getOrCreateTPValueForSCEVExpr(*Plan, TripCount[CurL], SE);
    
    // Create initial TFxUF values as default 1.
    LLVMContext &Ctx = CurL->getHeader()->getContext();
    Value *initialTFUF = ConstantInt::get(Ctx, APInt(32, 1));
    auto *tfuf = new TPValue(initialTFUF);
    Plan->TFxUF.insert({CurL, tfuf});

    // Set the tensor-trip-count.
    // Initially, we just set tensor-trip-count to be same with scalar trip-count.
    // TODO(yg0412.yun) need to fix.
    // Plan->TensorTripCount[CurL] = new TPValue(Plan->TripCount[CurL]->getLiveInIRValue());
  }

  // Create one synthetic PF value per loop dimension.
  // TODO(yg.yun) Future, it PF value should be set by TTI.
  for (unsigned D = 0; D < Pattern->getDepth(); ++D)
    Plan->DimPFs.push_back(
                 std::make_unique<TPSymbolicValue>("PF[" + std::to_string(D) + "]"));

  // Set the total loop-depth on TPlan.
  Plan->setDepth(Pattern->getDepth());
  // SEt the TripCouunt
  Plan->resetTripCount(TripCount);
  return Plan;
}

void TPlan::prepareToExecute(MapVector<Loop *, Value *> CanonicalIVStartValue,
                             TPTransformState &State) {

  // for (auto &BTCElem : BackedgeTakenCount) {
  //   if (State.CFG.PrevBB && TripCountT.count(BTCElem.first)) {
  //     IRBuilder<> Builder(State.CFG.PrevBB->getTerminator());
  //     auto *TCMO = Builder.CreateSub(
  //         TripCountT[BTCElem.first],
  //         ConstantInt::get(TripCountT[BTCElem.first]->getType(), 1),
  //         "trip.count.minus");
  //     BackedgeTakenCount[BTCElem.first]->setUnderlyingValue(TCMO);
  //   }
  // }

  // for (auto Elem : TensorTripCountT)
  //   if (Elem.second)
  //     TensorTripCount[Elem.first]->setUnderlyingValue(Elem.second);

  // if (State.CFG.PrevBB) {
  //   IRBuilder<> Builder(State.CFG.PrevBB->getTerminator());

  //   for (auto [TCVElem, TFElem, UFElem] : zip(TripCountT, State.TF, State.UF)) {
  //     if (TCVElem.second && TFElem.second && UFElem.second) {
  //       TFxUF[TCVElem.first]->setUnderlyingValue(createStepForTFElem(
  //           Builder, TCVElem.second->getType(), TFElem.second, UFElem.second));
  //     }
  //   }
  // }
}

/// Replace \p VPBB with a VPIRBasicBlock wrapping \p IRBB. All recipes from \p
/// VPBB are moved to the newly created VPIRBasicBlock.  VPBB must have a single
/// predecessor, which is rewired to the new VPIRBasicBlock. All successors of
/// VPBB, if any, are rewired to the new VPIRBasicBlock.
static void replaceTPBBWithIRTPBB(TPBasicBlock *TPBB, BasicBlock *IRBB) {
  TPIRBasicBlock *IRMiddleVPBB = new TPIRBasicBlock(IRBB);
  for (auto &R : make_early_inc_range(*TPBB))
    R.moveBefore(*IRMiddleVPBB, IRMiddleVPBB->end());
  TPBlockBase *PredTPBB = TPBB->getSinglePredecessor();
  TPBlockUtils::disconnectBlocks(PredTPBB, TPBB);
  TPBlockUtils::connectBlocks(PredTPBB, IRMiddleVPBB);
  for (auto *Succ : to_vector(TPBB->getSuccessors())) {
    TPBlockUtils::connectBlocks(IRMiddleVPBB, Succ);
    TPBlockUtils::disconnectBlocks(TPBB, Succ);
  }
  delete TPBB;
}

/// Generate the code inside the preheader and body of the vectorized loop.
/// Assumes a single pre-header basic-block was created for this. Introduce
/// additional basic-blocks as needed, and fill them all.
void TPlan::execute(TPTransformState *State) {
  // TODO(yuxin.an)
  TPBasicBlock *MiddleTPBB =
      cast<TPBasicBlock>(getTensorLoopRegion()->getSingleSuccessor());

  // BasicBlock *ScalarPh = MiddleBB->getSingleSuccessor();
  auto &MiddleSuccs = MiddleTPBB->getSuccessors();
  // assert((MiddleSuccs.size() == 1 || MiddleSuccs.size() == 2) &&
  //        "middle block has unexpected successors");
  TPBasicBlock *ScalarPhTPBB = cast<TPBasicBlock>(
      MiddleSuccs.size() == 1 ? MiddleSuccs[0] : MiddleSuccs[1]);
  // assert(!isa<TPIRBasicBlock>(ScalarPhTPBB) &&
  //        "scalar preheader cannot be wrapped already");

  replaceTPBBWithIRTPBB(ScalarPhTPBB, State->TBS.SPH);
  replaceTPBBWithIRTPBB(MiddleTPBB, State->TBS.MiddleB);

  if (Preheader) {
    State->TPBB2BB.insert({Preheader, State->TBS.TPH});
    State->BB2TPBB.insert({State->TBS.TPH, Preheader});
  }

  State->TPBB2BB.insert({Entry, State->TBS.TPH});
  State->BB2TPBB.insert({State->TBS.TPH, Entry});

  // Disconnect the middle block from its single successor (the scalar loop
  // header) in both the CFG and DT. The branch will be recreated during VPlan
  // execution.
  // auto *BrInst = new UnreachableInst(MiddleBB->getContext());
  // BrInst->insertBefore(MiddleBB->getTerminator());
  // MiddleBB->getTerminator()->eraseFromParent();
  // State->CFG.DTU.applyUpdates({{DominatorTree::Delete, MiddleBB, ScalarPh}});

  // replaceTPBBWithIRTPBB(Entry, TensorPreHeader);

  auto SplitBB = [&](BasicBlock *Old, Twine BBName) {
    return SplitBlock(Old, Old->getTerminator(), State->DT, State->LI, nullptr,
                      BBName);
  };

  State->TBS.TEntry = SplitBB(State->TBS.TPH, "tensor.entry");

  // State->TPBB2BB.insert({Loop2HeaderTPBB[LoopI.L], Body});
  // State->BB2TPBB.insert({Body, Loop2HeaderTPBB[LoopI.L]});

  BasicBlock *InsertPtFront = State->TBS.TEntry; // Front insert point
  BasicBlock *InsertPtBack = State->TBS.TEntry;  // Back insert point

  for (auto [Idx, L] : enumerate(getPattern()->Info.Loops)) {
    auto IdxR = Pattern->Info.LoopsR.size() - Idx - 1;

    if (!Idx) {
      auto *Body = SplitBB(InsertPtFront, "tensor.body" + Twine(Idx));
      State->TBS.Loop2HeadBB.insert({L, Body});
      State->TBS.Loop2LatchBB.insert({L, Body});
      InsertPtBack = Body;

      auto *CurTPBB = LoopIdx2HeaderTPBB[IdxR];
      State->TPBB2BB.insert({CurTPBB, Body});
      State->BB2TPBB.insert({Body, CurTPBB});
      State->BackedgeTPBB.insert({CurTPBB, CurTPBB});
      State->BackedgeBB.insert({Body, Body});

      LoopIdx2LatchTPBB[IdxR] = LoopIdx2HeaderTPBB[IdxR];

      if (LoopIdx2PreHeaderTPBB.count(IdxR)) {
        State->TPBB2BB.insert({LoopIdx2PreHeaderTPBB[IdxR], State->TBS.TEntry});
        State->BB2TPBB.insert({State->TBS.TEntry, LoopIdx2PreHeaderTPBB[IdxR]});
      }
    } else {
      auto *PreBlock = SplitBB(InsertPtFront, "tensor.body" + Twine(Idx));
      auto *PostBlock = SplitBB(InsertPtBack, "tensor.latch" + Twine(Idx));

      State->TBS.Loop2HeadBB.insert({L, PreBlock});
      State->TBS.Loop2LatchBB.insert({L, PostBlock});

      State->TPBB2BB.insert({LoopIdx2HeaderTPBB[IdxR], PreBlock});
      State->BB2TPBB.insert({PreBlock, LoopIdx2HeaderTPBB[IdxR]});

      State->TPBB2BB.insert({LoopIdx2LatchTPBB[IdxR], PostBlock});
      State->BB2TPBB.insert({PostBlock, LoopIdx2LatchTPBB[IdxR]});

      State->BackedgeTPBB.insert({LoopIdx2LatchTPBB[IdxR], LoopIdx2HeaderTPBB[IdxR]});
      State->BackedgeBB.insert({PostBlock, PreBlock});
      InsertPtBack = PostBlock;

      if (LoopIdx2PreHeaderTPBB.count(IdxR)) {
        State->TPBB2BB.insert({LoopIdx2PreHeaderTPBB[IdxR], PreBlock});
        State->BB2TPBB.insert({PreBlock, LoopIdx2PreHeaderTPBB[IdxR]});
      }
    }
  }

  State->TBS.TExiting = SplitBB(InsertPtBack, "tensor.exiting");

  // Generate code in the loop pre-header and body.
  for (TPBlockBase *Block : tp_depth_first_shallow(Entry)) {
    Block->execute(State);
  }
}

InstructionCost TPlan::cost(ElementCount VF, TPCostContext &Ctx) {
  // TODO(yuxin.an)
  llvm_unreachable("");
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void TPlan::printLiveIns(raw_ostream &O) const {
  TPSlotTracker SlotTracker(this);

  for (auto [idx, TFxUFElem] : enumerate(TFxUF)) {
    if (TFxUFElem.second->getNumUsers() > 0) {
      O << "\nLive-in ";
      TFxUFElem.second->printAsOperand(O, SlotTracker);
      O << " = TF." << idx << " * UF." << idx;
    }
  }

  for (auto [idx, TTCElem] : enumerate(TensorTripCount)) {
    if (TTCElem.second->getNumUsers() > 0) {
      O << "\nLive-in ";
      TTCElem.second->printAsOperand(O, SlotTracker);
      O << " = tensor-trip-count." << idx;
    }
  }

  for (auto [idx, BTCElem] : enumerate(BackedgeTakenCount)) {
    if (BTCElem.second && BTCElem.second->getNumUsers()) {
      O << "\nLive-in ";
      BTCElem.second->printAsOperand(O, SlotTracker);
      O << " = backedge-taken count." << idx;
    }
  }

  for (auto [idx, TCElem] : enumerate(TensorTripCount)) {
    O << "\n";
    if (TCElem.second->isLiveIn())
      O << "Live-in ";
    TCElem.second->printAsOperand(O, SlotTracker);
    O << " = original trip-count." << idx;
  }
  O << "\n";
}

LLVM_DUMP_METHOD
void TPlan::print(raw_ostream &O) const {
  TPSlotTracker SlotTracker(this);

  O << "TPlan '" << getName() << "' {";

  printLiveIns(O);

  if (!getPreheader()->empty()) {
    O << "\n";
    getPreheader()->print(O, "", SlotTracker);
  }

  for (const TPBlockBase *Block : tp_depth_first_shallow(getEntry())) {
    O << '\n';
    Block->print(O, "", SlotTracker);
  }

  if (!LiveOuts.empty())
    O << "\n";
  for (const auto &KV : LiveOuts) {
    KV.second->print(O, SlotTracker);
  }

  O << "}\n";
}

std::string TPlan::getName() const {
  auto PrintTF = [](raw_ostream &OS, TFTy TF) {
    if (!TF.empty()) {
      OS << "{";
      OS << &TF.begin()->second;
      for (auto TFElem : drop_begin(TF)) {
        OS << ",";
        OS << TFElem.second;
      }
      OS << "}";
    }
  };
  auto PrintTUF = [](raw_ostream &OS, TUFTy UF) {
    if (!UF.empty()) {
      OS << "{";
      OS << &UF.begin()->second;
      for (auto UFElem : drop_begin(UF)) {
        OS << ",";
        OS << UFElem.second;
      }
      OS << "}";
    }
  };

  std::string Out;
  raw_string_ostream RSO(Out);
  RSO << Name << " for ";
  // if (!TFs.empty()) {
  //   RSO << "TF={";
  //   PrintTF(RSO, TFs[0]);
  //   for (auto TF : drop_begin(TFs)) {
  //     RSO << ",";
  //     PrintTF(RSO, TF);
  //   }
  //   RSO << "},";
  // }

  // if (UFs.empty()) {
  //   RSO << "UF>=1";
  // } else {
  //   RSO << "UF={";
  //   PrintTUF(RSO, UFs[0]);
  //   for (auto UF : drop_begin(UFs)) {
  //     RSO << ",";
  //     PrintTUF(RSO, UF);
  //   }
  //   RSO << "}";
  // }

  return Out;
}

LLVM_DUMP_METHOD
void TPlan::printDOT(raw_ostream &O) const {
  // TODO(yuxin.an)
  llvm_unreachable("");
}

LLVM_DUMP_METHOD
void TPlan::dump() const { print(dbgs()); }
#endif

void TPlan::addLiveOut(PHINode *PN, TPValue *V) {
  // TODO(yuxin.an)
  llvm_unreachable("");
}

TPlan *TPlan::duplicate() {
  // TODO(yuxin.an)
  llvm_unreachable("");
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)

Twine TPlanPrinter::getUID(const TPBlockBase *Block) {
  return (isa<TPRegionBlock>(Block) ? "cluster_N" : "N") +
         Twine(getOrCreateBID(Block));
}

Twine TPlanPrinter::getOrCreateName(const TPBlockBase *Block) {
  const std::string &Name = Block->getName();
  if (!Name.empty())
    return Name;
  return "TPB" + Twine(getOrCreateBID(Block));
}

void TPlanPrinter::dump() {
  Depth = 1;
  bumpIndent(0);
  OS << "digraph TPlan {\n";
  OS << "graph [labelloc=t, fontsize=30; label=\"Tensorization Plan";
  if (!Plan.getName().empty())
    OS << "\\n" << DOT::EscapeString(Plan.getName());

  {
    // Print live-ins.
    std::string Str;
    raw_string_ostream SS(Str);
    Plan.printLiveIns(SS);
    SmallVector<StringRef, 0> Lines;
    StringRef(Str).rtrim('\n').split(Lines, "\n");
    for (auto Line : Lines)
      OS << DOT::EscapeString(Line.str()) << "\\n";
  }

  OS << "\"]\n";
  OS << "node [shape=rect, fontname=Courier, fontsize=30]\n";
  OS << "edge [fontname=Courier, fontsize=30]\n";
  OS << "compound=true\n";

  dumpBlock(Plan.getPreheader());

  for (const TPBlockBase *Block : tp_depth_first_shallow(Plan.getEntry()))
    dumpBlock(Block);

  OS << "}\n";
}


void TPReductionPHIRecipe::execute(TPTransformState &State) {
  const RecurrenceDescriptor &RdxDesc = getRecurrenceDescriptor();
  Value *Start = getStartValue()->getLiveInIRValue();
  Type *ScalarTy = Start->getType();
  
  int Dim = getDimIndex();
  unsigned PF = (Dim >= 0) ? State.Plan->getPFForDim(static_cast<unsigned>(Dim)) : 1;
  
  IRBuilder<> Builder(State.CurBB->getTerminator());
  
  Type *PhiTy = ScalarTy;
  if (PF > 1 && DimSet.any()) {
    PhiTy = VectorType::get(ScalarTy, PF, false);
  }
  
  PHINode *RdxPhi = PHINode::Create(PhiTy, 2, "rdx.phi");
  RdxPhi->insertBefore(State.CurBB->getTerminator());
  
  BasicBlock *Preheader = State.CurBB->getSinglePredecessor();
  Value *InitVal = Start;
  
  if (PF > 1 && isa<Constant>(Start)) {
    InitVal = Builder.CreateVectorSplat(PF, cast<Constant>(Start));
  }
  
  RdxPhi->addIncoming(InitVal, Preheader ? Preheader : State.TBS.TPH);
  
  State.TPValue2Value[this] = RdxPhi;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void TPReductionPHIRecipe::print(raw_ostream &O, const Twine &Indent,
                                 TPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-REDUCTION-PHI ";

  printAsOperand(O, SlotTracker);
  O << " = phi ";
  printOperands(O, SlotTracker);
}
#endif

//--------------------------------------------------------------------
// execute : 현재는 스텁(stub) 구현. 실제 로직은 나중에 채워도 된다.
//--------------------------------------------------------------------
void TPFirstOrderRecurrencePHIRecipe::execute(TPTransformState &State) {
  // FIXME(yg0412.yun)
  llvm_unreachable("");
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void TPFirstOrderRecurrencePHIRecipe::print(raw_ostream &O, const Twine &Indent,
                                            TPSlotTracker &SlotTracker) const {
  O << Indent << "FIRST-ORDER-RECURRENCE-PHI ";
  printAsOperand(O, SlotTracker);
  O << " = phi ";
  printOperands(O, SlotTracker);
}
#endif

void TPlanPrinter::dumpBlock(const TPBlockBase *Block) {
  if (const TPBasicBlock *BasicBlock = dyn_cast<TPBasicBlock>(Block))
    dumpBasicBlock(BasicBlock);
  else if (const TPRegionBlock *Region = dyn_cast<TPRegionBlock>(Block))
    dumpRegion(Region);
  else
    llvm_unreachable("Unsupported kind of VPBlock.");
}

void TPlanPrinter::drawEdge(const TPBlockBase *From, const TPBlockBase *To,
                            bool Hidden, const Twine &Label) {
  // Due to "dot" we print an edge between two regions as an edge between the
  // exiting basic block and the entry basic of the respective regions.
  const TPBlockBase *Tail = From->getExitingBasicBlock();
  const TPBlockBase *Head = To->getEntryBasicBlock();
  OS << Indent << getUID(Tail) << " -> " << getUID(Head);
  OS << " [ label=\"" << Label << '\"';
  if (Tail != From)
    OS << " ltail=" << getUID(From);
  if (Head != To)
    OS << " lhead=" << getUID(To);
  if (Hidden)
    OS << "; splines=none";
  OS << "]\n";
}

void TPlanPrinter::dumpEdges(const TPBlockBase *Block) {
  auto &Successors = Block->getSuccessors();
  if (Successors.size() == 1)
    drawEdge(Block, Successors.front(), false, "");
  else if (Successors.size() == 2) {
    drawEdge(Block, Successors.front(), false, "T");
    drawEdge(Block, Successors.back(), false, "F");
  } else {
    unsigned SuccessorNumber = 0;
    for (auto *Successor : Successors)
      drawEdge(Block, Successor, false, Twine(SuccessorNumber++));
  }
}

void TPlanPrinter::dumpBasicBlock(const TPBasicBlock *BasicBlock) {
  // Implement dot-formatted dump by performing plain-text dump into the
  // temporary storage followed by some post-processing.
  OS << Indent << getUID(BasicBlock) << " [label =\n";
  bumpIndent(1);
  std::string Str;
  raw_string_ostream SS(Str);
  // Use no indentation as we need to wrap the lines into quotes ourselves.
  BasicBlock->print(SS, "", SlotTracker);

  // We need to process each line of the output separately, so split
  // single-string plain-text dump.
  SmallVector<StringRef, 0> Lines;
  StringRef(Str).rtrim('\n').split(Lines, "\n");

  auto EmitLine = [&](StringRef Line, StringRef Suffix) {
    OS << Indent << '"' << DOT::EscapeString(Line.str()) << "\\l\"" << Suffix;
  };

  // Don't need the "+" after the last line.
  for (auto Line : make_range(Lines.begin(), Lines.end() - 1))
    EmitLine(Line, " +\n");
  EmitLine(Lines.back(), "\n");

  bumpIndent(-1);
  OS << Indent << "]\n";

  dumpEdges(BasicBlock);
}

void TPlanPrinter::dumpRegion(const TPRegionBlock *Region) {
  OS << Indent << "subgraph " << getUID(Region) << " {\n";
  bumpIndent(1);
  OS << Indent << "fontname=Courier\n"
     << Indent << "label=\""
     << DOT::EscapeString(Region->isReplicator() ? "<xVFxUF> " : "<x1> ")
     << DOT::EscapeString(Region->getName()) << "\"\n";
  // Dump the blocks of the region.
  assert(Region->getEntry() && "Region contains no inner blocks.");
  for (const TPBlockBase *Block : tp_depth_first_shallow(Region->getEntry()))
    dumpBlock(Block);
  bumpIndent(-1);
  OS << Indent << "}\n";
  dumpEdges(Region);
}

void TPlanIngredient::print(raw_ostream &O) const {
  if (auto *Inst = dyn_cast<Instruction>(V)) {
    if (!Inst->getType()->isVoidTy()) {
      Inst->printAsOperand(O, false);
      O << " = ";
    }
    O << Inst->getOpcodeName() << " ";
    unsigned E = Inst->getNumOperands();
    if (E > 0) {
      Inst->getOperand(0)->printAsOperand(O, false);
      for (unsigned I = 1; I < E; ++I)
        Inst->getOperand(I)->printAsOperand(O << ", ", false);
    }
  } else // !Inst
    V->printAsOperand(O, false);
}
