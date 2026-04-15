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
