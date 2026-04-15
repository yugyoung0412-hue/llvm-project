#include "llvm/Transforms/Tensorize/TPRecipeBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Tensorize/TPlan.h"

#define DEBUG_TYPE "loop-tensorize"

using namespace llvm;

iterator_range<mapped_iterator<Use *, std::function<TPValue *(Value *)>>>
TPRecipeBuilder::mapToTPValues(User::op_range Operands) {
  std::function<TPValue *(Value *)> Fn = [this](Value *Op) {
    if (auto *I = dyn_cast<Instruction>(Op)) {
      if (auto *R = Ingredient2Recipe.lookup(I))
        return R->getTPSingleValue();
    }
    return Plan.getOrAddLiveIn(Op);
  };
  return map_range(Operands, Fn);
}

TPValue *TPRecipeBuilder::mapToTPValue(Value *Op) {
  if (auto *I = dyn_cast<Instruction>(Op)) {
    if (auto *R = Ingredient2Recipe.lookup(I))
      return R->getTPSingleValue();
  }
  return Plan.getOrAddLiveIn(Op);
}

TPValue *TPRecipeBuilder::createEdgeMask(BasicBlock *Src, BasicBlock *Dst, Loop *L) {
  // YYG:REMOVE
  errs() << "[createEdgeMask]\n";
  errs() << "Src: " << *Src << "\n";
  errs() << "Dst: " << *Dst << "\n";

  assert(is_contained(predecessors(Dst), Src) && "Invalid edge");

  // Look for cached value.
  std::pair<BasicBlock *, BasicBlock *> Edge(Src, Dst);
  EdgeMaskCacheTy::iterator ECEntryIt = EdgeMaskCache.find(Edge);
  if (ECEntryIt != EdgeMaskCache.end())
    return ECEntryIt->second;

  // YYG::REMOVE
  errs() << "createEdgeMask\n";
  TPValue *SrcMask = getBlockInMask(Src);

  // The terminator has to be a branch inst!
  BranchInst *BI = dyn_cast<BranchInst>(Src->getTerminator());
  assert(BI && "Unexpected terminator found");

  if (!BI->isConditional() || BI->getSuccessor(0) == BI->getSuccessor(1))
    return EdgeMaskCache[Edge] = SrcMask;

  // If source is an exiting block, we know the exit edge is dynamically dead
  // in the vector loop, and thus we don't need to restrict the mask.  Avoid
  // adding uses of an otherwise potentially dead instruction.
  if (L->isLoopExiting(Src))
    return EdgeMaskCache[Edge] = SrcMask;

  TPValue *EdgeMask = getTPValueOrAddLiveIn(BI->getCondition(), Plan);

  assert(EdgeMask && "No Edge Mask found for condition");

  // If Dst is reached when false
  if (BI->getSuccessor(0) != Dst) {
    // YYG:REMOVE
    errs() << "!=Dst \n";
    EdgeMask = Builder.createNot(EdgeMask, BI->getDebugLoc());
  }

  if (SrcMask) { // Otherwise block in-mask is all-one, no need to AND.
    // The bitwise 'And' of SrcMask and EdgeMask introduces new UB if SrcMask
    // is false and EdgeMask is poison. Avoid that by using 'LogicalAnd'
    // instead which generates 'select i1 SrcMask, i1 EdgeMask, i1 false'.

    // YYG:REMOVE
    errs() << "SrcMask!!! \n";
    EdgeMask = Builder.createLogicalAnd(SrcMask, EdgeMask, BI->getDebugLoc());
  }

  return EdgeMaskCache[Edge] = EdgeMask;
}

TPValue *TPRecipeBuilder::getEdgeMask(BasicBlock *Src, BasicBlock *Dst) const {
  assert(is_contained(predecessors(Dst), Src) && "Invalid edge");

  // Look for cached value.
  std::pair<BasicBlock *, BasicBlock *> Edge(Src, Dst);
  EdgeMaskCacheTy::const_iterator ECEntryIt = EdgeMaskCache.find(Edge);
  assert(ECEntryIt != EdgeMaskCache.end() &&
         "looking up mask for edge which has not been created");
  return ECEntryIt->second;
}

void TPRecipeBuilder::createHeaderMask(Loop *L) {
  BasicBlock *Header = L->getHeader();

  LLVM_DEBUG(
      dbgs()
      << "[Warning] Please handle `TPRecipeBuilder::createHeaderMask` \n");
  if (!CM.foldTailByMasking()) {
    BlockMaskCache[Header] = nullptr;
    return;
  }

  llvm_unreachable("");
}

TPValue *TPRecipeBuilder::getBlockInMask(BasicBlock *BB) const {
  // Return the cached value.
  // BlockMaskCacheTy::const_iterator BCEntryIt = BlockMaskCache.find(BB);
  // assert(BCEntryIt != BlockMaskCache.end() &&
  //        "Trying to access mask for block without one.");
  // return BCEntryIt->second;
  return BlockMaskCache.lookup(BB);
}

void TPRecipeBuilder::createBlockInMask(BasicBlock *BB, Loop *L) {
  // unsigned total_depth = NestedOrigLoops[LoopDegree]->size();
  assert(L->contains(BB) && "Block is not a part of a loop");
  assert(BlockMaskCache.count(BB) == 0 && "Mask for block already computed");
  assert(L->getHeader() != BB &&
         "Loop header must have cached block mask");

  // All-one mask is modelled as no-mask following the convention for masked
  // load/store/gather/scatter. Initialize BlockMask to no-mask.
  TPValue *BlockMask = nullptr;
  
  // This is the block mask. We OR all incoming edges.
  for (auto *Predecessor : predecessors(BB)) {
    // YYG::REMOVE
    errs() << "Predecessor: " << *Predecessor << "\n";

    TPValue *EdgeMask = createEdgeMask(Predecessor, BB, L);
    errs() << "EdgeMask: \n";
    EdgeMask->dump();

    if (!EdgeMask) { // Mask of predecessor is all-one so mask of block is too.
      BlockMaskCache[BB] = EdgeMask;
      return;
    }

    if (!BlockMask) { // BlockMask has its initialized nullptr value.
      BlockMask = EdgeMask;
      continue;
    }

    BlockMask = Builder.createOr(BlockMask, EdgeMask, {});    
  }

  // YYG:REMOVE
  errs() << "[createBlockInMask] BlockMask: \n";
  BlockMask->dump();
  BlockMaskCache[BB] = BlockMask;
}

TPWidenMemoryRecipe *
TPRecipeBuilder::tryToWidenMemory(Instruction *I, ArrayRef<TPValue *> Operands,
                                  TFRange &Range) {
  errs() << "in   tryToWidenMemory\n";
  assert((isa<LoadInst>(I) || isa<StoreInst>(I)) &&
         "Must be called with either a load or store");

  TPValue *Mask = nullptr;

  /// FIXME(not general)
  bool Reverse = false; // !FIXME(yuxin.an)
  bool Consecutive = false;

  TPValue *Ptr = isa<LoadInst>(I) ? Operands[0] : Operands[1];

  if (Consecutive) {
    auto *GEP = dyn_cast<GetElementPtrInst>(
        Ptr->getUnderlyingValue()->stripPointerCasts());
    errs() << "GEP: " << *GEP << "\n";
    auto *VectorPtr = new TPVectorPointerRecipe(
        Ptr, getLoadStoreType(I), Reverse, GEP ? GEP->isInBounds() : false,
        I->getDebugLoc());
    Builder.getInsertBlock()->appendRecipe(VectorPtr);
    Ptr = VectorPtr;
    errs() << "After creating VectorPtr\n";
  }
  errs() << "After Consecutive \n";
  if (LoadInst *Load = dyn_cast<LoadInst>(I))
    return new TPWidenLoadRecipe(*Load, Ptr, Mask, Consecutive, Reverse,
                                 I->getDebugLoc());
  errs() << "Load: \n";
  StoreInst *Store = cast<StoreInst>(I);
  return new TPWidenStoreRecipe(*Store, Ptr, Operands[0], Mask, Consecutive,
                                Reverse, I->getDebugLoc());

  LLVM_DEBUG(
      dbgs()
      << "[Warning] Please handle `TPRecipeBuilder::tryToWidenMemory` \n");
  // TODO(yuxin.an)
  llvm_unreachable("");
}

/// Creates a VPWidenIntOrFpInductionRecpipe for \p Phi. If needed, it will also
/// insert a recipe to expand the step for the induction recipe.
static TPWidenIntOrFpInductionRecipe *
createWidenInductionRecipes(PHINode *Phi, Instruction *PhiOrTrunc,
                            TPValue *Start, const InductionDescriptor &IndDesc,
                            TPlan &Plan, ScalarEvolution &SE, Loop &OrigLoop) {
  // YYG:REMOVE
  errs() << "[createWidenInductionRecipes\n]";
  errs() << "OrigLoop.getLoopPreheader(): " << *(OrigLoop.getLoopPreheader()) << "\n";
  errs() << "IndDesc.getStartValue(): " << *(IndDesc.getStartValue()) << "\n";
  assert(IndDesc.getStartValue() ==
         Phi->getIncomingValueForBlock(OrigLoop.getLoopPreheader()) && "here1");
  // YYG:REMOVE
  errs() << "[createWidenInductionRecipes\n]";
  assert(SE.isLoopInvariant(IndDesc.getStep(), &OrigLoop) &&
         "step must be loop invariant");
  // YYG:REMOVE
  errs() << "[createWidenInductionRecipes\n]";

  TPValue *Step =
      tputils::getOrCreateTPValueForSCEVExpr(Plan, IndDesc.getStep(), SE);
  if (auto *TruncI = dyn_cast<TruncInst>(PhiOrTrunc)) {
    return new TPWidenIntOrFpInductionRecipe(Phi, Start, Step, IndDesc, TruncI);
  }
  assert(isa<PHINode>(PhiOrTrunc) && "must be a phi node here");
  return new TPWidenIntOrFpInductionRecipe(Phi, Start, Step, IndDesc);

  // TODO(yuxin.an)
  llvm_unreachable("");
}

void TPRecipeBuilder::fixHeaderPhis() {
  // The whole TPlan has been built at this point so all input Values must
  // have a TPRecipe counterpart. Fix TPlan header phis by adding their
  // corresponding backedge operands.
  for (TPHeaderPHIRecipe *R : PhisToFix) {
    auto *PN = cast<PHINode>(R->getUnderlyingValue());
    // Find the loop that contains this PHI (PHI lives in the loop header).
    Loop *PhiLoop = nullptr;
    for (Loop *L : NestedOrigLoops) {
      if (L->getHeader() == PN->getParent()) {
        PhiLoop = L;
        break;
      }
    }
    assert(PhiLoop && "Header PHI must belong to one of the nested loops");
    BasicBlock *OrigLatch = PhiLoop->getLoopLatch();
    TPRecipeBase *IncR =
        getRecipe(cast<Instruction>(PN->getIncomingValueForBlock(OrigLatch)));
    R->addOperand(IncR->getTPSingleValue());
  }
}

TPHeaderPHIRecipe *TPRecipeBuilder::tryToOptimizeInductionPHI(
    PHINode *Phi, ArrayRef<TPValue *> Operands, TFRange &Range) {

  Loop *L = Plan.getPattern()->Info.PHI2Loop[Phi];

  if (auto *II = Legal->getIntOrFpInductionDescriptor(Phi)) {
    errs() << "getINtOrFpInductionDescriptor(Phi)\n";
    return createWidenInductionRecipes(Phi, Phi, Operands[0], *II, Plan,
                                       *PSE.getSE(), *L);
  }
  // Check if this is pointer induction. If so, build the recipe for it.
  if (auto *II = Legal->getPointerInductionDescriptor(Phi)) {
    // YYG:REMOVE
    errs() << "Legal->getPointerInductionDescriptor(Phi)\n";
    TPValue *Step = tputils::getOrCreateTPValueForSCEVExpr(Plan, II->getStep(),
                                                           *PSE.getSE());
    // auto DecisionLambda = [&](ElementCount TF) -> bool {
    //   // map 에 entry 가 없을 경우를 대비해 기본값 1을 넣는다.
    //   if (!TF.isScalable() && TF.getFixedValue() == 0)
    //       TF = ElementCount::getFixed(1);

    //   return CM.isScalarAfterVectorization(Phi, TF);
    // };
    // return new TPWidenPointerInductionRecipe(
    //     Phi, Operands[0], Step, *II,
    //     LoopTensorizePlanner::getDecisionAndClampRange(
    //         DecisionLambda, Range));
    // Decision lambda : 현재 루프에 대한 VF 하나만 받는다.
    auto DecisionLambda = [&](llvm::ElementCount TF) -> bool {
        // 0 이거나 스케일러 0 인 경우 1 로 보정 (기본값)
        if (!TF.isScalable() && TF.getFixedValue() == 0)
            TF = llvm::ElementCount::getFixed(1);
        return CM.isScalarAfterVectorization(Phi, TF);
    };
    return new TPWidenPointerInductionRecipe(
            Phi, Operands[0], Step, *II,
            LoopTensorizePlanner::getDecisionAndClampRange(
                    DecisionLambda, Range, L));
  }
  // YYG::REMOVE
  errs() << "!Legl->getPointerINductionDescriptor(Phi)\n";
  return nullptr;
}

TPWidenIntOrFpInductionRecipe *TPRecipeBuilder::tryToOptimizeInductionTruncate(
    TruncInst *I, ArrayRef<TPValue *> Operands, TFRange &Range) {
  // TODO(yuxin.an)
  llvm_unreachable("");
}

TPBlendRecipe *TPRecipeBuilder::tryToBlend(PHINode *Phi,
                                           ArrayRef<TPValue *> Operands) {
  // YYG:REMOVE
  errs() << "[tryToBlend] Phi : " << *Phi << "\n";

  unsigned NumIncoming = Phi->getNumIncomingValues();

  // We know that all PHIs in non-header blocks are converted into selects, so
  // we don't have to worry about the insertion order and we can just use the
  // builder. At this point we generate the predication tree. There may be
  // duplications since this is a simple recursive scan, but future
  // optimizations will clean it up.
  // TODO: At the moment the first mask is always skipped, but it would be
  // better to skip the most expensive mask.
  SmallVector<TPValue *, 2> OperandsWithMask;

  for (unsigned In = 0; In < NumIncoming; In++) {
    // YYG::REMOVE
    errs() << "Operands[In]: " << *(Operands[In]) << "\n";
    
    OperandsWithMask.push_back(Operands[In]);
    TPValue *EdgeMask =
        getEdgeMask(Phi->getIncomingBlock(In), Phi->getParent());
    if (!EdgeMask) {
      assert(In == 0 && "Both null and non-null edge masks found");
      assert(all_equal(Operands) &&
             "Distinct incoming values with one having a full mask");
      break;
    }
    if (In == 0)
      continue;
    OperandsWithMask.push_back(EdgeMask);
  }
  return new TPBlendRecipe(Phi, OperandsWithMask);
}

TPWidenCallRecipe *TPRecipeBuilder::tryToWidenCall(CallInst *CI,
                                                   ArrayRef<TPValue *> Operands,
                                                   TFRange &Range) {
  // TODO(yuxin.an)
  llvm_unreachable("");
}

TPMatrixCallRecipe *
TPRecipeBuilder::tryToMatrixCall(Instruction *I, ArrayRef<TPValue *> Operands,
                                 TFRange &Range) {
  Intrinsic::ID ID = Intrinsic::matrix_multiply;

  return new TPMatrixCallRecipe(I, Operands, ID, I->getDebugLoc());
}

bool TPRecipeBuilder::shouldWiden(Instruction *I, TFRange &Range, Loop *L) const {
  assert(!isa<BranchInst>(I) && !isa<PHINode>(I) && !isa<LoadInst>(I) &&
         !isa<StoreInst>(I) && "Instruction should have been handled earlier");

  LLVM_DEBUG(
      dbgs() << "[Warning] Please handle `TPRecipeBuilder::shouldWiden` \n");

  // ----------------- yuxin's work ----------------------------------------//
  // if (isa<ICmpInst>(I) || (I->getOpcode() == Instruction::Mul) ||
  //     (I->getOpcode() == Instruction::Add) || isa<GetElementPtrInst>(I)) {
  //   return false;
  // }

  // if (I->getOpcode() == Instruction::FAdd ||
  //     I->getOpcode() == Instruction::FMul) {
  //   return true;
  // }
  // ----------------- yuxin's work ----------------------------------------//

  /// FIXME(yg0412.yun)
  return true;
  // ----------------- yuxin's work ----------------------------------------//
  // Instruction should be widened, unless it is scalar after vectorization,
  // scalarization is profitable or it is predicated.
  // auto WillScalarize = [this, I, L](ElementCount VF) -> bool {
  //   return CM.isScalarAfterVectorization(I, VF) ||
  //          CM.isProfitableToScalarize(I, VF) ||
  //          CM.isScalarWithPredication(I, VF, L);
  // };
  // return !LoopTensorizePlanner::getDecisionAndClampRange(WillScalarize,
  //                                                            Range, L);
  llvm_unreachable("");
}

TPWidenRecipe *TPRecipeBuilder::tryToWiden(Instruction *I,
                                           ArrayRef<TPValue *> Operands,
                                           TPBasicBlock *TPBB, Loop *L) {
  // YYG:REMOVE
  errs() << "[tryToWiden]\n";

  switch (I->getOpcode()) {
  default:
    return nullptr;
  case Instruction::SDiv:
  case Instruction::UDiv:
  case Instruction::SRem:
  case Instruction::URem: {
    // If not provably safe, use a select to form a safe divisor before widening the
    // div/rem operation itself.  Otherwise fall through to general handling below.
    if (CM.isPredicatedInst(I, L)) {
      // YYG:REMOVE
      errs() << "CM.isPredicatedInst(I, L) \n";

      SmallVector<TPValue *> Ops(Operands.begin(), Operands.end());
      TPValue *Mask = getBlockInMask(I->getParent());
      TPValue *One =
          Plan.getOrAddLiveIn(ConstantInt::get(I->getType(), 1u, false));
      auto *SafeRHS = Builder.createSelect(Mask, Ops[1], One, I->getDebugLoc());
      Ops[1] = SafeRHS;
      return new TPWidenRecipe(*I, make_range(Ops.begin(), Ops.end()));
    }
    [[fallthrough]];
  }
  case Instruction::Add:
  case Instruction::And:
  case Instruction::AShr:
  case Instruction::FAdd:
  case Instruction::FCmp:
  case Instruction::FDiv:
  case Instruction::FMul:
  case Instruction::FNeg:
  case Instruction::FRem:
  case Instruction::FSub:
  case Instruction::ICmp:
  case Instruction::LShr:
  case Instruction::Mul:
  case Instruction::Or:
  case Instruction::Select:
  case Instruction::Shl:
  case Instruction::Sub:
  case Instruction::Xor:
  case Instruction::Freeze:
    return new TPWidenRecipe(*I, make_range(Operands.begin(), Operands.end()));
  };
}

TPReplicateRecipe *TPRecipeBuilder::handleReplication(Instruction *I,
                                                      TFRange &Range, Loop *L) {
  // For determining the IsUniform, we should know the tensorization factor
  // and saved uniform instruction after applying that tensorization factor on inustructions.
  // --> 잠만, 지금도 TF 후보들 받고 있으니까 그걸로 계산하게 해야하나? 
  // --> TTI로 가져오기도 전에 TF 후보들을 미리 선별해서 거르는게 맞을까? 
  // bool IsUniform = LoopTensorizePlanner::getDecisionAndClampRange(
  //   [&](ElementCount VF) { return CM.isUniformAfterVectorization(I, VF); },
  //   Range, L);
  bool IsUniform = true;
  bool IsPredicated = CM.isPredicatedInst(I, L);

  // Even if the instruction is not marked as uniform, there are certain
  // intrinsic calls that can be effectively treated as such, so we check for
  // them here. Conservatively, we only do this for scalable vectors, since
  // for fixed-width VFs we can always fall back on full scalarization.
  if (/* && Range.Start.isScalable() */ isa<IntrinsicInst>(I)) {
    switch (cast<IntrinsicInst>(I)->getIntrinsicID()) {
    case Intrinsic::assume:
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
      // For scalable vectors if one of the operands is variant then we still
      // want to mark as uniform, which will generate one instruction for just
      // the first lane of the vector. We can't scalarize the call in the same
      // way as for fixed-width vectors because we don't know how many lanes
      // there are.
      //
      // The reasons for doing it this way for scalable vectors are:
      //   1. For the assume intrinsic generating the instruction for the first
      //      lane is still be better than not generating any at all. For
      //      example, the input may be a splat across all lanes.
      //   2. For the lifetime start/end intrinsics the pointer operand only
      //      does anything useful when the input comes from a stack object,
      //      which suggests it should always be uniform. For non-stack objects
      //      the effect is to poison the object, which still allows us to
      //      remove the call.
      IsUniform = true;
      break;
    default:
      break;
    }
  }

  // TPValue *BlockInMask = nullptr;
  // if (isa<ICmpInst>(I) || (I->getOpcode() == Instruction::Mul) ||
  //     (I->getOpcode() == Instruction::Add) || isa<GetElementPtrInst>(I)) {
  //   IsUniform = true;
  //   IsPredicated = false;
  // } else {
  //   llvm_unreachable("");
  // }
  TPValue *BlockInMask = nullptr;
  if (!IsPredicated) {
    // Finalize the recipe for Instr, first if it is not predicated.
    LLVM_DEBUG(dbgs() << "LV: Scalarizing:" << *I << "\n");
  } else {
    LLVM_DEBUG(dbgs() << "LV: Scalarizing and predicating:" << *I << "\n");
    // Instructions marked for predication are replicated and a mask operand is
    // added initially. Masked replicate recipes will later be placed under an
    // if-then construct to prevent side-effects. Generate recipes to compute
    // the block mask for this region.
    BlockInMask = getBlockInMask(I->getParent());
  }

  // Note that there is some custom logic to mark some intrinsics as uniform
  // manually above for scalable vectors, which this assert needs to account for
  // as well.
  // assert((Range.Start.isScalar() || !IsUniform || !IsPredicated ||
  //         (Range.Start.isScalable() && isa<IntrinsicInst>(I))) &&
  //        "Should not predicate a uniform recipe");
  auto *Recipe = new TPReplicateRecipe(I, mapToTPValues(I->operands()),
                                       IsUniform, BlockInMask);

  LLVM_DEBUG(
      dbgs()
      << "[Warning] Please handle `TPRecipeBuilder::handleReplication` \n");

  return Recipe;
}

TPRecipeBase *
TPRecipeBuilder::tryToCreateWidenRecipe(Instruction *Instr,
                                        ArrayRef<TPValue *> Operands,
                                        TFRange &Range, TPBasicBlock *TPBB, unsigned LoopDegree) {
  // First, check for specific widening recipes that deal with inductions, Phi
  // nodes, calls and memory operations.
  // YYG:REMOVE
  errs() << "[tryToCreateWidenRecipe] Instr: " << *Instr << "\n";
  unsigned TotalLoopDegree = NestedOrigLoops.size();
  unsigned outermostloop = TotalLoopDegree - LoopDegree;
  Loop *OrigLoop = NestedOrigLoops[outermostloop];
  TPRecipeBase *Recipe;
  if (auto *Phi = dyn_cast<PHINode>(Instr)) {
    errs() << "depth: " << TotalLoopDegree << ", LoopDegree: " << LoopDegree << "\n";
    errs() << "outermostloop: " << outermostloop << "\n";
    OrigLoop->dump();
    errs() << "Phi->getParent(): " << *(Phi->getParent()) << "\n"; 
    errs() << "Phi->getHeader(): " << *(OrigLoop->getHeader()) << "\n";
    
    // Technically, if the PhiNode is not from its own body, and another edge brings such PhiNode,
    // this PhiNode should be tryToBlend(). Plus, in nested-loop case, such PhiNode can be LCSSA-PHI 
    // which existing on latch/exiting block and one of the operand comes from 
    if (Phi->getParent() != OrigLoop->getHeader()) {
      // In nested-loop, LCSSA-PHI must be inside of latch block.
      // Because latch block is exiting block at the same time to escape the next outer-loop.
      // Therefore, if LCSSA-PHI is detected on latch/exiting block which one of the operands comes from
      // outer-loop's latch, we can wrap it to ExtractFromEnd or FirstElement & ExtractElement.
      // if (OrigLoop->isLoopLatch(Phi->getParent()) && )
      return tryToBlend(Phi, Operands);
    }
    if ((Recipe = tryToOptimizeInductionPHI(Phi, Operands, Range))) {
      // YYG:REMOVE
      errs() << "Scucessfully create to tryToOptimizeInductionPHI\n";
      return Recipe;
    }

    TPHeaderPHIRecipe *PhiRecipe = nullptr;
    // YYG::REMOVE
    errs() << "Legal->isReductionVariable(Phi): " << Legal->isReductionVariable(Phi) << "\n";
    // 아, get-Reduction 정보 관련해서 함수를 먼저 Call 안해서 Reduction으로 잡히지 않네..
    // assert((Legal->isReductionVariable(Phi) ||
    //         Legal->isFixedOrderRecurrence(Phi)) &&
    //        "can only widen reductions and fixed-order recurrences here");
    TPValue *StartV = Operands[0];
    if (Legal->isReductionVariable(Phi)) {
      // YYG:REMOVE
      errs() << "isReductionVariable(Phi)\n";

      const RecurrenceDescriptor &RdxDesc =
          Legal->getReductionVars().find(Phi)->second;
      assert(RdxDesc.getRecurrenceStartValue() ==
            Phi->getIncomingValueForBlock(OrigLoop->getLoopPreheader()));
      
      // TODO(yg0412.yun) In-Loop Reduction 수정!
      PhiRecipe = new TPReductionPHIRecipe(Phi, RdxDesc, *StartV,
                                          /* CM.isInLoopReduction(Phi) = */ true,
                                          /* CM.useOrderedReductions(RdxDesc) = */ true);
    } else {
      // TODO: Currently fixed-order recurrences are modeled as chains of
      // first-order recurrences. If there are no users of the intermediate
      // recurrences in the chain, the fixed order recurrence should be modeled
      // directly, enabling more efficient codegen.
      PhiRecipe = new TPFirstOrderRecurrencePHIRecipe(Phi, *StartV);
    }

    PhisToFix.push_back(PhiRecipe);
    return PhiRecipe;
  }
  if (isa<TruncInst>(Instr) && (Recipe = tryToOptimizeInductionTruncate(
                                    cast<TruncInst>(Instr), Operands, Range)))
    return Recipe;

  // All widen recipes below deal only with VF > 1.
  // if (false) // !FIXME(yuxin.an)
  //   return nullptr;

  // All widen recipes below deal only with VF > 1.
  // if (LoopTensorizePlanner::getDecisionAndClampRange(
  //         [&](ElementCount VF) { return VF.isScalar(); }, Range))
  //   return nullptr;
  
  if (auto *CI = dyn_cast<CallInst>(Instr))
    return tryToWidenCall(CI, Operands, Range);

  // !FIXME(yuxin.an)
  // if (Instr->getOpcode() == Instruction::FMul) {
  //   return tryToMatrixCall(Instr, Operands, Range);
  // }

  if (isa<LoadInst>(Instr) || isa<StoreInst>(Instr)) {
    return tryToWidenMemory(Instr, Operands, Range);
  }

  /// FIXME(not general)
  if (!shouldWiden(Instr, Range, OrigLoop))
    return nullptr;

  if (auto *GEP = dyn_cast<GetElementPtrInst>(Instr))
    return new TPWidenGEPRecipe(GEP,
                                make_range(Operands.begin(), Operands.end()));

  if (auto *SI = dyn_cast<SelectInst>(Instr)) {
    return new TPWidenSelectRecipe(
        *SI, make_range(Operands.begin(), Operands.end()));
  }

  if (auto *CI = dyn_cast<CastInst>(Instr)) {
    return new TPWidenCastRecipe(CI->getOpcode(), Operands[0], CI->getType(),
                                 *CI);
  }

  return tryToWiden(Instr, Operands, TPBB, OrigLoop);
}
