//===- TPlanner.cpp - A Loop Vectorizer ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tensorize/TPlanner.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/Transforms/Tensorize/TPlanCFG.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
// #include "llvm/IR/IntrinsicsGAIA.h" // GAIA not available in this build
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Tensorize/TPRecipeBuilder.h"
#include "llvm/Transforms/Tensorize/TPattern.h"
#include "llvm/Transforms/Tensorize/TPlan.h"
#include "llvm/Transforms/Tensorize/TPlanTransforms.h"
#include "llvm/Transforms/Tensorize/TPlanVerifier.h"
#include "llvm/Transforms/Tensorize/TensorizeCommon.h"
#include "llvm/Transforms/Tensorize/TPRecipeMatcher.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>

using namespace llvm;

namespace llvm {
extern cl::opt<bool> EnableTPlanNativePath;
class instruction;
class TPlan;

#define DEBUG_TYPE "tplan"

namespace {
static void addLiveOutsForFirstOrderRecurrences(TPlan &Plan) {
  TPRegionBlock *TensorRegion = Plan.getTensorLoopRegion();

  TPBasicBlock *ScalarPHTPBB = nullptr;

  auto *MiddleTPBB = cast<TPBasicBlock>(TensorRegion->getSingleSuccessor());
  for (TPBlockBase *Succ : MiddleTPBB->getSuccessors()) {
    if (isa<TPIRBasicBlock>(Succ))
      continue;
    assert(!ScalarPHTPBB && "Two candidates for ScalarPHTPBB?");
    ScalarPHTPBB = cast<TPBasicBlock>(Succ);
  }
  if (!ScalarPHTPBB)
    return;

  TPBuilder ScalarPHBuilder(ScalarPHTPBB);
  TPBuilder MiddleBuilder(MiddleTPBB);

  if (auto *Terminator = MiddleTPBB->getTerminator()) {
    auto *Condition = dyn_cast<TPInstruction>(Terminator->getOperand(0));
    assert((!Condition || Condition->getParent() == MiddleTPBB) &&
           "Condition expected in MiddleVPBB");
    MiddleBuilder.setInsertPoint(Condition ? Condition : Terminator);
  }

  TPValue *OneTPV = Plan.getOrAddLiveIn(
      ConstantInt::get(Plan.getCanonicalIV()->getScalarType(), 1));

  LLVM_DEBUG(
      dbgs()
      << "[Warning] Please handle `addLiveOutsForFirstOrderRecurrences` \n");
}
} // namespace


static bool le(ElementCount L, ElementCount R) {
  if (!L.isScalable() && !R.isScalable())
    return L.getFixedValue() <= R.getFixedValue();
  return false;
}

bool LoopTensorizePlanner::getDecisionAndClampRange(
        const std::function<bool(ElementCount)> &Predicate,
        TFRange &Range,
        Loop *CurLoop)            // ← 현재 루프를 인자로 받는다
{
    assert(CurLoop && "CurLoop must be supplied");
    //assert(!Range.isEmpty() && "Trying to test empty TF range.");

    // 1) 현재 루프에 대한 시작/끝 값 꺼내기
    ElementCount StartVF = Range.Start.lookup(CurLoop);
    ElementCount EndVF   = Range.End.lookup(CurLoop);

    // 2) 시작값에 대해 predicate 실행
    bool PredicateAtRangeStart = Predicate(StartVF);

    // 3) 시작값의 2배부터 End까지 순회하면서 predicate 가 바뀌는지 확인
    for (ElementCount TmpVF = StartVF * 2; le(TmpVF, EndVF);
         TmpVF = TmpVF * 2) {
      if (Predicate(TmpVF) != PredicateAtRangeStart) {
          // 현재 루프에 대한 End 를 갱신한다.
          // MapVector 에 엔트리가 없으면 새로 만든다 (operator[] 사용).
          Range.End[CurLoop] = TmpVF;
          break;
      }
  }

    return PredicateAtRangeStart;
}

TensorizationFactor
LoopTensorizePlanner::planInTPlanNativePath(SmallVector<ElementCount> UserTF) {
  SmallVector<ElementCount> TF = UserTF;

  auto PrintTF = [](raw_ostream &OS, SmallVector<ElementCount> TF,
                    std::string Prefix = "", std::string Suffix = "") {
    OS << Prefix << "{";
    for (auto &TFElem : TF) {
      OS << TFElem;
      if (!(&TFElem == &(TF.back())))
        OS << ", ";
    }
    OS << "}" << Suffix;
  };

  // TODO(yuxin.an): implement complete check

  for (auto TFElem : TF)
    assert(isPowerOf2_32(TFElem.getKnownMinValue()) &&
           "TF needs to be a power of two");

  LLVM_DEBUG(PrintTF(dbgs(), TF, "LT: Using TF ", " to build VPlans.\n"));

  buildTPlans(TF, TF);

  llvm_unreachable("");
}

void LoopTensorizePlanner::buildTPlans(SmallVector<ElementCount> MinTF,
                                       SmallVector<ElementCount> MaxTF) {

  llvm_unreachable("");
}

void LoopTensorizePlanner::setMaxTF(TensorizePattern *tp, TFTy *MaxTFMap) {
  // assert(Loops.size() == MaxTFMap->size() && "TLP: UserTFMap should match up
  // with NestedLoops' size.");
  if (!MaxTFMap)
    TPlanDL.setMaxTF(tp, MaxTFMap);
}

bool LoopTensorizePlanner::createNestedLoopTPlan(bool UseTensorType) {
  auto PrintTF = [](raw_ostream &OS, TFTy TF, std::string Prefix = "",
                    std::string Suffix = "") {
    if (!TF.empty()) {
      OS << Prefix << "{";
      OS << TF.begin()->second;
      for (auto TFElem : drop_begin(TF)) {
        OS << ",";
        OS << TFElem.second;
      }
      OS << "}" << Suffix;
    }
  };

  bool UserTFIsLegal = true;
  // // TODO(yuxin.an):  TF check is simplified compared to Vectorize.
  // for (auto TFElem : UserTF) {
  //   if (TFElem.second.isZero() ||
  //       !isPowerOf2_32(TFElem.second.getKnownMinValue())) {
  //     LLVM_DEBUG(dbgs() << ""); // TODO (yuxin.an)
  //     UserTFIsLegal = false;
  //     break;
  //   }
  // }

  if (UserTFIsLegal) {
    LLVM_DEBUG(PrintTF(dbgs(), TPlanDL.MaxTF, "LT: Using user TF ", ".\n"));
    buildTPlansWithTPRecipes(/* MinTF */ TPlanDL.MaxTF,
                             /* MaxTF */ TPlanDL.MaxTF, UseTensorType);
    LLVM_DEBUG(printPlans(dbgs()));
    dbgs() << "[Info] buildTPlansWithTPRecipes end.\n";
  }

  return true;
}

void LoopTensorizePlanner::buildTPlansWithTPRecipes(TFTy MinTF, TFTy MaxTF,
                                                    bool UseTensorType) {
  // TODO(yuxin.an): Confirm whether it is possible to not rely on the pattern
  assert((TPlanDL.pattern && TPlanDL.pattern->Status.CanTensorize) &&
         "Pattern needs to be specified and matched.");

  TFTy MaxTFTimes2;
  for (auto TFElem : MaxTF)
    MaxTFTimes2.insert({TFElem.first, TFElem.second * 2});

  // 지금 loop마다 TF를 설정하고 있음. 
  // Collect the instructions (and their associated costs) that will be more
  // profitable to scalarize.
  // CM.collectInLoopReductions();
  // if (CM.selectUserVectorizationFactor(UserVF)) {
  //   LLVM_DEBUG(dbgs() << "LV: Using user VF " << UserVF << ".\n");
  //   buildVPlansWithVPRecipes(UserVF, UserVF);
  //   if (!hasPlanWithVF(UserVF)) {
  //     LLVM_DEBUG(dbgs() << "LV: No VPlan could be built for " << UserVF
  //                       << ".\n");
  //     return std::nullopt;
  //   }

  //   LLVM_DEBUG(printPlans(dbgs()));
  //   return {{UserVF, 0, 0}};
  // } else
  //   reportVectorizationInfo("UserVF ignored because of invalid costs.",
  //                           "InvalidCost", ORE, OrigLoop);


  // !FIXME(yuxin.an): Confirm implementation logic
  TFRange SubRange = {MinTF, MaxTFTimes2};
  // for (auto [MinTF, MaxTFT2] : zip(MinTF, MaxTFTimes2)) {
  //   TFRange SubRangeElem = {MinTF, MaxTFT2};
  //   SubRange.push_back(SubRangeElem);
  // }

  // for (const auto &TF : SubRange) {
  //   // Collect Uniform and Scalar instructions after vectorization with VF.
  //   CM.collectUniformsAndScalars(TF);

  //   // Collect the instructions (and their associated costs) that will be more
  //   // profitable to scalarize.
  //   if (TF.isTensor())
  //     CM.collectInstsToScalarize(TF);
  // }

  if (auto Plan = tryToBuildTPlanWithTPRecipes(SubRange, UseTensorType)) {
    if (true) {
      // TODO(yg0412.yun) Need to move optimize to before TPlanTransform::execute();
      TPlanTransforms::optimize(*Plan, *Loop2PSE.begin()->second->getSE());
      LLVM_DEBUG(dbgs() << "[Info] `TPlanTransforms::optimize` end\n");
      // TODO(yg0412.yun)
      // turn on below code comments
      // assert(verifyTPlanIsValid(*Plan) && "TPlan is invalid");
      TPlans.push_back(std::move(Plan));
    }
  }
}

// Add the necessary canonical IV and branch recipes required to control the
// loop.
static void addCanonicalIVRecipes(TPlan &Plan, Type *IdxTy, bool HasNUW,
                                  DebugLoc DL) {

  // TODO(yuxin.an): refactor

  TPRegionBlock *TopRegion = Plan.getTensorLoopRegion();
  // YYG::REMOVE
  errs() << "TopRegion: \n";
  TopRegion->dump();

  // Unless calculating IdxR (reverse index), order is outer-most -> inner-most.
  for (auto [LIdx2HElem, LIdx2LElem] :
       zip(Plan.LoopIdx2HeaderTPBB, Plan.LoopIdx2LatchTPBB)) {
    // Traveling from outer-most loop. 
    Loop *CurLoop = Plan.LoopIdx2Loop[LIdx2HElem.first];
    // YYG::REMOVE
    errs() << "[addCanonicalize] CUrL: \n";
    CurLoop->dump();

    TPBasicBlock *Header = LIdx2HElem.second;
    TPBasicBlock *Latch = LIdx2LElem.second;

    Value *StartIdx = ConstantInt::get(IdxTy, 0);
    auto *StartV = Plan.getOrAddLiveIn(StartIdx);

    // Add a VPCanonicalIVPHIRecipe starting at 0 to the header.
    auto *CanonicalIVPHI = new TPCanonicalIVPHIRecipe(StartV, DL);
    Header->insert(CanonicalIVPHI, Header->begin());
    // YYG::REMOVE
    errs() << "after inserting CanonicalIVPhi\n";
    Plan.dump();

    TPBuilder Builder(Latch);
    // Add a VPInstruction to increment the scalar canonical IV by VF * UF.
    // Initially the induction increment is guaranteed to not wrap, but that may
    // change later, e.g. when tail-folding, when the flags need to be dropped.

    auto *CanonicalIVIncrement = Builder.createOverflowingOp(
        Instruction::Add, {CanonicalIVPHI, Plan.getTFxUF()[CurLoop]},
        {HasNUW, false}, DL, "index.next");
    CanonicalIVPHI->addOperand(CanonicalIVIncrement);
    // YYG::REMOVE
    errs() << "after inserting CanonicalIVPHI->addOperand\n";
    Plan.dump();

    // Add the BranchOnCount VPInstruction to the latch.
    // YYG::REMOVE
    errs() << "Plan.getTensorTripCount()[CurLoop]: " << *(Plan.getTensorTripCount()[CurLoop]) << "\n";

    Builder.createNaryOp(
        TPInstruction::BranchOnCount,
        {CanonicalIVIncrement, Plan.getTensorTripCount()[CurLoop]}, DL);
    
    // todo(yg0412.yun)
    // getTripCount에 실제 trip count가 들어가 있음. 
    // Builder.createNaryOp(
    //     TPInstruction::BranchOnCount,
    //     {CanonicalIVIncrement, Plan.getTripCount()[CurLoop]}, DL);
  }

  Plan.dump();

  LLVM_DEBUG(dbgs() << "[Warning] Please optimize `addCanonicalIVRecipes` \n");
}

void LoopTensorizePlanner::printExclusiveLoops() {
  for (auto &KV : ExclusiveLoops) {
    unsigned depth = KV.first;
    const CanonicalizedLoopInfo &Info = KV.second;

    llvm::errs() << "Depth " << depth << ":\n";
    Info.L->dump();
    llvm::errs() << "  Header    : " << Info.Header->getName() << "\n";
    llvm::errs() << "  Preheader : "
                 << (Info.Preheader ? Info.Preheader->getName() : "<none>")
                 << "\n";
    llvm::errs() << "  Latch     : " << Info.Latch->getName() << "\n";

    // (Optional) OwnBody 출력
    llvm::errs() << "  OwnBody (" << Info.OwnBody.size() << " BB): ";
    for (llvm::BasicBlock *BB : Info.OwnBody)
      llvm::errs() << BB->getName() << " ";
    llvm::errs() << "\n";
  }
}

/// L 에 속한 블록 중 하위 루프가 차지하는 블록을 제외하고
/// L 자체가 직접 갖는 블록만 OwnBody 에 넣는다.
void LoopTensorizePlanner::collectOwnBody(
    Loop *L, SmallVectorImpl<BasicBlock *> &OwnBody) {
  // 1) L 이 가지고 있는 모든 BB 를 집합에 넣는다.
  llvm::SmallPtrSet<BasicBlock *, 32> All;
  for (BasicBlock *BB : L->blocks())
    All.insert(BB);

  // 2) 하위 루프가 차지하는 BB 를 전부 지운다.
  std::function<void(Loop *)> eraseInner = [&](Loop *Sub) {
    for (BasicBlock *BB : Sub->blocks())
      All.erase(BB);
    for (Loop *Inner : Sub->getSubLoops())
      eraseInner(Inner);
  };
  for (Loop *Sub : L->getSubLoops())
    eraseInner(Sub);

  errs() << "After eraseInner \n";
  // 3) 남은 블록을 결과에 복사
  OwnBody.append(All.begin(), All.end());
}

// void LoopTensorizePlanner::CloneLoop(CanonicalizedLoopInfo *Info) {
//   ValueToValueMapTy VMap;

//   Function *F = Info->L->getHeader()->getParent();
//   // BasicBlock *NewPreHeader = CloneBasicBlock(Info->Preheader, VMap,
//   ".cloned", F);
//   // BasicBlock *NewHeader = CloneBasicBlock(Info->Header, VMap, ".cloned",
//   F);
//   // BasicBlock *NewLatch = CloneBasicBlock(Info->Latch, VMap, ".cloned", F);
//   // BasicBlock *NewExit = CloneBasicBlock(Info->Exit, VMap, ".cloned", F);
//   // errs() << "Info->Preheader: " << *(Info->Preheader) << "\n";
//   // errs() << "Info->Header: " << *(Info->Header) << "\n";
//   // errs() << "Info->Latch: " << *(Info->Latch) << "\n";
//   // errs() << "Info->Exit: " << *(Info->Exit) << "\n";

//   SmallVector<BasicBlock *, 4> NewBodyBlocks;
//   for (auto *BB : Info->OwnBody) {
//     // Clone Exclusive Loops inside of OwnBody.
//     errs() << "Info->OwnBody: " << *BB << "\n";
//     BasicBlock *Cloned;
//     if (BB == Info->Preheader)
//       Cloned = CloneBasicBlock(BB, VMap, "Preheader.cloned", F);
//     if (BB == Info->Header)
//       Cloned = CloneBasicBlock(BB, VMap, "Header.cloned", F);
//     if (BB == Info->Latch)
//       Cloned = CloneBasicBlock(BB, VMap, "Latch.cloned", F);
//     if (BB == Info->Exit)
//       Cloned = CloneBasicBlock(BB, VMap, "Exit.cloned", F);
//     NewBodyBlocks.push_back(Cloned);
//   }

//   // SSA remap
//   SmallVector<BasicBlock *, 8> AllNewBlocks;
//   errs() << "NewBodyBlocks.size() : " << NewBodyBlocks.size() << "\n";
//   AllNewBlocks.append(NewBodyBlocks.begin(), NewBodyBlocks.end());
//   remapInstructionsInBlocks(AllNewBlocks, VMap);

//   // // -------2. CFG ----------
//   // // Preheader -> NewHeader
//   // BranchInst::Create(NewHeader, NewPreHeader);

//   // // NewLatch -> NewHeader (backedge)
//   // if (Instruction *TI = NewLatch->getTerminator())
//   //   TI->eraseFromParent();
//   // BranchInst::Create(NewHeader, NewLatch);

//   // // NewLatch -> NewExist (exit edge)
//   // //IRBuilder<>(NewLatch).CreateCondBr("", NewHeader, NewExist);

//   // // ---3. LoopInfo update
//   errs() << "Before creating NewLoop!\n";
//   Loop *NewLoop = LI->AllocateLoop();
//   LI->addTopLevelLoop(NewLoop);

//   errs() << "Before creating AllNewBlocks!\n";
//   // You should know, for nested-loop, there could be no loop.body
//   // (not always) except inner-most loop.
//   // Therefore, we might create cloneLoop with Header and Latch, only.
//   // Preheader is not usually included.
//   errs() << "AllNewBlocks.size() : " << AllNewBlocks.size() << "\n";
//   for (auto *BB : AllNewBlocks) {
//     errs() << "AllNewBlocks: " << *BB << "\n";
//     NewLoop->addBasicBlockToLoop(BB, *LI);
//   }
//   errs() << "After creating AllNewBlocks!\n";

//   Info->ClonedeLoop = NewLoop;

//   errs() << "Succesfully generate CloneLoop() \n";

// }

// void LoopTensorizePlanner::attachBlockNumber(BasicBlock *BB, unsigned Num) {
//   LLVMContext &Ctx = BB->getContext();
//   MDNode *MD = MDNode::get(Ctx, MDSTring::get(Ctx, std::to_string(Num)));
//   BB->getTerminator()->setMetadata("block.number", MD);
// }

// void LoopTensorizePlanner::attachClonedBlockNumbers(Loop *OrigLoop,
// ValueToValueMapTy &VMap) {
//   for (auto *BB : OriginalLoop->getBlocks()) {
//     if (auto *ClonedBB = dyn_cast_or_null<BasicBlock>(VMap[BB])) {
//       unsigned Num = OrigBlockNumbers[BB];
//       attachBlockNumber(ClonedBB, Num);
//     }
//   }
// }

// Below `extractSingleLoop` is managed by LI. It is more likely to fall into
// Stack dump! Loop* LoopTensorizePlanner::extractSingleLoop(Loop *OrigLoop,
// LLVMContext &Ctx) {
//   errs() << "extractSingleLoop \n";
//   Loop *NewLoop = LI->AllocateLoop();
//   LI->addTopLevelLoop(NewLoop);

//   BasicBlock *Header = OrigLoop->getHeader();
//   NewLoop->addBasicBlockToLoop(Header, *LI);
//   errs() << "Header \n";

//   ArrayRef<BasicBlock *> LoopBlocks = OrigLoop->getBlocks();

//   for (BasicBlock *BB : LoopBlocks) {
//     if (LI->getLoopFor(BB) == OrigLoop) {
//       errs() << "LI->getLoopFor(BB) == OrigLoop \n";
//       if (BB != Header)
//         NewLoop->addBasicBlockToLoop(BB, *LI);
//     }
//   }

//   errs() << "endend \n";

//   if (BasicBlock *Latch = OrigLoop->getLoopLatch()) {
//     if (OrigLoop->contains(Latch))
//       NewLoop->addBasicBlockToLoop(Latch, *LI);
//   }

//   // Enroll loop as TopLevel on LoopInfo
//   LI->addTopLevelLoop(NewLoop);
//   return NewLoop;
// }

void LoopTensorizePlanner::fillExclusiveLoops() {

  std::function<void(Loop *)> DFS = [&](Loop *L) {
    unsigned Depth = L->getLoopDepth(); // 1‑based depth (LoopDegree)
    errs() << "DFS========\n";
    L->dump();

    // DenseMap::operator[] 로 자동 생성(또는 기존 객체를 반환)한다.
    CanonicalizedLoopInfo &Info = ExclusiveLoops[Depth];
    errs() << "After creating Info\n";

    // 현재 루프와 기본 메타데이터를 저장
    Info.L = L;
    Info.Preheader =
        L->getLoopPreheader(); // LoopSimplifyPass가 실행됐으면 non‑null
    Info.Header = L->getHeader();   // 언제나 non‑null
    Info.Latch = L->getLoopLatch(); // 언제나 non‑null
    Info.Exit = L->getExitBlock();

    errs() << "After getExitBlock Info\n";
    assert(Info.Preheader && Info.Header && Info.Latch && Info.Exit &&
           "LT::Canonical loop required!");

    // (선택) 현재 루프가 직접 소유하는 BB 를 OwnBody 에 저장
    collectOwnBody(L, Info.OwnBody);
    errs() << "After collectOwnBody Info\n";

    // Creating Info.ExclusiveLoop;
    // CloneLoop(&Info);
    // errs() << "After CloneLoop Info\n";
    // Info.ClonedeLoop->dump();
    // errs() << " = = = \n";

    // 자식 루프도 재귀적으로 처리
    // for (Loop *Sub : L->getSubLoops())
    //   DFS(Sub);
  };

  // LoopInfo 에는 최상위 루프만 들어 있기 때문에, 최상위부터 DFS 시작
  for (Loop *Top : Loops)
    DFS(Top);
}

bool LoopTensorizePlanner::ApplyPattern(TPlanPtr &tplan,
                                        TPRecipeBuilder *RecipeBuilder,
                                        TPBasicBlock *TPBB,
                                        bool UseTensorType) {
  // Check if user specify pattern for LoopTensorizer
  // TensorizePattern *rawPtr = TPlanDL.pattern.get();
  // if (isa<TargetAutoPattern>(rawPtr))
  //   return false;

  // FIXME (yg0412.yun)
  // need to fix passing TPlans[0]
  bool success = TPlanDL.pattern->tryToBuildTPlanWithTPRecipes(
      tplan, RecipeBuilder, TPBB, UseTensorType);
  return success;
}

TPlanPtr
LoopTensorizePlanner::tryToBuildTPlanWithTPRecipes(TFRange &Range,
                                                   bool UseTensorType) {
  SmallPtrSet<const InterleaveGroup<Instruction> *, 1> InterleaveGroups;

  auto &Ctx = Loop2PSE.begin()->second->getSE()->getContext();

  auto *I64Ty = Type::getInt64Ty(Ctx);

  TPlanPtr Plan = TPlan::createInitialTPlan(
      createTripCountSCEV(I64Ty, *Loop2PSE.begin()->second->getSE(), Loops),
      *Loop2PSE.begin()->second->getSE(),
      /*RequiresScalarEpilogueCheck=*/false,
      /*CM.foldTailByMasking()=*/false, TPlanDL.pattern);
  
  //YYG:REMOVE
  LLVM_DEBUG(dbgs() << "Initial Plan\n");
  Plan->dump();
  LLVM_DEBUG(dbgs() << "============\n"); 

  DebugLoc DL = DebugLoc();

  addCanonicalIVRecipes(*Plan, I64Ty, /*HasNUW=*/true, DL);
  
  // YYG::REMOVE
  LLVM_DEBUG(dbgs() << "After addCanonicalIVRecipes! \n");
  Plan->dump();

  /// Loops.front() is inner-most loop
  TPRecipeBuilder RecipeBuilder(*Plan, Loops, TLI, Legal, CM,
                                *Loop2PSE[Loops.front()], Builder);

  // TPBasicBlock *HeaderTPBB =
  // Plan->getTensorLoopRegion()->getEntryBasicBlock(); TPBasicBlock *TPBB =
  // HeaderTPBB; BasicBlock *HeaderBB = OrigLoop->getHeader();

  DenseMap<BasicBlock *, Loop *> BB2Loop;
  for (Loop *L : TPlanDL.pattern->Info.Loops) {
    BB2Loop.insert({L->getLoopPreheader(), L});
    BB2Loop.insert({L->getHeader(), L});
    BB2Loop.insert({L->getLoopLatch(), L});
  }

  TPBasicBlock *TPBB = nullptr;
  // SmallVector<BasicBlock *> HeaderBBs;

  // for (auto *L : TPlanDL.pattern->Info.LoopsR)
  //   HeaderBBs.push_back(L->getHeader());

  DenseMap<Loop *, TPValue *> Loop2Induction;

  /// Represents the loop-induction variables
  DenseMap<PHINode *, TPValue *> Induction2TPRecipe;

  // 이미 순회한 BB를 기록할 집합
  SmallPtrSet<BasicBlock *, 8> VisitedBBs;   // 8 은 예상 최대 개수, 필요하면 늘리세요

  const auto &LoopsInfo = TPlanDL.pattern->Info.Loops;
  
  // Below `if` block will define what's the pre-header for each nested-loops.
  // For building inital PreHeaderTPBB, re-named it when it detected.
  // In case of inner-most loop, loop body is header/latch/exiting basic block at the same time.
  // Thus, we skipt to inner-most loop case.
  for (size_t Idx = 0; Idx < LoopsInfo.size(); ++Idx) {
    Loop *CurL = LoopsInfo[Idx];
    CurL->dump();

    LoopBlocksDFS DFS(CurL);
    DFS.perform(LI);
    
    BasicBlock *HeaderBB = CurL->getHeader();
    for (BasicBlock *BB : make_range(DFS.beginRPO(), DFS.endRPO())) {
      //YYG:REMOVE
      errs() << "[tryTOBuild] BB: " << *BB << "\n";

      if (BB == HeaderBB && BB != CurL->getLoopLatch()) {
        /// If there's no tensor-preheader case,
        /// ex. after loop-unrolling, there's no pre-header per header
        /// Just collapsed to inner-most loop in this case.
        if (succ_size(HeaderBB) == 1)
          continue;
        
        /// Otherwise, there must be tensor-preheader case.
        assert(succ_size(HeaderBB) == 2 && "Header must have two successors for outer-loop of nested-loop.");
        
        /// If the headerBB has branch instruction for multiple successors,
        /// the loop is nested-loop and one of them is inner-loop's pre-header.
        /// In that case, we should set TPBB to point tensorPreHeaderTPBB.
        Instruction *Term = BB->getTerminator();
        assert(isa<BranchInst>(Term) && cast<BranchInst>(Term)->isConditional() && "Header of nested-loop must have multiple successors with its conditional branch instruction.");

        for (auto *succs : successors(HeaderBB)) {
          // skip for latch block
          if (succs == CurL->getLoopLatch()) continue;
          if (VisitedBBs.contains(succs))
            continue;          
          
          if (Idx==0) {
            Instruction *SuccsTerm = succs->getTerminator();
            // if it satisfies below conditions, it must jumps to latch/exiting block
            // inside of inner-most loop,
            // then it might be masking by vector predication.
            if (isa<BranchInst>(SuccsTerm) &&
                 cast<BranchInst>(SuccsTerm)->isUnconditional() &&
                 cast<BranchInst>(SuccsTerm)->getSuccessor(0) == CurL->getLoopLatch() )
              succs->setName("tensor.body." + Twine(Idx));
          }
          else {
            
            //YYG:REMOVE
            errs() << "[creating tensor.ph] succs: " << *succs << "\n";
            // To point its inner-loop, we set its level as Idx-1
            succs->setName("tensor.ph" + Twine(Idx-1));
            // YYG:REMOVE
            errs() << "Twine(Idx-1): " << succs->getName() << "\n";
          }
          VisitedBBs.insert(succs);
        }
      }
    }
  }

  VisitedBBs.clear();
  
  // From inner-most loop, 
  // for (size_t Idx = 0; Idx < LoopsInfo.size(); ++Idx) {

  // Stage 1. Get the outer-most Loop
  Loop *OuterMostLoop = LoopsInfo.back();
  assert(OuterMostLoop->getLoopDepth() == 1 && "CurL is not the most-outer loop");

  // Stage 2-1. DFS of Dominator Tree to traverse Dominator Order.
  // for (auto *Node : depth_first(DT->getRootNode())) {
  //   BasicBlock *BB = Node->getBlock();    
  //   // BB 처리
  //   errs() << "Visiting: " << *BB << "\n";
  // }

  // Stage 2-2. DFS the outer-most Loop and traverse each BasicBlock accroding to Reverse-DFS-post-order.
  // For Reducibla-CFG, DFS RPO gaurantees Dominator Order.
  // If A dominates B, then RPO pops out A before B. (Visiting def BB -> use BB)
  // But in most of case, natural nested-loop is born to be irreducible CFG.
  // TODO(yg0412.yun) Change the name of SDFS to DFS.
  LoopBlocksDFS SDFS(OuterMostLoop); 
  SDFS.perform(LI);
  for (BasicBlock *BB : make_range(SDFS.beginRPO(), SDFS.endRPO())) {
    // YYG:REMOVE
    errs() << "[RPODFS] *BB: " << *BB << "\n";

    // inner most-loop that firstly shows BB. 
    // In this case, BB is assigned to that inner most-loop level, not current loop-level.
    Loop *CurL = LI->getLoopFor(BB);
    // YYG::REMOVE
    CurL->dump();
    size_t totalLoopSize = LoopsInfo.size();
    size_t Idx = totalLoopSize - CurL->getLoopDepth();
    // YYG::REMOVE
    errs() << "Idx: " << Idx << "\n";


    // Stage 3. Pick appropriate TPBB for inserting TPRecipe.
    TPIRBasicBlock *HeaderTPBB = Plan->LoopIdx2HeaderTPBB[Idx];
    TPBasicBlock *preHeaderTPBB, *ExitingTPBB;
    if (Idx != 0) {
      preHeaderTPBB = Plan->LoopIdx2PreHeaderTPBB[Idx-1];
      ExitingTPBB = Plan->LoopIdx2ExitingTPBB[Idx-1];
    }
    TPBasicBlock *LatchTPBB = Plan->LoopIdx2LatchTPBB[Idx];

    // Detect the insertion point of each BB. 
    BasicBlock *HeaderBB = CurL->getHeader();
    if (BB == HeaderBB)
      TPBB = HeaderTPBB;
    else if (BB->getName() == ("tensor.ph" + Twine(Idx-1)).str())
      TPBB = preHeaderTPBB;
    else if (BB->getName() == ("tensor.body." + Twine(Idx)).str()) {
      TPBasicBlock *BodyBlock = new TPBasicBlock("tensor.body." + Twine(Idx));

      // We assume TPBB is pointing to HeaderTPBB
      TPBlockUtils::connectBlocks(TPBB, BodyBlock);
      TPBB = BodyBlock;
    }
    else if (BB == CurL->getLoopLatch())
      TPBB = LatchTPBB;
    else
      TPBB = ExitingTPBB;
    // If inner-most loop is only one block which is header, latch, 
    // and exiting at the same time. Then, categorize it as LatchBB.
    if (BB == HeaderBB && BB == CurL->getLoopLatch())
      TPBB = LatchTPBB;
    if (TPBB != LatchTPBB)
      Builder.setInsertPoint(TPBB);
    else {
      TPRecipeBase *BranchOnCount = TPBB->getTerminator();
      Builder.setInsertPoint(TPBB, BranchOnCount->getIterator());
    }


    // Stage 4. Trigger Mask in BB.
    // TODO(yg0412.yun), In here, we don't consider CM.foldTailByMasking(), Yet.
    bool NeedsBlends = BB != HeaderBB && !BB->phis().empty();
    // YYG::REMOVE
    errs() << "NeedsBlends: " << NeedsBlends << "\n";
    bool NeedsMasks = Legal->blockNeedsPredication(BB, CurL) || NeedsBlends;
    // YYG:REMOVE
    errs() << "NeedsMasks: " << NeedsMasks << "\n";
    // Even the TPBB is mapped to LatchTPBB, it can be categorized 
    // as header, too. Thus, checking BB whether than mapped TPBB.
    if (BB == HeaderBB) { //TPBB == HeaderTPBB) {
      errs() << "[OOOOO] RecipeBuilder.createHeaderMask(CurL)\n";
      RecipeBuilder.createHeaderMask(CurL);
    }
    else if (NeedsMasks) {
      errs() << "[XXXXX] RecipeBuilder.createBlockInMask(BB, CurL)\n";
      RecipeBuilder.createBlockInMask(BB, CurL);
    }

    // Introduce each ingredient into TPlan.
    for (Instruction &I : BB->instructionsWithoutDebug(false)) {
      Instruction *Instr = &I;
      SmallVector<TPValue *, 4> Operands;
      auto *Phi = dyn_cast<PHINode>(Instr);

      // YYG:REMOVE
      errs() << "[BUILD TPBB] *BB: " << *BB << "\n";
      errs() << "instr:=== " << *Instr << "\n";
      // FIXME(yg0412.yun) yuxin's work
      // if (Instr->getOpcode() == Instruction::FAdd)
      //   continue;

      // Here, we build the operands of PHI recipes,
      // and add founded phi on TPlanDL.pattern.
      if (Phi) {
        // yyg::remove
        errs() << "Phi: " << *Phi << "\n";

        unsigned NumIncoming = Phi->getNumIncomingValues();
        for (unsigned int i=0; i<NumIncoming; i++) {
          Operands.push_back(Plan->getOrAddLiveIn(
                Phi->getIncomingValueForBlock(Phi->getIncomingBlock(i))));
        }
        TPlanDL.pattern->Info.Loop2PHI[CurL].insert(Phi);
      } else {
        // Non-phi instruction case.
        auto OpRange = RecipeBuilder.mapToTPValues(Instr->operands());
        Operands = {OpRange.begin(), OpRange.end()};
      }
      
      // Invariant stores inside loop will be deleted and a single store
      // with the final reduction value will be added to the exit block
      StoreInst *SI;
      if ((SI = dyn_cast<StoreInst>(&I)) &&
        Legal->isInvariantAddressOfReduction(SI->getPointerOperand(), CurL))
        continue;
      
      
      TPRecipeBase *Recipe;
      // In here, if we meet the branch instruction, we firstly wrap it as replicate VPInstruction.
      // But later, it could be changed into VPBranchOnMaskRecipe as it should be or not.
      // Otherwise, other instructions are widening at first and then if it fails, 
      // it is changed to Replicate VPInstruction. 
      if (isa<BranchInst>(Instr)) {
        // We don't need to specify unconditial branch 
        // because inital TPlan has default successor.
        if (cast<BranchInst>(Instr)->isUnconditional())
          continue;
        Recipe = RecipeBuilder.handleReplication(Instr, Range, CurL);
      }
      else {
        errs() << "[Before] Size of Operands: " << Operands.size() << "\n";
        Recipe = RecipeBuilder.tryToCreateWidenRecipe(Instr, Operands, Range, TPBB, CurL->getLoopDepth());
        if (!Recipe)
          Recipe = RecipeBuilder.handleReplication(Instr, Range, CurL);
      }
      
      int Dim = Idx;
      // Set the LoopDepth on each Recipe (same with PHIRecipe)
      // DimIndex will point to its loop depth.
      Recipe->setDimIndex(Dim);
      RecipeBuilder.setRecipe(Instr, Recipe);

      //if (Instr->getOpcode() == Instruction::PHI) {
      if (Phi) {
        // non-loop-induction variable is also linked 
        // Loop2Induction[CurL] = Recipe->getTPSingleValue();
        TPValue *SingleValue = Recipe->getTPSingleValue();
        if (SingleValue) {
          // yyg;remove
          errs() << "[Phi: " << *Phi << "\n";
          SingleValue->dump();
          TPlanDL.pattern->Info.insertInduction2TPRecipe(Phi, Recipe);
        } else {
          LLVM_DEBUG(dbgs() << "[Warning] Skipping null TPValue insertion for Phi: " << *Phi << "\n");
        }
      }
      
      if (isa<TPHeaderPHIRecipe>(Recipe))
        Recipe->insertBefore(*HeaderTPBB, HeaderTPBB->getFirstNonPhi());
      else if (TPBB == LatchTPBB)
        TPBB->insert(Recipe, TPBB->getTerminator()->getIterator());
      else
        TPBB->appendRecipe(Recipe);
      
      Plan->dump();
    }

  }
  
  RecipeBuilder.fixHeaderPhis();

  // YYG:REMOVE
  errs() << "[Final]\n";

  // LoopBlocksDFS DFS(OrigLoop);
  Loop *TPLoop = TPlanDL.pattern->Info.Loops.back();
  LoopBlocksDFS DFS(TPLoop);
  DFS.perform(LI);
  for (BasicBlock *BB : make_range(DFS.beginRPO(), DFS.endRPO())) {
    bool isHeader = (BB == TPLoop->getHeader());
    bool isLatch = TPLoop->isLoopLatch(BB);
    bool isInLoop = TPLoop->contains(BB);
    bool isBody   = isInLoop && !isHeader && !isLatch;

    bool isExiting = TPLoop->isLoopExiting(BB);
    
    if (isHeader)        errs() << "  HEADER   : " << BB->getName() << "\n";
    else if (isLatch)    errs() << "  LATCH    : " << BB->getName() << "\n";
    else if (isBody)     errs() << "  BODY     : " << BB->getName() << "\n";
    else if (isExiting)  errs() << "  EXITING  : " << BB->getName() << "\n";
    else                 errs() << "  UNKNOWN  : " << BB->getName() << "\n";
  }

  SuccessToApplyPattern =
      ApplyPattern(Plan, &RecipeBuilder, TPBB, UseTensorType);
  LLVM_DEBUG(
      dbgs() << "[Warning] Please handle "
                "`LoopTensorizePlanner::tryToBuildTPlanWithTPRecipes` \n");

  return Plan;
}

// TODO(yuxin.an)
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void LoopTensorizePlanner::printPlans(raw_ostream &O) {
  for (const auto &Plan : TPlans)
    Plan->print(O);
}
#endif

EmissionPolicy LoopTensorizePlanner::buildEmissionPolicy(const TPlan &Plan) {
  // Collect contraction dims from all contraction recipes through TPRecipe.
  // A dim with a dynamic TC is DynamicTiled iff it appears as a ContractDim
  // in at least one contraction recipe.
  // For example, a GEMM with contractDim=0 (K), this produces ContractionDims = {0}.

  // YYG::REMOVE
  errs() << "[buildEmissionPolicy] \n";

  SmallBitVector ContractionDims;
  ReversePostOrderTraversal<TPBlockDeepTraversalWrapper<const TPBlockBase *>>
    RPOT(TPBlockDeepTraversalWrapper<const TPBlockBase *>(Plan.getEntry()));

  // From outer-most loop
  for (const TPBasicBlock *TPB : TPBlockUtils::blocksOnly<const TPBasicBlock>(RPOT)) {
    auto *TPBB = dyn_cast<const TPBasicBlock>(TPB);
    if (!TPBB) continue;

    // From outer -> inner
    for (const TPRecipeBase &Recipe : *TPBB) {
      if (Recipe.getTensorOpKind() != TensorOpKind::Contraction || Recipe.getContractDim() < 0)
        continue;
      unsigned CD = static_cast<unsigned>(Recipe.getContractDim());
      if (CD >= ContractionDims.size())
        ContractionDims.resize(CD + 1, false);
      ContractionDims.set(CD);
    }
  }

  EmissionPolicy Policy;

  for (const auto &[D, CurL] : Plan.LoopIdx2Loop) {
    const SCEV *BTC = Plan.getTripCount()[CurL];
    if (!BTC)
      continue; // Unknown TC — emit inline.
    // YYG::REMOVE
    errs() << "D: " << D << "\n";
    errs() << "[buildEmissionPolicy] CurL: \n";
    CurL->dump();
    errs() << "BTC: " << *BTC << "\n";

    unsigned PF = Plan.getPFForDim(D);
    // YYG::REMOVE
    errs() << "PF: " << PF << "\n";

    // When the `BTC` is static-constant TC, it can dyn_cast to SCEVConstant,
    // Otherwise, if it is dynamic-runtime TC, it should not be casted to SCEVConst.
    // Instead, it is SCEVUnknown.
    if (const auto *C = dyn_cast<SCEVConstant>(BTC)) {
      uint64_t RealTC = C->getValue()->getZExtValue() + 1;
      if (RealTC <= static_cast<uint64_t>(PF))
        continue; // TC fits in one tile — inline.
      Policy.Specs.push_back({D, PF, DimEmitMode::StaticTiled});
    } else {
      // Dynamic TC.
      bool IsContraction = D < ContractionDims.size() && ContractionDims.test(D);
      // YYG::REMOVE
      errs() << "IsContraction: " << IsContraction << "\n";

      if (!IsContraction) {
        // Output dim with runtime TC: use umin-bounded static tiling so the
        // tiling loop remains well-formed regardless of the runtime value.
        if (PF == 0)
          continue;
        Policy.Specs.push_back({D, PF, DimEmitMode::StaticTiled});
      } else {
        // Contraction dim with runtime TC: emit fixed-tile tensor.body loop.
        // Use Plan PF as the tile size; emitContraction() may refine via TTI.
        if (PF == 0)
          continue;
        Policy.Specs.push_back({D, PF, DimEmitMode::DynamicTiled});
      }
    }
  }
  return Policy;
}

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
TensorizedLoopSkeleton LoopTensorizePlanner::createTensorizedLoopSkeleton(Loop *OutermostLoop,
                                                           Value *RuntimeTC,
                                                           unsigned PF,
                                                           LoopInfo &LI,
                                                           DominatorTree &DT,
                                                           ValueToValueMapTy &VMap) {
  TensorizedLoopSkeleton Skel;

  // ---- Precondition checks ------------------------------------------------
  // Validates that the loop has:
  // 1. A unique preheader (single entry block)
  // 2. A single exit block
  // 3. The preheader has exactly one predecessor (OrigPred)
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
  // It creats a scalar fallback copy of the entire loop.
  // The clone's blocks get a .scalar suffix.
  // VMap maps original values to cloned values.
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
  // It fixes up all operand references in the cloned blocks.
  remapInstructionsInBlocks(ClonedBlocks, VMap);

  // The clone's preheader is the VMap image of OrigPreheader.
  BasicBlock *ScalarPreheader = cast<BasicBlock>(VMap[OrigPreheader]);

  // ---- Step 2: Create GuardBB and wire it between OrigPred and OrigPreheader
  // Creates a new tensor.buard basic block and wires it into the DCFG:
  // - Redirects OrigPred's terminator from OrigPreheader -> GuardBB
  // - Fixes PHI nodes in both the original and scalar preheaders to reference GuardBB as their predecessor
  // - Emits the branch: if (TC>=u PF) -> goto tensor path (original loop); 
  //       else -> goto scalar fallback (cloned scalar loop)
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
  // Tells LLVm's dominator tree that GuardBB is dominated by OrigPred, 
  // and both preheaders are now dominated by GuardBB.
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

void LoopTensorizePlanner::executePlan(
    TFTy BestTF, TUFTy BestUF, TPlan &BestTPlan, LoopTensorizer &LT,
    DominatorTree *DT, bool UseTensorType, bool IsEpilogueVectorization,
    const DenseMap<const SCEV *, Value *> *ExpandedSCEVs) {

  // YYG:REMOVE
  errs() << "[executePlan]   \n";

  // TPRecipePatternMatcher_match(BestTPlan, CM, *SE, *LI);

  auto PrintTF = [](raw_ostream &OS, TFTy TF) {
    if (!TF.empty()) {
      OS << "{";
      OS << TF.begin()->second;
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
      OS << UF.begin()->second;
      for (auto UFElem : drop_begin(UF)) {
        OS << ",";
        OS << UFElem.second;
      }
      OS << "}";
    }
  };

  // TODO (yg0412.yun) Need to implement for unroll-loop with mask predication.
  // TPlanTransforms::optimizeForTFAndUF(BestTPlan,
  //                                     *Loop2PSE.begin()->second->getSE());

  LLVM_DEBUG(dbgs() << "Executing best plan with TF=");
  LLVM_DEBUG(PrintTF(dbgs(), BestTF));
  LLVM_DEBUG(dbgs() << ", UF=");
  LLVM_DEBUG(PrintTUF(dbgs(), BestUF));
  LLVM_DEBUG(dbgs() << "\n");

  BestTPlan.setName("Final TPlan");
  LLVM_DEBUG(BestTPlan.dump());

  // // Perform the actual loop transformation.
  TPTransformState State(BestTF, BestUF, LI, DT, LT.Builder, &LT, &BestTPlan,
                         Loops.front()->getHeader()->getContext());
#ifdef EXPENSIVE_CHECKS
  assert(DT->verify(DominatorTree::VerificationLevel::Fast));
#endif

  // Classify every recipe by DimSet patterns.
  TensorOpKindMatcher TensorOpKind(BestTPlan, *SE, *LI);
  // Save the TensorOpKind in TPRecipe.
  TensorOpKind.match();
  // YYG::REMOVE
  errs() << "After match() \n";
  BestTPlan.dump();

  SCEVExpander Expander(*SE, Loops.front()->getHeader()->getModule()->getDataLayout(), "tplan.stride");
  State.SE = SE;
  State.Expander = &Expander;
  State.TTI = &TTI;
  State.DimToLoop = BestTPlan.LoopIdx2Loop;

  // Build the EmissionPolicy upfront - before execute() - so that both the
  // skeleton guard insertion and emitContraction() share the same dim
  // classification derived from TPlan's PF/TC data.
  State.Policy = buildEmissionPolicy(BestTPlan);
  LLVM_DEBUG({
  dbgs() << "LTP::executePlan: EmissionPolicy (" << State.Policy.Specs.size()
          << " specs):\n";
    for (const auto &S : State.Policy.Specs) {
      StringRef Mode = S.Mode == DimEmitMode::Inline       ? "Inline"
                      : S.Mode == DimEmitMode::StaticTiled  ? "StaticTiled"
                                                            : "DynamicTiled";
      dbgs() << "  dim=" << S.Dim << " PF=" << S.PF << " mode=" << Mode << "\n";
    }
  });

  // // If any dim needs a runtime profitability guard, insert the skeleton ONCE
  // // before the outermost tensor loop — here, before execute(), not inside
  // // emitContraction(). Mirrors VPlan's createVectorizedLoopSkeleton() call
  // // site: the guard fires once, not once per M×N iteration.
  // if (State.Policy.needsGuard()) {
  //   // Find the outermost loop in the tensor nest (highest dim index in
  //   // DimToLoop — outermost in DimIdx convention).
  //   // TODO(yg0412.yun) Below finding mechanism can be replaced
  //   // wihtout iterating for-loop.
  //   Loop *OutermostLoop = nullptr;
  //   unsigned MaxDim = 0;
  //   for (const auto &[D, L] : State.DimToLoop) {
  //     if (!OutermostLoop || D > MaxDim) {
  //       MaxDim = D;
  //       OutermostLoop = L;
  //     }
  //   }

  //   // YYG::REMOVE
  //   errs() << "OutermostLoop: \n";
  //   OutermostLoop->dump();
  //   // For each DynamicTiled dim, the PF in the Policy (from Plan.getPFForDim)
  //   // serves as the guard threshold: TC >=u PF means at least one full tile
  //   // can run on the tensor path.
  //   for (const DimEmissionSpec &Spec : State.Policy.Specs) {
  //     if (Spec.Mode != DimEmitMode::DynamicTiled)
  //       continue;

  //     if (!OutermostLoop)
  //       break;
      
  //     BasicBlock *OuterPH   = OutermostLoop->getLoopPreheader();
  //     BasicBlock *OuterPred = OuterPH ? OuterPH->getSinglePredecessor() : nullptr;
  //     errs() << "OuterPH: " << *OuterPH << "\n";
  //     errs() << "OuterPred: " << *OuterPred << "\n";
  //     if (!OuterPred || !OutermostLoop->getExitBlock())
  //       break;

  //     const SCEV *BTCSCEV = BestTPlan.getTCForDim(Spec.Dim);
  //     if (!BTCSCEV)
  //       break;
  //     // YYG::REMOVE
  //     errs() << "Spec.Dim: " << Spec.Dim << ", BTCSCEV: " << *BTCSCEV << "\n";

  //     // Expand the dim's backedge-taken count in OuterPred so it dominates
  //     // the guard block that createTensorizedLoopSkeleton() will insert.
  //     Instruction *ExpandAt = OuterPred->getTerminator();
  //     Value *GuardBTC = Expander.expandCodeFor(
  //         BTCSCEV, Type::getInt64Ty(BTCSCEV->getType()->getContext()), ExpandAt);
  //     IRBuilder<> PredB(ExpandAt);
  //     Value *GuardTC =
  //         PredB.CreateAdd(GuardBTC, PredB.getInt64(1), "tc.guard");

  //     ValueToValueMapTy SkelVMap;
  //     TensorizedLoopSkeleton Skel = createTensorizedLoopSkeleton(
  //         OutermostLoop, GuardTC, Spec.PF, *LI, *DT, SkelVMap);
  //     LLVM_DEBUG({
  //       if (Skel.Valid)
  //         dbgs() << "LTP::executePlan: profitability guard inserted before "
  //                << OutermostLoop->getName() << " (dim=" << Spec.Dim
  //                << " PF=" << Spec.PF << ")\n";
  //       else
  //         dbgs() << "LTP::executePlan: skeleton creation failed for dim "
  //                << Spec.Dim << "; proceeding without guard\n";
  //     });
  //     // Currently only one DynamicTiled dim is supported per lowering.
  //     break;
  //   }
  // }

  // Transform TPlan structure (no IR mutation here).
  // TPlanTransforms inserts TPGuardBlock + TPTilingRegion nodes,
  // sets State.TilingTCVal, and marks IsSubsumed on absorbed recipes.
  TPlanTransforms Transforms(BestTPlan, State.Policy, Expander, Builder, State.DimToLoop);
  Transforms.transform(State);

  MapVector<Loop *, Value *> CanonicalIVStartValue;

  // std::tie(State.CFG.PrevBB, CanonicalIVStartValue) =
  //     LT.createTensorizedLoopSkeleton(ExpandedSCEVs ? *ExpandedSCEVs
  //                                                   : State.ExpandedSCEVs);

  LT.createTensorLoopSkeleton("");
  State.TBS.EntryB = LT.EntryB;
  State.TBS.TPH = LT.LoopTensorPreHeader;
  State.TBS.MiddleB = LT.LoopMiddleBlock;
  State.TBS.SPH = LT.LoopScalarPreHeader;

#ifdef EXPENSIVE_CHECKS
  assert(DT->verify(DominatorTree::VerificationLevel::Fast));
#endif

  // TODO(yuxin.an)

  LT.printDebugTracesAtStart();

  // Compute TensorTripCountV[L] = floor(TC[L] / TF[L]) * TF[L] = TC[L] - (TC[L] % TF[L]).
  // Mirrors VPlan's VectorTripCountV: pre-computed outside prepareToExecute.
  MapVector<Loop *, Value *> TensorTripCountV;
  {
    IRBuilder<> TTCBuilder(State.TBS.TPH->getTerminator());
    for (auto &[L, TC_SCEV] : BestTPlan.getTripCount()) {
      Value *TCVal = tputils::getOrCreateTPValueForSCEVExpr(
          BestTPlan, TC_SCEV, *SE)->getLiveInIRValue();
      unsigned TFVal = State.TF.count(L) ? State.TF[L].getKnownMinValue() : 1;
      Value *TF_IR = ConstantInt::get(TCVal->getType(), TFVal);
      Value *Rem = TTCBuilder.CreateURem(TCVal, TF_IR, "tc.rem");
      Value *TTC = TTCBuilder.CreateSub(TCVal, Rem, "tensor.trip.count");
      TensorTripCountV[L] = TTC;
    }
  }

  //===------------------------------------------------===//
  //
  // Notice: any optimization or new instruction that go
  // into the code below should also be implemented in
  // the cost-model.
  //
  //===------------------------------------------------===//

  // 2. Copy and widen instructions from the old loop into the new loop.

  dbgs() << "------------TPlan\n";
  BestTPlan.dump();
  dbgs() << "------------TPlan\n";

  BestTPlan.prepareToExecute(TensorTripCountV, CanonicalIVStartValue, State);

  BestTPlan.execute(&State);
  // YYG::REMOVE
  errs() << "BestTPlan.execute: \n";
  BestTPlan.dump();

  // 3. Fix the vectorized code: take care of header phi's, live-outs,
  //    predication, updating analyses.
  // LT.fixTensorizedLoop(State, BestTPlan);

  // LT.adaptForTarget(State, UseTensorType);
}

void LoopTensorizer::fixTensorizedLoop(TPTransformState &State, TPlan &Plan) {
  auto GetPhi = [](BasicBlock *BB) { return cast<PHINode>(&BB->front()); };

  for (auto [LIdx2HElem, LIdx2LElem] :
       zip(Plan.LoopIdx2HeaderTPBB, Plan.LoopIdx2LatchTPBB)) {
    auto *HeaderBB = State.TPBB2BB[LIdx2HElem.second];
    auto *LatchBB = State.TPBB2BB[LIdx2LElem.second];
    auto *PHI = GetPhi(HeaderBB);
    auto *IdxAdd = State.IdxAddMap[LatchBB];
    PHI->addIncoming(IdxAdd, LatchBB);
  }
}

void LoopTensorizer::adaptForTarget(TPTransformState &State,
                                    bool UseTensorType) {
  auto *InnermostBB = State.TBS.Loop2HeadBB[Pattern->Loops[0]];
  LLVMContext &Ctx = InnermostBB->getContext();

  auto ConstInt = [&Ctx](unsigned N, int64_t Val) {
    return ConstantInt::get(Type::getIntNTy(Ctx, N), Val);
  };
  auto CI8 = [ConstInt](int64_t Val) { return ConstInt(8, Val); };
  auto CI16 = [ConstInt](int64_t Val) { return ConstInt(16, Val); };
  auto CI32 = [ConstInt](int64_t Val) { return ConstInt(32, Val); };

  auto GetSExtVal = [](Value *Val) {
    return cast<ConstantInt>(Val)->getSExtValue();
  };

  auto GetAddrType = [](Value *Val) {
    // memory space(mlir) / address space(llvm): (DRAM:0, SRAM:1)
    // address type for GAIA: (NoOp:0, DRAM:1, SRAM:2, vFIFO:3)
    const SmallDenseMap<unsigned, unsigned> AddrSpace2AddrType{
        {0, 1}, // DRAM
        {1, 2}, // SRAM
    };

    auto *PtrTy =
        cast<PointerType>(cast<LoadInst>(Val)->getPointerOperandType());
    unsigned AddrSpace = PtrTy->getAddressSpace();
    return AddrSpace2AddrType.lookup(AddrSpace);
  };

  // GAIA-only block — commented out (IntrinsicsGAIA.h / Triple::gaia not available)
  // if (ArchType == Triple::ArchType::gaia) { ... }
}

// std::pair<BasicBlock *, MapVector<Loop *, Value *>>
// LoopTensorizer::createTensorizedLoopSkeleton(
//     const SCEV2ValueTy &ExpandedSCEVs) {
//   /*
//    In this function we generate a new loop. The new loop will contain
//    the vectorized instructions while the old loop will continue to run the
//    scalar remainder.

//        [ ] <-- old preheader - loop iteration number check and SCEVs in Plan's
//      /  |      preheader are expanded here. Eventually all required SCEV
//     /   |      expansion should happen here.
//    /    v
//   |    [ ] <-- vector loop bypass (may consist of multiple blocks).
//   |  /  |
//   | /   v
//   ||   [ ]     <-- vector pre header.
//   |/    |
//   |     v
//   |    [  ] \
//   |    [  ]_|   <-- vector loop (created during VPlan execution).
//   |     |
//   |     v
//   \   -[ ]   <--- middle-block (wrapped in VPIRBasicBlock with the branch to
//    |    |                       successors created during VPlan execution)
//    \/   |
//    /\   v
//    | ->[ ]     <--- new preheader (wrapped in VPIRBasicBlock).
//    |    |
//  (opt)  v      <-- edge from middle to exit iff epilogue is not required.
//    |   [ ] \
//    |   [ ]_|   <-- old scalar loop to handle remainder (scalar epilogue).
//     \   |
//      \  v
//       >[ ]     <-- exit block(s). (wrapped in VPIRBasicBlock)
//    ...
//    */

//   // Create an empty vector loop, and prepare basic blocks for the runtime
//   // checks.
//   // createTensorLoopSkeleton("");

//   // // Now, compare the new count to zero. If it is zero skip the vector loop
//   // // and jump to the scalar loop. This check also covers the case where the
//   // // backedge-taken count is uint##_max: adding one to it will overflow
//   // // leading to an incorrect trip count of zero. In this (rare) case we will
//   // // also jump to the scalar loop.
//   // emitIterationCountCheck(LoopScalarPreHeader);

//   // // Generate the code to check any assumptions that we've made for SCEV
//   // // expressions.
//   // emitSCEVChecks(LoopScalarPreHeader);

//   // // Generate the code that checks in runtime if arrays overlap. We put the
//   // // checks into a separate block to make the more common case of few elements
//   // // faster.
//   // emitMemRuntimeChecks(LoopScalarPreHeader);

//   // // Emit phis for the new starting index of the scalar loop.
//   // createInductionResumeValues(ExpandedSCEVs);

//   // // (maxim.o): Emit code to bypass scalar loops altogether.
//   // emitScalarLoopBypassCode();

//   // MapVector<Loop *, Value *> Res;

//   // // LT has no more getTripCount()
//   // // for (auto Elem : getTripCount())
//   // //   Res.insert({Elem.first, nullptr});

//   // return {LoopTensorPreHeader, Res};
// }

void LoopTensorizer::createTensorLoopSkeleton(StringRef Prefix) {
  // YYG::REMOVE
  errs() << "[createTensorLoopSkeleton]\n";

  auto SplitBB = [&](BasicBlock *Old, Twine BBName) {
    // From Old->getTerminator() instructions moves to a new block.
    // The two blocks are joined by an unconditional branch.
    return SplitBlock(Old, /* splitPt= */ Old->getTerminator(), DT, LI, /* MemorySSAUpdetaer= */ nullptr, BBName /* Before = false*/);
  };

  Loop *OutermostLoop = Pattern->Info.Loops.back();
  // YYG::REMOVE
  errs() << "OutermostLoop: \n";
  OutermostLoop->dump();

  LoopScalarBody = OutermostLoop->getHeader();
  // YYG::REMOVE
  errs() << "LoopScalarBody: " << *LoopScalarBody << "\n";

  EntryB = OutermostLoop->getLoopPreheader();
  // YYG::REMOVE
  errs() << "EntryB: " << *EntryB << "\n";
  assert(EntryB && "Invalid loop structure");

  LoopExitBlock = OutermostLoop->getUniqueExitBlock(); // may be nullptr
  // YYG::REMOVE
  errs() << "LoopExitBlock: " << *LoopExitBlock << "\n";
  // TODO(yuxin.an)
  assert((LoopExitBlock) && "multiple exit loop without required epilogue?");

  LoopTensorPreHeader = SplitBB(EntryB, "tensor.ph"); // 원래 preheader가 뭐지?
  // YYG::REMOVE
  errs() << "LoopTensorPreHeader: " << *LoopTensorPreHeader << "\n";

  LoopMiddleBlock = SplitBB(LoopTensorPreHeader, "middle.block");
  // YYG::REMOVE
  errs() << "LoopMiddleBlock: " << *LoopMiddleBlock << "\n";

  LoopScalarPreHeader = SplitBB(LoopMiddleBlock, "scalar.ph");
  // YYG::REMOVE
  errs() << "LoopScalarPreHeader: " << *LoopScalarPreHeader << "\n";
}

void LoopTensorizer::emitIterationCountCheck(BasicBlock *Bypass) {
  assert(Bypass && "Expected valid bypass basic block.");
  // Value *Count = getTripCount();
  // // Reuse existing vector loop preheader for TC checks.
  // // Note that new preheader block is generated for vector loop.
  // BasicBlock *const TCCheckBlock = LoopVectorPreHeader;
  // IRBuilder<> Builder(TCCheckBlock->getTerminator());

  
}

void LoopTensorizer::emitScalarLoopBypassCode() {
  Loop *OutermostLoop = Pattern->Info.Loops.back();
  LoopExitBlock = OutermostLoop->getUniqueExitBlock(); // may be nullptr
  assert((LoopExitBlock) && "multiple exit loop without required epilogue?");

  BranchInst *Branch = cast<BranchInst>(LoopScalarPreHeader->getTerminator());
  assert(Branch->isUnconditional() &&
         "scalar preheader must have exactly one successor!");

  llvm::Value *TrueCondition =
      llvm::ConstantInt::get(llvm::Type::getInt1Ty(Branch->getContext()), 1);

  IRBuilder<> Builder(Branch);
  // Set branch condition to always True so we effectively skip the whole
  // scalar loop.
  BranchInst *NewBranch = Builder.CreateCondBr(TrueCondition, LoopExitBlock,
                                               Branch->getSuccessor(0));
  Branch->replaceAllUsesWith(NewBranch);
  Branch->eraseFromParent();
}

BasicBlock *LoopTensorizer::emitSCEVChecks(BasicBlock *Bypass) {
  LLVM_DEBUG(dbgs() << "[Warning] Please handle "
                       "`LoopTensorizer::emitSCEVChecks` \n");
  return nullptr;
}

BasicBlock *LoopTensorizer::emitMemRuntimeChecks(BasicBlock *Bypass) {
  if (EnableTPlanNativePath)
    return nullptr;
  llvm_unreachable("");
}

/// Create a new ICmp VPInstruction with predicate \p Pred and operands \p A
/// and \p B.
/// TODO: add createFCmp when needed.
TPValue *TPBuilder::createICmp(CmpInst::Predicate Pred, TPValue *A, TPValue *B,
                    DebugLoc DL, const Twine &Name) {
  // YYG:REMOVE
  errs() << "[TPBuilder::createICmp]\n";
  assert(Pred >= CmpInst::FIRST_ICMP_PREDICATE &&
          Pred <= CmpInst::LAST_ICMP_PREDICATE && "invalid predicate");
  // YYG:REMOVE
  errs() << "[TPBuilder::createICmp]\n";
  return tryInsertInstruction(
      new TPInstruction(Instruction::ICmp, Pred, A, B, DL, Name));
}

/// Return a value for Step multiplied by VF.
Value *createStepForTFElem(IRBuilderBase &B, Type *Ty, ElementCount TFElem,
                           int64_t Step) {
  assert(Ty->isIntegerTy() && "Expected an integer step");
  return B.CreateElementCount(Ty, TFElem.multiplyCoefficientBy(Step));
}

MapVector<Loop *, Value *>
LoopTensorizer::getOrCreateTensorTripCount(BasicBlock *InsertBlock) {
  // YYG:REMOVE
  // errs() << "getOrCreateTensorTripCount\n";

  // if (!TensorTripCount.empty())
  //   return TensorTripCount;

  // MapVector<Loop *, Value *> TC = getTripCount();

  // for (auto TCElem : TC) {
  //   IRBuilder<> Builder(InsertBlock->getTerminator());
  //   Type *Ty = TCElem.second->getType();
  //   Value *Step =
  //       createStepForTFElem(Builder, Ty, TF[TCElem.first], UF[TCElem.first]);

  //   Value *R = Builder.CreateURem(TCElem.second, Step, "n.mod.vf");

  //   // !FIXME(yuxin.an)
  //   if (true) {
  //     auto *IsZero = Builder.CreateICmpEQ(R, ConstantInt::get(R->getType(), 0));
  //     R = Builder.CreateSelect(IsZero, Step, R);
  //   }

  //   auto *Temp = Builder.CreateSub(TCElem.second, R, "n.vec");
  //   TensorTripCount.insert({TCElem.first, Temp});
  // }

  // return TensorTripCount;
  llvm_unreachable("Need to Implement getOrCreateTensorTripCount.");
}

PHINode *LoopTensorizer::createInductionResumeValue(
    PHINode *OrigPhi, const InductionDescriptor &II, Value *Step,
    ArrayRef<BasicBlock *> BypassBlocks,
    std::pair<BasicBlock *, Value *> AdditionalBypass) {
  llvm_unreachable("");
}

void LoopTensorizer::createInductionResumeValues(
    const SCEV2ValueTy &ExpandedSCEVs,
    std::pair<BasicBlock *, Value *> AdditionalBypass) {
  llvm_unreachable("Need to Implement createInductionResumeValues.");
  // getOrCreateTensorTripCount(LoopTensorPreHeader);

  // auto GetLoopIdxPHI = [](Loop *L) {
  //   return cast<PHINode>(&L->getBlocks().front()->front());
  // };

  // if (ArchType == Triple::ArchType::gaia) {
  //   BranchInst *BI = BranchInst::Create(LoopExitBlock);
  //   ReplaceInstWithInst(LoopMiddleBlock->getTerminator(), BI);
  // }

  // int outIdx = Pattern->Info.Loops.size() - 1;
  // int midIdx = Pattern->Info.Loops.size() - 2;
  // Loop *MiddleL = Pattern->Info.Loops[midIdx];
  // Loop *OutermostL = Pattern->Info.Loops[outIdx];

  // PHINode *MiddleLPhi = GetLoopIdxPHI(MiddleL);
  // PHINode *OutermostLPhi = GetLoopIdxPHI(OutermostL);

  // auto *BrBlock = MiddleL->getLoopPreheader();
  // auto *BrBlockFork =
  //     SplitBlock(BrBlock, BrBlock->getTerminator(), DT, LI, nullptr, "br.fork");
  // auto *BrBlockForkBRI = cast<BranchInst>(BrBlockFork->getTerminator());

  // IRBuilder<> BuilderBrBlock(BrBlock->getTerminator());

  // auto *OutermostLIdxICmp =
  //     BuilderBrBlock.CreateICmpULT(OutermostLPhi, TensorTripCount[OutermostL]);

  // BranchInst *BI = BranchInst::Create(BrBlockForkBRI->getSuccessor(0),
  //                                     BrBlockFork, OutermostLIdxICmp);
  // ReplaceInstWithInst(BrBlock->getTerminator(), BI);
  // MiddleLPhi->addIncoming(TensorTripCount[MiddleL], BrBlock);
}

} // namespace llvm
