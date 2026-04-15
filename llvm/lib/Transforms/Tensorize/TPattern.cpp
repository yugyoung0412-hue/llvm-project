#include "llvm/Transforms/Tensorize/TPattern.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Tensorize/LoopTensorize.h"
#include "llvm/Transforms/Tensorize/TPRecipeBuilder.h"
#include "llvm/Transforms/Tensorize/TPlanCFG.h"
#include "llvm/Transforms/Utils/SizeOpts.h"
#include <cassert>

using namespace llvm;
using namespace PatternMatch;

#define LT_NAME "loop-tensorize"
#define DEBUG_TYPE LT_NAME

#ifndef IF_RET_FALSE
#define IF_RET_FALSE(condition)                                                \
  if (condition)                                                               \
    return false;
#endif

namespace llvm {

template <typename Opnd0, typename Opnd1>
inline typename PatternMatch::m_Intrinsic_Ty<Opnd0, Opnd1>::Ty
m_Maximum(const Opnd0 &Op0, const Opnd1 &Op1) {
  return m_Intrinsic<Intrinsic::maximum>(Op0, Op1);
}

bool TargetAutoPattern::isSemanticMatch() { llvm_unreachable(""); }

TargetAutoPattern::~TargetAutoPattern() = default;

// TPBasicBlock *getOuterMostBodyTPBB(TPlanPtr *tplan,
//                                   SmallVector<BasicBlock *> HeaderBBs) {
//   assert(!HeaderBBs.empty() && "Header list must not be empty");

//   // get the HeaderBB for innermost
//   // Fixme (yg0412.yun) need to check this is outer-most loop
//   TPBasicBlock *OutermostHeader = HeaderBBs.back();

//   TPBasicBlock *HeaderTPBB = tplan->get()->HeaderTPBB2Loop[OutermostHeader];
//   assert(HeaderTPBB && "Header must already be in the map");

//   TPBlockBase *Succ = HeaderTPBB->getSingleSuccessor();
//   assert(Succ && "Header must have a single successor (the body block)");
//   return cast<TPBasicBlock>(Succ);
// }

SmallVector<unsigned> getTPValueShape(const TPSingleDefRecipe &V,
                                      const TPlan &Plan) {
  SmallVector<unsigned> Shape;
  for (int D = V.DimSet.find_first(); D >=0; D = V.DimSet.find_next(D))
    Shape.push_back(Plan.getPFForDim(static_cast<unsigned>(D)));
  return Shape;
}

bool TargetAutoPattern::tryToBuildTPlanWithTPRecipes(
    TPlanPtr &tplan, TPRecipeBuilder *RecipeBuilder, TPBasicBlock *TPBB,
    bool UseTensorType) {
  /// FIXME(yg0412.yun) paste all the plain BBs inside of tplan 
  auto TFxUF = tplan->getTFxUF();
  bool SuccessToSetDim = false;
  bool SuccessToSearch = false;
  bool SuccessToSetPF = false;

  // YYG::REMOVE
  errs() << "[TargetAutoPattern::!tryToBuildTPlanWithTPRecipes] \n";

  // ------------------ Stage 1. Set DimSet -----------------------------------//
  SmallVector<TPRecipeBase *, 32> Worklist;

  ReversePostOrderTraversal<TPBlockDeepTraversalWrapper<TPBlockBase *>>
      RPOT(TPBlockDeepTraversalWrapper<TPBlockBase *>(tplan.get()->getEntry()));
  
  // From outer-most loop
  for (TPBasicBlock *TPB : TPBlockUtils::blocksOnly<TPBasicBlock>(RPOT)) {
    auto *TPBB = dyn_cast<TPBasicBlock>(TPB);
    if (!TPBB) continue;

    // From outer -> inner
    for (TPRecipeBase &Recipe : *TPBB) {
      Recipe.dump();
      int Dim = Recipe.getDimIndex();
      // Dim can be unset for CANONICAL-INDUCTION, branch
      if (Dim == -1) continue;
      
      // DimSet is a SmallBitVector - It stores bits indexed from 0 to size-1. 
      // To set bit Dim, the vector must have at least Dim + 1 bits.
      Recipe.DimSet.resize(std::max(Recipe.DimSet.size(), (size_t)(Dim + 1)));
      Recipe.DimSet.set(Dim);
     
      Worklist.push_back(&Recipe);
    }
  }
  
  // ------------------ Stage 2. BFS Union Propagation to Fixpoint -----------------------------------//
  while (!Worklist.empty()) {
    TPRecipeBase *VRecipe = Worklist.pop_back_val();
    
    // YYG::REMOVE
    errs() << "== Def Chain ==";
    VRecipe->dump();
    unsigned NeedSize = VRecipe->DimSet.size();
    errs() << "V's size: " << NeedSize << "\n";

    // TPSingleDefRecipe only has `users()`
    // for (TPUser *U : dyn_cast<TPValue>(VRecipe)->users()) {
    if (auto *SingleDef = dyn_cast<TPSingleDefRecipe>(VRecipe)) {
      for (TPUser *U : SingleDef->users()) {

        // TP-value which defined when TPlan generation.
        // if (dyn_cast<TPLiveOut>(U)) continue;
        // if (U->getTPUserID() == TPUSer::TPUserID::LiveOut) continue;
        
        // TPUser -> TPRecipeBase to get the DimSet.
        TPRecipeBase *URecipe = dyn_cast<TPRecipeBase>(U);
        if (!URecipe) {
          errs() << "Error: DV is not a TPRecipeBase\n";
          continue;
        }

        errs() << "U: ";
        VRecipe->dump();
        errs() << "URecipe: ";
        URecipe->dump();
        errs() << "URecipe's size before: " << URecipe->DimSet.size() << ", NeedSize: " << NeedSize << "\n";

        if (URecipe->DimSet.size() < NeedSize)
          URecipe->DimSet.resize(NeedSize);
        
        SmallBitVector Before = URecipe->DimSet;
        URecipe->DimSet |= VRecipe->DimSet;
        
        errs() << "URecipe's size after: " << URecipe->DimSet.size() << "\n";
        errs() << "DimSet changed: " << (URecipe->DimSet != Before ? "YES" : "NO") << "\n";
        
        if (URecipe->DimSet != Before)
          Worklist.push_back(URecipe);
      }
    } else {
      // For multi-def recipes, currently only interleaved loads.
      // TPValue *ToCheck =
      //     VRecipe->getNumDefinedValues() >= 1 ? VRecipe->getTPValue(0) : VRecipe->getOperand(1);
      // Store: 0개, Load: 1개
      errs() << " VRecipe->getNumDefinedValues() : " << VRecipe->getNumDefinedValues() << "\n";
      for (unsigned i = 0; i < VRecipe->getNumDefinedValues(); ++i) {
        for (TPUser *U : VRecipe->getTPValue(i)->users()) {
          TPRecipeBase *Recipe = nullptr;
          TPSingleDefRecipe *DV = nullptr;

          if ((Recipe = dyn_cast<TPRecipeBase>(U))) {
            DV = Recipe->getDefinedValue(); // 아, 이건 TPSingleDefRecipe를 반환 ㅜ.ㅜ

            if (!DV) continue;
          } else if (dyn_cast<TPLiveOut>(U)) {
            continue;
          } else {
            errs() << "warning: Unknown TPUser type, skipping.\n";
            continue;
          }

          auto *DVRecipe = dyn_cast<TPRecipeBase>(DV);
          if (!DVRecipe) {
            errs() << "Error: DV is not a TPRecipeBase!\n";
            continue;
          }

          errs() << "[Else] U: ";
          Recipe->dump();
          errs() << "DV: ";
          DVRecipe->dump();
          // YYG::REMOVE
          errs() << "DVRecipe's DimSet: " ;
          for (int D = DVRecipe->DimSet.find_first(); D >= 0; D = DVRecipe->DimSet.find_next(D))
            errs() << D << "\n";
          errs() << "DV's size before: " << DVRecipe->DimSet.size() << ", NeedSize: " << NeedSize << "\n";

          if (DVRecipe->DimSet.size() < NeedSize)
            DVRecipe->DimSet.resize(NeedSize);

          SmallBitVector Before = DVRecipe->DimSet;
          DVRecipe->DimSet |= VRecipe->DimSet;

          errs() << "DVRecipe's size after: " << DVRecipe->DimSet.size() << "\n";
          errs() << "DimSet changed: " << (DVRecipe->DimSet != Before ? "YES" : "NO") << "\n";

          if (DVRecipe->DimSet != Before)
            Worklist.push_back(DVRecipe);
        }
      }
      errs() << "Skipping: TPRecipe is not TPSingleDefRecipe! \n";
    }
  }
  // Currently, DimSet has been propagated to all TPRecipes.
  SuccessToSetDim = true;

  // -------------------- Stage 3. Search Logic --------------------------------------------- //
  // - search for the best parallel factor (PF) for each TPRecipe.
  // 필수 조건: Search Process (Algorithm), Cost Model
  // -------------------- TODO(yg0412.yun) -------------------------------------------------- //
  SuccessToSearch = true;

  // -------------------- Stage 4. Applying PF to loop-induction variable ------------------- //
  // - set parallel factor (PF) to each Dim, same effect with
  // setting PF to each loop induction-variables.
  // `Dim` represents the index of loop-induction variable. 
  for (unsigned dim = 0; dim < tplan.get()->getDepth(); dim++) {

    // Only set the PF to DimPF. Thus, PF is not printed after this.
    // You can check applied PF when lowering them (ex. execute()). 
    // TODO(yg.yun) set PF through TTI projection.
    tplan.get()->setDimPF(dim, /* (a.k.a tile_size as)PF=*/ 256 * (dim+1));
  }
  SuccessToSetPF = true;

  // YYG::REMOVE
  errs() << "After BFS propagation! \n";
  tplan->dump();

  return SuccessToSetDim && SuccessToSearch && SuccessToSetPF;
}

bool GEMMPattern::tryToBuildTPlanWithTPRecipes(TPlanPtr &tplan,
                                               TPRecipeBuilder *RecipeBuilder,
                                               TPBasicBlock *TPBB,
                                               bool UseTensorType) {
  // TPRecipeBuilder RecipeBuilder(*tplan, Loops.front(), TLI, Legal, CM,
  //                             *Loop2PSE[Loops.front()], Builder);

  Loop *OutermostL = Info.Loops[2];
  Loop *MiddleL = Info.Loops[1];
  Loop *InnermostL = Info.Loops[0];

  auto *OutermostPhi = Info.Loop2PHI_tmp[OutermostL];
  auto *MiddlePhi = Info.Loop2PHI_tmp[MiddleL];

  // TPBB: InnerMostBody after building initial TPlan
  // SmallVector<BasicBlock *> HeaderBBs;

  // for (auto *L : TPlanDL.pattern->Info.LoopsR)
  //   HeaderBBs.push_back(L->getHeader());
  // // FIXME (yg0412.yun) getInnerMostBodyTPBB->getOuterMostBodyTPBB
  // TPBasicBlock *TPBB = getOuterMostBodyTPBB(&tplan, HeaderBBs);

  auto TFxUF = tplan->getTFxUF();
  bool success;

  TPRecipeBase *AIdxRecipe = new TPNewInstrRecipe(
      Instruction::Mul,
      {RecipeBuilder->mapToTPValue(OutermostPhi), TFxUF[InnermostL]});
  TPBB->appendRecipe(AIdxRecipe);

  TPRecipeBase *CIdxRecipeSub = new TPNewInstrRecipe(
      Instruction::Mul, {RecipeBuilder->mapToTPValue(OutermostPhi),
                         tplan->getOrCreateBackedgeTakenCount()[MiddleL]});
  TPBB->appendRecipe(CIdxRecipeSub);

  TPRecipeBase *CIdxRecipe = new TPNewInstrRecipe(
      Instruction::Add, {CIdxRecipeSub->getTPSingleValue(),
                         RecipeBuilder->mapToTPValue(MiddlePhi)});
  TPBB->appendRecipe(CIdxRecipe);

  DenseMap<GetElementPtrInst *, TPValue *> GEPIdx{
      {GEMMInfo.GEPs[0], AIdxRecipe->getTPSingleValue()}, // 해당 연산의 결과
      {GEMMInfo.GEPs[1], RecipeBuilder->mapToTPValue(MiddlePhi)},
      {GEMMInfo.GEPs[2], CIdxRecipe->getTPSingleValue()},
  };

  SmallVector<TPValue *> MatrixLoads;
  TPValue *Res, *CurGEP;

  if (UseTensorType) {
    for (Instruction &I : drop_end(
             InnermostL->getLoopLatch()->instructionsWithoutDebug(false))) {
      Instruction *Instr = &I;
      if (auto *GEPI = dyn_cast<GetElementPtrInst>(Instr)) {

        if (!GEPIdx.count(GEPI))
          continue;

        auto *GEPRecipe = new TPNewInstrRecipe(
            Instruction::GetElementPtr,
            {tplan->getOrAddLiveIn(Instr->getOperand(0)), GEPIdx[GEPI]});
        RecipeBuilder->setRecipe(Instr, GEPRecipe);
        TPBB->appendRecipe(GEPRecipe);
        CurGEP = GEPRecipe->getTPSingleValue();
      } else if (auto *LoadI = dyn_cast<LoadInst>(Instr)) {
        Intrinsic::ID ID = Intrinsic::tensor_new_load;
        SmallVector<TPValue *> Operands;

        if (LoadI == GEMMInfo.LoadA) {
          Operands.append({CurGEP,
                           tplan->getOrCreateBackedgeTakenCount()[InnermostL],
                           TFxUF[OutermostL], TFxUF[InnermostL]});
        } else if (LoadI == GEMMInfo.LoadB) {
          Operands.append({CurGEP,
                           tplan->getOrCreateBackedgeTakenCount()[MiddleL],
                           TFxUF[InnermostL], TFxUF[MiddleL]});
        } else {
          continue;
        }

        auto *MatrixLoadRecipe = new TPMatrixCallRecipe(
            Instr, make_range(Operands.begin(), Operands.end()), ID,
            Instr->getDebugLoc());
        RecipeBuilder->setRecipe(Instr, MatrixLoadRecipe);
        TPBB->appendRecipe(MatrixLoadRecipe);
        MatrixLoads.push_back(MatrixLoadRecipe->getTPSingleValue());
      } else if (auto *StoreI = dyn_cast<StoreInst>(Instr)) {
        Intrinsic::ID ID = Intrinsic::tensor_new_store;
        SmallVector<TPValue *, 4> Operands{
            Res, CurGEP, tplan->getOrCreateBackedgeTakenCount()[MiddleL],
            TFxUF[MiddleL], TFxUF[OutermostL]};
        auto *StoreRecipe = new TPMatrixCallRecipe(
            Instr, make_range(Operands.begin(), Operands.end()), ID,
            Instr->getDebugLoc());
        RecipeBuilder->setRecipe(Instr, StoreRecipe);
        TPBB->appendRecipe(StoreRecipe);
      } else if (Instr->getOpcode() == Instruction::FMul) {
        Intrinsic::ID ID = Intrinsic::tensor_multiply;
        SmallVector<TPValue *, 4> Operands, Operands2;

        dbgs() << "MatrixLoads: " << MatrixLoads.size() << "\n";

        Operands.append({MatrixLoads[0], MatrixLoads[1], TFxUF[OutermostL],
                         TFxUF[InnermostL], TFxUF[MiddleL]});
        auto *MMRecipe = new TPMatrixCallRecipe(
            Instr, make_range(Operands.begin(), Operands.end()), ID,
            Instr->getDebugLoc());
        RecipeBuilder->setRecipe(Instr, MMRecipe);
        TPBB->appendRecipe(MMRecipe);
        Res = MMRecipe->getTPSingleValue();
      }
    }
    success = true;
  } else {
    for (Instruction &I : drop_end(
             InnermostL->getLoopLatch()->instructionsWithoutDebug(false))) {
      Instruction *Instr = &I;
      if (auto *GEPI = dyn_cast<GetElementPtrInst>(Instr)) {

        if (!GEPIdx.count(GEPI))
          continue;

        auto *GEPRecipe = new TPNewInstrRecipe(
            Instruction::GetElementPtr,
            {tplan->getOrAddLiveIn(Instr->getOperand(0)), GEPIdx[GEPI]});
        RecipeBuilder->setRecipe(Instr, GEPRecipe);
        TPBB->appendRecipe(GEPRecipe);
        CurGEP = GEPRecipe->getTPSingleValue();
      } else if (auto *LoadI = dyn_cast<LoadInst>(Instr)) {
        Intrinsic::ID ID = Intrinsic::matrix_column_major_load_addr_space_ext;
        SmallVector<TPValue *> Operands;

        if (LoadI == GEMMInfo.LoadA) {
          Operands.append({CurGEP,
                           tplan->getOrCreateBackedgeTakenCount()[InnermostL],
                           TFxUF[InnermostL], TFxUF[OutermostL]});
        } else if (LoadI == GEMMInfo.LoadB) {
          Operands.append({CurGEP,
                           tplan->getOrCreateBackedgeTakenCount()[MiddleL],
                           TFxUF[MiddleL], TFxUF[InnermostL]});
        } else {
          continue;
        }

        auto *MatrixLoadRecipe = new TPMatrixCallRecipe(
            Instr, make_range(Operands.begin(), Operands.end()), ID,
            Instr->getDebugLoc());
        RecipeBuilder->setRecipe(Instr, MatrixLoadRecipe);
        TPBB->appendRecipe(MatrixLoadRecipe);

        Operands.clear();

        if (LoadI == GEMMInfo.LoadA)
          Operands.append({MatrixLoadRecipe->getTPSingleValue(),
                           TFxUF[OutermostL], TFxUF[InnermostL]});
        else if (LoadI == GEMMInfo.LoadB)
          Operands.append({MatrixLoadRecipe->getTPSingleValue(),
                           TFxUF[InnermostL], TFxUF[MiddleL]});
        else
          continue;

        auto *TransposeRecipe = new TPMatrixCallRecipe(
            Instr, make_range(Operands.begin(), Operands.end()),
            Intrinsic::matrix_transpose, Instr->getDebugLoc());
        TPBB->appendRecipe(TransposeRecipe);
        MatrixLoads.push_back(TransposeRecipe->getTPSingleValue());
      } else if (auto *StoreI = dyn_cast<StoreInst>(Instr)) {
        Intrinsic::ID ID = Intrinsic::matrix_column_major_store_addr_space_ext;
        SmallVector<TPValue *, 4> Operands{
            Res, CurGEP, tplan->getOrCreateBackedgeTakenCount()[MiddleL],
            TFxUF[MiddleL], TFxUF[OutermostL]};
        auto *StoreRecipe = new TPMatrixCallRecipe(
            Instr, make_range(Operands.begin(), Operands.end()), ID,
            Instr->getDebugLoc());
        RecipeBuilder->setRecipe(Instr, StoreRecipe);
        TPBB->appendRecipe(StoreRecipe);
      } else if (Instr->getOpcode() == Instruction::FMul) {
        Intrinsic::ID ID = Intrinsic::matrix_multiply;
        SmallVector<TPValue *, 4> Operands, Operands2;

        dbgs() << "MatrixLoads: " << MatrixLoads.size() << "\n";

        Operands.append({MatrixLoads[0], MatrixLoads[1], TFxUF[OutermostL],
                         TFxUF[InnermostL], TFxUF[MiddleL]});
        auto *MMRecipe = new TPMatrixCallRecipe(
            Instr, make_range(Operands.begin(), Operands.end()), ID,
            Instr->getDebugLoc());
        RecipeBuilder->setRecipe(Instr, MMRecipe);
        TPBB->appendRecipe(MMRecipe);

        Operands2.append(
            {MMRecipe->getTPSingleValue(), TFxUF[OutermostL], TFxUF[MiddleL]});

        auto *TransposeRecipe = new TPMatrixCallRecipe(
            Instr, make_range(Operands2.begin(), Operands2.end()),
            Intrinsic::matrix_transpose, Instr->getDebugLoc());
        TPBB->appendRecipe(TransposeRecipe);

        Res = TransposeRecipe->getTPSingleValue();
      }
    }
    success = true;
  }

  return success;
}

bool GEMMPattern::isSemanticMatch() {
  IF_RET_FALSE(!Status.CanTensorize)

  BasicBlock *InnermostLatch = Loops.front()->getLoopLatch();
  IF_RET_FALSE(!InnermostLatch)

  auto GetPHI = [](Value *LHS, Value *RHS) {
    return isa<PHINode>(LHS) ? cast<PHINode>(LHS) : cast<PHINode>(RHS);
  };

  auto GetPHIFromLoop = [](Loop *L) {
    return cast<PHINode>(&L->getBlocks().front()->front());
  };

  bool Res = false;

  for (auto &I : *InnermostLatch) {
    Value *Vals[30]; // Just for temporary variables

    if ( // Depth 0:
        match(&I, m_Store(/*ValueOp=*/m_Value(Vals[0]),
                          /*PointerOp=*/m_Value(Vals[1]))) &&
        // Depth 1:
        match(Vals[0], m_FAdd(m_Value(Vals[2]), m_Value(Vals[3]))) &&
        match(Vals[1], m_GEP(/*PointerOp=*/m_Value(Vals[4]),
                             /*IndexOp=*/m_Value(Vals[5]))) &&
        // Depth 2:
        (match(Vals[2], m_FMul(m_Value(Vals[6]), m_Value(Vals[7]))) ||
         match(Vals[3], m_FMul(m_Value(Vals[6]), m_Value(Vals[7])))) &&
        (match(Vals[2], m_Load(/*PointerOp=*/m_Value(Vals[8]))) ||
         match(Vals[3], m_Load(/*PointerOp=*/m_Value(Vals[8])))) &&
        // Depth 3:
        match(Vals[6], m_Load(/*PointerOp=*/m_Value(Vals[9]))) &&
        match(Vals[7], m_Load(/*PointerOp=*/m_Value(Vals[10]))) &&
        match(Vals[8], m_GEP(/*PointerOp=*/m_Value(Vals[11]),
                             /*IndexOp=*/m_Value(Vals[12]))) &&
        // Depth 4:
        match(Vals[9], m_GEP(/*PointerOp=*/m_Value(Vals[13]),
                             /*IndexOp=*/m_Value(Vals[14]))) &&
        match(Vals[10], m_GEP(/*PointerOp=*/m_Value(Vals[15]),
                              /*IndexOp=*/m_Value(Vals[16]))) &&
        // Value Check:
        (Vals[4] == Vals[11] && Vals[4] != Vals[13] && Vals[4] != Vals[15]) &&
        // Index Check for Matrix C:
        match(Vals[5], m_Add(m_Value(Vals[17]), m_Value(Vals[18]))) &&
        (match(Vals[17], m_Mul(m_Value(Vals[19]), m_Value(Vals[20]))) ||
         match(Vals[18], m_Mul(m_Value(Vals[19]), m_Value(Vals[20])))) &&
        // Index Check for Matrix A:
        match(Vals[14], m_Add(m_Value(Vals[21]), m_Value(Vals[22]))) &&
        (match(Vals[21], m_Mul(m_Value(Vals[23]), m_Value(Vals[24]))) ||
         match(Vals[22], m_Mul(m_Value(Vals[23]), m_Value(Vals[24])))) &&
        // Index Check for Matrix B:
        match(Vals[16], m_Add(m_Value(Vals[25]), m_Value(Vals[26]))) &&
        (match(Vals[25], m_Mul(m_Value(Vals[27]), m_Value(Vals[28]))) ||
         match(Vals[26], m_Mul(m_Value(Vals[27]), m_Value(Vals[28]))))) {
      auto *FAddInstr = cast<Instruction>(Vals[0]);
      Info.ElementTy = FAddInstr->getType();
      Res = true;

      GEMMInfo.GEPs.append({cast<GetElementPtrInst>(Vals[9]),
                            cast<GetElementPtrInst>(Vals[10]),
                            cast<GetElementPtrInst>(Vals[1])});

      GEMMInfo.LoadA = cast<LoadInst>(Vals[6]);
      GEMMInfo.LoadB = cast<LoadInst>(Vals[7]);

      // PHIs
      PHINode *InnermostPHI = GetPHIFromLoop(Loops[0]);
      PHINode *MiddlePHI = GetPHIFromLoop(Loops[1]);
      PHINode *OutermostPHI = GetPHIFromLoop(Loops[2]);

      // Transpose Info for matrix A
      PHINode *MatrixAPHI0 = GetPHI(Vals[23], Vals[24]);
      PHINode *MatrixAPHI1 = GetPHI(Vals[21], Vals[22]);
      if (MatrixAPHI0 == OutermostPHI && MatrixAPHI1 == InnermostPHI)
        GEMMInfo.TransposeA = false;
      else if (MatrixAPHI0 == InnermostPHI && MatrixAPHI1 == OutermostPHI)
        GEMMInfo.TransposeA = true;
      else
        return false;

      // Transpose Info for matrix B
      PHINode *MatrixBPHI0 = GetPHI(Vals[27], Vals[28]);
      PHINode *MatrixBPHI1 = GetPHI(Vals[25], Vals[26]);
      if (MatrixBPHI0 == InnermostPHI && MatrixBPHI1 == MiddlePHI)
        GEMMInfo.TransposeB = false;
      else if (MatrixBPHI0 == MiddlePHI && MatrixBPHI1 == InnermostPHI)
        GEMMInfo.TransposeB = true;
      else
        return false;
    }
  }
  return Res;
}

bool ElementWiseTensorizePattern::tryToBuildTPlanWithTPRecipes(
    TPlanPtr &tplan, TPRecipeBuilder *RecipeBuilder, TPBasicBlock *TPBB,
    bool UseTensorType) {
  // TPRecipeBuilder RecipeBuilder(*tplan, Loops.front(), TLI, Legal, CM,
  //                             *Loop2PSE[Loops.front()], Builder);

  Loop *OutermostL = Info.Loops[1];
  Loop *InnermostL = Info.Loops[0];

  auto *OutermostPhi = Info.Loop2PHI_tmp[OutermostL];
  auto *InnermostPhi = Info.Loop2PHI_tmp[InnermostL];

  auto TFxUF = tplan->getTFxUF();
  bool success;

  TPRecipeBase *AIdxRecipe = new TPNewInstrRecipe(
      Instruction::Mul,
      {RecipeBuilder->mapToTPValue(OutermostPhi), TFxUF[InnermostL]});
  TPBB->appendRecipe(AIdxRecipe);

  TPRecipeBase *CIdxRecipeSub = new TPNewInstrRecipe(
      Instruction::Mul, {RecipeBuilder->mapToTPValue(OutermostPhi),
                         tplan->getOrCreateBackedgeTakenCount()[InnermostL]});
  TPBB->appendRecipe(CIdxRecipeSub);

  TPRecipeBase *CIdxRecipe = new TPNewInstrRecipe(
      Instruction::Add, {CIdxRecipeSub->getTPSingleValue(),
                         RecipeBuilder->mapToTPValue(InnermostPhi)});
  TPBB->appendRecipe(CIdxRecipe);

  DenseMap<GetElementPtrInst *, TPValue *> GEPIdx{
      {eleWiseInfo.GEPs[0], AIdxRecipe->getTPSingleValue()},
      {eleWiseInfo.GEPs[1], CIdxRecipe->getTPSingleValue()},
  };

  SmallVector<TPValue *> MatrixLoads;
  TPValue *Res, *CurGEP;

  if (UseTensorType) {
    for (Instruction &I : drop_end(
             InnermostL->getLoopLatch()->instructionsWithoutDebug(false))) {
      Instruction *Instr = &I;
      if (auto *GEPI = dyn_cast<GetElementPtrInst>(Instr)) {

        if (!GEPIdx.count(GEPI))
          continue;

        auto *GEPRecipe = new TPNewInstrRecipe(
            Instruction::GetElementPtr,
            {tplan->getOrAddLiveIn(Instr->getOperand(0)), GEPIdx[GEPI]});
        RecipeBuilder->setRecipe(Instr, GEPRecipe);
        TPBB->appendRecipe(GEPRecipe);
        CurGEP = GEPRecipe->getTPSingleValue();
      } else if (auto *LoadI = dyn_cast<LoadInst>(Instr)) {
        Intrinsic::ID ID = Intrinsic::tensor_new_load;
        SmallVector<TPValue *> Operands;

        if (LoadI == eleWiseInfo.LoadA) {
          Operands.append({CurGEP,
                           tplan->getOrCreateBackedgeTakenCount()[InnermostL],
                           TFxUF[OutermostL], TFxUF[InnermostL]});
        } else if (LoadI == eleWiseInfo.LoadB) {
          Operands.append({CurGEP,
                           tplan->getOrCreateBackedgeTakenCount()[InnermostL],
                           TFxUF[OutermostL], TFxUF[InnermostL]});
        } else {
          continue;
        }

        auto *MatrixLoadRecipe = new TPMatrixCallRecipe(
            Instr, make_range(Operands.begin(), Operands.end()), ID,
            Instr->getDebugLoc());
        RecipeBuilder->setRecipe(Instr, MatrixLoadRecipe);
        TPBB->appendRecipe(MatrixLoadRecipe);
        MatrixLoads.push_back(MatrixLoadRecipe->getTPSingleValue());
      } else if (auto *StoreI = dyn_cast<StoreInst>(Instr)) {
        Intrinsic::ID ID = Intrinsic::tensor_new_store;
        SmallVector<TPValue *, 4> Operands{
            Res, CurGEP, tplan->getOrCreateBackedgeTakenCount()[OutermostL],
            TFxUF[OutermostL], TFxUF[InnermostL]};
        auto *StoreRecipe = new TPMatrixCallRecipe(
            Instr, make_range(Operands.begin(), Operands.end()), ID,
            Instr->getDebugLoc());
        RecipeBuilder->setRecipe(Instr, StoreRecipe);
        TPBB->appendRecipe(StoreRecipe);
      } //----------instruction------------------------------------
      else if (Instr->getOpcode() == Instruction::FAdd ||
               Instr->getOpcode() == Instruction::FSub ||
               Instr->getOpcode() == Instruction::FMul) {
        if (MatrixLoads.size() < 2)
          continue;
        Intrinsic::ID ID = Intrinsic::not_intrinsic;
        switch (Instr->getOpcode()) {
        case Instruction::FAdd:
          ID = Intrinsic::tensor_add;
          break;
        case Instruction::FSub:
          ID = Intrinsic::tensor_sub;
          break;
        case Instruction::FMul:
          ID = Intrinsic::tensor_mul;
          break;
        default:
          llvm_unreachable("Unsupported floating‑point opcode");
        }
        SmallVector<TPValue *, 4> Operands, Operands2;
        dbgs() << "MatrixLoads: " << MatrixLoads.size() << "\n";
        Operands.append({MatrixLoads[0], MatrixLoads[1], TFxUF[OutermostL],
                         TFxUF[InnermostL]});
        auto *MMRecipe = new TPMatrixCallRecipe(
            Instr, make_range(Operands.begin(), Operands.end()), ID,
            Instr->getDebugLoc());
        RecipeBuilder->setRecipe(Instr, MMRecipe);
        TPBB->appendRecipe(MMRecipe);
        Res = MMRecipe->getTPSingleValue();
      } //-----------intrinsic----------------
      else if (auto *CI = dyn_cast<CallInst>(&I)) {
        Intrinsic::ID ID;
        switch (CI->getIntrinsicID()) {
        case Intrinsic::maximum:
          ID = Intrinsic::tensor_maximum;
          break;
        case Intrinsic::sqrt:
          ID = Intrinsic::tensor_sqrt;
          break;
        case Intrinsic::fabs:
          ID = Intrinsic::tensor_abs;
          break;
        default:
          llvm_unreachable("Unsupported Intrinsics!");
          break;
        }
        SmallVector<TPValue *, 4> Operands, Operands2;
        dbgs() << "MatrixLoads: " << MatrixLoads.size() << "\n";
        if (CI->getIntrinsicID() == Intrinsic::maximum) {
          Operands.append({MatrixLoads[0], MatrixLoads[1], TFxUF[OutermostL],
                           TFxUF[InnermostL]});
        } else if (CI->getIntrinsicID() == Intrinsic::sqrt ||
                   CI->getIntrinsicID() == Intrinsic::fabs) {
          Operands.append(
              {MatrixLoads[0], TFxUF[OutermostL], TFxUF[InnermostL]});
        }

        auto *MMRecipe = new TPMatrixCallRecipe(
            Instr, make_range(Operands.begin(), Operands.end()), ID,
            Instr->getDebugLoc());
        RecipeBuilder->setRecipe(Instr, MMRecipe);
        TPBB->appendRecipe(MMRecipe);
        Res = MMRecipe->getTPSingleValue();
      }
    }
    success = true;
  }
  return success;
}

bool ElementWiseTensorizePattern::isSemanticMatch() {
  IF_RET_FALSE(!Status.CanTensorize)

  BasicBlock *InnermostLatch = Loops.front()->getLoopLatch();
  IF_RET_FALSE(!InnermostLatch)

  auto GetPHI = [](Value *LHS, Value *RHS) {
    return isa<PHINode>(LHS) ? cast<PHINode>(LHS) : cast<PHINode>(RHS);
  };

  auto GetPHIFromLoop = [](Loop *L) {
    return cast<PHINode>(&L->getBlocks().front()->front());
  };

  bool Res = false;

  for (auto &I : *InnermostLatch) {
    Instruction *CoreInst = nullptr;
    Value *Vals[10] = {nullptr};
    ConstantInt *ShiftOffset = nullptr;
    llvm::errs() << I << "\n";

    ElementWisePatternInfo tmpInfo;
    tmpInfo.LoadA = nullptr;
    tmpInfo.LoadB = nullptr;
    tmpInfo.GEPs.clear();

    // -----------------------------------------------------------------
    // ① match Store( GEP(...), BinOp(...) )
    // -----------------------------------------------------------------
    if (!match(&I, m_Store(m_Instruction(CoreInst), m_Value(Vals[0]))))
      continue;

    // -----------------------------------------------------------------
    // ② match「Element‑Wise」 (Add / Sub / Mul / Div / Max)
    // -----------------------------------------------------------------
    bool IsElemWiseOp =
        match(CoreInst, m_FAdd(m_Value(Vals[2]), m_Value(Vals[3]))) ||
        match(CoreInst, m_FSub(m_Value(Vals[2]), m_Value(Vals[3]))) ||
        match(CoreInst, m_FMul(m_Value(Vals[2]), m_Value(Vals[3]))) ||
        match(CoreInst, m_FDiv(m_Value(Vals[2]), m_Value(Vals[3]))) ||
        match(CoreInst, m_Maximum(m_Value(Vals[2]), m_Value(Vals[3])));

    bool UnaryOp = (match(Vals[0], m_GEP(m_Value(Vals[1]), m_Value())) &&
                    (match(CoreInst, m_Sqrt(m_Value(Vals[2]))) ||
                     match(CoreInst, m_FAbs(m_Value(Vals[2])))) &&
                    (match(Vals[2], m_Load(m_Value(Vals[3]))) &&
                     match(Vals[3], m_GEP(m_Value(Vals[4]), m_Value()))) &&
                    // Value Check
                    Vals[1] != Vals[4]);

    if (IsElemWiseOp) {
      // -----------------------------------------------------------------
      // ③ match LoadA / LoadB and GEP
      // -----------------------------------------------------------------
      if (match(Vals[0], m_GEP(m_Value(Vals[1]), m_Value())) && // Store GEP
          match(Vals[2], m_Load(m_Value(Vals[4]))) && // First operand Load
          match(Vals[3], m_Load(m_Value(Vals[5]))) && // Second operand Load
          match(Vals[4], m_GEP(m_Value(Vals[6]), m_Value())) && // LoadA GEP
          match(Vals[5], m_GEP(m_Value(Vals[7]), m_Value()))) { // LoadB GEP

        // -------------------------------------------------------------
        // 4) write info into temp data structure
        // -------------------------------------------------------------
        tmpInfo.LoadA = dyn_cast<LoadInst>(Vals[2]);
        tmpInfo.LoadB = dyn_cast<LoadInst>(Vals[3]);
        llvm::errs() << *tmpInfo.LoadA << "  -->  "
                     << *tmpInfo.LoadA->getPointerOperand() << "\n";
        llvm::errs() << *tmpInfo.LoadB << "  -->  "
                     << *tmpInfo.LoadB->getPointerOperand() << "\n";

        // GEP[0] = Store（Matrix C）
        // GEP[1] = A GEP，GEP[2] = B GEP
        tmpInfo.GEPs.push_back(dyn_cast<GetElementPtrInst>(Vals[0])); // C
        tmpInfo.GEPs.push_back(dyn_cast<GetElementPtrInst>(Vals[4])); // A
        tmpInfo.GEPs.push_back(dyn_cast<GetElementPtrInst>(Vals[5])); // B
        auto *FInstr = cast<Instruction>(Vals[0]);
        Info.ElementTy = FInstr->getType();
        Status.CanTensorize = true;
        eleWiseInfo = tmpInfo;
        return true;
      }
    }
    if (UnaryOp) {
      tmpInfo.LoadA = dyn_cast<LoadInst>(Vals[2]);

      if (auto *GEP0 = dyn_cast<GetElementPtrInst>(Vals[0]))
        tmpInfo.GEPs.push_back(GEP0);
      if (auto *GEP1 = dyn_cast<GetElementPtrInst>(Vals[3]))
        tmpInfo.GEPs.push_back(GEP1);
      auto *FInstr = cast<Instruction>(Vals[0]);
      Info.ElementTy = FInstr->getType();
      Status.CanTensorize = true;
      eleWiseInfo = tmpInfo;
      return true;
    }
  }
  return Res;
}

static StoreInst *findStoreFromValue(Value *V,
                                     const SmallVectorImpl<Loop *> &Loops) {
  SmallPtrSet<Value *, 8> Visited;   // 무한 순환 방지

  std::function<StoreInst *(Value *)> DFS = [&](Value *Cur) -> StoreInst * {
    if (!Cur || !Visited.insert(Cur).second) return nullptr;   // already seen

    // ① 바로 Store 인가?
    if (auto *SI = dyn_cast<StoreInst>(Cur))
      return SI;

    // ② phi ? (다른 phi 로 연결될 가능성)
    if (auto *PN = dyn_cast<PHINode>(Cur)) {
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
        if (StoreInst *SI = DFS(PN->getIncomingValue(i)))
          return SI;
      }
    }

    // ③ BitCast / PtrToInt / inttoptr 등은 무시하고 내부값을 타고 내려간다.
    if (auto *CI = dyn_cast<CastInst>(Cur))
      return DFS(CI->getOperand(0));

    // ④ SelectInst (조건부 선택) – 두 피연산 다 확인
    if (auto *SI = dyn_cast<SelectInst>(Cur)) {
      if (StoreInst *ST = DFS(SI->getTrueValue()))  return ST;
      if (StoreInst *ST = DFS(SI->getFalseValue())) return ST;
    }

    if (!Cur->hasUseList())
      return nullptr;

    // ⑤ 일반 인스트럭션 – 모든 user 를 살펴보고 Store 로 이어지는지 본다.
    for (User *U : Cur->users())
      if (auto *ST = DFS(U))
        return ST;

    return nullptr;        // 여기까지 오면 Store 를 찾지 못했다.
  };

  return DFS(V);
}

static int64_t getConstInt(Value *V) {
  if (auto *C = dyn_cast<ConstantInt>(V))
    return C->getSExtValue();
  // could be a SExt/ZeroExt of constant
  if (auto *SE = dyn_cast<SExtInst>(V))
    if (auto *C = dyn_cast<ConstantInt>(SE->getOperand(0)))
       return C->getSExtValue();
  // else fallback
  return -1;
}

static GetElementPtrInst *propagateGEP(GetElementPtrInst *Gep) {
  GetElementPtrInst *ResGep = Gep;
  while (auto *Temp = dyn_cast<GetElementPtrInst>(ResGep->getOperand(0)))
    ResGep = Temp;

  return ResGep;
}

bool ConvolutionTensorizePattern::isSemanticMatch() {
  // -----------------------------------------------------------------
  // 1) Very early sanity checks – identical to GEMMPattern
  // -----------------------------------------------------------------
  IF_RET_FALSE(!Status.CanTensorize);

  // The innermost loop latch is where the final store to the output tensor
  // lives (the `store float %151, float* %155` in the IR).
  Loop *Innermost = Loops.front();
  IF_RET_FALSE(!Innermost);

  // -----------------------------------------------------------------
  // 2) Walk through the latch and try to recognise the convolution pattern.
  // -----------------------------------------------------------------
  bool Res = false;

  SmallPtrSet<BasicBlock *, 8> VisitedBBs;
  for (Loop *L : Loops) {
    for (BasicBlock *BB : L->getBlocks()) {
      if (VisitedBBs.contains(BB)) continue;

      for (Instruction &I : *BB) {
        Value *Vals[35];          // enough room for all temporaries

        if (!match(&I,
                   m_FAdd(m_Value(Vals[0]),   // previous partial sum (load)
                          m_Value(Vals[1]))) // product = FMul(...)
            ) continue;

        // ---------------------------------------------------------
        // Depth 1 – The product must be an FMul of **two loads**.
        // ---------------------------------------------------------
        Value *MulNode = nullptr;
        if (match(Vals[0],
                  m_FMul(m_Value(Vals[2]), m_Value(Vals[3]))) ) {
          CI.FMulInst = &I;
          MulNode = Vals[0];
        }
        else if (match(Vals[1],
                  m_FMul(m_Value(Vals[2]), m_Value(Vals[3]))) ) {
          CI.FMulInst = cast<Instruction>(Vals[1]);
          MulNode = Vals[1];
        } else
          continue;

        // ---------------------------------------------------------
        // Depth 2 – Each operand of the FMul must be a Load.
        // ---------------------------------------------------------
        if (!match(Vals[2], m_Load(m_Value(Vals[4]))) ||
            !match(Vals[3], m_Load(m_Value(Vals[5]))) )
          continue;

        // ---------------------------------------------------------
        // Depth 3 – The loads must come from two distinct base tensors.
        // ---------------------------------------------------------
        if (!match(Vals[4],
                   m_GEP(m_Value(Vals[6]),   // input base
                         m_Value(Vals[7]))) // linearised input index)
            ) continue;
        if (!match(Vals[5],
                   m_GEP(m_Value(Vals[8]),   // weight base
                         m_Value(Vals[9]))) // linearised weight index)
            ) continue;

        // // ---------------------------------------------------------
        // // Depth 4 – The accumulated value across nested-loop (the other operand of the FAdd)
        // // ---------------------------------------------------------        
        // if (!match(Vals[0], m_Load(m_Value(Vals[10]))) &&
        //     !match(Vals[1], m_Load(m_Value(Vals[11]))) )
        //   continue;   // the FAdd must read the running sum
        // // YYG::REMOVE
        // errs() << "Match m_Load! \n";

        // ---------------------------------------------------------
        // Depth 5 – Index arithmetic for **input, weight** tensor (same as before)
        // ---------------------------------------------------------
        if (!(
          // height component (oh*stride_h + kh - pad_h)
          match(Vals[7],                    // GEP's index
                m_AShr(m_Value(Vals[15]),    // %87  loop‑level offset
                      m_Value(Vals[16])))  // %66  kernel‑height offset)
          && match(Vals[15],                // SHL
                  m_Shl(m_Value(Vals[17]),    // %84  output‑height var
                        m_Value(Vals[18])))   // %9   stride_h)
          && match(Vals[17],
                  m_Add(m_Value(Vals[19]),    // %66  kernel‑height var
                        m_Value(Vals[20])))   // %8   padding_h)

          // width component (ow*stride_w + kw - pad_w)
          && match(Vals[9],
                m_Add(m_Value(Vals[21]),    // %89  loop‑level offset
                      m_Value(Vals[22])))   // %67  kernel‑width offset)
          && match(Vals[21],
                  m_Mul(m_Value(Vals[23]),    // %84  output‑width var
                        m_Value(Vals[24])))   // %10  stride_w)
          )) continue;

        // FAdd 결과값(Vals[0] 은 I 자체) → phi‑chain → Store ?
        StoreInst *Store = findStoreFromValue(Vals[2], Loops);
        if (!Store) continue;   // 저장을 찾지 못하면 패턴이 아니다
        CI.InstStore = Store;

        // store <value::PHI> , <ptr::GEP>
        if (!match(Store, m_Store(m_Value(Vals[27]), m_Value(Vals[28])))) continue;
        // -----------------------------------------------------------------
        //   output 주소 : GEP (outputBase::load, idxOut::ashr)
        // -----------------------------------------------------------------
        if (!match(Vals[28], m_GEP(m_Value(Vals[29]), m_Value(Vals[30])))) continue;

        // -----------------------------------------------------------------
        //   All checks passed – fill the ConvolutionInfo structure.
        // -----------------------------------------------------------------
        CI.ElementTy = Store->getValueOperand()->getType();          // float for the GGML case
        Info.ElementTy = CI.ElementTy;

        // Remember the three GEPs that give us the *addresses* of the three
        // tensors (input, weight, output).  They are used later by the
        // tensorizer to replace the whole nest with a single intrinsic call.
        CI.GEPs.append({
            propagateGEP(cast<GetElementPtrInst>(Vals[5])), // input  GEP
            propagateGEP(cast<GetElementPtrInst>(Vals[4])), // weight GEP
            propagateGEP(cast<GetElementPtrInst>(Vals[28])) // output GEP
        });
        CI.LoadInput  = cast<LoadInst>(Vals[2]);
        CI.LoadWeight = cast<LoadInst>(Vals[3]);
        // If using out-lined version ggml_conv_2d then
        CI.Accumulator = Vals[29];
        // If not, use this for output tensor
        //CI.Accumulator = cast<LoadInst>(Vals[29]); // FAdd Value

        // 6) Kernel transposition 옵션 – 현재 패턴에서는 사용되지 않음
        CI.TransposeKernel = false;

        // -----------------------------------------------------------------
        //   All done – signal a positive match.
        // -----------------------------------------------------------------
        Res = true;
        break;               // there is only one store in the latch, so we can stop
        }                     // end of "if (match store ... )"
      VisitedBBs.insert(BB);
    }                       // end of for‑loop over latch
  }
  return Res;
}
