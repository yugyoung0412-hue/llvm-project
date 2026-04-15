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


bool ConvolutionTensorizePattern::tryToBuildTPlanWithTPRecipes(TPlanPtr &tplan,
                                               TPRecipeBuilder *RecipeBuilder,
                                               TPBasicBlock *TPBB,
                                               bool UseTensorType) {
  
  
  // Current TPlan is not general at all.
  // Below implementation will create new TPlan with calling llvm.tensor.conv2d intrinsic.
  // This is temporal solution for testing end-to-end building of GGML for NPU backend.

  // step 1. create new TPlan rather than remove tplan's TPBBs.
  TPlanPtr        Plan = nullptr;
  TPIRBasicBlock *Entry = new TPIRBasicBlock(Loops[0]->getLoopPreheader());
  TPBasicBlock *TensorPreheader = new TPBasicBlock("tensor.ph6");
  Plan = std::make_unique<TPlan>(Entry, TensorPreheader, tplan->getPattern());
  // Plan->TripCount[Loops[0]] =
  //     tputils::getOrCreateTPValueForSCEVExpr(*Plan, /* TripCount[Loops[0]]=*/16, SE);

  // Header / Latch
  // Header와 BODY를 분리해서 따로 만들어야 됨!
  BasicBlock *IRHeaderBlock = Loops[0]->getHeader();
  auto *HeaderTPBB = new TPIRBasicBlock(IRHeaderBlock, "tensor.header6");
  TPBasicBlock *LatchTPBB  = new TPBasicBlock("tensor.latch6");
  TPRegionBlock *CurRegion = new TPRegionBlock(HeaderTPBB, LatchTPBB,
                              "tensor loop6", false);
  // Middle / ScalarPH
  TPBasicBlock *MiddleTPBB = new TPBasicBlock("middle.block6");
  TPBasicBlock *ScalarPH   = new TPBasicBlock("scalar.ph6");
  TPBlockUtils::connectBlocks(MiddleTPBB, ScalarPH);
  // HeaderTPBB -> LatchTPBB 
  TPBlockUtils::connectBlocks(HeaderTPBB, LatchTPBB);
  // CurRegion -> MiddleTPBB
  TPBlockUtils::insertBlockAfter(MiddleTPBB, CurRegion);
  // TensorPreheader -> Header (of CurRegion)
  TPBlockUtils::insertBlockAfter(CurRegion, TensorPreheader);

  // YYG::REMOVE
  errs() << "TPlan: \n";
  Plan->dump();

  // step 2. calling llvm.tensor.load intrinsic for input tensor (A_instr), weight tensor (B_instr)
  bool success;
  // auto TFxUF = tplan->getTFxUF();
  auto TFxUF = Plan->getTFxUF();
  Loop *OutermostL = Info.Loops[2];
  Loop *MiddleL = Info.Loops[1];
  Loop *InnermostL = Info.Loops[0];

  SmallVector<TPValue *> MatrixLoads;
  TPValue *Res, *CurGEP;
  
  auto *OutermostPhi = Info.Loop2PHI_tmp[OutermostL];
  auto *MiddlePhi = Info.Loop2PHI_tmp[MiddleL];
  
  // ConstantInt *ConstInt = ConstantInt::get(Type::getInt32Ty(Context), 42);
  GetElementPtrInst *A_Instr = CI.GEPs[0];
  ConstantInt *Input_N_IC = ConstantInt::get(Type::getInt32Ty(A_Instr->getContext()), 256);
  ConstantInt *Input_IH_IW = ConstantInt::get(Type::getInt32Ty(A_Instr->getContext()), 128);
  TPValue *Const_N_IC = new TPValue(Input_N_IC);
  TPValue *Const_IH_IW = new TPValue(Input_IH_IW);
  TPRecipeBase *AIdxRecipe = new TPNewInstrRecipe(
      Instruction::Mul,
      {Const_N_IC, Const_IH_IW});
  LatchTPBB->appendRecipe(AIdxRecipe);

  GetElementPtrInst *B_Instr = CI.GEPs[1];

  int KH = 3;
  int KW = 3;
  ConstantInt *Input_OC_IC = ConstantInt::get(Type::getInt32Ty(B_Instr->getContext()), 512);
  ConstantInt *Input_KH_KW = ConstantInt::get(Type::getInt32Ty(B_Instr->getContext()), KH * KW);
  TPValue *Const_OC_IC = new TPValue(Input_OC_IC);
  TPValue *Const_KH_KW = new TPValue(Input_KH_KW);
  TPRecipeBase *BIdxRecipe =
      new TPNewInstrRecipe(Instruction::Mul, {Const_OC_IC, Const_KH_KW});
  LatchTPBB->appendRecipe(BIdxRecipe);

  if (UseTensorType) {
    auto *GEPRecipe = new TPNewInstrRecipe(
            Instruction::GetElementPtr,
            {Plan->getOrAddLiveIn( CI.GEPs[0]->getOperand(0)), AIdxRecipe->getTPSingleValue()});
    // RecipeBuilder->setRecipe(A_Instr, GEPRecipe);
    LatchTPBB->appendRecipe(GEPRecipe);
    CurGEP = GEPRecipe->getTPSingleValue();

    ConstantInt *Zero = ConstantInt::get(Type::getInt32Ty(CI.Accumulator->getContext()), 0);
    ConstantInt *One =
        ConstantInt::get(Type::getInt32Ty(CI.Accumulator->getContext()), 1);
    TPValue *ConstOne = new TPValue(One);

    // LoadInstr of Input Tensor
    Intrinsic::ID ID = Intrinsic::tensor_new_load;
    SmallVector<TPValue *> Operands;
    Operands.append({CurGEP, ConstOne, Const_N_IC,
                     // Plan->getOrCreateBackedgeTakenCount()[InnermostL],
                     Const_IH_IW, Const_IH_IW});
    // TFxUF[OutermostL], TFxUF[InnermostL]});
    auto *MatrixLoadRecipe = new TPMatrixCallRecipe(
        A_Instr, make_range(Operands.begin(), Operands.end()), ID,
        A_Instr->getDebugLoc());
    // RecipeBuilder->setRecipe(A_Instr, MatrixLoadRecipe);
    LatchTPBB->appendRecipe(MatrixLoadRecipe);
    MatrixLoads.push_back(MatrixLoadRecipe->getTPSingleValue());

    auto *BGEPRecipe = new TPNewInstrRecipe(
            Instruction::GetElementPtr,
            {Plan->getOrAddLiveIn( CI.GEPs[1]->getOperand(0)), BIdxRecipe->getTPSingleValue()});
    LatchTPBB->appendRecipe(BGEPRecipe);
    CurGEP = BGEPRecipe->getTPSingleValue();

    ConstantInt *Input_KH = ConstantInt::get(Type::getInt32Ty(B_Instr->getContext()), KH);
    ConstantInt *Input_KW = ConstantInt::get(Type::getInt32Ty(B_Instr->getContext()), KW);
    TPValue *KernelH= new TPValue(Input_KH);
    TPValue *KernelW = new TPValue(Input_KW);

    // LoadInstr of Weight Tensor
    Intrinsic::ID BID = Intrinsic::tensor_new_load;
    SmallVector<TPValue *> BOperands;
    BOperands.append({CurGEP, Const_OC_IC, Const_N_IC,
                      // Plan->getOrCreateBackedgeTakenCount()[InnermostL],
                      KernelH, KernelW});
    // TFxUF[OutermostL], TFxUF[InnermostL]});
    auto *BMatrixLoadRecipe = new TPMatrixCallRecipe(
        B_Instr, make_range(BOperands.begin(), BOperands.end()), BID,
        B_Instr->getDebugLoc());
    LatchTPBB->appendRecipe(BMatrixLoadRecipe);
    MatrixLoads.push_back(BMatrixLoadRecipe->getTPSingleValue());

    // 3. Call llvm.tensor.conv2d intrinsic
    Intrinsic::ID ConvID = Intrinsic::tensor_convolution_2d;
    SmallVector<TPValue *, 4> Op, Op2;

    TPValue *PadOne = new TPValue(One);
    TPValue *StrideOne = new TPValue(One);
    TPValue *DilationZero = new TPValue(Zero);
    TPValue *ConstPadTy = new TPValue(Zero);

    // Here, stride means memory-layout related stride for NPU.Cov
    Op.append({MatrixLoads[0], MatrixLoads[1], StrideOne, StrideOne, PadOne,
               PadOne, PadOne, PadOne, DilationZero, DilationZero, ConstPadTy,
               KernelH, KernelW});
    // %tensor_conv2d = call <4, <3 x 5 x 5 x 6 x half>,  [2, 3, 4, 5], NCHW,
    // SRAM, 2222>
    //                  @llvm.tensor.conv2d.t3x5x5x6xf16x2222.t3x5x5x6xf16x2222.t3x5x5x6xf16x2222.t1x4xi32x0.t1x4xi32x0.t1x4xi32x0
    //                  (<4, <3 x 5 x 5 x 6 x half>,  [2, 3, 4, 5], NCHW, SRAM,
    //                  2222> %Ifm,
    //                    <4, <3 x 5 x 5 x 6 x half>,  [2, 3, 4, 5], NCHW, SRAM,
    //                    2222> %Weight, <2, <1 x 4 x i32>> %Stride, <2, <1 x 4
    //                    x i32>> %PadSize, <2, <1 x 4 x i32>> %DilationRate,
    //                    i32 %PadTy)

    auto *ConvRecipe = new TPMatrixCallRecipe(
        CI.FMulInst, make_range(Op.begin(), Op.end()), ConvID,
        CI.FMulInst->getDebugLoc());
    LatchTPBB->appendRecipe(ConvRecipe);
        
    // If we need transpose to store output matrix,
    // auto *TransposeRecipe = new TPMatrixCallRecipe(
    //     Instr, make_range(Operands2.begin(), Operands2.end()),
    //     Intrinsic::matrix_transpose, Instr->getDebugLoc());
    // TPBB->appendRecipe(TransposeRecipe);
    // Res = TransposeRecipe->getTPSingleValue();
    
    // 4. Store output tensor
    Res = ConvRecipe;
    
    //Intrinsic::ID StoreID = Intrinsic::matrix_column_major_store_addr_space_ext;
    // Intrinsic::ID StoreID = Intrinsic::matrix_column_major_store;
    Intrinsic::ID StoreID = Intrinsic::tensor_new_store;
    SmallVector<TPValue *, 4> StoreOperands{
        Res, CurGEP,
        Const_OC_IC, // Plan->getOrCreateBackedgeTakenCount()[MiddleL],
        Const_IH_IW, Const_IH_IW};
    // TFxUF[MiddleL], TFxUF[OutermostL]};
    auto *StoreRecipe = new TPMatrixCallRecipe(
        CI.InstStore, make_range(StoreOperands.begin(), StoreOperands.end()),
        StoreID, CI.InstStore->getDebugLoc());
    // RecipeBuilder->setRecipe(Instr, StoreRecipe);
    LatchTPBB->appendRecipe(StoreRecipe);
    // YYG::REMOVE
    errs() << "Calling intrinsic end!\n";
    Plan->dump();
    success = true;
  }
  // tensor intrinsic else
  // ...

  tplan = std::move(Plan);
  return success;
}



// bool ConvolutionTensorizePattern::tryToBuildTPlanWithTPRecipes(const TPlanPtr &tplan,
//                                                TPRecipeBuilder *RecipeBuilder,
//                                                TPBasicBlock *TPBB,
//                                                bool UseTensorType) {
//   // -----------------------------------------------------------------
//   // 1️⃣  Grab the three relevant loops (outer‑most → innermost)
//   // -----------------------------------------------------------------
//   // The original TPlan that you posted had 7 nested loops, but the
//   // *convolution* we want to emit only needs the three outer loops that
//   // correspond to the output tensor dimensions:
//   //   N  (batch)        → OutermostL
//   //   OC (output chan) → MiddleL
//   //   OH,OW (spatial)  → InnermostL   (inner‑most of the three)
//   Loop *OutermostL = Info.Loops[2];   // N‑loop
//   Loop *MiddleL    = Info.Loops[1];   // OC‑loop
//   Loop *InnermostL = Info.Loops[0];   // (H,W)‑loop (the innermost of the three)

//   // -----------------------------------------------------------------
//   // 2️⃣  Grab the phi‑nodes that represent the canonical induction
//   // -----------------------------------------------------------------
//   // They are already materialised by the TPlan (see the “CANONICAL‑INDUCTION”
//   // lines in the IR you posted).  We turn them into TP‑values that can be
//   // used as indices for the tensor intrinsics.
//   auto *OuterPhi   = Info.Loop2PHI[OutermostL];   // %tp<%21> in the IR
//   auto *MiddlePhi  = Info.Loop2PHI[MiddleL];      // %tp<%22>
//   auto *InnerPhi   = Info.Loop2PHI[InnermostL];   // %tp<%26>

//   // -----------------------------------------------------------------
//   // 3️⃣  Build the three “index” recipes (AIdx, CIdx, BIdx) that are
//   //     required later when we build the GEPs for the three tensors.
//   // -----------------------------------------------------------------
//   // In the GEMM code these were called AIdx, CIdx, … – we keep the same
//   // naming because the convolution intrinsic expects *three* runtime
//   // indices that are multiplied by the corresponding *tensor‑factor*
//   // (TFxUF) values.
//   //   AIdx = OuterPhi * UF[Innermost]          // batch dimension
//   //   BIdx = OuterPhi * UF[Middle]            // output‑channel dimension
//   //   CIdx = MiddlePhi + OuterPhi * UF[Middle]  // spatial base index
//   //
//   // (The exact arithmetic is not important for the intrinsic – it only
//   //  needs a *single* tensor value, but we keep the same recipes so that
//   //  the generated schedule remains identical to the original TPlan.)

//   // 3.1 AIdx  (batch index)
//   TPRecipeBase *AIdxRecipe = new TPNewInstrRecipe(
//       Instruction::Mul,
//       {RecipeBuilder->mapToTPValue(OuterPhi),
//        tplan->getTFxUF()[InnermostL]});                 // TFxUF[Innermost]
//   TPBB->appendRecipe(AIdxRecipe);

//   // 3.2 BIdx  (output‑channel index, i.e. "CIdx" in GEMM)
//   TPRecipeBase *BIdxRecipeSub = new TPNewInstrRecipe(
//       Instruction::Mul,
//       {RecipeBuilder->mapToTPValue(OuterPhi),
//        tplan->getOrCreateBackedgeTakenCount()[MiddleL]}); // back‑edge count = trip‑count
//   TPBB->appendRecipe(BIdxRecipeSub);

//   TPRecipeBase *CIdxRecipe = new TPNewInstrRecipe(
//       Instruction::Add,
//       {CIdxRecipe->getTPSingleValue(),
//        RecipeBuilder->mapToTPValue(MiddlePhi)});        // + middle‑phi
//   TPBB->appendRecipe(CIdxRecipe);

//   // -----------------------------------------------------------------
//   // 4️⃣  Build GEP recipes for the three tensor live‑ins
//   // -----------------------------------------------------------------
//   // The live‑ins that the original TPlan created are:
//   //   %Ifm   – the input feature map tensor
//   //   %Weight – the convolution kernel tensor
//   //   %Ofm   – the output tensor (where we store the result)
//   //
//   // The map (GEP → index) is identical to the GEMM case: the first GEP
//   // (input tensor) uses AIdx, the second (weight) uses MiddlePhi, the third
//   // (output) uses CIdx.
//   DenseMap<GetElementPtrInst *, TPValue *> GEPIdx{
//       {CI.GEPs[0], AIdxRecipe->getTPSingleValue()},      // Ifm GEP
//       {CI.GEPs[1], RecipeBuilder->mapToTPValue(MiddlePhi)}, // Weight GEP
//       {CI.GEPs[2], CIdxRecipe->getTPSingleValue()},     // Ofm GEP
//   };

//   // -----------------------------------------------------------------
//   // 5️⃣  Walk the innermost loop body and replace the whole
//   //     load‑multiply‑store sequence with a single conv2d intrinsic.
//   // -----------------------------------------------------------------
//   // The code mirrors the structure of the GEMM builder, but *instead*
//   // of emitting three separate matrix‑load / matrix‑multiply /
//   // matrix‑store recipes we emit just one call to
//   //   llvm.tensor.conv2d
//   // which itself takes the two input tensors (Ifm, Weight) and the
//   // convolution meta‑data.
//   //
//   // -----------------------------------------------------------------
//   // 5‑a  Gather the two input tensors (Ifm and Weight) as TP‑values.
//   // -----------------------------------------------------------------
//   TPValue *IfmTensor = nullptr;
//   TPValue *WeightTensor = nullptr;
//   SmallVector<TPValue *> CurGEPs;               // temporary holder for each GEP
//   SmallVector<TPValue *> MatrixLoads;          // we keep the two loads only
//   TPValue *Res = nullptr;                      // the result of the conv2d call
//   TPValue *CurGEP = nullptr;                   // the most‑recent GEP value

//   // -----------------------------------------------------------------
//   // 5‑b  Walk the instructions that belong to the innermost loop latch.
//   // -----------------------------------------------------------------
//   // The loop latch contains the *last* iteration of the inner loop – that is
//   // exactly the block that the TPlan IR prints as “… latch6”, “… latch5”, …
//   // for the original GEMM.  We use the same helper (`drop_end`) that the GEMM
//   // version uses to skip the terminator and debug‑info.
//   // -----------------------------------------------------------------
//   for (Instruction &I :
//        drop_end(InnermostL->getLoopLatch()->instructionsWithoutDebug(false))) {
//     Instruction *Inst = &I;

//     // -------------------------------------------------------------
//     // 5‑c  GetElementPtr => create a GEP recipe and remember it.
//     // -------------------------------------------------------------
//     if (auto *GEPI = dyn_cast<GetElementPtrInst>(Inst)) {
//       if (!GEPIdx.count(GEPI))
//         continue;                                   // not one of the three tensors

//       auto *GEPRecipe = new TPNewInstrRecipe(
//           Instruction::GetElementPtr,
//           {tplan->getOrAddLiveIn(GEPI->getOperand(0)),   // base tensor
//            GEPIdx[GEPI]});                              // runtime index
//       RecipeBuilder->setRecipe(GEPI, GEPRecipe);
//       TPBB->appendRecipe(GEPRecipe);
//       CurGEP = GEPRecipe->getTPSingleValue();
//       CurGEPs.push_back(CurGEP);
//       continue;
//     }

//     // -------------------------------------------------------------
//     // 5‑d  LoadInst => create a tensor‑load recipe.
//     // -------------------------------------------------------------
//     if (auto *LoadI = dyn_cast<LoadInst>(Inst)) {
//       // The convolution intrinsic expects tensors, not raw scalars.
//       // We therefore use the *tensor_new_load* intrinsic that is already
//       // defined for the tensor‑type path.
//       Intrinsic::ID LoadID = Intrinsic::tensor_new_load;
//       SmallVector<TPValue *> Operands;

//       // Decide whether this load is the input feature map or the kernel.
//       if (LoadI == CI.LoadIfm) {
//         // %Ifm pointer + back‑edge count of the innermost loop
//         Operands.append({CurGEP,
//                          tplan->getOrCreateBackedgeTakenCount()[InnermostL],
//                          tplan->getTFxUF()[OutermostL],
//                          tplan->getTFxUF()[InnermostL]});
//         // Remember the tensor value – we will feed it to the conv2d call.
//         IfmTensor = new TPMatrixCallRecipe(
//             LoadI, make_range(Operands.begin(), Operands.end()), LoadID,
//             LoadI->getDebugLoc())->getTPSingleValue();
//         // Also push it onto MatrixLoads so that the same vector can be used
//         // later (the GEMM code expects a vector of loads).
//         MatrixLoads.push_back(IfmTensor);
//       } else if (LoadI == CI.LoadWeight) {
//         Operands.append({CurGEP,
//                          tplan->getOrCreateBackedgeTakenCount()[MiddleL],
//                          tplan->getTFxUF()[InnermostL],
//                          tplan->getTFxUF()[MiddleL]});
//         WeightTensor = new TPMatrixCallRecipe(
//             LoadI, make_range(Operands.begin(), Operands.end()), LoadID,
//             LoadI->getDebugLoc())->getTPSingleValue();
//         MatrixLoads.push_back(WeightTensor);
//       } else {
//         // Any other load (e.g. a scalar constant) is not part of the conv2d.
//         // We simply forward it as a normal LLVM load (the existing TPlan
//         // already inserted a recipe for it if needed).  For simplicity we
//         // ignore that case here because the convolution IR you posted does
//         // not contain any extra scalar loads.
//         continue;
//       }

//       // Store the recipe so that later stages (debug, verification) can see
//       // it.
//       auto *LoadRecipe = new TPMatrixCallRecipe(
//           LoadI, make_range(Operands.begin(), Operands.end()), LoadID,
//           LoadI->getDebugLoc());
//       RecipeBuilder->setRecipe(LoadI, LoadRecipe);
//       TPBB->appendRecipe(LoadRecipe);
//       continue;
//     }

//     // -------------------------------------------------------------
//     // 5‑e  StoreInst => we will NOT emit a store here.  The actual
//     //                write‑back to the output tensor is done *after*
//     //                the conv2d intrinsic, using the tensor_new_store
//     //                intrinsic.
//     // -------------------------------------------------------------
//     if (auto *StoreI = dyn_cast<StoreInst>(Inst)) {
//       // Remember the destination GEP so that we can use it after the
//       // conv2d call.
//       // The GEP that computes the destination address is the *third*
//       // entry in `ConvInfo.GEPs` (the one that used CIdxRecipe in the GEMM
//       // version).
//       if (StoreI->getPointerOperand() == CI.GEPs[2]) {
//         // The pointer operand is a GEP that we already emitted a recipe for.
//         // The TPValue for that GEP is the last element we pushed in CurGEPs.
//         // Save it for the store that follows the conv2d.
//         CurGEP = CurGEPs.back();
//       }
//       // We do **not** create a store recipe here – it will be emitted
//       // after the conv2d intrinsic as shown later.
//       continue;
//     }

//     // -------------------------------------------------------------
//     // 5‑f  Any other opcode (e.g. a redundant FMul) is ignored,
//     //      because the whole computation is performed by the intrinsic.
//     // -------------------------------------------------------------
//   } // end for‑loop over latch instructions

//   // -----------------------------------------------------------------
//   // 6️⃣  If we have both tensors, emit the conv2d intrinsic.
//   // -----------------------------------------------------------------
//   if (!IfmTensor || !WeightTensor) {
//     // Something went wrong – the expected loads were not found.
//     dbgs() << "ConvPattern: missing Ifm or Weight tensor load.\n";
//     return false;
//   }

//   // -----------------------------------------------------------------
//   // 6‑a  Prepare the meta‑data operands (stride, pad, dilation, padTy)
//   // -----------------------------------------------------------------
//   // All of them are constant tensors of shape <2, <x x i32>> (as shown
//   // in the intrinsic prototype).  We create them once with the
//   // TP‑builder’s helper `createConstantTensor` (the helper exists in the
//   // LLVM‑TVM/MLIR stack; if it does not exist in your code‑base you can
//   // replace it with a normal `ConstantInt` + GEP pair – the exact API is
//   // out of scope for this snippet).
//   //
//   // The concrete values are:
//   //   stride   = {1, 1}
//   //   padSize  = {0, 0}
//   //   dilation = {1, 1}
//   //   padTy    = 0   (no “explicit” padding mode)
//   // -----------------------------------------------------------------
//   // Helper that builds a <2, <x x i32>> constant tensor.
//   auto makeI32Vec2 = [&](int a, int b) -> TPValue * {
//     // Create a constant struct <2 x i32> = <i32 a, i32 b>
//     Constant *C = ConstantStruct::get(
//         StructType::get(Type::getInt32Ty(Inst->getContext()),
//                         Type::getInt32Ty(Inst->getContext())),
//         {ConstantInt::get(Type::getInt32Ty(Inst->getContext()), a),
//          ConstantInt::get(Type::getInt32Ty(Inst->getContext()), b)});
//     // Wrap it into a tensor value that the recipe system understands.
//     return RecipeBuilder->createTensorConstant(C);
//   };

//   TPValue *Stride   = makeI32Vec2(1, 1);
//   TPValue *PadSize  = makeI32Vec2(0, 0);
//   TPValue *Dilation = makeI32Vec2(1, 1);
//   TPValue *PadTy    = ConstantInt::get(Type::getInt32Ty(Inst->getContext()), 0);

//   // -----------------------------------------------------------------
//   // 6‑b  Assemble the operand list for llvm.tensor.conv2d
//   // -----------------------------------------------------------------
//   // The prototype you posted is:
//   //   %tensor_conv2d = call <4, <3 x 5 x 5 x 6 x half>>, [2,3,4,5],
//   //                    NCHW, SRAM, 2222> @llvm.tensor.conv2d.t3x5x5x6xf16x2222...
//   //   (   %Ifm, %Weight, %Stride, %PadSize, %DilationRate, %PadTy )
//   // In the TP‑builder world we only need to pass the **TPValues** that
//   // represent those tensors.
//   // -----------------------------------------------------------------
//   SmallVector<TPValue *> Conv2dOperands{
//       IfmTensor,          // Input feature map
//       WeightTensor,       // Kernel
//       Stride,             // stride      (const tensor <2 x i32>)
//       PadSize,            // pad size    (const tensor <2 x i32>)
//       Dilation,           // dilation    (const tensor <2 x i32>)
//       PadTy               // pad type    (scalar i32)
//   };

//   // -----------------------------------------------------------------
//   // 6‑c  Emit the conv2d call recipe.
//   // -----------------------------------------------------------------
//   Intrinsic::ID Conv2dID = Intrinsic::tensor_conv2d; // <-- the name you gave
//   auto *Conv2dRecipe = new TPMatrixCallRecipe(
//       /*Inst=*/nullptr,                     // no underlying LLVM instruction
//       make_range(Conv2dOperands.begin(), Conv2dOperands.end()),
//       Conv2dID,
//       /*DebugLoc=*/DebugLoc());             // no debug info needed here
//   // The recipe itself produces a TensorValue that we store in `Res`.
//   Res = Conv2dRecipe->getTPSingleValue();

//   // Register the recipe with the builder so that later passes can see it.
//   TPBB->appendRecipe(Conv2dRecipe);
//   RecipeBuilder->registerTensorIntrinsic(Conv2dID, Conv2dRecipe);

//   // -----------------------------------------------------------------
//   // 7️⃣  Store the result back to the output tensor (`Ofm`).
//   // -----------------------------------------------------------------
//   // The output tensor lives in `ConvInfo.GEPs[2]`.  Its GEP recipe was
//   // already emitted when we processed the GEP instructions above, and the
//   // TPValue for that GEP is held in `CurGEP` (the last value we saw that
//   // matched that GEP).
//   Intrinsic::ID StoreID = Intrinsic::tensor_new_store;
//   SmallVector<TPValue *, 4> StoreOps{
//       Res,                                 // value to store
//       CurGEP,                              // destination GEP
//       tplan->getOrCreateBackedgeTakenCount()[MiddleL],
//       tplan->getTFxUF()[MiddleL],
//       tplan->getTFxUF()[OutermostL]};

//   auto *StoreRecipe = new TPMatrixCallRecipe(
//       /*Inst=*/nullptr, make_range(StoreOps.begin(), StoreOps.end()),
//       StoreID, DebugLoc());
//   TPBB->appendRecipe(StoreRecipe);
//   RecipeBuilder->setRecipe(CI.StoreOfm, StoreRecipe); // optional bookkeeping

//   // -----------------------------------------------------------------
//   8️⃣  Done – return success.
//   -----------------------------------------------------------------
//   return true;
// }

} // namespace llvm
