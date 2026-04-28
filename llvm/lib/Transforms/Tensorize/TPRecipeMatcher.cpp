//===- TPRecipeMatcher.cpp - Pattern matching for TPlan recipes -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tensorize/TPRecipeMatcher.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Tensorize/TPlanCFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Tensorize/TPlan.h"

using namespace llvm;

#define DEBUG_TYPE "tplan-matcher"

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Returns the DimSet of \p V, or an empty bitset for live-ins/synthetics.
const SmallBitVector &getDimSet(const TPValue *V) {
  SmallBitVector Empty;
  if (V->getDefiningRecipe() == nullptr) {
    // Live-in TPValue
    return Empty;
  }
  if (const auto *SDR = dyn_cast<TPRecipeBase>(V->getDefiningRecipe()))
    return SDR->DimSet;
  return Empty;
}

TPRecipeBase *TensorOpKindMatcher::skipIntermediateRecipes(TPValue *V) {
  // if (auto *DefRecipe = RV->getDefiningRecipe()) {

  while (V) {
    auto *SDR = dyn_cast<TPSingleDefRecipe>(V->getDefiningRecipe());
    if (!SDR)
      return nullptr;
    
    // SDR IS the recipe — no getDefiningRecipe() needed
    if (SDR->getTPDefID() == TPRecipeBase::TPWidenCastSC) {
      V = SDR->getOperand(0); // skip cast, follow source
      continue;
    }
    if (SDR->getTPDefID() == TPRecipeBase::TPWidenSC &&
        SDR->getNumOperands() == 1) {
      // Single-operand Widen = unary op; skip it.
      if (isa<UnaryOperator>(cast<TPWidenRecipe>(SDR)->getUnderlyingInstr())) {
        V = SDR->getOperand(0);
        continue;
      }
    }
    return SDR; // Not an intermediate — stop here.
  }
  return nullptr;
}

bool TensorOpKindMatcher::isAddLike(const TPRecipeBase *R) {
  if (!R || R->getTPDefID() != TPRecipeBase::TPWidenSC)
    return false;
  auto *WR = cast<TPWidenRecipe>(R);
  unsigned Op = WR->getUnderlyingInstr()->getOpcode();
  return Op == Instruction::FAdd || Op == Instruction::Add;
}

bool TensorOpKindMatcher::isMulLike(const TPRecipeBase *R) {
  if (!R || R->getTPDefID() != TPRecipeBase::TPWidenSC)
    return false;
  auto *WR = cast<TPWidenRecipe>(R);
  unsigned Op = WR->getUnderlyingInstr()->getOpcode();
  return Op == Instruction::FMul || Op == Instruction::Mul;
}

bool TensorOpKindMatcher::isReductionUpdate(const TPRecipeBase *R) {
  if (!R || R->getTPDefID() != TPRecipeBase::TPWidenSC)
    return false;
  auto *WR = cast<TPWidenRecipe>(R);

  if (!isa<BinaryOperator>(WR->getUnderlyingInstr()))
    return false;
  
  if (!isAddLike(R))
    return false;

  for (TPValue *Op : R->operands()) {
    auto *RV = dyn_cast<TPValue>(Op);
    if (!RV) continue;
    if (auto *DefRecipe = RV->getDefiningRecipe()) {
      RV->getDefiningRecipe()->dump();
      if (isa<TPReductionPHIRecipe>(RV->getDefiningRecipe())) {
        TPRecipeBase *R = DefRecipe;
        plan->setReductionDims(R->DimSet);
        // YYG::REMOVE
        errs() << "ReductionDim's DimSet: ";
        auto Reduction = plan->getReductionDims();
        for (int D = Reduction.find_first(); D >= 0; D = Reduction.find_next(D))
          errs() << D << "\n";
        return true;
      }
    }
  }
  return false;
}

TPValue *TensorOpKindMatcher::getReductionInput(const TPRecipeBase *R) {
  for (TPValue *Op : R->operands()) {
    auto *RV = dyn_cast<TPValue>(Op);
    if (!RV || !isa<TPReductionPHIRecipe>(RV->getDefiningRecipe()))
      return Op;
  }
  return nullptr;
}

RecipeClassification TensorOpKindMatcher::classifyReduction(const TPRecipeBase &R) {
  // The logic step-by-step
  // 1. Find the non-PHI input of the reduction update (e.g., the fadd's operand that isn't the reduction PHI).
  // 2. Skip intermediate unary ops (like fneg) to find the "producer."
  // 3. Check if the producer is an fmul. If not -> PlainReduction.
  // 4. If it is an fmul, get the DimSets of its two operands (D0, D1).
  // 5. Compute Shared = D0 n D1 n ReductionDims. If any bit is set,
  // that dimension is a contraction dimension (it appears in both operands AND is reduced over).
  // Return Contraction with that dim index.
  // 6. Otherwise -> PlainReduction.
  // YYG::REMOVE
  errs() << "[classifyReduction]\n";

  TPValue *Input = getReductionInput(&R);
  // YYG::REMOVE
  errs() << "*Input: " << *Input << "\n";

  TPRecipeBase *Producer = skipIntermediateRecipes(Input);
  errs() << "Producer: \n";
  Producer->dump();

  if (Producer && isMulLike(Producer)) {
    // YYG::REMOVE
    errs() << "Producer->getOperand(0): " << *(Producer->getOperand(0)) << "\n";
    errs() << "Producer->getOperand(1): " << *(Producer->getOperand(1)) << "\n";
    // Producer->getOperand(0): WIDEN ir<%105> = load ir<%104>
    // Producer->getOperand(1): WIDEN ir<%109> = load ir<%108>

    const SmallBitVector &D0 = getDimSet(Producer->getOperand(0));
    const SmallBitVector &D1 = getDimSet(Producer->getOperand(1));
    errs() << "D0's DimSet: ";
    for (int D = D0.find_first(); D >= 0; D = D0.find_next(D))
      errs() << D << "\n";
    
    errs() << "D1's DimSet: ";
    for (int D = D1.find_first(); D >= 0; D = D1.find_next(D))
      errs() << D << "\n";

    // Resize to common size for bitwise ops.
    unsigned N = std::max({D0.size(), D1.size(), // ReductionDim은 어디서 세팅하는거지? 
                           plan->getReductionDims().size()});
    // YYG::REMOVE
    errs() << "plan->getReductionDims().size(): " << plan->getReductionDims().size() << "\n";
    errs() << "N: " << N << "\n";

    SmallBitVector SharedFinal = D0;
    SharedFinal.resize(N);
    SmallBitVector D1r = D1;
    D1r.resize(N);
    SharedFinal &= D1r; // Shared = D0 & D1
    errs() << "SharedFinal's DimSet: " ;
    for (int D = SharedFinal.find_first(); D >= 0; D = SharedFinal.find_next(D))
      errs() << D << "\n";

    SmallBitVector RD = plan->getReductionDims(); // 이게 0라서 0로 잡힘! 
    RD.resize(N);
    SharedFinal &= RD; // Shared & ReductionDims
    errs() << "SharedFinal's DimSet After ReductionDims: " ;
    for (int D = SharedFinal.find_first(); D >= 0; D = SharedFinal.find_next(D))
      errs() << D << "\n";

    if (SharedFinal.any()) {
      int ContractDim = SharedFinal.find_first();
      // YYG::REMOVE
      errs() << "ContractDim: " << ContractDim << "\n";

      return {TensorOpKind::Contraction, ContractDim,
              /* FusedMulRecipe = */ const_cast<TPRecipeBase *>(Producer)};
    }
  }
  return {TensorOpKind::PlainReduction, -1, nullptr};
}

/// Classify a binary op recipe (non-reduction).
TensorOpKind TensorOpKindMatcher::classifyBinaryOp(const TPRecipeBase &R) {
  // YYG::REMOVE
  errs() << "[classifyBinaryOp] R: \n";
  R.dump();

  const SmallBitVector &D0 = getDimSet(R.getOperand(0));
  const SmallBitVector &D1 = getDimSet(R.getOperand(1));
  errs() << "D0's DimSet: " ;
  for (int D = D0.find_first(); D >= 0; D = D0.find_next(D))
    errs() << D << "\n";
  errs() << "D1's DimSet: \n";
  for (int D = D1.find_first(); D >= 0; D = D1.find_next(D))
    errs() << D << "\n";

  if (D0.none() && D1.none())
    return TensorOpKind::Scalar;

  // Resize to equal length for comparison.
  unsigned N = std::max(D0.size(), D1.size());
  SmallBitVector A = D0, B = D1;
  A.resize(N); B.resize(N);

  // Compare value.
  if (A == B)
    return TensorOpKind::ElementWise;

  // Subset check.
  SmallBitVector Intersection = A;
  Intersection &= B;
  if (Intersection == A) return TensorOpKind::BroadcastBinary; // A ⊆ B
  if (Intersection == B) return TensorOpKind::BroadcastBinary; // B ⊆ A

  // YYG::REMOVE
  errs() << "REMOVE!\n";

  // Disjoint check.
  // Outer-product only works when fmul/mul instr.
  if (Intersection.none()) {
    auto *Inst = R.getDefinedValue()->getUnderlyingInstr();
    unsigned OC = Inst->getOpcode();
    if (OC == Instruction::FMul || OC == Instruction::Mul)
      return TensorOpKind::OuterProduct;
  }

  // Partial overlap: this binary op may be the fused mul of a contraction.
  // Return Scalar conservatively; the second pass in TPRecipePatternMatcher_match
  // will correct fused muls to Contraction.
  LLVM_DEBUG(dbgs()
             << "TPRecipeMatcher: partial-overlap binary op classified as "
                "Scalar (may be corrected by second pass)\n");
  return TensorOpKind::Scalar;
}

SmallVector<unsigned> getTPValueShape(const TPSingleDefRecipe &V,
                                             const TPlan &Plan) {
  SmallVector<unsigned> Shape;
  for (int D = V.DimSet.find_first(); D >= 0; D = V.DimSet.find_next(D))
    Shape.push_back(Plan.getPFForDim(static_cast<unsigned>(D)));
  return Shape;
}

SmallVector<const SCEV *> getTPValueStrides(const TPSingleDefRecipe &V,
                                                   const TPlan &Plan,
                                                   ScalarEvolution &SE) {
  SmallVector<const SCEV *> Strides;
  for (int D = V.DimSet.find_first(); D >= 0; D = V.DimSet.find_next(D))
    Strides.push_back(V.getMemStride(static_cast<unsigned>(D), Plan, SE));
  return Strides;
}

void TensorOpKindMatcher::populateSCEVStridesFromIndex(
    DenseMap<unsigned, const SCEV *> &MemStrides,
    const SmallBitVector &DimSet,
    Value *GEPIdx,
    const MapVector<unsigned, Loop *> &DimToLoop) {

  const SCEV *IdxSCEV = SE->getSCEV(GEPIdx);

  DenseMap<const Loop *, const SCEV *> LoopStep;
  const SCEV *S = IdxSCEV;

  // If loop-induction variables are not canonicalized, whlie-loop for below logic can be work.
  // Right now, it is deprecated.
  // The while loop in populateSCEVStridesFromIndex unwraps this nesting layer by layer:
  // S = {0} <- AddRec? no -> stop
  // S ={0, + N}<Loop_i> <- AddRec? yes -> record Loop_i strides=N
  // S = { {0, +, N}, +, 1 }<Loop_k> <- AddRec? yes -> record Loop_k stride=1.
  // Each layer gives one {Loop, stride} pair.
  // That's why this pattern naturally handles any depth of loop nesting.
  if (const auto *AR = dyn_cast<SCEVAddRecExpr>(S)) {
    LoopStep[AR->getLoop()] = AR->getStepRecurrence(*SE);
    S = AR->getStart();
  }
  else return;

  // Load instruction's DimSet
  for (int D = DimSet.find_first(); D >= 0; D = DimSet.find_next(D)) {
    // Finding Loop for Dim d.
    auto DIt = DimToLoop.find(static_cast<unsigned>(D));
    if (DIt == DimToLoop.end())
      continue;

    auto SIt = LoopStep.find(DIt->second);
    if (SIt != LoopStep.end()) {
      // After this runs, each load/store recipe has its MemStrides map populated
      // - e.g., { dim0 -> 4, dim1 -> 1} mapping, "dim0 steps by 4 elements per iteration,
      // dim1 steps by 1.". This information is later used to determine leading
      // dimensions (LDA/LDB) and contiguity for tensor operations like matrix multiply.
      MemStrides[static_cast<unsigned>(D)] = SIt->second;
    }
  }
}


void TensorOpKindMatcher::populateSCEVStrides(TPWidenLoadRecipe &LR,
                                 const MapVector<unsigned, Loop *> &DimToLoop) {
  // YYG::REMOVE
  errs() << "populateSCEVStrides\n";

  auto *Load = cast<LoadInst>(LR.getUnderlyingValue());
  auto *GEP = dyn_cast<GetElementPtrInst>(Load->getPointerOperand()->stripPointerCasts());
  // YYG::REMOVE
  errs() << "Load: " << *Load << "\n";
  errs() << "Load->getPointerOperand(): " << *(Load->getPointerOperand()) << "\n";

  // Only cares about Ptr[Idx] for flat GEP.
  if (!GEP || GEP->getNumIndices() != 1)
    return;
  errs() << "GEP: " << *GEP << "\n";
  errs() << "GEP->getOperand(1): " << *(GEP->getOperand(1)) << "\n";

  // nested-GEP
  auto *Ptr = Load->getPointerOperand()->stripPointerCasts();
  unsigned MaxDepth = 32;
  while (Ptr && MaxDepth-- > 0) {
    auto *GEP = dyn_cast<GetElementPtrInst>(Ptr);
    if (!GEP)
      break;
    if (GEP->getNumIndices() == 1) {
      // YYG::REMOVE
      errs() << "[Load] GEP: " << *GEP << "\n";
      populateSCEVStridesFromIndex(LR.MemStrides, LR.DimSet,
                                    GEP->getOperand(1), DimToLoop);
    }
    Ptr = GEP->getPointerOperand()->stripPointerCasts();
  }
}

void TensorOpKindMatcher::populateSCEVStrides(TPWidenStoreRecipe &SR,
                                 const MapVector<unsigned, Loop *> &DimToLoop) {
  // YYG::REMOVE
  errs() << "SR: \n";
  SR.dump();
  errs() << "SR.getOperand(1): " << *(SR.getOperand(1)) << "\n"; // TPValue
  if (auto *ValDR = SR.getOperand(1)->getDefiningRecipe()) {
    // YYG::REMOVE
    errs() << "ValDR:\n";
    ValDR->dump();
    SR.DimSet = ValDR->DimSet;

    for (int D = ValDR->DimSet.find_first(); D >= 0; D = ValDR->DimSet.find_next(D)) {
      // YYG::REMOVE
      errs() << "D: " << D << "\n";
    }
  }
  if (SR.DimSet.none())
    return;
  TPValue *Addr = SR.getAddr();
  // YYG::REMOVE
  Addr->dump();

  TPValue *Underlying = Addr;
  while (auto *Cast = dyn_cast<TPWidenCastRecipe>(Underlying)) {
    if (Cast->getOpcode() != Instruction::BitCast)
      break;
    Underlying = Cast->getOperand(0);
  }

  if (auto *GEPRecipe = dyn_cast<TPWidenGEPRecipe>(Underlying)) {
    Instruction *IRInst = GEPRecipe->getUnderlyingInstr();\
    auto *GEP = dyn_cast<GetElementPtrInst>(IRInst);
    
    // Only cares about Ptr[Idx] for flat GEP.
    if (!GEP || GEP->getNumIndices() != 1)
      return;
    populateSCEVStridesFromIndex(SR.MemStrides, SR.DimSet,
                                  /* Index= */ GEP->getOperand(1), DimToLoop);
  }
  return;
}

void TensorOpKindMatcher::match() {
  // Build dim→Loop mapping from IV recipes before classifying.
  MapVector<unsigned, Loop *> DimToLoop = plan->LoopIdx2Loop; //buildDimToLoop(Plan, LI);
  SmallVector<TPRecipeBase *, 32> Worklist;

  ReversePostOrderTraversal<TPBlockDeepTraversalWrapper<TPBlockBase *>>
      RPOT(TPBlockDeepTraversalWrapper<TPBlockBase *>(plan->getEntry()));
  
  // From outer-most loop,
  // Figuring out the memory access stride per loop dimension for load/store recipes,
  // using LLVM's Scalar Evolution (SCEV) analysis. This tells the tensor plan how many elements
  // each loop iteration steps through in memory.
  for (TPBasicBlock *BB : TPBlockUtils::blocksOnly<TPBasicBlock>(RPOT)) {
    for (TPRecipeBase &R : *BB) {
      // Populate SCEV strides on load/store recipes before classification.
      if (auto *LR = dyn_cast<TPWidenLoadRecipe>(&R))
        populateSCEVStrides(*const_cast<TPWidenLoadRecipe *>(LR), DimToLoop);
      else if (auto *SR = dyn_cast<TPWidenStoreRecipe>(&R))
        populateSCEVStrides(*const_cast<TPWidenStoreRecipe *>(SR), DimToLoop);
      
      RecipeClassification C;
      
      // Find FAdd/Add instruction and set Plan.ReductionDim according to its instr.DimSet.
      if (isReductionUpdate(&R)) {
        // Only for FMul and Mul instruction, categorize them as FusedMulRecipe.
        C = classifyReduction(R);
      } else if (R.getTPDefID() == TPRecipeBase::TPWidenSC &&
                 isa<BinaryOperator>(
                     cast<TPWidenRecipe>(R).getUnderlyingInstr()) &&
                 R.getNumOperands() == 2) {
        C.Kind = classifyBinaryOp(R);
      }
      // else: load, store, cast, PHI, canonical IV → default Scalar
      // Out->operator[](&R) = C;      
      R.setTensorOpKind(C);
    }
  }
}
