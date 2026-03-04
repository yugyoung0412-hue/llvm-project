//===- TPlanBuilder.cpp - Build TPlan from LoopNestInfo -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implements TPlanBuilder_build(): converts a LoopNestInfo (produced by
/// analyzeLoopNest()) into an initial scalar TPlan that downstream passes
/// (TPlanWidener, TPlanLowering) can operate on.
///
/// Phases:
///   1. buildHeaderPHIs  – one TPHeaderPHIRecipe per induction variable
///   2. buildBody        – TPWidenLoad/Store from MemAccesses
///   3. buildPatternRecipes – inject tensor-specific recipes (e.g. MatMul)
///   4. buildLatchAndExit  – TPBranchOnCountRecipe in latch
///   5. resolvePF
///
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TPlan.h"
#include "llvm/Transforms/Vectorize/TPRecipe.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PointerType.h"
#include "llvm/IR/Type.h"

using namespace llvm;

namespace {

class TPlanBuilderImpl {
  const LoopNestInfo &Info;
  const PatternHint &Hint;
  const TensorOpDesc &Op;
  ArrayRef<unsigned> ExplicitPF;
  LLVMContext &Ctx;
  std::unique_ptr<TPlan> Plan;

  // Header PHI recipes indexed by dimension (for cross-phase references).
  SmallVector<TPHeaderPHIRecipe *> PHIRecipes;

public:
  TPlanBuilderImpl(const LoopNestInfo &Info, const PatternHint &Hint,
                   const TensorOpDesc &Op, ArrayRef<unsigned> ExplicitPF,
                   LLVMContext &Ctx)
      : Info(Info), Hint(Hint), Op(Op), ExplicitPF(ExplicitPF), Ctx(Ctx),
        Plan(std::make_unique<TPlan>()) {}

  std::unique_ptr<TPlan> build() {
    buildHeaderPHIs();
    buildBody();
    buildPatternRecipes();
    buildLatchAndExit();
    Plan->resolvePF(Op, ExplicitPF, {});
    return std::move(Plan);
  }

private:
  /// Phase 1: emit one TPHeaderPHIRecipe per induction variable.
  void buildHeaderPHIs() {
    Type *I64 = Type::getInt64Ty(Ctx);
    TPBasicBlock *Header = Plan->getVectorBody()->getHeader();
    for (unsigned Dim = 0, E = Info.IVs.size(); Dim < E; ++Dim) {
      // Start and Step values are represented as null TPValues here;
      // the lowering pass will wire the actual SCEV-expanded values.
      auto *R = new TPHeaderPHIRecipe(Dim, /*PF=*/1, /*Start=*/nullptr,
                                      /*Step=*/nullptr, I64);
      Header->appendRecipe(R);
      PHIRecipes.push_back(R);
    }
  }

  /// Phase 2: emit load/store recipes from MemAccesses.
  void buildBody() {
    TPBasicBlock *Header = Plan->getVectorBody()->getHeader();
    Type *PtrTy = PointerType::getUnqual(Ctx);
    for (const MemAccess &MA : Info.Accesses) {
      // Represent the base pointer as a null TPValue; the lowering pass
      // will materialize the actual GEP.
      TPValue *PtrVal = nullptr;
      if (MA.Kind == AccessKind::Read || MA.Kind == AccessKind::ReadWrite) {
        auto *Load = new TPWidenLoadRecipe(ptrPlaceholder(), Align(1),
                                          MA.ElemType ? MA.ElemType
                                                      : Type::getFloatTy(Ctx));
        Header->appendRecipe(Load);
        (void)PtrTy;
      }
      if (MA.Kind == AccessKind::Write || MA.Kind == AccessKind::ReadWrite) {
        auto *Store = new TPWidenStoreRecipe(ptrPlaceholder(), /*Val=*/nullptr,
                                             Align(1));
        Header->appendRecipe(Store);
      }
    }
  }

  /// Phase 3: inject pattern-specific tensor op recipes.
  void buildPatternRecipes() {
    if (Hint.Kind != PatternKind::GEMM)
      return;
    // Inject a MatMul recipe when three or more PHIs exist (M/N/K dims).
    if (PHIRecipes.size() < 3)
      return;
    Type *F32 = Type::getFloatTy(Ctx);
    // Use the ISA op descriptor dimensions if provided, otherwise default 1.
    unsigned M = Op.M ? Op.M : 1;
    unsigned K = Op.K ? Op.K : 1;
    unsigned N = Op.N ? Op.N : 1;
    auto *MM = new TPMatMulRecipe(M, K, N, Intrinsic::matrix_multiply,
                                  /*A=*/nullptr, /*B=*/nullptr,
                                  /*Accum=*/nullptr, F32);
    Plan->getVectorBody()->getHeader()->appendRecipe(MM);
    Plan->setPattern(PatternKind::GEMM);
    if (Op.IntrinsicID != Intrinsic::not_intrinsic)
      Plan->setSelectedOp(Op);
  }

  /// Phase 4: add a branch-on-count recipe in the latch.
  void buildLatchAndExit() {
    if (PHIRecipes.empty())
      return;
    // Use the first (outermost) PHI as the loop IV; trip count is null
    // (SCEV-expanded at lowering time).
    auto *Branch =
        new TPBranchOnCountRecipe(PHIRecipes[0], /*TripCount=*/nullptr);
    Plan->getVectorBody()->getLatch()->appendRecipe(Branch);
  }

  /// Helper: return a stable null-standing placeholder pointer value.
  /// In the real builder this would be a TPExpandSCEVRecipe; for now nullptr.
  TPValue *ptrPlaceholder() { return nullptr; }
};

} // anonymous namespace

std::unique_ptr<TPlan> llvm::TPlanBuilder_build(const LoopNestInfo &Info,
                                                  const PatternHint &Hint,
                                                  const TensorOpDesc &Op,
                                                  ArrayRef<unsigned> ExplicitPF,
                                                  LLVMContext &Ctx) {
  // Require at least one loop.
  if (Info.Loops.empty())
    return nullptr;
  return TPlanBuilderImpl(Info, Hint, Op, ExplicitPF, Ctx).build();
}
