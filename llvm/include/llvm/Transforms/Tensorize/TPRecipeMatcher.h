//===- TPRecipeMatcher.h - Pattern matching for TPlan recipes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// Defines TensorOpKind, RecipeClassification, and RecipeClassMap.
/// No includes from TPlan.h or TPRecipe.h to avoid circular dependencies.

/// Declares TPRecipePatternMatcher_match(), getTPValueShape(), and
/// getTPValueStrides(). Requires TPlanWidener_widen() to have been called first.
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_TENSORIZE_TRECIPEMATCHER_H
#define LLVM_TRANSFORMS_TENSORIZE_TRECIPEMATCHER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/IR/Instructions.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {
class LoopInfo;
class SCEV;
class Loop;
class ScalarEvolution;
class TPValue;
class TPlan;
class TPSingleDefRecipe;
class TPRecipeBase;

struct TPWidenLoadRecipe;
struct TPWidenStoreRecipe;

/// Classification of a recipe's tensor operation semantics.
enum class TensorOpKind {
  Scalar,           ///< DimSet empty — scalar op, no tensor parallelism
  ElementWise,      ///< Binary op, both operand DimSets equal
  BroadcastBinary,  ///< Binary op, one DimSet is strict subset of the other
  OuterProduct,     ///< Binary op, operand DimSets are disjoint
  Contraction,      ///< Reduction update of mul-like op sharing a reduction dim
  PlainReduction,   ///< Reduction update with no fuseable mul-like producer
};

struct RecipeClassification {
  TensorOpKind  Kind           = TensorOpKind::Scalar;
  int           ContractDim    = -1;         ///< Loop-dim index; Contraction only
  TPRecipeBase *FusedMulRecipe = nullptr;    ///< Pre-resolved mul recipe; Contraction only
};

class TensorOpKindMatcher {
public:
    TPlan *plan;
    ScalarEvolution *SE;
    LoopInfo *LI;

    TensorOpKindMatcher(TPlan &Plan, ScalarEvolution &SE, LoopInfo &LI) :
        plan(&Plan), SE(&SE), LI(&LI) {}

    ~TensorOpKindMatcher() = default;

    /// Returns the tensor shape of \p V: { Plan.getPFForDim(d) for d in V.DimSet }.
    /// Returns {} for scalar (empty DimSet) values.
    /// Requires TPlanWidener_widen() to have been called first.
    SmallVector<unsigned> getTPValueShape(const TPSingleDefRecipe &V,
                                        const TPlan &Plan);

    /// Returns the effective memory stride for each dim in V.DimSet (innermost
    /// first) as SCEV expressions in element units. Each entry is
    /// V.getMemStride(D, Plan, SE): a SCEV from MemStrides if populated by
    /// TPRecipePatternMatcher_match(), else a dense-default SCEV constant.
    SmallVector<const SCEV *> getTPValueStrides(const TPSingleDefRecipe &V,
                                                const TPlan &Plan,
                                                ScalarEvolution &SE);


    /// Classify every recipe in \p Plan into a TensorOpKind, and populate
    /// MemStrides on load/store recipes using SCEV GEP-index analysis.
    /// Requires TPlanWidener_widen() to have been called first.
    /// Results are written into \p Out (existing entries are overwritten).
    void match();

    /// Extract per-dimension element-count strides from \p GEPIdx (the flat index
    /// expression of a single-index GEP) and store them in \p MemStrides.
    void populateSCEVStridesFromIndex(
        DenseMap<unsigned, const SCEV *> &MemStrides,
        const SmallBitVector &DimSet,
        Value *GEPIdx,
        const MapVector<unsigned, Loop *> &DimToLoop);

    /// Populate MemStrides on a load recipe by analysing the GEP index of
    /// its load instruction's pointer operand.
    void populateSCEVStrides(TPWidenLoadRecipe &LR,
                            const MapVector<unsigned, Loop *> &DimToLoop);

    /// Populate DimSet + MemStrides on a store recipe. DimSet is copied from
    /// the stored-value operand's DimSet; strides come from the store's GEP index.
    void populateSCEVStrides(TPWidenStoreRecipe &SR,
                            const MapVector<unsigned, Loop *> &DimToLoop);
    
    /// True if \p R is a reduction update recipe:
    /// a Widen recipe (fadd/fsub etc.) whose one operand is defined by a
    /// TPReductionPHIRecipe.
    bool isReductionUpdate(const TPRecipeBase *R);

    /// Classify a single reduction-update recipe.
    /// Determines whether a reduction is a contraction (i.e., a tensor contraction like
    /// matrix multiply's dot-product dimension) or just a plain reduction (like a simple sum).
    /// It does this by inspecting the dimension sets of the multiply's operation.
    RecipeClassification classifyReduction(const TPRecipeBase &R);
    
    /// Returns the non-PHI operand of a reduction update recipe.
    /// Precondition: isReductionUpdate(R) == true.
    TPValue *getReductionInput(const TPRecipeBase *R);

    /// Walk past WidenCast and scalar unary (single-operand Widen) recipes.
    /// Returns the first recipe that is NOT an intermediate, or nullptr if
    /// the chain bottoms out at a live-in / synthetic value.
    /// Null-safe: returns nullptr when given nullptr.
    /// Unary ops like fneg don't change the algebraic structure - they're just
    /// sign flips (or similar) sitting between the multiply and the add.
    /// The function needs to find the actual producer that determines the contraction pattern.
    /// Concrete exmple: C[i][j] += -A[i][k] * B[k][j]
    /// ex. %mul = fmul float %a, %b        : DimSEt = { 0, 1, 2 }
    ///     %neg = fneg float %mul,         : DimSet = { 0, 1, 2 } <- unary, same dims
    ///     %sum = fadd float %acc, %neg    : reduction update
    /// Without skiiping, getReductionInput(%sum) returns %neg. Then isMulLike(%neg) -> false.
    TPRecipeBase *skipIntermediateRecipes(TPValue *V);
    
    /// True if \p R is an FAdd/Add-like recipe.
    bool isAddLike(const TPRecipeBase *R);

    /// True if \p R is an FMul/Mul-like recipe.
    bool isMulLike(const TPRecipeBase *R);
    
    /// Classify a binary op recipe (non-reduction).
    TensorOpKind classifyBinaryOp(const TPRecipeBase &R);

};
} // namespace llvm
#endif // LLVM_TRANSFORMS_TENSORIZE_TRECIPEMATCHER_H
