#ifndef LLVM_TRANSFORMS_TENSORIZE_TPLANANALYSIS_H
#define LLVM_TRANSFORMS_TENSORIZE_TPLANANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace llvm {

class LLVMContext;
class TPValue;
class TPBlendRecipe;
class TPInstruction;
class TPWidenRecipe;
class TPWidenCallRecipe;
class TPWidenIntOrFpInductionRecipe;
class TPWidenMemoryRecipe;
struct TPWidenSelectRecipe;
class TPReplicateRecipe;
class TPRecipeBase;
class TPlan;
class Type;

/// An analysis for type-inference for VPValues.
/// It infers the scalar type for a given VPValue by bottom-up traversing
/// through defining recipes until root nodes with known types are reached (e.g.
/// live-ins or load recipes). The types are then propagated top down through
/// operations.
/// Note that the analysis caches the inferred types. A new analysis object must
/// be constructed once a VPlan has been modified in a way that invalidates any
/// of the previously inferred types.
class TPTypeAnalysis {
  DenseMap<const TPValue *, Type *> CachedTypes;
  /// Type of the canonical induction variable. Used for all VPValues without
  /// any underlying IR value (like the vector trip count or the backedge-taken
  /// count).
  Type *CanonicalIVTy;
  LLVMContext &Ctx;

  Type *inferScalarTypeForRecipe(const TPBlendRecipe *R);
  Type *inferScalarTypeForRecipe(const TPInstruction *R);
  Type *inferScalarTypeForRecipe(const TPWidenCallRecipe *R);
  Type *inferScalarTypeForRecipe(const TPWidenRecipe *R);
  Type *inferScalarTypeForRecipe(const TPWidenIntOrFpInductionRecipe *R);
  Type *inferScalarTypeForRecipe(const TPWidenMemoryRecipe *R);
  Type *inferScalarTypeForRecipe(const TPWidenSelectRecipe *R);
  Type *inferScalarTypeForRecipe(const TPReplicateRecipe *R);

public:
  TPTypeAnalysis(Type *CanonicalIVTy, LLVMContext &Ctx)
      : CanonicalIVTy(CanonicalIVTy), Ctx(Ctx) {}

  /// Infer the type of \p V. Returns the scalar type of \p V.
  Type *inferScalarType(const TPValue *V);

  /// Return the LLVMContext used by the analysis.
  LLVMContext &getContext() { return Ctx; }
};

// Collect a VPlan's ephemeral recipes (those used only by an assume).
void collectEphemeralRecipesForTPlan(TPlan &Plan,
                                     DenseSet<TPRecipeBase *> &EphRecipes);

} // namespace llvm

#endif // LLVM_TRANSFORMS_TENSORIZE_TPLANANALYSIS_H
