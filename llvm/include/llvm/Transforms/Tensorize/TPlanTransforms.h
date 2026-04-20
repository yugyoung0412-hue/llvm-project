#ifndef LLVM_TRANSFORMS_TENSORIZE_TPLANTRANSFORMS_H
#define LLVM_TRANSFORMS_TENSORIZE_TPLANTRANSFORMS_H

#include "TPlan.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Transforms/Tensorize/TensorizeCommon.h"

namespace llvm {

class InductionDescriptor;
class Instruction;
class PHINode;
class ScalarEvolution;
class PredicatedScalarEvolution;
class TargetLibraryInfo;
class TPBuilder;
class TPTilingRegion;

//===----------------------------------------------------------------------===//
// TPlanTransformer — rewrites the TPlan tree before execute()
//===----------------------------------------------------------------------===//

/// Rewrites the TPlan tree to replace IR-level surgery with explicit TPlan
/// nodes. After transform():
///  - The innermost TPRegionBlock (K-loop) has TilingOverride set to a
///    TPTilingRegion that owns the tiling loop structure.
///  - If Policy has a DynamicTiled dim: a TPGuardBlock wraps the root.
///  - Body recipes that are absorbed by tensor.contract have IsSubsumed=true.
///  - State.TilingTCVal is populated with the expanded trip-count Value*.
///
/// IR is NOT modified by transform(); only the TPlan tree changes.
class TPlanTransforms  {
  TPlan &Plan;
  const EmissionPolicy &Policy;
  SCEVExpander &Expander;
  TPBuilder &Builder;
  Loop *OutermostLoop;

public:
  TPlanTransforms(TPlan &P, const EmissionPolicy &Pol,
                   SCEVExpander &Exp, TPBuilder &B,
                   const MapVector<unsigned, Loop *> &DimToLoop)
      : Plan(P), Policy(Pol), Expander(Exp), Builder(B) {
    OutermostLoop = nullptr;
    unsigned MaxDim = 0;
    for (const auto &[D, L] : DimToLoop)
      if (!OutermostLoop || D > MaxDim) { MaxDim = D; OutermostLoop = L; }
  }

  /// Replaces the VPInstructions in \p Plan with corresponding
  /// widen recipes.
  static void
  TPInstructionsToTPRecipes(TPlanPtr &Plan,
                            function_ref<const InductionDescriptor *(PHINode *)>
                                GetIntOrFpInductionDescriptor,
                            ScalarEvolution &SE, const TargetLibraryInfo &TLI);

  /// Sink users of fixed-order recurrences after the recipe defining their
  /// previous value. Then introduce FirstOrderRecurrenceSplice VPInstructions
  /// to combine the value from the recurrence phis and previous values. The
  /// current implementation assumes all users can be sunk after the previous
  /// value, which is enforced by earlier legality checks.
  /// \returns true if all users of fixed-order recurrences could be re-arranged
  /// as needed or false if it is not possible. In the latter case, \p Plan is
  /// not valid.
  static bool adjustFixedOrderRecurrences(TPlan &Plan, TPBuilder &Builder);

  /// Mark body recipes absorbed by the tensor intrinsic as IsSubsumed=true.
  /// \p TilingDim is the loop dimension index (DimIdx convention, 0=innermost)
  /// being tiled.  Only load/store and IV recipes whose DimIndex matches
  /// \p TilingDim are subsumed; arithmetic ops are subsumed unless they are
  /// the Contraction or PlainReduction anchor.  GEP, ReductionPHI, and
  /// PointerInduction recipes are never subsumed.
  void markSubsumedRecipes(TPBasicBlock *Body, unsigned TilingDim);

  /// Build a scalar epilogue block (K%PF iteration). TODO(yg0412.yun) The scalar epilogue
  /// for DynamicTiled dims isn't implemented yet.
  TPBasicBlock *buildScalarEpilogue(TPBasicBlock *Body);

  /// Replace the tiling-dim loop region with a TPTilingRegion by installing a
  /// TilingOverride on the corresponding TPRegionBlock.
  TPTilingRegion *replaceWithTilingRegion(TPRegionBlock *Innermost,
                                           const DimEmissionSpec &Spec);

  /// Clear NSW/NUW flags from reduction instructions if necessary.
  static void clearReductionWrapFlags(TPlan &Plan);

  /// Optimize \p Plan based on \p BestVF and \p BestUF. This may restrict the
  /// resulting plan to \p BestVF and \p BestUF.
  static void optimizeForTFAndUF(TPlan &Plan, ScalarEvolution &SE);

  /// Apply VPlan-to-VPlan optimizations to \p Plan, including induction recipe
  /// optimizations, dead recipe removal, replicate region optimizations and
  /// block merging.
  static void optimize(TPlan &Plan, ScalarEvolution &SE);

  /// Entry point: rewrite Plan according to Policy. Populates State.TilingTCVal.
  void transform(TPTransformState &State);

  /// Wrap predicated VPReplicateRecipes with a mask operand in an if-then
  /// region block and remove the mask operand. Optimize the created regions by
  /// iteratively sinking scalar operands into the region, followed by merging
  /// regions until no improvements are remaining.
  static void createAndOptimizeReplicateRegions(TPlan &Plan);

  /// Replace (ICMP_ULE, wide canonical IV, backedge-taken-count) checks with an
  /// (active-lane-mask recipe, wide canonical IV, trip-count). If \p
  /// UseActiveLaneMaskForControlFlow is true, introduce an
  /// VPActiveLaneMaskPHIRecipe. If \p DataAndControlFlowWithoutRuntimeCheck is
  /// true, no minimum-iteration runtime check will be created (during skeleton
  /// creation) and instead it is handled using active-lane-mask. \p
  /// DataAndControlFlowWithoutRuntimeCheck implies \p
  /// UseActiveLaneMaskForControlFlow.
  static void addActiveLaneMask(TPlan &Plan,
                                bool UseActiveLaneMaskForControlFlow,
                                bool DataAndControlFlowWithoutRuntimeCheck);

  /// Insert truncates and extends for any truncated recipe. Redundant casts
  /// will be folded later.
  static void
  truncateToMinimalBitwidths(TPlan &Plan,
                             const MapVector<Instruction *, uint64_t> &MinBWs,
                             LLVMContext &Ctx);

  /// Drop poison flags from recipes that may generate a poison value that is
  /// used after vectorization, even when their operands are not poison. Those
  /// recipes meet the following conditions:
  ///  * Contribute to the address computation of a recipe generating a widen
  ///    memory load/store (VPWidenMemoryInstructionRecipe or
  ///    VPInterleaveRecipe).
  ///  * Such a widen memory load/store has at least one underlying Instruction
  ///    that is in a basic block that needs predication and after vectorization
  ///    the generated instruction won't be predicated.
  /// Uses \p BlockNeedsPredication to check if a block needs predicating.
  /// TODO: Replace BlockNeedsPredication callback with retrieving info from
  ///       VPlan directly.
  static void dropPoisonGeneratingRecipes(
      TPlan &Plan, function_ref<bool(BasicBlock *)> BlockNeedsPredication);

  /// Add a VPEVLBasedIVPHIRecipe and related recipes to \p Plan and
  /// replaces all uses except the canonical IV increment of
  /// VPCanonicalIVPHIRecipe with a VPEVLBasedIVPHIRecipe.
  /// VPCanonicalIVPHIRecipe is only used to control the loop after
  /// this transformation.
  /// \returns true if the transformation succeeds, or false if it doesn't.
  static bool tryAddExplicitTensorLength(TPlan &Plan);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_TENSORIZE_TPLANTRANSFORMS_H
