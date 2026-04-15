#ifndef LLVM_TRANSFORMS_TENSORIZE_TPRECIPEBUILDER_H
#define LLVM_TRANSFORMS_TENSORIZE_TPRECIPEBUILDER_H

#include "TPlan.h"
#include "LoopTensorizeCostModel.h"
#include "TPlanner.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"

namespace llvm {

class LoopTensorizationLegality;
class LoopTensorizeCostModel;
class TargetLibraryInfo;

/// Helper class to create VPRecipies from IR instructions.
class TPRecipeBuilder {
  /// The VPlan new recipes are added to.
  TPlan &Plan;

  /// The loop that we evaluate.
  // Loop *OrigLoop;
  const SmallVector<Loop *> NestedOrigLoops;

  /// Target Library Info.
  const TargetLibraryInfo *TLI;

  /// The legality analysis.
  LoopTensorizationLegality *Legal;

  /// The profitablity analysis.
  LoopTensorizeCostModel &CM;

  PredicatedScalarEvolution &PSE;

  TPBuilder &Builder;

  // !TODO(yuxin.an): For GEMM
  Instruction *TemporaryInstr = nullptr;

  /// When we if-convert we need to create edge masks. We have to cache values
  /// so that we don't end up with exponential recursion/IR. Note that
  /// if-conversion currently takes place during VPlan-construction, so these
  /// caches are only used at that stage.
  using EdgeMaskCacheTy =
      DenseMap<std::pair<BasicBlock *, BasicBlock *>, TPValue *>;
  using BlockMaskCacheTy = DenseMap<BasicBlock *, TPValue *>;
  EdgeMaskCacheTy EdgeMaskCache;
  BlockMaskCacheTy BlockMaskCache;
  // VPlan construction support: Hold a mapping from ingredients to
  // their recipe.
  DenseMap<Instruction *, TPRecipeBase *> Ingredient2Recipe;

  /// Cross-iteration reduction & first-order recurrence phis for which we need
  /// to add the incoming value from the backedge after all recipes have been
  /// created.
  SmallVector<TPHeaderPHIRecipe *, 4> PhisToFix;

  /// Check if \p I can be widened at the start of \p Range and possibly
  /// decrease the range such that the returned value holds for the entire \p
  /// Range. The function should not be called for memory instructions or calls.
  bool shouldWiden(Instruction *I, TFRange &Range, Loop *L) const;

  /// Check if the load or store instruction \p I should widened for \p
  /// Range.Start and potentially masked. Such instructions are handled by a
  /// recipe that takes an additional VPInstruction for the mask.
  TPWidenMemoryRecipe *tryToWidenMemory(Instruction *I,
                                        ArrayRef<TPValue *> Operands,
                                        TFRange &Range);

  /// Check if an induction recipe should be constructed for \p Phi. If so build
  /// and return it. If not, return null.
  TPHeaderPHIRecipe *tryToOptimizeInductionPHI(PHINode *Phi,
                                               ArrayRef<TPValue *> Operands,
                                               TFRange &Range);

  /// Optimize the special case where the operand of \p I is a constant integer
  /// induction variable.
  TPWidenIntOrFpInductionRecipe *
  tryToOptimizeInductionTruncate(TruncInst *I, ArrayRef<TPValue *> Operands,
                                 TFRange &Range);

  /// Handle non-loop phi nodes. Return a new VPBlendRecipe otherwise. Currently
  /// all such phi nodes are turned into a sequence of select instructions as
  /// the vectorizer currently performs full if-conversion.
  TPBlendRecipe *tryToBlend(PHINode *Phi, ArrayRef<TPValue *> Operands);

  /// Handle call instructions. If \p CI can be widened for \p Range.Start,
  /// return a new VPWidenCallRecipe. Range.End may be decreased to ensure same
  /// decision from \p Range.Start to \p Range.End.
  TPWidenCallRecipe *tryToWidenCall(CallInst *CI, ArrayRef<TPValue *> Operands,
                                    TFRange &Range);

  TPMatrixCallRecipe *
  tryToMatrixCall(Instruction *I, ArrayRef<TPValue *> Operands, TFRange &Range);

  /// Check if \p I has an opcode that can be widened and return a VPWidenRecipe
  /// if it can. The function should only be called if the cost-model indicates
  /// that widening should be performed.
  TPWidenRecipe *tryToWiden(Instruction *I, ArrayRef<TPValue *> Operands,
                            TPBasicBlock *TPBB, Loop *L);

public:
  TPRecipeBuilder(TPlan &Plan, SmallVector<Loop *> OrigLoop, const TargetLibraryInfo *TLI,
                  LoopTensorizationLegality *Legal, LoopTensorizeCostModel &CM,
                  PredicatedScalarEvolution &PSE, TPBuilder &Builder)
      : Plan(Plan), NestedOrigLoops(OrigLoop), TLI(TLI), Legal(Legal), CM(CM),
        PSE(PSE), Builder(Builder) {}

  /// Create and return a widened recipe for \p I if one can be created within
  /// the given VF \p Range.
  TPRecipeBase *tryToCreateWidenRecipe(Instruction *Instr,
                                       ArrayRef<TPValue *> Operands,
                                       TFRange &Range, TPBasicBlock *VPBB, unsigned LoopDegree);

  /// Set the recipe created for given ingredient.
  void setRecipe(Instruction *I, TPRecipeBase *R) {
    assert(!Ingredient2Recipe.contains(I) &&
           "Cannot reset recipe for instruction.");
    Ingredient2Recipe[I] = R;
  }

  void fixHeaderPhis();

  /// Create the mask for the vector loop header block.
  void createHeaderMask(Loop *L);

  /// A helper function that computes the predicate of the block BB, assuming
  /// that the header block of the loop is set to True or the loop mask when
  /// tail folding.
  void createBlockInMask(BasicBlock *BB, Loop *L);

  /// Returns the *entry* mask for the block \p BB.
  TPValue *getBlockInMask(BasicBlock *BB) const;

  /// A helper function that computes the predicate of the edge between SRC
  /// and DST.
  TPValue *createEdgeMask(BasicBlock *Src, BasicBlock *Dst, Loop *L);

  /// A helper that returns the previously computed predicate of the edge
  /// between SRC and DST.
  TPValue *getEdgeMask(BasicBlock *Src, BasicBlock *Dst) const;

  /// Return the recipe created for given ingredient.
  TPRecipeBase *getRecipe(Instruction *I) {
    assert(Ingredient2Recipe.count(I) &&
           "Recording this ingredients recipe was not requested");
    assert(Ingredient2Recipe[I] != nullptr &&
           "Ingredient doesn't have a recipe");
    return Ingredient2Recipe[I];
  }

  /// Build a VPReplicationRecipe for \p I. If it is predicated, add the mask as
  /// last operand. Range.End may be decreased to ensure same recipe behavior
  /// from \p Range.Start to \p Range.End.
  TPReplicateRecipe *handleReplication(Instruction *I, TFRange &Range, Loop *L);

  /// Returns a range mapping the values of the range \p Operands to their
  /// corresponding VPValues.
  iterator_range<mapped_iterator<Use *, std::function<TPValue *(Value *)>>>
  mapToTPValues(User::op_range Operands);

  TPValue *mapToTPValue(Value *);

  TPValue *getTPValueOrAddLiveIn(Value *V, TPlan &Plan) {
    if (auto *I = dyn_cast<Instruction>(V)) {
      if (auto *R = Ingredient2Recipe.lookup(I))
        return R->getTPSingleValue();
    }
    return Plan.getOrAddLiveIn(V);
  }
};
} // namespace llvm

#endif // LLVM_TRANSFORMS_TENSORIZE_TPRECIPEBUILDER_H
