#include "llvm/Transforms/Tensorize/TPlanVerifier.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Tensorize/TPlan.h"
#include "llvm/Transforms/Tensorize/TPlanCFG.h"
#include "llvm/Transforms/Tensorize/TPlanDominatorTree.h"
#include <algorithm>

#define DEBUG_TYPE "loop-tensorize"

using namespace llvm;

namespace {
class TPlanVerifier {
  const TPDominatorTree &TPDT;

  SmallPtrSet<BasicBlock *, 8> WrappedIRBBs;

  // Verify that phi-like recipes are at the beginning of \p VPBB, with no
  // other recipes in between. Also check that only header blocks contain
  // VPHeaderPHIRecipes.
  bool verifyPhiRecipes(const TPBasicBlock *VPBB);

  bool verifyTPBasicBlock(const TPBasicBlock *VPBB);

  bool verifyBlock(const TPBlockBase *VPB);

  /// Helper function that verifies the CFG invariants of the VPBlockBases
  /// within
  /// \p Region. Checks in this function are generic for VPBlockBases. They are
  /// not specific for VPBasicBlocks or VPRegionBlocks.
  bool verifyBlocksInRegion(const TPRegionBlock *Region);

  /// Verify the CFG invariants of VPRegionBlock \p Region and its nested
  /// VPBlockBases. Do not recurse inside nested VPRegionBlocks.
  bool verifyRegion(const TPRegionBlock *Region);

  /// Verify the CFG invariants of VPRegionBlock \p Region and its nested
  /// VPBlockBases. Recurse inside nested VPRegionBlocks.
  bool verifyRegionRec(const TPRegionBlock *Region);

public:
  TPlanVerifier(TPDominatorTree &TPDT) : TPDT(TPDT) {}

  bool verify(const TPlan &Plan);
};
} // namespace

bool TPlanVerifier::verifyPhiRecipes(const TPBasicBlock *TPBB) {
  auto RecipeI = TPBB->begin();
  auto End = TPBB->end();
  unsigned NumActiveLaneMaskPhiRecipes = 0;
  const TPRegionBlock *ParentR = TPBB->getParent();

  SmallVector<TPBlockBase *> HeaderTPBVec;

  if (ParentR) {
    for (auto Elem : ParentR->getLoop2HeaderTPB())
      HeaderTPBVec.push_back(Elem.second);
  }

  bool IsHeaderTPBB =
      ParentR && !ParentR->isReplicator() && is_contained(HeaderTPBVec, TPBB);

  while (RecipeI != End && RecipeI->isPhi()) {
    if (isa<TPActiveLaneMaskPHIRecipe>(RecipeI))
      NumActiveLaneMaskPhiRecipes++;

    if (IsHeaderTPBB && !isa<TPHeaderPHIRecipe, TPWidenPHIRecipe>(*RecipeI)) {
      errs() << "Found non-header PHI recipe in header TPBB";
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
      errs() << ": ";
      RecipeI->dump();
#endif
      return false;
    }

    if (!IsHeaderTPBB && isa<TPHeaderPHIRecipe>(*RecipeI)) {
      errs() << "Found header PHI recipe in non-header TPBB";
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
      errs() << ": ";
      RecipeI->dump();
#endif
      return false;
    }

    RecipeI++;
  }

  if (NumActiveLaneMaskPhiRecipes > 1) {
    errs() << "There should be no more than one TPActiveLaneMaskPHIRecipe";
    return false;
  }

  while (RecipeI != End) {
    if (RecipeI->isPhi() && !isa<TPBlendRecipe>(&*RecipeI)) {
      errs() << "Found phi-like recipe after non-phi recipe";

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
      errs() << ": ";
      RecipeI->dump();
      errs() << "after\n";
      std::prev(RecipeI)->dump();
#endif
      return false;
    }
    RecipeI++;
  }
  return true;
}

bool TPlanVerifier::verifyTPBasicBlock(const TPBasicBlock *TPBB) {
  // YYG::REMOVE
  errs() << "[verifyTPBasicBlock]\n";
  TPBB->dump();

  if (!verifyPhiRecipes(TPBB))
    return false;

  // Verify that defs in TPBB dominate all their uses. The current
  // implementation is still incomplete.
  DenseMap<const TPRecipeBase *, unsigned> RecipeNumbering;
  unsigned Cnt = 0;
  for (const TPRecipeBase &R : *TPBB)
    RecipeNumbering[&R] = Cnt++;

  for (const TPRecipeBase &R : *TPBB) {
    // YYG:REMOVE
    errs() << "[TPRecipeBase] \n";
    R.dump();

    for (const TPValue *V : R.definedValues()) {
      for (const TPUser *U : V->users()) {
        auto *UI = dyn_cast<TPRecipeBase>(U);
        // YYG::REMOVE
        errs() << "*UI: " << "\n";
        UI->dump();

        // TODO: check dominance of incoming values for phis properly.
        if (!UI ||
            isa<TPHeaderPHIRecipe, TPWidenPHIRecipe, TPPredInstPHIRecipe>(UI))
          continue;

        // If the user is in the same block, check it comes after R in the
        // block.
        if (UI->getParent() == TPBB) {
          // YYG::REMOVE
          errs() << "UI->getParend(): ";
          UI->getParent()->dump();

          if (RecipeNumbering[UI] < RecipeNumbering[&R]) {
            errs() << "Use before def!\n";
            return false;
          }
          continue;
        }

        if (!TPDT.dominates(TPBB, UI->getParent())) {
          errs() << "Use before def!\n";
          return false;
        }
      }
    }
  }

  auto *IRBB = dyn_cast<TPIRBasicBlock>(TPBB);
  if (!IRBB)
    return true;

  if (!WrappedIRBBs.insert(IRBB->getIRBasicBlock()).second) {
    errs() << "Same IR basic block used by multiple wrapper blocks!\n";
    return false;
  }

  TPBlockBase *MiddleBB =
      IRBB->getPlan()->getTensorLoopRegion()->getSingleSuccessor();
  if (IRBB != IRBB->getPlan()->getPreheader() &&
      IRBB->getSinglePredecessor() != MiddleBB) {
    errs() << "TPIRBasicBlock can only be used as pre-header or a successor of "
              "middle-block at the moment!\n";
    return false;
  }
  return true;
}

/// Utility function that checks whether \p TPBlockVec has duplicate
/// TPBlockBases.
static bool hasDuplicates(const SmallVectorImpl<TPBlockBase *> &TPBlockVec) {
  SmallDenseSet<const TPBlockBase *, 8> TPBlockSet;
  for (const auto *Block : TPBlockVec) {
    if (TPBlockSet.count(Block))
      return true;
    TPBlockSet.insert(Block);
  }
  return false;
}

bool TPlanVerifier::verifyBlock(const TPBlockBase *TPB) {
  auto *TPBB = dyn_cast<TPBasicBlock>(TPB);
  // Check block's condition bit.
  if (TPB->getNumSuccessors() > 1) {
    if (!TPBB || !TPBB->getTerminator()) {
      errs() << "Block has multiple successors but doesn't "
                "have a proper branch recipe!\n";
      return false;
    }
  } else {
    if (TPBB && TPBB->getTerminator()) {
      errs() << "Unexpected branch recipe!\n";
      return false;
    }
  }

  // Check block's successors.
  const auto &Successors = TPB->getSuccessors();
  // There must be only one instance of a successor in block's successor list.
  // TODO: This won't work for switch statements.
  if (hasDuplicates(Successors)) {
    errs() << "Multiple instances of the same successor.\n";
    return false;
  }

  for (const TPBlockBase *Succ : Successors) {
    // There must be a bi-directional link between block and successor.
    const auto &SuccPreds = Succ->getPredecessors();
    if (!is_contained(SuccPreds, TPB)) {
      errs() << "Missing predecessor link.\n";
      return false;
    }
  }

  // Check block's predecessors.
  const auto &Predecessors = TPB->getPredecessors();
  // There must be only one instance of a predecessor in block's predecessor
  // list.
  // TODO: This won't work for switch statements.
  if (hasDuplicates(Predecessors)) {
    errs() << "Multiple instances of the same predecessor.\n";
    return false;
  }

  for (const TPBlockBase *Pred : Predecessors) {
    // Block and predecessor must be inside the same region.
    if (Pred->getParent() != TPB->getParent()) {
      errs() << "Predecessor is not in the same region.\n";
      return false;
    }

    // There must be a bi-directional link between block and predecessor.
    const auto &PredSuccs = Pred->getSuccessors();
    if (!is_contained(PredSuccs, TPB)) {
      errs() << "Missing successor link.\n";
      return false;
    }
  }
  return !TPBB || verifyTPBasicBlock(TPBB);
}

bool TPlanVerifier::verifyBlocksInRegion(const TPRegionBlock *Region) {
  for (const TPBlockBase *TPB : tp_depth_first_shallow(Region->getEntry())) {
    // Check block's parent.
    if (TPB->getParent() != Region) {
      errs() << "TPBlockBase has wrong parent\n";
      return false;
    }

    if (!verifyBlock(TPB))
      return false;
  }
  return true;
}

bool TPlanVerifier::verifyRegion(const TPRegionBlock *Region) {
  const TPBlockBase *Entry = Region->getEntry();
  const TPBlockBase *Exiting = Region->getExiting();

  // Entry and Exiting shouldn't have any predecessor/successor, respectively.
  if (Entry->getNumPredecessors() != 0) {
    errs() << "region entry block has predecessors\n";
    return false;
  }
  if (Exiting->getNumSuccessors() != 0) {
    errs() << "region exiting block has successors\n";
    return false;
  }

  return verifyBlocksInRegion(Region);
}

bool TPlanVerifier::verifyRegionRec(const TPRegionBlock *Region) {
  // Recurse inside nested regions and check all blocks inside the region.
  return verifyRegion(Region) &&
         all_of(tp_depth_first_shallow(Region->getEntry()),
                [this](const TPBlockBase *TPB) {
                  const auto *SubRegion = dyn_cast<TPRegionBlock>(TPB);
                  return !SubRegion || verifyRegionRec(SubRegion);
                });
}

bool TPlanVerifier::verify(const TPlan &Plan) {
  //! FIXME(yuxin.an)
  // YG:REMOVE
  errs() << "[Info] [verify]\n";

  if (any_of(tp_depth_first_shallow(Plan.getEntry()),
             [this](const TPBlockBase *TPB) { return !verifyBlock(TPB); }))
    return false;

  const TPRegionBlock *TopRegion = Plan.getTensorLoopRegion();
  if (!verifyRegionRec(TopRegion))
    return false;

  if (TopRegion->getParent()) {
    errs() << "TPlan Top Region should have no parent.\n";
    return false;
  }

  const TPBasicBlock *Entry = dyn_cast<TPBasicBlock>(TopRegion->getEntry());
  if (!Entry) {
    errs() << "TPlan entry block is not a TPBasicBlock\n";
    return false;
  }

  for (auto LIdx2HElem : Plan.LoopIdx2HeaderTPBB) {
    auto *HeaderTPBB = LIdx2HElem.second;
    if (!isa<TPCanonicalIVPHIRecipe>(&*HeaderTPBB->begin())) {
      errs() << "TPlan tensor loop header does not start with a "
                "TPCanonicalIVPHIRecipe\n";
      return false;
    }
  }

  dbgs() << "[Warning] Please handle `TPlanVerifier::verify()` \n";

  // const TPBasicBlock *Exiting =
  // dyn_cast<TPBasicBlock>(TopRegion->getExiting()); if (!Exiting) {
  //   errs() << "TPlan exiting block is not a TPBasicBlock\n";
  //   return false;
  // }

  // if (Exiting->empty()) {
  //   errs() << "TPlan tensor loop exiting block must end with BranchOnCount or
  //   "
  //             "BranchOnCond TPInstruction but is empty\n";
  //   return false;
  // }

  // auto *LastInst = dyn_cast<TPInstruction>(std::prev(Exiting->end()));
  // if (!LastInst || (LastInst->getOpcode() != TPInstruction::BranchOnCount &&
  //                   LastInst->getOpcode() != TPInstruction::BranchOnCond)) {
  //   errs() << "TPlan tensor loop exit must end with BranchOnCount or "
  //             "BranchOnCond TPInstruction\n";
  //   return false;
  // }

  // for (const auto &KV : Plan.getLiveOuts())
  //   if (KV.second->getNumOperands() != 1) {
  //     errs() << "live outs must have a single operand\n";
  //     return false;
  //   }

  return true;
}

bool llvm::verifyTPlanIsValid(const TPlan &Plan) {
  TPDominatorTree TPDT;
  TPDT.recalculate(const_cast<TPlan &>(Plan));
  TPlanVerifier Verifier(TPDT);
  return Verifier.verify(Plan);
}
