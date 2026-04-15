#ifndef LLVM_TRANSFORMS_TENSORIZE_TPLANVERIFIER_H
#define LLVM_TRANSFORMS_TENSORIZE_TPLANVERIFIER_H

namespace llvm {
class TPlan;

/// Verify invariants for general VPlans. Currently it checks the following:
/// 1. Region/Block verification: Check the Region/Block verification
/// invariants for every region in the H-CFG.
/// 2. all phi-like recipes must be at the beginning of a block, with no other
/// recipes in between. Note that currently there is still an exception for
/// VPBlendRecipes.
bool verifyTPlanIsValid(const TPlan &Plan);

} // namespace llvm

#endif // LLVM_TRANSFORMS_TENSORIZE_TPLANVERIFIER_H
