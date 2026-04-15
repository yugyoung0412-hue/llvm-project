#ifndef LLVM_ANALYSIS_TENSORUTILS_H
#define LLVM_ANALYSIS_TENSORUTILS_H

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/VFABIDemangler.h"
#include "llvm/Support/CheckedArithmetic.h"

namespace llvm {

/// Compute the access-group list of access groups that @p Inst1 and @p Inst2
/// are both in. If either instruction does not access memory at all, it is
/// considered to be in every list.
///
/// If the list contains just one access group, it is returned directly. If the
/// list is empty, returns nullptr.
MDNode *intersectAccessGroups(const Instruction *Inst1,
                              const Instruction *Inst2);

/// Specifically, let Kinds = [MD_tbaa, MD_alias_scope, MD_noalias, MD_fpmath,
/// MD_nontemporal, MD_access_group, MD_mmra].
/// For K in Kinds, we get the MDNode for K from each of the
/// elements of VL, compute their "intersection" (i.e., the most generic
/// metadata value that covers all of the individual values), and set I's
/// metadata for M equal to the intersection value.
///
/// This function always sets a (possibly null) value for each K in Kinds.
Instruction *propagateMetadata(Instruction *I, ArrayRef<Value *> VL);
} // namespace llvm

#endif
