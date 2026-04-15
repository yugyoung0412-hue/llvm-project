#ifndef LLVM_TRANSFORMS_TENSORIZE_TPLANDOMINATORTREE_H
#define LLVM_TRANSFORMS_TENSORIZE_TPLANDOMINATORTREE_H

#include "TPlan.h"
#include "TPlanCFG.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/GenericDomTree.h"

namespace llvm {

template <> struct DomTreeNodeTraits<TPBlockBase> {
  using NodeType = TPBlockBase;
  using NodePtr = TPBlockBase *;
  using ParentPtr = TPlan *;

  static NodePtr getEntryNode(ParentPtr Parent) { return Parent->getEntry(); }
  static ParentPtr getParent(NodePtr B) { return B->getPlan(); }
};

///
/// Template specialization of the standard LLVM dominator tree utility for
/// VPBlockBases.
using TPDominatorTree = DomTreeBase<TPBlockBase>;

using TPDomTreeNode = DomTreeNodeBase<TPBlockBase>;

/// Template specializations of GraphTraits for VPDomTreeNode.
template <>
struct GraphTraits<TPDomTreeNode *>
    : public DomTreeGraphTraitsBase<TPDomTreeNode,
                                    TPDomTreeNode::const_iterator> {};

template <>
struct GraphTraits<const TPDomTreeNode *>
    : public DomTreeGraphTraitsBase<const TPDomTreeNode,
                                    TPDomTreeNode::const_iterator> {};
} // namespace llvm
#endif // LLVM_TRANSFORMS_TENSORIZE_TPLANDOMINATORTREE_H
