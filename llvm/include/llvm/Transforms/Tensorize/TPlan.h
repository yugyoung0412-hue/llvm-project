//===- TPlan.h - Represent A Vectorizer Plan --------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_TENSORIZE_TPLAN_H
#define LLVM_TRANSFORMS_TENSORIZE_TPLAN_H

#include "TPattern.h"
#include "TPlanAnalysis.h"
#include "TPlanValue.h"
#include "TensorizeCommon.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TensorUtils.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/FMF.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/Transforms/Tensorize/TPRecipeMatcher.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <string>

namespace llvm {

class BasicBlock;
class DominatorTree;
class TargetTransformInfo;
class SCEVExpander;
class LoopTensorizer;
class IRBuilderBase;
class LoopInfo;
class raw_ostream;
class RecurrenceDescriptor;
class SCEV;
class Type;
class TPBasicBlock;
class TPRegionBlock;
class TPlan;
class TPReplicateRecipe;
class TPlanSlp;
class Talue;
class LoopTensorizeCostModel;
class LoopVersioning;

struct TPCostContext;

namespace Intrinsic {
typedef unsigned ID;
}

/// Returns a calculation for the total number of elements for a given \p VF.
/// For fixed width vectors this value is a constant, whereas for scalable
/// vectors it is an expression determined at runtime.
Value *getRuntimeTF(IRBuilderBase &B, Type *Ty, ElementCount VF);

/// Return a value for Step multiplied by VF.
Value *createStepForTFElem(IRBuilderBase &B, Type *Ty, ElementCount TFElem,
                           int64_t Step);

/// A helper function that returns the reciprocal of the block probability of
/// predicated blocks. If we return X, we are assuming the predicated block
/// will execute once for every X iterations of the loop header.
///
/// TODO: We should use actual block probability here, if available. Currently,
///       we always assume predicated blocks have a 50% chance of executing.
inline unsigned getReciprocalPredBlockProb() { return 2; }

MapVector<Loop *, SCEV *> createTripCountSCEV(Type *IdxTy, ScalarEvolution &SE,
                                              SmallVector<Loop *> Loops);

/// A range of powers-of-2 vectorization factors with fixed start and
/// adjustable end. The range includes start and excludes end, e.g.,:
/// [1, 16) = {1, 2, 4, 8}
struct TFRange {
  // A power of 2. 
  // using TFTy = MapVector<Loop *, ElementCount>
  const TFTy Start;
  ElementCount tmp_start;
  ElementCount tmp_end;

  // A power of 2. If End <= Start range is empty.
  TFTy End;

  bool isEmpty() const {
    if (Start.empty() || End.empty())
      return true;
    for (auto [StartElem, EndElem] : zip(Start, End)) {
      if (EndElem.second.getKnownMinValue() <=
          StartElem.second.getKnownMinValue())
        return true;
    }
    return false;
  }

  TFRange(ElementCount Start, ElementCount End) : tmp_start(Start), tmp_end(End) {
    assert(Start.isScalable() == End.isScalable() &&
           "Both Start and End should have the same scalable flag");
    assert(isPowerOf2_32(Start.getKnownMinValue()) &&
           "Expected Start to be a power of 2");
    assert(isPowerOf2_32(End.getKnownMinValue()) &&
           "Expected End to be a power of 2");
  }

  TFRange(TFTy Start, TFTy End) : Start(Start), End(End) {
    for (auto [StartElem, EndElem] : zip(Start, End)) {
      // assert(StartElem.second.isScalable() == EndElem.second.isScalable() &&
      //        "Both Start and End should have the same scalable flag");
      // assert(isPowerOf2_32(StartElem.second.getKnownMinValue()) &&
      //        "Expected Start to be a power of 2");
      // assert(isPowerOf2_32(EndElem.second.getKnownMinValue()) &&
      //        "Expected End to be a power of 2");
    }
  }

  /// Iterator to iterate over vectorization factors in a VFRange.
  class iterator
      : public iterator_facade_base<iterator, std::forward_iterator_tag, TFTy> {
    TFTy TF;

  public:
    iterator(TFTy TF) : TF(TF) {}

    bool operator==(const iterator &Other) const {
      for (auto [TFElem, OtherTFElem] : zip(TF, *Other)) {
        if (TFElem.first != OtherTFElem.first ||
            TFElem.second != OtherTFElem.second)
          return false;
      }
      return true;
    }

    TFTy operator*() const { return TF; }

    iterator &operator++() {
      // TODO(yuxin.an): Need to check.
      for (auto &Elem : TF)
        Elem.second *= 2;
      return *this;
    }
  };

  iterator begin() { return iterator(Start); }
  iterator end() {
    for (auto Elem : End)
      assert(isPowerOf2_32(Elem.second.getKnownMinValue()));

    return iterator(End);
  }
};

using TPlanPtr = std::unique_ptr<TPlan>;

class TPLane { // yuxin.an: L156
public:
  /// Kind describes how to interpret Lane.
  enum class Kind : uint8_t {
    /// For First, Lane is the index into the first N elements of a
    /// fixed-vector <N x <ElTy>> or a scalable vector <vscale x N x <ElTy>>.
    First,
    /// For ScalableLast, Lane is the offset from the start of the last
    /// N-element subvector in a scalable vector <vscale x N x <ElTy>>. For
    /// example, a Lane of 0 corresponds to lane `(vscale - 1) * N`, a Lane of
    /// 1 corresponds to `((vscale - 1) * N) + 1`, etc.
    ScalableLast
  };

private:
  /// in [0..VF)
  unsigned Lane;

  /// Indicates how the Lane should be interpreted, as described above.
  Kind LaneKind;

public:
  TPLane(unsigned Lane, Kind LaneKind) : Lane(Lane), LaneKind(LaneKind) {}

  static TPLane getFirstLane() { return TPLane(0, TPLane::Kind::First); }

  static TPLane getLaneFromEnd(const ElementCount &VF, unsigned Offset) {
    assert(Offset > 0 && Offset <= VF.getKnownMinValue() &&
           "trying to extract with invalid offset");
    unsigned LaneOffset = VF.getKnownMinValue() - Offset;
    Kind LaneKind;
    if (VF.isScalable())
      // In this case 'LaneOffset' refers to the offset from the start of the
      // last subvector with VF.getKnownMinValue() elements.
      LaneKind = TPLane::Kind::ScalableLast;
    else
      LaneKind = TPLane::Kind::First;
    return TPLane(LaneOffset, LaneKind);
  }

  static TPLane getLastLaneForTF(const ElementCount &VF) {
    return getLaneFromEnd(VF, 1);
  }

  /// Returns a compile-time known value for the lane index and asserts if the
  /// lane can only be calculated at runtime.
  unsigned getKnownLane() const {
    assert(LaneKind == Kind::First);
    return Lane;
  }

  /// Returns an expression describing the lane index that can be used at
  /// runtime.
  Value *getAsRuntimeExpr(IRBuilderBase &Builder, const ElementCount &VF) const;

  /// Returns the Kind of lane offset.
  Kind getKind() const { return LaneKind; }

  /// Returns true if this is the first lane of the whole vector.
  bool isFirstLane() const { return Lane == 0 && LaneKind == Kind::First; }

  /// Maps the lane to a cache index based on \p VF.
  unsigned mapToCacheIndex(const ElementCount &VF) const {
    switch (LaneKind) {
    case TPLane::Kind::ScalableLast:
      assert(VF.isScalable() && Lane < VF.getKnownMinValue());
      return VF.getKnownMinValue() + Lane;
    default:
      assert(Lane < VF.getKnownMinValue());
      return Lane;
    }
  }

  /// Returns the maxmimum number of lanes that we are able to consider
  /// caching for \p VF.
  static unsigned getNumCachedLanes(const ElementCount &VF) {
    return VF.getKnownMinValue() * (VF.isScalable() ? 2 : 1);
  }
};

struct TPIteration { // yuxin.an: L238
  unsigned Part;

  TPLane Lane;

  TPIteration(unsigned Part, unsigned Lane,
              TPLane::Kind Kind = TPLane::Kind::First)
      : Part(Part), Lane(Lane, Kind) {}

  TPIteration(unsigned Part, const TPLane &Lane) : Part(Part), Lane(Lane) {}

  bool isFirstIteration() const { return Part == 0 && Lane.isFirstLane(); }
};

struct TensorBlocks {
  BasicBlock *EntryB = nullptr;  // Entry Block
  BasicBlock *MiddleB = nullptr; // Middle Block
  BasicBlock *SPH = nullptr;     // Scalar PreHeader
  BasicBlock *TPH = nullptr;     // Tensor PreHeader
  BasicBlock *TEntry = nullptr;
  BasicBlock *TExiting = nullptr;
  DenseMap<Loop *, BasicBlock *> Loop2HeadBB;
  DenseMap<Loop *, BasicBlock *> Loop2LatchBB;
};

//===----------------------------------------------------------------------===//
// EmissionPolicy — per-dim lowering intent built from TPlan before execute()
//===----------------------------------------------------------------------===//

/// How a tensor dimension should be emitted during lowering.
enum class DimEmitMode {
  Inline,       ///< TC <= PF for this dim: no tiling loop needed.
  StaticTiled,  ///< TC > PF (known at compile time) or dynamic output dim:
                ///< emit umin-bounded tiling loop via emitTilingLoop().
  DynamicTiled, ///< Reduction dim with runtime TC: emit fixed-tile loop
                ///< (tensor.body) + epilogue tiers + scalar remainder.
};

/// Per-dimension emission specification built by buildEmissionPolicy().
/// Dim indices use the DimIdx convention (innermost=0, outermost=Depth-1).
struct DimEmissionSpec {
  unsigned    Dim;                     ///< Dimension index.
  unsigned    PF;                      ///< Tile size from Plan.getPFForDim(Dim).
  DimEmitMode Mode = DimEmitMode::Inline;
};

/// Upfront per-lowering emission plan: classifies every tensor dim before
/// execute() runs. Built by buildEmissionPolicy() in LoopTensorizePlanner::executePlan().
///
/// Seperates the "what to emit" decision (here, driven by TPlan's PF/TC data)
/// from the "how to emit" mechanics (inside emitContraction()).
struct EmissionPolicy {
  SmallVector<DimEmissionSpec, 4> Specs;

  /// True iff any dim uses DynamicTiled mode.
  /// When true, LoopTensorizePlanner::executePlan() must call createTensorizedLoopSkeleton()
  /// before exeucte() to insert a runtime profitability guard.
  bool needsGuard() const {
    return llvm::any_of(Specs, [](const DimEmissionSpec &S) {
      return S.Mode == DimEmitMode::DynamicTiled;
    });
  }

  /// Returns the spec for dim \p Dim, or nullptr if not present.
  const DimEmissionSpec *getSpec(unsigned Dim) const {
    for (const auto &S : Specs)
      if (S.Dim == Dim)
        return &S;
    return nullptr;
  }

  /// True if any dim requires a tiling loop (Static or Dynamic).
  bool needsTiling() const {
    return llvm::any_of(Specs, [](const DimEmissionSpec &S) {
      return S.Mode != DimEmitMode::Inline;
    });
  }
};

// yuxin.an: L255
struct TPTransformState {
  // TODO(yuxin.an)
  TPTransformState(TFTy VF, TUFTy UF, LoopInfo *LI, DominatorTree *DT,
                   IRBuilderBase &Builder, LoopTensorizer *LT, TPlan *Plan,
                   LLVMContext &Ctx);

  /// The chosen Vectorization and Unroll Factors of the loop being vectorized.
  TFTy TF;
  TUFTy UF;

  const TargetTransformInfo *TTI;
  /// TPlan dim index -> Loop*.
  /// Used by decomposePtrForDims() in emitContraction().
  MapVector<unsigned, Loop *> DimToLoop;

  DenseMap<const TPDef *, Value *> ValueMap;

  /// Upfront emission policy built by buildEmissionPolicy() before execute().
  /// Consumed by emitContraction() to classify dims without re-running checkDim().
  EmissionPolicy Policy;

  /// Hold the indices to generate specific scalar instructions. Null indicates
  /// that all instances are to be generated, using either scalar or vector
  /// instructions.
  std::optional<TPIteration> Instance;

  struct DataState {
    /// A type for vectorized values in the new loop. Each value from the
    /// original loop, when vectorized, is represented by UF vector values in
    /// the new unrolled loop, where UF is the unroll factor.
    typedef SmallVector<Value *, 2> PerPartValuesTy;

    DenseMap<TPValue *, PerPartValuesTy> PerPartOutput;

    using ScalarsPerPartValuesTy = SmallVector<SmallVector<Value *, 4>, 2>;
    DenseMap<TPValue *, ScalarsPerPartValuesTy> PerPartScalars;
  } Data;

  /// Get the generated vector Value for a given VPValue \p Def and a given \p
  /// Part if \p IsScalar is false, otherwise return the generated scalar
  /// for \p Part. \See set.
  Value *get(TPValue *Def, unsigned Part, bool IsScalar = false);

  /// Get the generated Value for a given VPValue and given Part and Lane.
  Value *get(TPValue *Def, const TPIteration &Instance, Loop *L);

  /// Get the generated Value for a given VPValue and given Part and Lane.
  DenseMap<Loop *, Value *> get(DenseMap<Loop *, TPValue *> Def,
                                const TPIteration &Instance);

  Value *getValue(const TPDef *V) const { return ValueMap.lookup(V); }
  void setValue(const TPDef *V, Value *IRV) { ValueMap[V] = IRV; }

  bool hasTensorValue(TPValue *Def, unsigned Part) {
    auto I = Data.PerPartOutput.find(Def);
    return I != Data.PerPartOutput.end() && Part < I->second.size() &&
           I->second[Part];
  }

  bool hasScalarValue(TPValue *Def, TPIteration Instance, Loop *L) {
    auto I = Data.PerPartScalars.find(Def);
    if (I == Data.PerPartScalars.end())
      return false;
    unsigned CacheIdx = Instance.Lane.mapToCacheIndex(TF[L]);
    return Instance.Part < I->second.size() &&
           CacheIdx < I->second[Instance.Part].size() &&
           I->second[Instance.Part][CacheIdx];
  }

  /// Set by TPlanTransformer before execute() for the tiling dim's trip-count.
  /// TPTilingRegion::execute() reads this to compute tile loop bounds.
  Value *TilingTCVal = nullptr;

  /// Set the generated vector Value for a given VPValue and a given Part, if \p
  /// IsScalar is false. If \p IsScalar is true, set the scalar in (Part, 0).
  void set(TPValue *Def, Value *V, unsigned Part, bool IsScalar = false) {
    // TODO(yuxin.an)
    llvm_unreachable("");
  }

  /// Reset an existing vector value for \p Def and a given \p Part.
  void reset(TPValue *Def, Value *V, unsigned Part) {
    // TODO(yuxin.an):
    llvm_unreachable("");
  }

  /// Set the generated scalar \p V for \p Def and the given \p Instance.
  void set(TPValue *Def, Value *V, const TPIteration &Instance) {
    // TODO(yuxin.an):
    llvm_unreachable("");
  }

  /// Reset an existing scalar value for \p Def and a given \p Instance.
  void reset(TPValue *Def, Value *V, const TPIteration &Instance) {
    // TODO(yuxin.an):
    llvm_unreachable("");
  }

  /// Add additional metadata to \p To that was not present on \p Orig.
  ///
  /// Currently this is used to add the noalias annotations based on the
  /// inserted memchecks.  Use this for instructions that are *cloned* into the
  /// vector loop.
  void addNewMetadata(Instruction *To, const Instruction *Orig);

  /// Add metadata from one instruction to another.
  ///
  /// This includes both the original MDs from \p From and additional ones (\see
  /// addNewMetadata).  Use this for *newly created* instructions in the vector
  /// loop.
  void addMetadata(Value *To, Instruction *From);

  /// Set the debug location in the builder using the debug location \p DL.
  void setDebugLocFrom(DebugLoc DL);

  /// Construct the vector value of a scalarized value \p V one lane at a time.
  void packScalarIntoTensorValue(TPValue *Def, const TPIteration &Instance);

  struct CFGState {
    /// The previous VPBasicBlock visited. Initially set to null.
    TPBasicBlock *PrevTPBB = nullptr;

    /// The previous IR BasicBlock created or used. Initially set to the new
    /// header BasicBlock.
    BasicBlock *PrevBB = nullptr;

    /// The last IR BasicBlock in the output IR. Set to the exit block of the
    /// vector loop.
    BasicBlock *ExitBB = nullptr;

    /// A mapping of each VPBasicBlock to the corresponding BasicBlock. In case
    /// of replication, maps the BasicBlock of the last replica created.
    SmallDenseMap<TPBasicBlock *, BasicBlock *> TPBB2IRBB;

    /// Updater for the DominatorTree.
    DomTreeUpdater DTU;

    CFGState(DominatorTree *DT)
        : DTU(DT, DomTreeUpdater::UpdateStrategy::Lazy) {}

    /// Returns the BasicBlock* mapped to the pre-header of the loop region
    /// containing \p R.
    BasicBlock *getPreheaderBBFor(TPRecipeBase *R);
  } CFG;

  /// Hold a pointer to LoopInfo to register new basic blocks in the loop.
  LoopInfo *LI;

  DominatorTree *DT;

  /// Hold a reference to the IRBuilder used to generate output IR code.
  IRBuilderBase &Builder;

  /// Hold a pointer to InnerLoopVectorizer to reuse its IR generation methods.
  LoopTensorizer *LT;

  /// Pointer to the VPlan code is generated for.
  TPlan *Plan;

  DenseMap<Loop *, Loop *> Loop2TensorLoop;

  Loop *CurLoop = nullptr;

  /// The loop object for the current parent region, or nullptr.
  Loop *CurrentTensorLoop = nullptr;

  /// LoopVersioning.  It's only set up (non-null) if memchecks were
  /// used.
  ///
  /// This is currently only used to add no-alias metadata based on the
  /// memchecks.  The actually versioning is performed manually.
  LoopVersioning *LVer = nullptr;

  ScalarEvolution *SE;

  SCEVExpander *Expander;

  /// Map SCEVs to their expanded values. Populated when executing
  /// VPExpandSCEVRecipes.
  DenseMap<const SCEV *, Value *> ExpandedSCEVs;

  /// VPlan-based type analysis.
  TPTypeAnalysis TypeAnalysis;

  TensorBlocks TBS;
  DenseMap<TPBasicBlock *, BasicBlock *> TPBB2BB;
  DenseMap<BasicBlock *, TPBasicBlock *> BB2TPBB;
  DenseMap<TPValue *, Value *> TPValue2Value;
  DenseMap<TPBasicBlock *, TPBasicBlock *> BackedgeTPBB;
  DenseMap<BasicBlock *, BasicBlock *> BackedgeBB;
  BasicBlock *CurBB;
  DenseMap<BasicBlock *, Value *> IdxAddMap;
  DenseMap<PHINode *, Value *> BackedgeValues;
};

class TPBlockBase { // yuxin.an: L437
  // TODO(yuxin.an)
  friend class TPBlockUtils;

  const unsigned char SubclassID; ///< Subclass identifier (for isa/dyn_cast).

  /// An optional name for the block.
  std::string Name;

  /// The immediate VPRegionBlock which this VPBlockBase belongs to, or null if
  /// it is a topmost VPBlockBase.
  TPRegionBlock *Parent = nullptr;

  /// List of predecessor blocks.
  SmallVector<TPBlockBase *, 1> Predecessors;

  /// List of successor blocks.
  SmallVector<TPBlockBase *, 1> Successors;

  /// VPlan containing the block. Can only be set on the entry block of the
  /// plan.
  TPlan *Plan = nullptr;

  /// Add \p Successor as the last successor to this block.
  void appendSuccessor(TPBlockBase *Successor) {
    assert(Successor && "Cannot add nullptr successor!");
    Successors.push_back(Successor);
  }

  /// Add \p Predecessor as the last predecessor to this block.
  void appendPredecessor(TPBlockBase *Predecessor) {
    assert(Predecessor && "Cannot add nullptr predecessor!");
    Predecessors.push_back(Predecessor);
  }

  /// Remove \p Predecessor from the predecessors of this block.
  void removePredecessor(TPBlockBase *Predecessor) {
    auto Pos = find(Predecessors, Predecessor);
    assert(Pos && "Predecessor does not exist");
    Predecessors.erase(Pos);
  }

  /// Remove \p Successor from the successors of this block.
  void removeSuccessor(TPBlockBase *Successor) {
    auto Pos = find(Successors, Successor);
    assert(Pos && "Successor does not exist");
    Successors.erase(Pos);
  }

protected:
  TPBlockBase(const unsigned char SC, const std::string &N)
      : SubclassID(SC), Name(N) {}

public:
  /// An enumeration for keeping track of the concrete subclass of VPBlockBase
  /// that are actually instantiated. Values of this enumeration are kept in the
  /// SubclassID field of the VPBlockBase objects. They are used for concrete
  /// type identification.
  using TPBlockTy = enum { TPRegionBlockSC, TPBasicBlockSC, TPIRBasicBlockSC, TPGuardBlockSC, TPTilingRegionSC };

  using TPBlocksTy = SmallVectorImpl<TPBlockBase *>;

  virtual ~TPBlockBase() = default;

  const std::string &getName() const { return Name; }

  void setName(const Twine &newName) { Name = newName.str(); }

  /// \return an ID for the concrete type of this object.
  /// This is used to implement the classof checks. This should not be used
  /// for any other purpose, as the values may change as LLVM evolves.
  unsigned getTPBlockID() const { return SubclassID; }

  TPRegionBlock *getParent() { return Parent; }
  const TPRegionBlock *getParent() const { return Parent; }

  /// \return A pointer to the plan containing the current block.
  TPlan *getPlan();
  const TPlan *getPlan() const;

  /// Sets the pointer of the plan containing the block. The block must be the
  /// entry block into the VPlan.
  void setPlan(TPlan *ParentPlan);

  void setParent(TPRegionBlock *P) { Parent = P; }

  /// \return the VPBasicBlock that is the entry of this VPBlockBase,
  /// recursively, if the latter is a VPRegionBlock. Otherwise, if this
  /// VPBlockBase is a VPBasicBlock, it is returned.
  const TPBasicBlock *getEntryBasicBlock() const;
  TPBasicBlock *getEntryBasicBlock();

  /// \return the VPBasicBlock that is the exiting this VPBlockBase,
  /// recursively, if the latter is a VPRegionBlock. Otherwise, if this
  /// VPBlockBase is a VPBasicBlock, it is returned.
  const TPBasicBlock *getExitingBasicBlock() const;
  TPBasicBlock *getExitingBasicBlock();

  const TPBlocksTy &getSuccessors() const { return Successors; }
  TPBlocksTy &getSuccessors() { return Successors; }

  iterator_range<TPBlockBase **> successors() { return Successors; }

  const TPBlocksTy &getPredecessors() const { return Predecessors; }
  TPBlocksTy &getPredecessors() { return Predecessors; }

  /// \return the successor of this VPBlockBase if it has a single successor.
  /// Otherwise return a null pointer.
  TPBlockBase *getSingleSuccessor() const {
    // YYG:REMOVE
    errs() << "[getSingleSuccessor] Successors.size(): " << Successors.size() << "\n";
    return (Successors.size() == 1 ? *Successors.begin() : nullptr);
  }

  /// \return the predecessor of this VPBlockBase if it has a single
  /// predecessor. Otherwise return a null pointer.
  TPBlockBase *getSinglePredecessor() const {
    return (Predecessors.size() == 1 ? *Predecessors.begin() : nullptr);
  }

  size_t getNumSuccessors() const { return Successors.size(); }
  size_t getNumPredecessors() const { return Predecessors.size(); }

  /// An Enclosing Block of a block B is any block containing B, including B
  /// itself. \return the closest enclosing block starting from "this", which
  /// has successors. \return the root enclosing block if all enclosing blocks
  /// have no successors.
  TPBlockBase *getEnclosingBlockWithSuccessors();

  /// \return the closest enclosing block starting from "this", which has
  /// predecessors. \return the root enclosing block if all enclosing blocks
  /// have no predecessors.
  TPBlockBase *getEnclosingBlockWithPredecessors();

  /// \return the successors either attached directly to this VPBlockBase or, if
  /// this VPBlockBase is the exit block of a VPRegionBlock and has no
  /// successors of its own, search recursively for the first enclosing
  /// VPRegionBlock that has successors and return them. If no such
  /// VPRegionBlock exists, return the (empty) successors of the topmost
  /// VPBlockBase reached.
  const TPBlocksTy &getHierarchicalSuccessors() {
    return getEnclosingBlockWithSuccessors()->getSuccessors();
  }

  /// \return the hierarchical successor of this VPBlockBase if it has a single
  /// hierarchical successor. Otherwise return a null pointer.
  TPBlockBase *getSingleHierarchicalSuccessor() {
    return getEnclosingBlockWithSuccessors()->getSingleSuccessor();
  }

  /// \return the predecessors either attached directly to this VPBlockBase or,
  /// if this VPBlockBase is the entry block of a VPRegionBlock and has no
  /// predecessors of its own, search recursively for the first enclosing
  /// VPRegionBlock that has predecessors and return them. If no such
  /// VPRegionBlock exists, return the (empty) predecessors of the topmost
  /// VPBlockBase reached.
  const TPBlocksTy &getHierarchicalPredecessors() {
    return getEnclosingBlockWithPredecessors()->getPredecessors();
  }

  /// \return the hierarchical predecessor of this VPBlockBase if it has a
  /// single hierarchical predecessor. Otherwise return a null pointer.
  TPBlockBase *getSingleHierarchicalPredecessor() {
    return getEnclosingBlockWithPredecessors()->getSinglePredecessor();
  }

  /// Set a given VPBlockBase \p Successor as the single successor of this
  /// VPBlockBase. This VPBlockBase is not added as predecessor of \p Successor.
  /// This VPBlockBase must have no successors.
  void setOneSuccessor(TPBlockBase *Successor) {
    assert(Successors.empty() && "Setting one successor when others exist.");
    assert(Successor->getParent() == getParent() &&
           "connected blocks must have the same parent");
    appendSuccessor(Successor);
  }

  /// Set two given VPBlockBases \p IfTrue and \p IfFalse to be the two
  /// successors of this VPBlockBase. This VPBlockBase is not added as
  /// predecessor of \p IfTrue or \p IfFalse. This VPBlockBase must have no
  /// successors.
  void setTwoSuccessors(TPBlockBase *IfTrue, TPBlockBase *IfFalse) {
    assert(Successors.empty() && "Setting two successors when others exist.");
    appendSuccessor(IfTrue);
    appendSuccessor(IfFalse);
  }

  /// Set each VPBasicBlock in \p NewPreds as predecessor of this VPBlockBase.
  /// This VPBlockBase must have no predecessors. This VPBlockBase is not added
  /// as successor of any VPBasicBlock in \p NewPreds.
  void setPredecessors(ArrayRef<TPBlockBase *> NewPreds) {
    assert(Predecessors.empty() && "Block predecessors already set.");
    for (auto *Pred : NewPreds)
      appendPredecessor(Pred);
  }

  /// Set each VPBasicBlock in \p NewSuccss as successor of this VPBlockBase.
  /// This VPBlockBase must have no successors. This VPBlockBase is not added
  /// as predecessor of any VPBasicBlock in \p NewSuccs.
  void setSuccessors(ArrayRef<TPBlockBase *> NewSuccs) {
    // assert(Successors.empty() && "Block successors already set.");
    for (auto *Succ : NewSuccs)
      appendSuccessor(Succ);
  }

  /// Remove all the predecessor of this block.
  void clearPredecessors() { Predecessors.clear(); }

  /// Remove all the successors of this block.
  void clearSuccessors() { Successors.clear(); }

  /// The method which generates the output IR that correspond to this
  /// VPBlockBase, thereby "executing" the VPlan.
  virtual void execute(TPTransformState *State) = 0;

  /// Return the cost of the block.
  // TODO(yuxin.an): Need to confirm VF -> TF.
  virtual InstructionCost cost(ElementCount VF, TPCostContext &Ctx) = 0;

  /// Delete all blocks reachable from a given VPBlockBase, inclusive.
  static void deleteCFG(TPBlockBase *Entry);

  /// Return true if it is legal to hoist instructions into this block.
  bool isLegalToHoistInto() {
    // There are currently no constraints that prevent an instruction to be
    // hoisted into a VPBlockBase.
    return true;
  }

  /// Replace all operands of VPUsers in the block with \p NewValue and also
  /// replaces all uses of VPValues defined in the block with NewValue.
  virtual void dropAllReferences(TPValue *NewValue) = 0;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void printAsOperand(raw_ostream &OS, bool PrintType) const {
    OS << getName();
  }

  /// Print plain-text dump of this VPBlockBase to \p O, prefixing all lines
  /// with \p Indent. \p SlotTracker is used to print unnamed VPValue's using
  /// consequtive numbers.
  ///
  /// Note that the numbering is applied to the whole VPlan, so printing
  /// individual blocks is consistent with the whole VPlan printing.
  virtual void print(raw_ostream &O, const Twine &Indent,
                     TPSlotTracker &SlotTracker) const = 0;

  /// Print plain-text dump of this VPlan to \p O.
  void print(raw_ostream &O) const {
    TPSlotTracker SlotTracker(getPlan());
    print(O, "", SlotTracker);
  }

  /// Print the successors of this block to \p O, prefixing all lines with \p
  /// Indent.
  void printSuccessors(raw_ostream &O, const Twine &Indent) const;

  /// Dump this VPBlockBase to dbgs().
  LLVM_DUMP_METHOD void dump() const { print(dbgs()); }
#endif

  /// Clone the current block and it's recipes without updating the operands of
  /// the cloned recipes, including all blocks in the single-entry single-exit
  /// region for VPRegionBlocks.
  virtual TPBlockBase *clone() = 0;
};

/// A value that is used outside the VPlan. The operand of the user needs to be
/// added to the associated phi node. The incoming block from VPlan is
/// determined by where the VPValue is defined: if it is defined by a recipe
/// outside a region, its parent block is used, otherwise the middle block is
/// used.
class TPLiveOut : public TPUser { // yuxin.an: L704
  PHINode *Phi;

public:
  TPLiveOut(PHINode *Phi, TPValue *Op)
      : TPUser({Op}, TPUser::TPUserID::LiveOut), Phi(Phi) {}

  static inline bool classof(const TPUser *U) {
    return U->getTPUserID() == TPUser::TPUserID::LiveOut;
  }

  /// Fix the wrapped phi node. This means adding an incoming value to exit
  /// block phi's from the vector loop via middle block (values from scalar loop
  /// already reach these phi's), and updating the value to scalar header phi's
  /// from the scalar preheader.
  void fixPhi(TPlan &Plan, TPTransformState &State);

  /// Returns true if the VPLiveOut uses scalars of operand \p Op.
  bool usesScalars(const TPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return true;
  }

  PHINode *getPhi() const { return Phi; }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the VPLiveOut to \p O.
  void print(raw_ostream &O, TPSlotTracker &SlotTracker) const;
#endif
};

/// Struct to hold various analysis needed for cost computations.
struct TPCostContext { // yuxin.an: L737
  // TODO(yuxin.an)
};

struct DimPF {
  //SmallVector<ElementCount, 8> PFs = { ElementCount::getFixed(1) };
  
  /// Loop dim indices this value.
  SmallBitVector DimSet;
};

// yuxin.an: L762
class TPRecipeBase : public ilist_node_with_parent<TPRecipeBase, TPBasicBlock>,
                     public TPDef,
                     public TPUser {
  // TODO(yuxin.an)
  friend TPBasicBlock;
  friend class TPBlockUtils;

  /// Each VPRecipe belongs to a single VPBasicBlock.
  TPBasicBlock *Parent = nullptr;

  /// The debug location for the recipe.
  DebugLoc DL;

  /// Parallel Factor (PF)
  ElementCount PF = ElementCount::getFixed(1);

  /// Represent index of loop-induction variable (0 = inner-most)
  int DimIndex = -1;
  
  RecipeClassification Out = { TensorOpKind::Scalar, -1, nullptr };

  /// When true, execute() is a no-op. Set by TPlanTransformer on recipes
  /// that are absorbed into a tensor op (e.g. loads/fmul/fadd subsumed by
  /// tensor.contract). ScalarEpilogue copies always have IsSubsumed=false.
  bool IsSubsumed = false;

public:
  TPRecipeBase(const unsigned char SC, ArrayRef<TPValue *> Operands,
               DebugLoc DL = {})
      : TPDef(SC), TPUser(Operands, TPUser::TPUserID::Recipe), DL(DL) {}

  template <typename IterT>
  TPRecipeBase(const unsigned char SC, iterator_range<IterT> Operands,
               DebugLoc DL = {})
      : TPDef(SC), TPUser(Operands, TPUser::TPUserID::Recipe), DL(DL) {}
  virtual ~TPRecipeBase() = default;

  bool isSubsumed() const { return IsSubsumed; }
  void setSubsumed(bool V = true) { IsSubsumed = V; }

  /// Clone the current recipe.
  virtual TPRecipeBase *clone() = 0;

  void setTensorOpKind(RecipeClassification New) { Out = New; }

  TensorOpKind getTensorOpKind() const { return Out.Kind; }

  int getContractDim() const { return Out.ContractDim; }

  TPRecipeBase* getFusedMulRecipe() const { return Out.FusedMulRecipe; }

  void setDimIndex(int dimIdx) { DimIndex = dimIdx; }

  int getDimIndex() { return DimIndex; }
  
  SmallBitVector DimSet;
  
  /// Loop dim indices this value spans;
  DimPF *dimPF;

  ElementCount getPF() const { return PF; }

  /// Apply PF
  void applyPF(ElementCount newPF) { this->PF = newPF; }
  
  void setPF(ElementCount NewPF) { PF = NewPF; }

  /// \return the VPBasicBlock which this VPRecipe belongs to.
  TPBasicBlock *getParent() { return Parent; }
  const TPBasicBlock *getParent() const { return Parent; }

  /// The method which generates the output IR instructions that correspond to
  /// this VPRecipe, thereby "executing" the VPlan.
  virtual void execute(TPTransformState &State) = 0;

  /// Return the cost of this recipe, taking into account if the cost
  /// computation should be skipped and the ForceTargetInstructionCost flag.
  /// Also takes care of printing the cost for debugging.
  virtual InstructionCost cost(ElementCount VF, TPCostContext &Ctx);

  /// Insert an unlinked recipe into a basic block immediately before
  /// the specified recipe.
  void insertBefore(TPRecipeBase *InsertPos);

  /// Returns the single-def value, or nullptr for stores.
  TPSingleDefRecipe *getDefinedValue();
  const TPSingleDefRecipe *getDefinedValue() const;

  /// Insert an unlinked recipe into \p BB immediately before the insertion
  /// point \p IP;
  void insertBefore(TPBasicBlock &BB, iplist<TPRecipeBase>::iterator IP);

  /// Insert an unlinked Recipe into a basic block immediately after
  /// the specified Recipe.
  void insertAfter(TPRecipeBase *InsertPos);

  /// Unlink this recipe from its current VPBasicBlock and insert it into
  /// the VPBasicBlock that MovePos lives in, right after MovePos.
  void moveAfter(TPRecipeBase *MovePos);

  /// Unlink this recipe and insert into BB before I.
  ///
  /// \pre I is a valid iterator into BB.
  void moveBefore(TPBasicBlock &BB, iplist<TPRecipeBase>::iterator I);

  /// This method unlinks 'this' from the containing basic block, but does not
  /// delete it.
  void removeFromParent();

  /// This method unlinks 'this' from the containing basic block and deletes it.
  ///
  /// \returns an iterator pointing to the element after the erased one
  iplist<TPRecipeBase>::iterator eraseFromParent();

  /// Method to support type inquiry through isa, cast, and dyn_cast.
  static inline bool classof(const TPDef *D) {
    // All VPDefs are also VPRecipeBases.
    return true;
  }

  static inline bool classof(const TPUser *U) {
    return U->getTPUserID() == TPUser::TPUserID::Recipe;
  }

  /// Returns true if the recipe may have side-effects.
  bool mayHaveSideEffects() const;

  /// Returns true for PHI-like recipes.
  bool isPhi() const {
    return getTPDefID() >= TPFirstPHISC && getTPDefID() <= TPLastPHISC;
  }

  /// Returns true if the recipe may read from memory.
  bool mayReadFromMemory() const;

  /// Returns true if the recipe may write to memory.
  bool mayWriteToMemory() const;

  /// Returns true if the recipe may read from or write to memory.
  bool mayReadOrWriteMemory() const {
    return mayReadFromMemory() || mayWriteToMemory();
  }

  /// Returns the debug location of the recipe.
  DebugLoc getDebugLoc() const { return DL; }

protected:
  /// Compute the cost of this recipe using the legacy cost model and the
  /// underlying instructions.
  InstructionCost computeCost(ElementCount VF, TPCostContext &Ctx) const;
};

// Helper macro to define common classof implementations for recipes.
// Inline definitions for TPRecipeBase member functions.
inline TPSingleDefRecipe *TPRecipeBase::getDefinedValue() {
  return dyn_cast<TPSingleDefRecipe>(this);
}
inline const TPSingleDefRecipe *TPRecipeBase::getDefinedValue() const {
  return dyn_cast<TPSingleDefRecipe>(this);
}

#define TP_CLASSOF_IMPL(TPDefID)                                               \
  static inline bool classof(const TPDef *D) {                                 \
    return D->getTPDefID() == TPDefID;                                         \
  }                                                                            \
  static inline bool classof(const TPValue *V) {                               \
    auto *R = V->getDefiningRecipe();                                          \
    return R && R->getTPDefID() == TPDefID;                                    \
  }                                                                            \
  static inline bool classof(const TPUser *U) {                                \
    auto *R = dyn_cast<TPRecipeBase>(U);                                       \
    return R && R->getTPDefID() == TPDefID;                                    \
  }                                                                            \
  static inline bool classof(const TPRecipeBase *R) {                          \
    return R->getTPDefID() == TPDefID;                                         \
  }                                                                            \
  static inline bool classof(const TPSingleDefRecipe *R) {                     \
    return R->getTPDefID() == TPDefID;                                         \
  }

/// VPSingleDef is a base class for recipes for modeling a sequence of one or
/// more output IR that define a single result VPValue.
/// Note that VPRecipeBase must be inherited from before VPValue.
class TPSingleDefRecipe : public TPRecipeBase, public TPValue {
public:
  template <typename IterT>
  TPSingleDefRecipe(const unsigned char SC, IterT Operands, DebugLoc DL = {})
      : TPRecipeBase(SC, Operands, DL), TPValue(this) {}

  TPSingleDefRecipe(const unsigned char SC, ArrayRef<TPValue *> Operands,
                    DebugLoc DL = {})
      : TPRecipeBase(SC, Operands, DL), TPValue(this) {}

  template <typename IterT>
  TPSingleDefRecipe(const unsigned char SC, IterT Operands, Value *UV,
                    DebugLoc DL = {})
      : TPRecipeBase(SC, Operands, DL), TPValue(this, UV) {}

  TPSingleDefRecipe(const unsigned char SC, ArrayRef<TPValue *> Operands,
                    Value *UV, DebugLoc DL = {})
      : TPRecipeBase(SC, Operands, DL), TPValue(this, UV) {}

  static inline bool classof(const TPRecipeBase *R) {
    switch (R->getTPDefID()) {
    case TPRecipeBase::TPDerivedIVSC:
    case TPRecipeBase::TPEVLBasedIVPHISC:
    case TPRecipeBase::TPExpandSCEVSC:
    case TPRecipeBase::TPInstructionSC:
    case TPRecipeBase::TPReductionEVLSC:
    case TPRecipeBase::TPReductionSC:
    case TPRecipeBase::TPNewInstrSC:
    case TPRecipeBase::TPReplicateSC:
    case TPRecipeBase::TPScalarIVStepsSC:
    case TPRecipeBase::TPVectorPointerSC:
    case TPRecipeBase::TPWidenCallSC:
    case TPRecipeBase::TPMatrixCallSC:
    case TPRecipeBase::TPWidenCanonicalIVSC:
    case TPRecipeBase::TPWidenCastSC:
    case TPRecipeBase::TPWidenGEPSC:
    case TPRecipeBase::TPWidenSC:
    case TPRecipeBase::TPWidenSelectSC:
    case TPRecipeBase::TPBlendSC:
    case TPRecipeBase::TPPredInstPHISC:
    case TPRecipeBase::TPCanonicalIVPHISC:
    case TPRecipeBase::TPActiveLaneMaskPHISC:
    case TPRecipeBase::TPFirstOrderRecurrencePHISC:
    case TPRecipeBase::TPWidenPHISC:
    case TPRecipeBase::TPWidenIntOrFpInductionSC:
    case TPRecipeBase::TPWidenPointerInductionSC:
    case TPRecipeBase::TPReductionPHISC:
    case TPRecipeBase::TPScalarCastSC:
      return true;
    case TPRecipeBase::TPInterleaveSC:
    case TPRecipeBase::TPBranchOnMaskSC:
    case TPRecipeBase::TPWidenLoadEVLSC:
    case TPRecipeBase::TPWidenLoadSC:
    case TPRecipeBase::TPWidenStoreEVLSC:
    case TPRecipeBase::TPWidenStoreSC:
      // TODO: Widened stores don't define a value, but widened loads do. Split
      // the recipes to be able to make widened loads VPSingleDefRecipes.
      return false;
    }
    llvm_unreachable("Unhandled VPDefID");
  }

  /// DimSet is a bitset of loop indices that this recipe's defined value "spans"
  /// - i.e., which loop induction dimensions contributed to computing. It drives tensor semantics:
  /// - Empty ({}) -> scalar: no loop dimension, no tensor parallelism (TensorOpKind::Scalar)
  /// - Non-empty -> the value is a tensor over those dims; the pattern matcher (by TPRecipeMatcher)
  /// uses DimSet comparisons between operands to classify 
  /// the op (ElementWise, BroadcastBinary, OuterProduct, Contraction, etc.).
  /// getTPValueShape() (TPRecipeMatcher.h:21) maps it to actual sizes: { Plan.getPFForDim(d) for d in DimSet }.
  SmallBitVector DimSet;

  /// Per-dim memory stride overrides (load/store recipes only).
  /// Key: dim index (innermost=0). Value: SCEV stride expression in elements.
  /// Absent entry → dense default expressed as a SCEV constant.
  /// Populated by TPRecipePatternMatcher_match() via SCEV GEP-index analysis.
  DenseMap<unsigned, const SCEV *> MemStrides;

  static inline bool classof(const TPUser *U) {
    auto *R = dyn_cast<TPRecipeBase>(U);
    return R && classof(R);
  }

  virtual TPSingleDefRecipe *clone() override = 0;

  /// Returns the underlying instruction.
  Instruction *getUnderlyingInstr() {
    return cast<Instruction>(getUnderlyingValue());
  }
  const Instruction *getUnderlyingInstr() const {
    return cast<Instruction>(getUnderlyingValue());
  }

  /// Returns the effective memory stride for \p Dim as a SCEV expression.
  /// Returns MemStrides[Dim] if set, else SE.getConstant(getDenseStrideForDim(Dim)).
  const SCEV *getMemStride(unsigned Dim, const TPlan &Plan,
                            ScalarEvolution &SE) const;
};

/// Class to record LLVM IR flag for a recipe along with it.
class TPRecipeWithIRFlags : public TPSingleDefRecipe { // yuxin.an: L964
  enum class OperationType : unsigned char {
    Cmp,
    OverflowingBinOp,
    DisjointOp,
    PossiblyExactOp,
    GEPOp,
    FPMathOp,
    NonNegOp,
    Other
  };

public:
  struct WrapFlagsTy {
    char HasNUW : 1;
    char HasNSW : 1;

    WrapFlagsTy(bool HasNUW, bool HasNSW) : HasNUW(HasNUW), HasNSW(HasNSW) {}
  };

  struct DisjointFlagsTy {
    char IsDisjoint : 1;
    DisjointFlagsTy(bool IsDisjoint) : IsDisjoint(IsDisjoint) {}
  };

protected:
  struct GEPFlagsTy {
    char IsInBounds : 1;
    GEPFlagsTy(bool IsInBounds) : IsInBounds(IsInBounds) {}
  };

private:
  struct ExactFlagsTy {
    char IsExact : 1;
  };
  struct NonNegFlagsTy {
    char NonNeg : 1;
  };
  struct FastMathFlagsTy {
    char AllowReassoc : 1;
    char NoNaNs : 1;
    char NoInfs : 1;
    char NoSignedZeros : 1;
    char AllowReciprocal : 1;
    char AllowContract : 1;
    char ApproxFunc : 1;

    FastMathFlagsTy(const FastMathFlags &FMF);
  };

  OperationType OpType;

  union {
    CmpInst::Predicate CmpPredicate;
    WrapFlagsTy WrapFlags;
    DisjointFlagsTy DisjointFlags;
    ExactFlagsTy ExactFlags;
    GEPFlagsTy GEPFlags;
    NonNegFlagsTy NonNegFlags;
    FastMathFlagsTy FMFs;
    unsigned AllFlags;
  };

protected:
  void transferFlags(TPRecipeWithIRFlags &Other) {
    OpType = Other.OpType;
    AllFlags = Other.AllFlags;
  }

public:
  template <typename IterT>
  TPRecipeWithIRFlags(const unsigned char SC, IterT Operands, DebugLoc DL = {})
      : TPSingleDefRecipe(SC, Operands, DL) {
    OpType = OperationType::Other;
    AllFlags = 0;
  }

  template <typename IterT>
  TPRecipeWithIRFlags(const unsigned char SC, IterT Operands, Instruction &I)
      : TPSingleDefRecipe(SC, Operands, &I, I.getDebugLoc()) {
    if (auto *Op = dyn_cast<CmpInst>(&I)) {
      OpType = OperationType::Cmp;
      CmpPredicate = Op->getPredicate();
    } else if (auto *Op = dyn_cast<PossiblyDisjointInst>(&I)) {
      OpType = OperationType::DisjointOp;
      DisjointFlags.IsDisjoint = Op->isDisjoint();
    } else if (auto *Op = dyn_cast<OverflowingBinaryOperator>(&I)) {
      OpType = OperationType::OverflowingBinOp;
      WrapFlags = {Op->hasNoUnsignedWrap(), Op->hasNoSignedWrap()};
    } else if (auto *Op = dyn_cast<PossiblyExactOperator>(&I)) {
      OpType = OperationType::PossiblyExactOp;
      ExactFlags.IsExact = Op->isExact();
    } else if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
      OpType = OperationType::GEPOp;
      GEPFlags.IsInBounds = GEP->isInBounds();
    } else if (auto *PNNI = dyn_cast<PossiblyNonNegInst>(&I)) {
      OpType = OperationType::NonNegOp;
      NonNegFlags.NonNeg = PNNI->hasNonNeg();
    } else if (auto *Op = dyn_cast<FPMathOperator>(&I)) {
      OpType = OperationType::FPMathOp;
      FMFs = Op->getFastMathFlags();
    } else {
      OpType = OperationType::Other;
      AllFlags = 0;
    }
  }

  template <typename IterT>
  TPRecipeWithIRFlags(const unsigned char SC, IterT Operands,
                      CmpInst::Predicate Pred, DebugLoc DL = {})
      : TPSingleDefRecipe(SC, Operands, DL), OpType(OperationType::Cmp),
        CmpPredicate(Pred) {}

  template <typename IterT>
  TPRecipeWithIRFlags(const unsigned char SC, IterT Operands,
                      WrapFlagsTy WrapFlags, DebugLoc DL = {})
      : TPSingleDefRecipe(SC, Operands, DL),
        OpType(OperationType::OverflowingBinOp), WrapFlags(WrapFlags) {}

  template <typename IterT>
  TPRecipeWithIRFlags(const unsigned char SC, IterT Operands,
                      FastMathFlags FMFs, DebugLoc DL = {})
      : TPSingleDefRecipe(SC, Operands, DL), OpType(OperationType::FPMathOp),
        FMFs(FMFs) {}

  template <typename IterT>
  TPRecipeWithIRFlags(const unsigned char SC, IterT Operands,
                      DisjointFlagsTy DisjointFlags, DebugLoc DL = {})
      : TPSingleDefRecipe(SC, Operands, DL), OpType(OperationType::DisjointOp),
        DisjointFlags(DisjointFlags) {}

protected:
  template <typename IterT>
  TPRecipeWithIRFlags(const unsigned char SC, IterT Operands,
                      GEPFlagsTy GEPFlags, DebugLoc DL = {})
      : TPSingleDefRecipe(SC, Operands, DL), OpType(OperationType::GEPOp),
        GEPFlags(GEPFlags) {}

public:
  static inline bool classof(const TPRecipeBase *R) {
    return R->getTPDefID() == TPRecipeBase::TPInstructionSC ||
           R->getTPDefID() == TPRecipeBase::TPWidenSC ||
           R->getTPDefID() == TPRecipeBase::TPWidenGEPSC ||
           R->getTPDefID() == TPRecipeBase::TPWidenCastSC ||
           R->getTPDefID() == TPRecipeBase::TPReplicateSC ||
           R->getTPDefID() == TPRecipeBase::TPVectorPointerSC;
  }

  static inline bool classof(const TPUser *U) {
    auto *R = dyn_cast<TPRecipeBase>(U);
    return R && classof(R);
  }

  /// Drop all poison-generating flags.
  void dropPoisonGeneratingFlags() {
    // NOTE: This needs to be kept in-sync with
    // Instruction::dropPoisonGeneratingFlags.
    switch (OpType) {
    case OperationType::OverflowingBinOp:
      WrapFlags.HasNUW = false;
      WrapFlags.HasNSW = false;
      break;
    case OperationType::DisjointOp:
      break;
    case OperationType::PossiblyExactOp:
      ExactFlags.IsExact = false;
      break;
    case OperationType::GEPOp:
      GEPFlags.IsInBounds = false;
      break;
    case OperationType::FPMathOp:
      FMFs.NoNaNs = false;
      FMFs.NoInfs = false;
      break;
    case OperationType::NonNegOp:
      NonNegFlags.NonNeg = false;
      break;
    case OperationType::Cmp:
    case OperationType::Other:
      break;
    }
  }

  /// Set the IR flags for \p I.
  void setFlags(Instruction *I) const {
    switch (OpType) {
    switch (OpType) {
    case OperationType::OverflowingBinOp:
      I->setHasNoUnsignedWrap(WrapFlags.HasNUW);
      I->setHasNoSignedWrap(WrapFlags.HasNSW);
      break;
    case OperationType::DisjointOp:
      cast<PossiblyDisjointInst>(I)->setIsDisjoint(DisjointFlags.IsDisjoint);
      break;
    case OperationType::PossiblyExactOp:
      I->setIsExact(ExactFlags.IsExact);
      break;
    case OperationType::GEPOp:
      // TODO(gep_nowrap): Track the full GEPNoWrapFlags in VPlan.
      cast<GetElementPtrInst>(I)->setNoWrapFlags(
          GEPFlags.IsInBounds ? GEPNoWrapFlags::inBounds()
                              : GEPNoWrapFlags::none());
      break;
    case OperationType::FPMathOp:
