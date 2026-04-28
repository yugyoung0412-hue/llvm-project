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
      I->setHasAllowReassoc(FMFs.AllowReassoc);
      I->setHasNoNaNs(FMFs.NoNaNs);
      I->setHasNoInfs(FMFs.NoInfs);
      I->setHasNoSignedZeros(FMFs.NoSignedZeros);
      I->setHasAllowReciprocal(FMFs.AllowReciprocal);
      I->setHasAllowContract(FMFs.AllowContract);
      I->setHasApproxFunc(FMFs.ApproxFunc);
      break;
    case OperationType::NonNegOp:
      I->setNonNeg(NonNegFlags.NonNeg);
      break;
    case OperationType::Cmp:
    case OperationType::Other:
      break;
    }
  }

  CmpInst::Predicate getPredicate() const {
    assert(OpType == OperationType::Cmp &&
           "recipe doesn't have a compare predicate");
    return CmpPredicate;
  }

  bool isInBounds() const {
    assert(OpType == OperationType::GEPOp &&
           "recipe doesn't have inbounds flag");
    return GEPFlags.IsInBounds;
  }

  /// Returns true if the recipe has fast-math flags.
  bool hasFastMathFlags() const { return OpType == OperationType::FPMathOp; }

  FastMathFlags getFastMathFlags() const;

  bool hasNoUnsignedWrap() const {
    assert(OpType == OperationType::OverflowingBinOp &&
           "recipe doesn't have a NUW flag");
    return WrapFlags.HasNUW;
  }

  bool hasNoSignedWrap() const {
    assert(OpType == OperationType::OverflowingBinOp &&
           "recipe doesn't have a NSW flag");
    return WrapFlags.HasNSW;
  }

  bool isDisjoint() const {
    assert(OpType == OperationType::DisjointOp &&
           "recipe cannot have a disjoing flag");
    return DisjointFlags.IsDisjoint;
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void printFlags(raw_ostream &O) const;
#endif
};

/// This is a concrete Recipe that models a single VPlan-level instruction.
/// While as any Recipe it may generate a sequence of IR instructions when
/// executed, these instructions would always form a single-def expression as
/// the VPInstruction is also a single def-use vertex.
class TPInstruction : public TPRecipeWithIRFlags { // yuxin.an: L1229
  // TODO(yuxin.an)
  // friend class VPlanSlp;

public:
  /// VPlan opcodes, extending LLVM IR with idiomatics instructions.
  enum {
    FirstOrderRecurrenceSplice =
        Instruction::OtherOpsEnd + 1, // Combines the incoming and previous
                                      // values of a first-order recurrence.
    Not,
    SLPLoad,
    SLPStore,
    ActiveLaneMask,
    ExplicitTensorLength,
    /// Creates a scalar phi in a leaf VPBB with a single predecessor in VPlan.
    /// The first operand is the incoming value from the predecessor in VPlan,
    /// the second operand is the incoming value for all other predecessors
    /// (which are currently not modeled in VPlan).
    ResumePhi,
    CalculateTripCountMinusVF,
    // Increment the canonical IV separately for each unrolled part.
    CanonicalIVIncrementForPart,
    BranchOnCount,
    BranchOnCond,
    ComputeReductionResult,
    // Takes the VPValue to extract from as first operand and the lane or part
    // to extract as second operand, counting from the end starting with 1 for
    // last. The second operand must be a positive constant and <= VF when
    // extracting from a vector or <= UF when extracting from an unrolled
    // scalar.
    ExtractFromEnd,
    LogicalAnd, // Non-poison propagating logical And.
    // Add an offset in bytes (second operand) to a base pointer (first
    // operand). Only generates scalar values (either for the first lane only or
    // for all lanes, depending on its uses).
    PtrAdd,
  };

private:
  typedef unsigned char OpcodeTy;
  OpcodeTy Opcode;

  /// An optional name that can be used for the generated IR instruction.
  const std::string Name;

  /// Returns true if this VPInstruction generates scalar values for all lanes.
  /// Most VPInstructions generate a single value per part, either vector or
  /// scalar. VPReplicateRecipe takes care of generating multiple (scalar)
  /// values per all lanes, stemming from an original ingredient. This method
  /// identifies the (rare) cases of VPInstructions that do so as well, w/o an
  /// underlying ingredient.
  bool doesGeneratePerAllLanes() const;

  /// Returns true if we can generate a scalar for the first lane only if
  /// needed.
  bool canGenerateScalarForFirstLane() const;

  /// Utility methods serving execute(): generates a single instance of the
  /// modeled instruction for a given part. \returns the generated value for \p
  /// Part. In some cases an existing value is returned rather than a generated
  /// one.
  Value *generatePerPart(TPTransformState &State, unsigned Part);

  /// Utility methods serving execute(): generates a scalar single instance of
  /// the modeled instruction for a given lane. \returns the scalar generated
  /// value for lane \p Lane.
  Value *generatePerLane(TPTransformState &State, const TPIteration &Lane);

#if !defined(NDEBUG)
  /// Return true if the VPInstruction is a floating point math operation, i.e.
  /// has fast-math flags.
  bool isFPMathOp() const;
#endif

public:
  TPInstruction(unsigned Opcode, ArrayRef<TPValue *> Operands, DebugLoc DL,
                const Twine &Name = "")
      : TPRecipeWithIRFlags(TPDef::TPInstructionSC, Operands, DL),
        Opcode(Opcode), Name(Name.str()) {}

  TPInstruction(unsigned Opcode, std::initializer_list<TPValue *> Operands,
                DebugLoc DL = {}, const Twine &Name = "")
      : TPInstruction(Opcode, ArrayRef<TPValue *>(Operands), DL, Name) {}

  TPInstruction(unsigned Opcode, CmpInst::Predicate Pred, TPValue *A,
                TPValue *B, DebugLoc DL = {}, const Twine &Name = "");

  TPInstruction(unsigned Opcode, std::initializer_list<TPValue *> Operands,
                WrapFlagsTy WrapFlags, DebugLoc DL = {}, const Twine &Name = "")
      : TPRecipeWithIRFlags(TPDef::TPInstructionSC, Operands, WrapFlags, DL),
        Opcode(Opcode), Name(Name.str()) {}

  TPInstruction(unsigned Opcode, std::initializer_list<TPValue *> Operands,
                DisjointFlagsTy DisjointFlag, DebugLoc DL = {},
                const Twine &Name = "")
      : TPRecipeWithIRFlags(TPDef::TPInstructionSC, Operands, DisjointFlag, DL),
        Opcode(Opcode), Name(Name.str()) {
    assert(Opcode == Instruction::Or && "only OR opcodes can be disjoint");
  }

  TPInstruction(unsigned Opcode, std::initializer_list<TPValue *> Operands,
                FastMathFlags FMFs, DebugLoc DL = {}, const Twine &Name = "");

  TP_CLASSOF_IMPL(TPDef::TPInstructionSC)

  TPInstruction *clone() override {
    SmallVector<TPValue *, 2> Operands(operands());
    auto *New = new TPInstruction(Opcode, Operands, getDebugLoc(), Name);
    New->transferFlags(*this);
    return New;
  }

  unsigned getOpcode() const { return Opcode; }

  /// Generate the instruction.
  /// TODO: We currently execute only per-part unless a specific instance is
  /// provided.
  void execute(TPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the VPInstruction to \p O.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override;

  /// Print the VPInstruction to dbgs() (for debugging).
  LLVM_DUMP_METHOD void dump() const;
#endif

  /// Return true if this instruction may modify memory.
  bool mayWriteToMemory() const {
    // TODO: we can use attributes of the called function to rule out memory
    //       modifications.
    return Opcode == Instruction::Store || Opcode == Instruction::Call ||
           Opcode == Instruction::Invoke || Opcode == SLPStore;
  }

  bool hasResult() const {
    // CallInst may or may not have a result, depending on the called function.
    // Conservatively return calls have results for now.
    switch (getOpcode()) {
    case Instruction::Ret:
    case Instruction::Br:
    case Instruction::Store:
    case Instruction::Switch:
    case Instruction::IndirectBr:
    case Instruction::Resume:
    case Instruction::CatchRet:
    case Instruction::Unreachable:
    case Instruction::Fence:
    case Instruction::AtomicRMW:
    case TPInstruction::BranchOnCond:
    case TPInstruction::BranchOnCount:
      return false;
    default:
      return true;
    }
  }

  /// Returns true if the recipe only uses the first lane of operand \p Op.
  bool onlyFirstLaneUsed(const TPValue *Op) const override;

  /// Returns true if the recipe only uses the first part of operand \p Op.
  bool onlyFirstPartUsed(const TPValue *Op) const override;

  // TODO(yuxin.an)
  /// Returns true if this VPInstruction produces a scalar value from a vector,
  /// e.g. by performing a reduction or extracting a lane.
  bool isTensorToScalar() const;

  /// Returns true if this VPInstruction's operands are single scalars and the
  /// result is also a single scalar.
  bool isSingleScalar() const;
};

/// VPWidenRecipe is a recipe for producing a widened instruction using the
/// opcode and operands of the recipe. This recipe covers most of the
/// traditional vectorization cases where each recipe transforms into a
/// vectorized version of itself.
class TPWidenRecipe : public TPRecipeWithIRFlags {
  unsigned Opcode;

public:
  template <typename IterT>
  TPWidenRecipe(Instruction &I, iterator_range<IterT> Operands)
      : TPRecipeWithIRFlags(TPDef::TPWidenSC, Operands, I),
        Opcode(I.getOpcode()) {}

  ~TPWidenRecipe() override = default;

  TPWidenRecipe *clone() override {
    auto *R = new TPWidenRecipe(*getUnderlyingInstr(), operands());
    R->transferFlags(*this);
    return R;
  }

  TP_CLASSOF_IMPL(TPDef::TPWidenSC)

  /// Produce a widened instruction using the opcode and operands of the recipe,
  /// processing State.VF elements.
  void execute(TPTransformState &State) override;

  unsigned getOpcode() const { return Opcode; }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override;
#endif
};

/// VPWidenCastRecipe is a recipe to create vector cast instructions.
class TPWidenCastRecipe : public TPRecipeWithIRFlags { // yuxin.an: L1439
  /// Cast instruction opcode.
  Instruction::CastOps Opcode;

  /// Result type for the cast.
  Type *ResultTy;

public:
  TPWidenCastRecipe(Instruction::CastOps Opcode, TPValue *Op, Type *ResultTy,
                    CastInst &UI)
      : TPRecipeWithIRFlags(TPDef::TPWidenCastSC, Op, UI), Opcode(Opcode),
        ResultTy(ResultTy) {
    assert(UI.getOpcode() == Opcode &&
           "opcode of underlying cast doesn't match");
  }

  TPWidenCastRecipe(Instruction::CastOps Opcode, TPValue *Op, Type *ResultTy)
      : TPRecipeWithIRFlags(TPDef::TPWidenCastSC, Op), Opcode(Opcode),
        ResultTy(ResultTy) {}

  ~TPWidenCastRecipe() override = default;

  TPWidenCastRecipe *clone() override {
    if (auto *UV = getUnderlyingValue())
      return new TPWidenCastRecipe(Opcode, getOperand(0), ResultTy,
                                   *cast<CastInst>(UV));

    return new TPWidenCastRecipe(Opcode, getOperand(0), ResultTy);
  }

  TP_CLASSOF_IMPL(TPDef::TPWidenCastSC)

  /// Produce widened copies of the cast.
  void execute(TPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override;
#endif

  Instruction::CastOps getOpcode() const { return Opcode; }

  /// Returns the result type of the cast.
  Type *getResultType() const { return ResultTy; }
};

/// VPScalarCastRecipe is a recipe to create scalar cast instructions.
class TPScalarCastRecipe : public TPSingleDefRecipe {
  Instruction::CastOps Opcode;

  Type *ResultTy;

  Value *generate(TPTransformState &State, unsigned Part);

public:
  TPScalarCastRecipe(Instruction::CastOps Opcode, TPValue *Op, Type *ResultTy)
      : TPSingleDefRecipe(TPDef::TPScalarCastSC, {Op}), Opcode(Opcode),
        ResultTy(ResultTy) {}

  ~TPScalarCastRecipe() override = default;

  TPScalarCastRecipe *clone() override {
    return new TPScalarCastRecipe(Opcode, getOperand(0), ResultTy);
  }

  TP_CLASSOF_IMPL(TPDef::TPScalarCastSC)

  void execute(TPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override;
#endif

  /// Returns the result type of the cast.
  Type *getResultType() const { return ResultTy; }

  bool onlyFirstLaneUsed(const TPValue *Op) const override {
    // At the moment, only uniform codegen is implemented.
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return true;
  }
};

/// A recipe for widening Call instructions.
class TPWidenCallRecipe : public TPSingleDefRecipe { // yuxin.an: L1526
  /// ID of the vector intrinsic to call when widening the call. If set the
  /// Intrinsic::not_intrinsic, a library call will be used instead.
  Intrinsic::ID VectorIntrinsicID;
  /// If this recipe represents a library call, Variant stores a pointer to
  /// the chosen function. There is a 1:1 mapping between a given VF and the
  /// chosen vectorized variant, so there will be a different vplan for each
  /// VF with a valid variant.
  Function *Variant;

public:
  template <typename IterT>
  TPWidenCallRecipe(Value *UV, iterator_range<IterT> CallArguments,
                    Intrinsic::ID VectorIntrinsicID, DebugLoc DL = {},
                    Function *Variant = nullptr)
      : TPSingleDefRecipe(TPDef::TPWidenCallSC, CallArguments, UV, DL),
        VectorIntrinsicID(VectorIntrinsicID), Variant(Variant) {
    assert(
        isa<Function>(getOperand(getNumOperands() - 1)->getLiveInIRValue()) &&
        "last operand must be the called function");
  }

  ~TPWidenCallRecipe() override = default;

  TPWidenCallRecipe *clone() override {
    return new TPWidenCallRecipe(getUnderlyingValue(), operands(),
                                 VectorIntrinsicID, getDebugLoc(), Variant);
  }

  TP_CLASSOF_IMPL(TPDef::TPWidenCallSC)

  /// Produce a widened version of the call instruction.
  void execute(TPTransformState &State) override;

  Function *getCalledScalarFunction() const {
    return cast<Function>(getOperand(getNumOperands() - 1)->getLiveInIRValue());
  }

  operand_range arg_operands() {
    return make_range(op_begin(), op_begin() + getNumOperands() - 1);
  }
  const_operand_range arg_operands() const {
    return make_range(op_begin(), op_begin() + getNumOperands() - 1);
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override;
#endif
};

class TPMatrixCallRecipe : public TPSingleDefRecipe { // yuxin.an: L1526
  /// ID of the vector intrinsic to call when widening the call. If set the
  /// Intrinsic::not_intrinsic, a library call will be used instead.
  Intrinsic::ID MatrixIntrinsicID;
  /// If this recipe represents a library call, Variant stores a pointer to
  /// the chosen function. There is a 1:1 mapping between a given VF and the
  /// chosen vectorized variant, so there will be a different vplan for each
  /// VF with a valid variant.
  Function *Variant;

public:
  template <typename IterT>
  TPMatrixCallRecipe(Value *UV, iterator_range<IterT> CallArguments,
                     Intrinsic::ID MatrixIntrinsicID, DebugLoc DL = {},
                     Function *Variant = nullptr)
      : TPSingleDefRecipe(TPDef::TPMatrixCallSC, CallArguments, UV, DL),
        MatrixIntrinsicID(MatrixIntrinsicID), Variant(Variant) {}

  TPMatrixCallRecipe(Value *UV, ArrayRef<TPValue *> CallArguments,
                     Intrinsic::ID MatrixIntrinsicID, DebugLoc DL = {},
                     Function *Variant = nullptr)
      : TPSingleDefRecipe(TPDef::TPMatrixCallSC, CallArguments, UV, DL),
        MatrixIntrinsicID(MatrixIntrinsicID), Variant(Variant) {}

  ~TPMatrixCallRecipe() override = default;

  TPMatrixCallRecipe *clone() override {
    return new TPMatrixCallRecipe(getUnderlyingValue(), operands(),
                                  MatrixIntrinsicID, getDebugLoc(), Variant);
  }

  TP_CLASSOF_IMPL(TPDef::TPMatrixCallSC)

  /// Produce a widened version of the call instruction.
  void execute(TPTransformState &State) override;

  Function *getCalledScalarFunction() const {
    return cast<Function>(getOperand(getNumOperands() - 1)->getLiveInIRValue());
  }

  operand_range arg_operands() {
    // !FIXME(yuxin.an)
    return make_range(op_begin(), op_begin() + getNumOperands());
  }
  const_operand_range arg_operands() const {
    // !FIXME(yuxin.an)
    return make_range(op_begin(), op_begin() + getNumOperands());
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override;
#endif
};

/// A recipe for widening select instructions.
struct TPWidenSelectRecipe : public TPSingleDefRecipe {
  template <typename IterT>
  TPWidenSelectRecipe(SelectInst &I, iterator_range<IterT> Operands)
      : TPSingleDefRecipe(TPDef::TPWidenSelectSC, Operands, &I,
                          I.getDebugLoc()) {}

  ~TPWidenSelectRecipe() override = default;

  TPWidenSelectRecipe *clone() override {
    return new TPWidenSelectRecipe(*cast<SelectInst>(getUnderlyingInstr()),
                                   operands());
  }

  TP_CLASSOF_IMPL(TPDef::TPWidenSelectSC)

  /// Produce a widened version of the select instruction.
  void execute(TPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override;
#endif

  TPValue *getCond() const { return getOperand(0); }

  bool isInvariantCond() const {
    return getCond()->isDefinedOutsideVectorRegions();
  }
};

/// A recipe for handling GEP instructions.
class TPWidenGEPRecipe : public TPRecipeWithIRFlags { // yuxin.an: L1613
  bool isPointerLoopInvariant() const {
    return getOperand(0)->isDefinedOutsideVectorRegions();
  }

  bool isIndexLoopInvariant(unsigned I) const {
    return getOperand(I + 1)->isDefinedOutsideVectorRegions();
  }

  bool areAllOperandsInvariant() const {
    return all_of(operands(), [](TPValue *Op) {
      return Op->isDefinedOutsideVectorRegions();
    });
  }

public:
  template <typename IterT>
  TPWidenGEPRecipe(GetElementPtrInst *GEP, iterator_range<IterT> Operands)
      : TPRecipeWithIRFlags(TPDef::TPWidenGEPSC, Operands, *GEP) {}

  ~TPWidenGEPRecipe() override = default;

  TPWidenGEPRecipe *clone() override {
    return new TPWidenGEPRecipe(cast<GetElementPtrInst>(getUnderlyingInstr()),
                                operands());
  }

  TP_CLASSOF_IMPL(TPDef::TPWidenGEPSC)

  /// Generate the gep nodes.
  void execute(TPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override;
#endif
};

/// A recipe to compute the pointers for widened memory accesses of IndexTy for
/// all parts. If IsReverse is true, compute pointers for accessing the input in
/// reverse order per part.
class TPVectorPointerRecipe : public TPRecipeWithIRFlags { // yuxin.an: L1655
  Type *IndexedTy;
  bool IsReverse;

public:
  TPVectorPointerRecipe(TPValue *Ptr, Type *IndexedTy, bool IsReverse,
                        bool IsInBounds, DebugLoc DL)
      : TPRecipeWithIRFlags(TPDef::TPVectorPointerSC, ArrayRef<TPValue *>(Ptr),
                            GEPFlagsTy(IsInBounds), DL),
        IndexedTy(IndexedTy), IsReverse(IsReverse) {}

  TP_CLASSOF_IMPL(TPDef::TPVectorPointerSC)

  void execute(TPTransformState &State) override;

  bool onlyFirstLaneUsed(const TPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return true;
  }

  TPVectorPointerRecipe *clone() override {
    return new TPVectorPointerRecipe(getOperand(0), IndexedTy, IsReverse,
                                     isInBounds(), getDebugLoc());
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override;
#endif
};

/// A pure virtual base class for all recipes modeling header phis, including
/// phis for first order recurrences, pointer inductions and reductions. The
/// start value is the first operand of the recipe and the incoming value from
/// the backedge is the second operand.
///
/// Inductions are modeled using the following sub-classes:
///  * VPCanonicalIVPHIRecipe: Canonical scalar induction of the vector loop,
///    starting at a specified value (zero for the main vector loop, the resume
///    value for the epilogue vector loop) and stepping by 1. The induction
///    controls exiting of the vector loop by comparing against the vector trip
///    count. Produces a single scalar PHI for the induction value per
///    iteration.
///  * VPWidenIntOrFpInductionRecipe: Generates vector values for integer and
///    floating point inductions with arbitrary start and step values. Produces
///    a vector PHI per-part.
///  * VPDerivedIVRecipe: Converts the canonical IV value to the corresponding
///    value of an IV with different start and step values. Produces a single
///    scalar value per iteration
///  * VPScalarIVStepsRecipe: Generates scalar values per-lane based on a
///    canonical or derived induction.
///  * VPWidenPointerInductionRecipe: Generate vector and scalar values for a
///    pointer induction. Produces either a vector PHI per-part or scalar values
///    per-lane based on the canonical induction.
class TPHeaderPHIRecipe : public TPSingleDefRecipe { // yuxin.an: L1711
protected:
  TPHeaderPHIRecipe(unsigned char VPDefID, Instruction *UnderlyingInstr,
                    TPValue *Start = nullptr, DebugLoc DL = {})
      : TPSingleDefRecipe(VPDefID, ArrayRef<TPValue *>(), UnderlyingInstr, DL) {
    if (Start)
      addOperand(Start);
  }

public:
  ~TPHeaderPHIRecipe() override = default;

  /// Method to support type inquiry through isa, cast, and dyn_cast.
  static inline bool classof(const TPRecipeBase *B) {
    return B->getTPDefID() >= TPDef::TPFirstHeaderPHISC &&
           B->getTPDefID() <= TPDef::TPLastHeaderPHISC;
  }
  static inline bool classof(const TPValue *V) {
    auto *B = V->getDefiningRecipe();
    return B && B->getTPDefID() >= TPRecipeBase::TPFirstHeaderPHISC &&
           B->getTPDefID() <= TPRecipeBase::TPLastHeaderPHISC;
  }

  /// Generate the phi nodes.
  void execute(TPTransformState &State) override = 0;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override = 0;
#endif

  /// Returns the start value of the phi, if one is set.
  TPValue *getStartValue() {
    return getNumOperands() == 0 ? nullptr : getOperand(0);
  }
  TPValue *getStartValue() const {
    return getNumOperands() == 0 ? nullptr : getOperand(0);
  }

  /// Update the start value of the recipe.
  void setStartValue(TPValue *V) { setOperand(0, V); }

  /// Returns the incoming value from the loop backedge.
  virtual TPValue *getBackedgeValue() { return getOperand(1); }

  /// Returns the backedge value as a recipe. The backedge value is guaranteed
  /// to be a recipe.
  virtual TPRecipeBase &getBackedgeRecipe() {
    return *getBackedgeValue()->getDefiningRecipe();
  }
};

/// A recipe for handling first-order recurrence phis. The start value is the
/// first operand of the recipe and the incoming value from the backedge is the
/// second operand.
struct TPFirstOrderRecurrencePHIRecipe : public TPHeaderPHIRecipe {
  TPFirstOrderRecurrencePHIRecipe(PHINode *Phi, TPValue &Start)
      : TPHeaderPHIRecipe(TPDef::TPFirstOrderRecurrencePHISC, Phi, &Start) {}

  TP_CLASSOF_IMPL(TPDef::TPFirstOrderRecurrencePHISC)

  static inline bool classof(const TPHeaderPHIRecipe *R) {
    return R->getTPDefID() == TPDef::TPFirstOrderRecurrencePHISC;
  }

  TPFirstOrderRecurrencePHIRecipe *clone() override {
    return new TPFirstOrderRecurrencePHIRecipe(
        cast<PHINode>(getUnderlyingInstr()), *getOperand(0));
  }

  void execute(TPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override;
#endif
};

class TPWidenPointerInductionRecipe : public TPHeaderPHIRecipe {
  const InductionDescriptor &IndDesc;

  bool IsScalarAfterVectorization;

public:
  /// Create a new TPWidenPointerInductionRecipe for \p Phi with start value \p
  /// Start.
  TPWidenPointerInductionRecipe(PHINode *Phi, TPValue *Start, TPValue *Step,
                                const InductionDescriptor &IndDesc,
                                bool IsScalarAfterVectorization)
      : TPHeaderPHIRecipe(TPDef::TPWidenPointerInductionSC, Phi),
        IndDesc(IndDesc),
        IsScalarAfterVectorization(IsScalarAfterVectorization) {
    addOperand(Start);
    addOperand(Step);
  }

  ~TPWidenPointerInductionRecipe() override = default;

  TPWidenPointerInductionRecipe *clone() override {
    return new TPWidenPointerInductionRecipe(
        cast<PHINode>(getUnderlyingInstr()), getOperand(0), getOperand(1),
        IndDesc, IsScalarAfterVectorization);
  }

  TP_CLASSOF_IMPL(TPDef::TPWidenPointerInductionSC)

  /// Generate vector values for the pointer induction.
  void execute(TPTransformState &State) override;

  /// Returns true if only scalar values will be generated.
  bool onlyScalarsGenerated(bool IsScalable);

  /// Returns the induction descriptor for the recipe.
  const InductionDescriptor &getInductionDescriptor() const { return IndDesc; }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override;
#endif
};

/// A recipe for handling phi nodes of integer and floating-point inductions,
/// producing their vector values.
class TPWidenIntOrFpInductionRecipe
    : public TPHeaderPHIRecipe { // yuxin.an: L1768
  PHINode *IV;
  TruncInst *Trunc;
  const InductionDescriptor &IndDesc;

public:
  TPWidenIntOrFpInductionRecipe(PHINode *IV, TPValue *Start, TPValue *Step,
                                const InductionDescriptor &IndDesc)
      : TPHeaderPHIRecipe(TPDef::TPWidenIntOrFpInductionSC, IV, Start), IV(IV),
        Trunc(nullptr), IndDesc(IndDesc) {
    addOperand(Step);
  }

  TPWidenIntOrFpInductionRecipe(PHINode *IV, TPValue *Start, TPValue *Step,
                                const InductionDescriptor &IndDesc,
                                TruncInst *Trunc)
      : TPHeaderPHIRecipe(TPDef::TPWidenIntOrFpInductionSC, Trunc, Start),
        IV(IV), Trunc(Trunc), IndDesc(IndDesc) {
    addOperand(Step);
  }

  ~TPWidenIntOrFpInductionRecipe() override = default;

  TPWidenIntOrFpInductionRecipe *clone() override {
    return new TPWidenIntOrFpInductionRecipe(IV, getStartValue(),
                                             getStepValue(), IndDesc, Trunc);
  }

  TP_CLASSOF_IMPL(TPDef::TPWidenIntOrFpInductionSC)

  /// Generate the vectorized and scalarized versions of the phi node as
  /// needed by their users.
  void execute(TPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override;
#endif

  TPValue *getBackedgeValue() override {
    // TODO: All operands of base recipe must exist and be at same index in
    // derived recipe.
    llvm_unreachable(
        "VPWidenIntOrFpInductionRecipe generates its own backedge value");
  }

  TPRecipeBase &getBackedgeRecipe() override {
    // TODO: All operands of base recipe must exist and be at same index in
    // derived recipe.
    llvm_unreachable(
        "VPWidenIntOrFpInductionRecipe generates its own backedge value");
  }

  /// Returns the step value of the induction.
  TPValue *getStepValue() { return getOperand(1); }
  const TPValue *getStepValue() const { return getOperand(1); }

  /// Returns the first defined value as TruncInst, if it is one or nullptr
  /// otherwise.
  TruncInst *getTruncInst() { return Trunc; }
  const TruncInst *getTruncInst() const { return Trunc; }

  PHINode *getPHINode() { return IV; }

  /// Returns the induction descriptor for the recipe.
  const InductionDescriptor &getInductionDescriptor() const { return IndDesc; }

  /// Returns true if the induction is canonical, i.e. starting at 0 and
  /// incremented by UF * VF (= the original IV is incremented by 1) and has the
  /// same type as the canonical induction.
  bool isCanonical() const;

  /// Returns the scalar type of the induction.
  Type *getScalarType() const {
    return Trunc ? Trunc->getType() : IV->getType();
  }
};

/// A recipe for handling phis that are widened in the vector loop.
/// In the VPlan native path, all incoming VPValues & VPBasicBlock pairs are
/// managed in the recipe directly.
class TPWidenPHIRecipe : public TPSingleDefRecipe {
  /// List of incoming blocks. Only used in the VPlan native path.
  SmallVector<TPBasicBlock *, 2> IncomingBlocks;

public:
  /// Create a new VPWidenPHIRecipe for \p Phi with start value \p Start.
  TPWidenPHIRecipe(PHINode *Phi, TPValue *Start = nullptr)
      : TPSingleDefRecipe(TPDef::TPWidenPHISC, ArrayRef<TPValue *>(), Phi) {
    if (Start)
      addOperand(Start);
  }

  TPWidenPHIRecipe *clone() override {
    llvm_unreachable("cloning not implemented yet");
  }

  ~TPWidenPHIRecipe() override = default;

  TP_CLASSOF_IMPL(TPDef::TPWidenPHISC)

  /// Generate the phi/select nodes.
  void execute(TPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override;
#endif

  /// Adds a pair (\p IncomingV, \p IncomingBlock) to the phi.
  void addIncoming(TPValue *IncomingV, TPBasicBlock *IncomingBlock) {
    addOperand(IncomingV);
    IncomingBlocks.push_back(IncomingBlock);
  }

  /// Returns the \p I th incoming VPBasicBlock.
  TPBasicBlock *getIncomingBlock(unsigned I) { return IncomingBlocks[I]; }

  /// Returns the \p I th incoming VPValue.
  TPValue *getIncomingValue(unsigned I) { return getOperand(I); }
};

/// A recipe for handling reduction phis. The start value is the first operand
/// of the recipe and the incoming value from the backedge is the second
/// operand.
class TPReductionPHIRecipe : public TPHeaderPHIRecipe { // yuxin.an: L1966
  /// Descriptor for the reduction.
  const RecurrenceDescriptor &RdxDesc;

  /// The phi is part of an in-loop reduction.
  bool IsInLoop;

  /// The phi is part of an ordered reduction. Requires IsInLoop to be true.
  bool IsOrdered;

public:
  /// Create a new VPReductionPHIRecipe for the reduction \p Phi described by \p
  /// RdxDesc.
  TPReductionPHIRecipe(PHINode *Phi, const RecurrenceDescriptor &RdxDesc,
                       TPValue &Start, bool IsInLoop = false,
                       bool IsOrdered = false)
      : TPHeaderPHIRecipe(TPDef::TPReductionPHISC, Phi, &Start),
        RdxDesc(RdxDesc), IsInLoop(IsInLoop), IsOrdered(IsOrdered) {
    assert((!IsOrdered || IsInLoop) && "IsOrdered requires IsInLoop");
  }

  ~TPReductionPHIRecipe() override = default;

  TPReductionPHIRecipe *clone() override {
    auto *R =
        new TPReductionPHIRecipe(cast<PHINode>(getUnderlyingInstr()), RdxDesc,
                                 *getOperand(0), IsInLoop, IsOrdered);
    // YYG::REMOVE
    errs() << "[TPReductionPHIRecipe] *clone()\n";
    getBackedgeValue()->dump();
    R->addOperand(getBackedgeValue());
    return R;
  }

  TP_CLASSOF_IMPL(TPDef::TPReductionPHISC)

  static inline bool classof(const TPHeaderPHIRecipe *R) {
    return R->getTPDefID() == TPDef::TPReductionPHISC;
  }

  /// Generate the phi/select nodes.
  void execute(TPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override;
#endif

  const RecurrenceDescriptor &getRecurrenceDescriptor() const {
    return RdxDesc;
  }

  /// Returns true, if the phi is part of an ordered reduction.
  bool isOrdered() const { return IsOrdered; }

  /// Returns true, if the phi is part of an in-loop reduction.
  bool isInLoop() const { return IsInLoop; }
};

/// A recipe for vectorizing a phi-node as a sequence of mask-based select
/// instructions.
class TPBlendRecipe : public TPSingleDefRecipe {
public:
  /// The blend operation is a User of the incoming values and of their
  /// respective masks, ordered [I0, I1, M1, I2, M2, ...]. Note that the first
  /// incoming value does not have a mask associated.
  TPBlendRecipe(PHINode *Phi, ArrayRef<TPValue *> Operands)
      : TPSingleDefRecipe(TPDef::TPBlendSC, Operands, Phi, Phi->getDebugLoc()) {
    assert((Operands.size() + 1) % 2 == 0 &&
           "Expected an odd number of operands");
  }

  TPBlendRecipe *clone() override {
    SmallVector<TPValue *> Ops(operands());
    return new TPBlendRecipe(cast<PHINode>(getUnderlyingValue()), Ops);
  }

  TP_CLASSOF_IMPL(TPDef::TPBlendSC)

  /// Return the number of incoming values, taking into account that the first
  /// incoming value has no mask.
  unsigned getNumIncomingValues() const { return (getNumOperands() + 1) / 2; }

  /// Return incoming value number \p Idx.
  TPValue *getIncomingValue(unsigned Idx) const {
    return Idx == 0 ? getOperand(0) : getOperand(Idx * 2 - 1);
  }

  /// Return mask number \p Idx.
  TPValue *getMask(unsigned Idx) const {
    assert(Idx > 0 && "First index has no mask associated.");
    return getOperand(Idx * 2);
  }

  /// Generate the phi/select nodes.
  void execute(TPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override;
#endif

  /// Returns true if the recipe only uses the first lane of operand \p Op.
  bool onlyFirstLaneUsed(const TPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    // Recursing through Blend recipes only, must terminate at header phi's the
    // latest.
    return all_of(users(),
                  [this](TPUser *U) { return U->onlyFirstLaneUsed(this); });
  }
};

class TPNewInstrRecipe : public TPRecipeWithIRFlags {
  unsigned OpCode;

public:
  template <typename IterT>
  TPNewInstrRecipe(unsigned OpCode, iterator_range<IterT> Operands)
      : TPRecipeWithIRFlags(TPDef::TPNewInstrSC, Operands), OpCode(OpCode) {}

  TPNewInstrRecipe(unsigned OpCode, ArrayRef<TPValue *> Operands)
      : TPRecipeWithIRFlags(TPDef::TPNewInstrSC, Operands), OpCode(OpCode) {}

  ~TPNewInstrRecipe() override = default;

  TPNewInstrRecipe *clone() override {
    auto *Copy = new TPNewInstrRecipe(getOpcode(), operands());
    return Copy;
  }

  TP_CLASSOF_IMPL(TPDef::TPNewInstrSC)

  /// Generate replicas of the desired Ingredient. Replicas will be generated
  /// for all parts and lanes unless a specific part and lane are specified in
  /// the \p State.
  void execute(TPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override;
#endif

  unsigned getOpcode() const { return OpCode; }
};

/// VPReplicateRecipe replicates a given instruction producing multiple scalar
/// copies of the original scalar type, one per lane, instead of producing a
/// single copy of widened type for all lanes. If the instruction is known to be
/// uniform only one copy, per lane zero, will be generated.
class TPReplicateRecipe : public TPRecipeWithIRFlags { // yuxin.an: L2288
  /// Indicator if only a single replica per lane is needed.
  bool IsUniform;

  /// Indicator if the replicas are also predicated.
  bool IsPredicated;

public:
  template <typename IterT>
  TPReplicateRecipe(Instruction *I, iterator_range<IterT> Operands,
                    bool IsUniform, TPValue *Mask = nullptr)
      : TPRecipeWithIRFlags(TPDef::TPReplicateSC, Operands, *I),
        IsUniform(IsUniform), IsPredicated(Mask) {
    if (Mask)
      addOperand(Mask);
  }

  ~TPReplicateRecipe() override = default;

  TPReplicateRecipe *clone() override {
    auto *Copy =
        new TPReplicateRecipe(getUnderlyingInstr(), operands(), IsUniform,
                              isPredicated() ? getMask() : nullptr);
    Copy->transferFlags(*this);
    return Copy;
  }

  TP_CLASSOF_IMPL(TPDef::TPReplicateSC)

  /// Generate replicas of the desired Ingredient. Replicas will be generated
  /// for all parts and lanes unless a specific part and lane are specified in
  /// the \p State.
  void execute(TPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override;
#endif

  bool isUniform() const { return IsUniform; }

  bool isPredicated() const { return IsPredicated; }

  /// Returns true if the recipe only uses the first lane of operand \p Op.
  bool onlyFirstLaneUsed(const TPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return isUniform();
  }

  /// Returns true if the recipe uses scalars of operand \p Op.
  bool usesScalars(const TPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return true;
  }

  /// Returns true if the recipe is used by a widened recipe via an intervening
  /// VPPredInstPHIRecipe. In this case, the scalar values should also be packed
  /// in a vector.
  bool shouldPack() const;

  /// Return the mask of a predicated VPReplicateRecipe.
  TPValue *getMask() {
    assert(isPredicated() && "Trying to get the mask of a unpredicated recipe");
    return getOperand(getNumOperands() - 1);
  }

  unsigned getOpcode() const { return getUnderlyingInstr()->getOpcode(); }
};

/// A recipe for generating conditional branches on the bits of a mask.
class TPBranchOnMaskRecipe : public TPRecipeBase {
public:
  TPBranchOnMaskRecipe(TPValue *BlockInMask)
      : TPRecipeBase(TPDef::TPBranchOnMaskSC, {}) {
    if (BlockInMask) // nullptr means all-one mask.
      addOperand(BlockInMask);
  }

  TPBranchOnMaskRecipe *clone() override {
    return new TPBranchOnMaskRecipe(getOperand(0));
  }

  TP_CLASSOF_IMPL(TPDef::TPBranchOnMaskSC)

  /// Generate the extraction of the appropriate bit from the block mask and the
  /// conditional branch.
  void execute(TPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override {
    O << Indent << "BRANCH-ON-MASK ";
    if (TPValue *Mask = getMask())
      Mask->printAsOperand(O, SlotTracker);
    else
      O << " All-One";
  }
#endif

  /// Return the mask used by this recipe. Note that a full mask is represented
  /// by a nullptr.
  TPValue *getMask() const {
    assert(getNumOperands() <= 1 && "should have either 0 or 1 operands");
    // Mask is optional.
    return getNumOperands() == 1 ? getOperand(0) : nullptr;
  }

  /// Returns true if the recipe uses scalars of operand \p Op.
  bool usesScalars(const TPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return true;
  }
};

/// VPPredInstPHIRecipe is a recipe for generating the phi nodes needed when
/// control converges back from a Branch-on-Mask. The phi nodes are needed in
/// order to merge values that are set under such a branch and feed their uses.
/// The phi nodes can be scalar or vector depending on the users of the value.
/// This recipe works in concert with VPBranchOnMaskRecipe.
class TPPredInstPHIRecipe : public TPSingleDefRecipe { // yuxin.an: L2412
public:
  /// Construct a VPPredInstPHIRecipe given \p PredInst whose value needs a phi
  /// nodes after merging back from a Branch-on-Mask.
  TPPredInstPHIRecipe(TPValue *PredV)
      : TPSingleDefRecipe(TPDef::TPPredInstPHISC, PredV) {}
  ~TPPredInstPHIRecipe() override = default;

  TPPredInstPHIRecipe *clone() override {
    return new TPPredInstPHIRecipe(getOperand(0));
  }

  TP_CLASSOF_IMPL(TPDef::TPPredInstPHISC)

  /// Generates phi nodes for live-outs as needed to retain SSA form.
  void execute(TPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override;
#endif

  /// Returns true if the recipe uses scalars of operand \p Op.
  bool usesScalars(const TPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return true;
  }
};

/// A common base class for widening memory operations. An optional mask can be
/// provided as the last operand.
class TPWidenMemoryRecipe : public TPRecipeBase {
protected:
  Instruction &Ingredient;

  /// Whether the accessed addresses are consecutive.
  bool Consecutive;

  /// Whether the consecutive accessed addresses are in reverse order.
  bool Reverse;

  /// Whether the memory access is masked.
  bool IsMasked = false;

  void setMask(TPValue *Mask) {
    assert(!IsMasked && "cannot re-set mask");
    if (!Mask)
      return;
    addOperand(Mask);
    IsMasked = true;
  }

  TPWidenMemoryRecipe(const char unsigned SC, Instruction &I,
                      std::initializer_list<TPValue *> Operands,
                      bool Consecutive, bool Reverse, DebugLoc DL)
      : TPRecipeBase(SC, Operands, DL), Ingredient(I), Consecutive(Consecutive),
        Reverse(Reverse) {
    assert((Consecutive || !Reverse) && "Reverse implies consecutive");
  }

public:
  TPWidenMemoryRecipe *clone() override {
    llvm_unreachable("cloning not supported");
  }

  static inline bool classof(const TPRecipeBase *R) {
    return R->getTPDefID() == TPRecipeBase::TPWidenLoadSC ||
           R->getTPDefID() == TPRecipeBase::TPWidenStoreSC ||
           R->getTPDefID() == TPRecipeBase::TPWidenLoadEVLSC ||
           R->getTPDefID() == TPRecipeBase::TPWidenStoreEVLSC;
  }

  static inline bool classof(const TPUser *U) {
    auto *R = dyn_cast<TPRecipeBase>(U);
    return R && classof(R);
  }

  /// Return whether the loaded-from / stored-to addresses are consecutive.
  bool isConsecutive() const { return Consecutive; }

  /// Per-dim memory stride overrides in elements.
  /// Populated by TPRecipePatternMatcher_match() via SCEV GEP-index analysis.
  DenseMap<unsigned, const SCEV *> MemStrides;

  /// Returns the effective memory stride for \p Dim as a SCEV expression.
  /// A store produces no SSA value, so it can't be a TPSingleDefRecipe. But after the SCEV
  /// stride analysis was added, stores also need DimSet and MemStrides - specifically for computing the C matrix's leading
  /// dimension (LDC) during contraction lowering.
  const SCEV *getMemStride(unsigned Dim, const TPlan &Plan,
                            ScalarEvolution &SE) const;

  /// Return whether the consecutive loaded/stored addresses are in reverse
  /// order.
  bool isReverse() const { return Reverse; }

  /// Return the address accessed by this recipe.
  TPValue *getAddr() const { return getOperand(0); }

  /// Returns true if the recipe is masked.
  bool isMasked() const { return IsMasked; }

  /// Return the mask used by this recipe. Note that a full mask is represented
  /// by a nullptr.
  TPValue *getMask() const {
    // Mask is optional and therefore the last operand.
    return isMasked() ? getOperand(getNumOperands() - 1) : nullptr;
  }

  /// Generate the wide load/store.
  void execute(TPTransformState &State) override {
    llvm_unreachable("VPWidenMemoryRecipe should not be instantiated.");
  }

  Instruction &getIngredient() const { return Ingredient; }
};

/// A recipe for widening load operations, using the address to load from and an
/// optional mask.
struct TPWidenLoadRecipe final : public TPWidenMemoryRecipe,
                                 public TPValue { // yuxin.an: L2521
  TPWidenLoadRecipe(LoadInst &Load, TPValue *Addr, TPValue *Mask,
                    bool Consecutive, bool Reverse, DebugLoc DL)
      : TPWidenMemoryRecipe(TPDef::TPWidenLoadSC, Load, {Addr}, Consecutive,
                            Reverse, DL),
        TPValue(this, &Load) {
    setMask(Mask);
  }

  TPWidenLoadRecipe *clone() override {
    return new TPWidenLoadRecipe(cast<LoadInst>(Ingredient), getAddr(),
                                 getMask(), Consecutive, Reverse,
                                 getDebugLoc());
  }

  TP_CLASSOF_IMPL(TPDef::TPWidenLoadSC);

  /// Generate a wide load or gather.
  void execute(TPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override;
#endif

  /// Returns true if the recipe only uses the first lane of operand \p Op.
  bool onlyFirstLaneUsed(const TPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    // Widened, consecutive loads operations only demand the first lane of
    // their address.
    return Op == getAddr() && isConsecutive();
  }
};

/// A recipe for widening store operations, using the stored value, the address
/// to store to and an optional mask.
struct TPWidenStoreRecipe final : public TPWidenMemoryRecipe {
  TPWidenStoreRecipe(StoreInst &Store, TPValue *Addr, TPValue *StoredVal,
                     TPValue *Mask, bool Consecutive, bool Reverse, DebugLoc DL)
      : TPWidenMemoryRecipe(TPDef::TPWidenStoreSC, Store, {Addr, StoredVal},
                            Consecutive, Reverse, DL) {
    setMask(Mask);
  }

  TPWidenStoreRecipe *clone() override {
    return new TPWidenStoreRecipe(cast<StoreInst>(Ingredient), getAddr(),
                                  getStoredValue(), getMask(), Consecutive,
                                  Reverse, getDebugLoc());
  }

  TP_CLASSOF_IMPL(TPDef::TPWidenStoreSC);

  /// Return the value stored by this recipe.
  TPValue *getStoredValue() const { return getOperand(1); }

  /// Generate a wide store or scatter.
  void execute(TPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override;
#endif

  /// Returns true if the recipe only uses the first lane of operand \p Op.
  bool onlyFirstLaneUsed(const TPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    // Widened, consecutive stores only demand the first lane of their address,
    // unless the same operand is also stored.
    return Op == getAddr() && isConsecutive() && Op != getStoredValue();
  }
};

/// Recipe to expand a SCEV expression.
class TPExpandSCEVRecipe : public TPSingleDefRecipe { // yuxin.an: L2678
  const SCEV *Expr;
  ScalarEvolution &SE;

public:
  TPExpandSCEVRecipe(const SCEV *Expr, ScalarEvolution &SE)
      : TPSingleDefRecipe(TPDef::TPExpandSCEVSC, {}), Expr(Expr), SE(SE) {}

  ~TPExpandSCEVRecipe() override = default;

  TPExpandSCEVRecipe *clone() override {
    return new TPExpandSCEVRecipe(Expr, SE);
  }

  TP_CLASSOF_IMPL(TPDef::TPExpandSCEVSC)

  /// Generate a canonical vector induction variable of the vector loop, with
  void execute(TPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override;
#endif

  const SCEV *getSCEV() const { return Expr; }
};

/// Canonical scalar induction phi of the vector loop. Starting at the specified
/// start value (either 0 or the resume value when vectorizing the epilogue
/// loop). VPWidenCanonicalIVRecipe represents the vector version of the
/// canonical induction variable.
class TPCanonicalIVPHIRecipe : public TPHeaderPHIRecipe { // yuxin.an: L2710
public:
  TPCanonicalIVPHIRecipe(TPValue *StartV, DebugLoc DL)
      : TPHeaderPHIRecipe(TPDef::TPCanonicalIVPHISC, nullptr, StartV, DL) {}

  ~TPCanonicalIVPHIRecipe() override = default;

  TPCanonicalIVPHIRecipe *clone() override {
    auto *R = new TPCanonicalIVPHIRecipe(getOperand(0), getDebugLoc());
    R->addOperand(getBackedgeValue());
    return R;
  }

  TP_CLASSOF_IMPL(TPDef::TPCanonicalIVPHISC)

  static inline bool classof(const TPHeaderPHIRecipe *D) {
    return D->getTPDefID() == TPDef::TPCanonicalIVPHISC;
  }

  /// Generate the canonical scalar induction phi of the vector loop.
  void execute(TPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override;
#endif

  /// Returns the scalar type of the induction.
  Type *getScalarType() const {
    return getStartValue()->getLiveInIRValue()->getType();
  }

  /// Returns true if the recipe only uses the first lane of operand \p Op.
  bool onlyFirstLaneUsed(const TPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return true;
  }

  /// Returns true if the recipe only uses the first part of operand \p Op.
  bool onlyFirstPartUsed(const TPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return true;
  }

  /// Check if the induction described by \p Kind, /p Start and \p Step is
  /// canonical, i.e.  has the same start and step (of 1) as the canonical IV.
  bool isCanonical(InductionDescriptor::InductionKind Kind, TPValue *Start,
                   TPValue *Step) const;
};

/// A recipe for generating the active lane mask for the vector loop that is
/// used to predicate the vector operations.
/// TODO: It would be good to use the existing VPWidenPHIRecipe instead and
/// remove VPActiveLaneMaskPHIRecipe.
class TPActiveLaneMaskPHIRecipe : public TPHeaderPHIRecipe { // yuxin.an: L2767
public:
  TPActiveLaneMaskPHIRecipe(TPValue *StartMask, DebugLoc DL)
      : TPHeaderPHIRecipe(TPDef::TPActiveLaneMaskPHISC, nullptr, StartMask,
                          DL) {}

  ~TPActiveLaneMaskPHIRecipe() override = default;

  TPActiveLaneMaskPHIRecipe *clone() override {
    return new TPActiveLaneMaskPHIRecipe(getOperand(0), getDebugLoc());
  }

  TP_CLASSOF_IMPL(TPDef::TPActiveLaneMaskPHISC)

  static inline bool classof(const TPHeaderPHIRecipe *D) {
    return D->getTPDefID() == TPDef::TPActiveLaneMaskPHISC;
  }

  /// Generate the active lane mask phi of the vector loop.
  void execute(TPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override;
#endif
};

/// A Recipe for widening the canonical induction variable of the vector loop.
class TPWidenCanonicalIVRecipe : public TPSingleDefRecipe {
public:
  TPWidenCanonicalIVRecipe(TPCanonicalIVPHIRecipe *CanonicalIV)
      : TPSingleDefRecipe(TPDef::TPWidenCanonicalIVSC, {CanonicalIV}) {}

  ~TPWidenCanonicalIVRecipe() override = default;

  TPWidenCanonicalIVRecipe *clone() override {
    return new TPWidenCanonicalIVRecipe(
        cast<TPCanonicalIVPHIRecipe>(getOperand(0)));
  }

  TP_CLASSOF_IMPL(TPDef::TPWidenCanonicalIVSC)

  /// Generate a canonical vector induction variable of the vector loop, with
  /// start = {<Part*VF, Part*VF+1, ..., Part*VF+VF-1> for 0 <= Part < UF}, and
  /// step = <VF*UF, VF*UF, ..., VF*UF>.
  void execute(TPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override;
#endif
};

/// A recipe for converting the input value \p IV value to the corresponding
/// value of an IV with different start and step values, using Start + IV *
/// Step.
class TPDerivedIVRecipe : public TPSingleDefRecipe {
  /// Kind of the induction.
  const InductionDescriptor::InductionKind Kind;
  /// If not nullptr, the floating point induction binary operator. Must be set
  /// for floating point inductions.
  const FPMathOperator *FPBinOp;

public:
  TPDerivedIVRecipe(const InductionDescriptor &IndDesc, TPValue *Start,
                    TPCanonicalIVPHIRecipe *CanonicalIV, TPValue *Step)
      : TPDerivedIVRecipe(
            IndDesc.getKind(),
            dyn_cast_or_null<FPMathOperator>(IndDesc.getInductionBinOp()),
            Start, CanonicalIV, Step) {}

  TPDerivedIVRecipe(InductionDescriptor::InductionKind Kind,
                    const FPMathOperator *FPBinOp, TPValue *Start, TPValue *IV,
                    TPValue *Step)
      : TPSingleDefRecipe(TPDef::TPDerivedIVSC, {Start, IV, Step}), Kind(Kind),
        FPBinOp(FPBinOp) {}

  ~TPDerivedIVRecipe() override = default;

  TPDerivedIVRecipe *clone() override {
    return new TPDerivedIVRecipe(Kind, FPBinOp, getStartValue(), getOperand(1),
                                 getStepValue());
  }

  TP_CLASSOF_IMPL(TPDef::TPDerivedIVSC)

  /// Generate the transformed value of the induction at offset StartValue (1.
  /// operand) + IV (2. operand) * StepValue (3, operand).
  void execute(TPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override;
#endif

  Type *getScalarType() const {
    return getStartValue()->getLiveInIRValue()->getType();
  }

  TPValue *getStartValue() const { return getOperand(0); }
  TPValue *getStepValue() const { return getOperand(2); }

  /// Returns true if the recipe only uses the first lane of operand \p Op.
  bool onlyFirstLaneUsed(const TPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return true;
  }
};

/// A recipe for handling phi nodes of integer and floating-point inductions,
/// producing their scalar values.
class TPScalarIVStepsRecipe : public TPRecipeWithIRFlags { // yuxin.an: L2921
  Instruction::BinaryOps InductionOpcode;
  Loop *L;
  TPValue *IV;

public:
  TPScalarIVStepsRecipe(TPValue *IV, TPValue *Step,
                        Instruction::BinaryOps Opcode, FastMathFlags FMFs,
                        Loop *L)
      : TPRecipeWithIRFlags(TPDef::TPScalarIVStepsSC,
                            ArrayRef<TPValue *>({IV, Step}), FMFs),
        InductionOpcode(Opcode), L(L), IV(IV) {}

  TPScalarIVStepsRecipe(const InductionDescriptor &IndDesc, TPValue *IV,
                        TPValue *Step, Loop *L)
      : TPScalarIVStepsRecipe(
            IV, Step, IndDesc.getInductionOpcode(),
            dyn_cast_or_null<FPMathOperator>(IndDesc.getInductionBinOp())
                ? IndDesc.getInductionBinOp()->getFastMathFlags()
                : FastMathFlags(),
            L) {}

  ~TPScalarIVStepsRecipe() override = default;

  TPScalarIVStepsRecipe *clone() override {
    return new TPScalarIVStepsRecipe(
        getOperand(0), getOperand(1), InductionOpcode,
        hasFastMathFlags() ? getFastMathFlags() : FastMathFlags(), L);
  }

  TP_CLASSOF_IMPL(TPDef::TPScalarIVStepsSC)

  /// Generate the scalarized versions of the phi node as needed by their users.
  void execute(TPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override;
#endif

  TPValue *getStepValue() const { return getOperand(1); }

  /// Returns true if the recipe only uses the first lane of operand \p Op.
  bool onlyFirstLaneUsed(const TPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return true;
  }
};

/// VPBasicBlock serves as the leaf of the Hierarchical Control-Flow Graph. It
/// holds a sequence of zero or more VPRecipe's each representing a sequence of
/// output IR instructions. All PHI-like recipes must come before any non-PHI
/// recipes.
class TPBasicBlock : public TPBlockBase { // L2971
public:
  using RecipeListTy = iplist<TPRecipeBase>;

protected:
  /// The VPRecipes held in the order of output instructions to generate.
  RecipeListTy Recipes;

  TPBasicBlock(const unsigned char BlockSC, const Twine &Name = "")
      : TPBlockBase(BlockSC, Name.str()) {}

public:
  TPBasicBlock(const Twine &Name = "", TPRecipeBase *Recipe = nullptr)
      : TPBlockBase(TPBasicBlockSC, Name.str()) {
    if (Recipe)
      appendRecipe(Recipe);
  }

  ~TPBasicBlock() override {
    while (!Recipes.empty())
      Recipes.pop_back();
  }

  /// Instruction iterators...
  using iterator = RecipeListTy::iterator;
  using const_iterator = RecipeListTy::const_iterator;
  using reverse_iterator = RecipeListTy::reverse_iterator;
  using const_reverse_iterator = RecipeListTy::const_reverse_iterator;

  //===--------------------------------------------------------------------===//
  /// Recipe iterator methods
  ///
  inline iterator begin() { return Recipes.begin(); }
  inline const_iterator begin() const { return Recipes.begin(); }
  inline iterator end() { return Recipes.end(); }
  inline const_iterator end() const { return Recipes.end(); }

  inline reverse_iterator rbegin() { return Recipes.rbegin(); }
  inline const_reverse_iterator rbegin() const { return Recipes.rbegin(); }
  inline reverse_iterator rend() { return Recipes.rend(); }
  inline const_reverse_iterator rend() const { return Recipes.rend(); }

  inline size_t size() const { return Recipes.size(); }
  inline bool empty() const { return Recipes.empty(); }
  inline const TPRecipeBase &front() const { return Recipes.front(); }
  inline TPRecipeBase &front() { return Recipes.front(); }
  inline const TPRecipeBase &back() const { return Recipes.back(); }
  inline TPRecipeBase &back() { return Recipes.back(); }

  /// Returns a reference to the list of recipes.
  RecipeListTy &getRecipeList() { return Recipes; }

  /// Returns a pointer to a member of the recipe list.
  static RecipeListTy TPBasicBlock::*getSublistAccess(TPRecipeBase *) {
    return &TPBasicBlock::Recipes;
  }

  /// Method to support type inquiry through isa, cast, and dyn_cast.
  static inline bool classof(const TPBlockBase *V) {
    return V->getTPBlockID() == TPBlockBase::TPBasicBlockSC ||
           V->getTPBlockID() == TPBlockBase::TPIRBasicBlockSC;
  }

  void insert(TPRecipeBase *Recipe, iterator InsertPt) {
    assert(Recipe && "No recipe to append.");
    assert(!Recipe->Parent && "Recipe already in VPlan");
    Recipe->Parent = this;
    Recipes.insert(InsertPt, Recipe);
  }

  /// Augment the existing recipes of a VPBasicBlock with an additional
  /// \p Recipe as the last recipe.
  void appendRecipe(TPRecipeBase *Recipe) { insert(Recipe, end()); }

  /// The method which generates the output IR instructions that correspond to
  /// this VPBasicBlock, thereby "executing" the VPlan.
  void execute(TPTransformState *State) override;

  /// Return the cost of this VPBasicBlock.
  InstructionCost cost(ElementCount VF, TPCostContext &Ctx) override;

  /// Return the position of the first non-phi node recipe in the block.
  iterator getFirstNonPhi();

  /// Returns an iterator range over the PHI-like recipes in the block.
  iterator_range<iterator> phis() {
    return make_range(begin(), getFirstNonPhi());
  }

  void dropAllReferences(TPValue *NewValue) override;

  /// Split current block at \p SplitAt by inserting a new block between the
  /// current block and its successors and moving all recipes starting at
  /// SplitAt to the new block. Returns the new block.
  TPBasicBlock *splitAt(iterator SplitAt);

  TPRegionBlock *getEnclosingLoopRegion();

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print this VPBsicBlock to \p O, prefixing all lines with \p Indent. \p
  /// SlotTracker is used to print unnamed VPValue's using consequtive numbers.
  ///
  /// Note that the numbering is applied to the whole VPlan, so printing
  /// individual blocks is consistent with the whole VPlan printing.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override;
  using TPBlockBase::print; // Get the print(raw_stream &O) version.
#endif

  /// If the block has multiple successors, return the branch recipe terminating
  /// the block. If there are no or only a single successor, return nullptr;
  TPRecipeBase *getTerminator();
  const TPRecipeBase *getTerminator() const;

  /// Returns true if the block is exiting it's parent region.
  bool isExiting() const;

  /// Clone the current block and it's recipes, without updating the operands of
  /// the cloned recipes.
  TPBasicBlock *clone() override {
    auto *NewBlock = new TPBasicBlock(getName());
    for (TPRecipeBase &R : *this)
      NewBlock->appendRecipe(R.clone());
    return NewBlock;
  }

protected:
  /// Execute the recipes in the IR basic block \p BB.
  void executeRecipes(TPTransformState *State, BasicBlock *BB);

private:
  /// Create an IR BasicBlock to hold the output instructions generated by this
  /// VPBasicBlock, and return it. Update the CFGState accordingly.
  BasicBlock *createEmptyBasicBlock(TPTransformState::CFGState &CFG,
                                    TPTransformState *State);
};

class TPIRBasicBlock : public TPBasicBlock {
  BasicBlock *IRBB;

public:
  TPIRBasicBlock(BasicBlock *IRBB, const Twine &Name = "")
      : TPBasicBlock(TPIRBasicBlockSC,
                     (Twine("ir-bb<") + (IRBB->hasName() ? IRBB->getName() : Name) + Twine(">")).str()),
        IRBB(IRBB) {}

  ~TPIRBasicBlock() override {}

  static inline bool classof(const TPBlockBase *V) {
    return V->getTPBlockID() == TPBlockBase::TPIRBasicBlockSC;
  }

  /// The method which generates the output IR instructions that correspond to
  /// this VPBasicBlock, thereby "executing" the VPlan.
  void execute(TPTransformState *State) override;

  TPIRBasicBlock *clone() override {
    auto *NewBlock = new TPIRBasicBlock(IRBB);
    for (TPRecipeBase &R : Recipes)
      NewBlock->appendRecipe(R.clone());
    return NewBlock;
  }

  BasicBlock *getIRBasicBlock() const { return IRBB; }
};

class TPRegionBlock : public TPBlockBase { // yuxin.an: L3149
  // TODO(yuxin.an)
  /// Hold the Single Entry of the SESE region modelled by the VPRegionBlock.
  TPBlockBase *Entry;

  /// Hold the Single Exiting block of the SESE region modelled by the
  /// VPRegionBlock.
  TPBlockBase *Exiting;
  TPBlockBase *Middle;
  TPBlockBase *Scalar;
  TPRegionBlock *Inner = nullptr;

public:
  void setInner(TPRegionBlock *inner) { Inner = inner; }

  /// Accessor for the inner nested region (if any).
  TPRegionBlock *getInner() { return Inner; }
  const TPRegionBlock *getInner() const { return Inner; }

private:

  DenseMap<Loop *, TPBlockBase *> Loop2HeaderTPB;
  DenseMap<TPBlockBase *, Loop *> HeaderTPB2Loop;

  DenseMap<Loop *, TPBlockBase *> Loop2LatchTPB;
  DenseMap<TPBlockBase *, Loop *> LatchTPB2Loop;

  /// An indicator whether this region is to generate multiple replicated
  /// instances of output IR corresponding to its VPBlockBases.
  bool IsReplicator;

public:
  // 2) outer region that contains an inner region
  TPRegionBlock(TPBlockBase *E, TPBlockBase *L, TPRegionBlock *InnerRegion,
                const Twine &Name = "", bool Rep = false)
      : TPBlockBase(TPRegionBlockSC, Name.str()),
        Entry(E), Exiting(L), Inner(InnerRegion), IsReplicator(Rep) {
    Entry->setParent(this);
    Exiting->setParent(this);
    Inner->setParent(this);
  }

  TPRegionBlock(TPBlockBase *Entry, TPBlockBase *Exiting,
                const Twine &Name = "", bool IsReplicator = false)
      : TPBlockBase(TPRegionBlockSC, Name.str()), Entry(Entry), Exiting(Exiting),
        IsReplicator(IsReplicator) {
    // assert(Entry->getPredecessors().empty() && "Entry block has predecessors."); 
    // assert(Exiting->getSuccessors().empty() && "Exit block has successors.");
    Entry->setParent(this);
    Exiting->setParent(this);
  }

  TPRegionBlock(TPBlockBase *Entry, TPBlockBase *Exiting,
                DenseMap<Loop *, TPBlockBase *> Loop2HeaderTPB,
                DenseMap<Loop *, TPBlockBase *> Loop2LatchTPB,
                const std::string &Name = "", bool IsReplicator = false);

  TPRegionBlock(const std::string &Name = "", bool IsReplicator = false)
      : TPBlockBase(TPRegionBlockSC, Name), Entry(nullptr), Exiting(nullptr),
        Loop2HeaderTPB({}), HeaderTPB2Loop({}), Loop2LatchTPB({}),
        LatchTPB2Loop({}), IsReplicator(IsReplicator) {}

  ~TPRegionBlock() override {
    if (Entry) {
      TPValue DummyValue;
      Entry->dropAllReferences(&DummyValue);
      deleteCFG(Entry);
    }
  }

  /// Method to support type inquiry through isa, cast, and dyn_cast.
  static inline bool classof(const TPBlockBase *V) {
    return V->getTPBlockID() == TPBlockBase::TPRegionBlockSC;
  }

  const TPBlockBase *getEntry() const { return Entry; }
  TPBlockBase *getEntry() { return Entry; }
  const TPBlockBase *getScalar() const { return Scalar; }
  TPBlockBase *getScalar() { return Scalar; }

  /// Set \p EntryBlock as the entry VPBlockBase of this VPRegionBlock. \p
  /// EntryBlock must have no predecessors.
  void setEntry(TPBlockBase *EntryBlock) {
    assert(EntryBlock->getPredecessors().empty() &&
           "Entry block cannot have predecessors.");
    Entry = EntryBlock;
    EntryBlock->setParent(this);
  }

  const TPBlockBase *getExiting() const { return Exiting; }
  TPBlockBase *getExiting() { return Exiting; }

  /// Set \p ExitingBlock as the exiting VPBlockBase of this VPRegionBlock. \p
  /// ExitingBlock must have no successors.
  void setExiting(TPBlockBase *ExitingBlock) {
    assert(ExitingBlock->getSuccessors().empty() &&
           "Exit block cannot have successors.");
    Exiting = ExitingBlock;
    ExitingBlock->setParent(this);
  }

  // TPBasicBlock *getBB(unsigned idx) {
  //   auto *EntryBB = getEntryBasicBlock();
  //   auto *Res = EntryBB;
  //   if (idx) {
  //     for (unsigned i = 0; i < idx; i++) {
  //       Res = dyn_cast<TPBasicBlock>(Res->getSingleSuccessor());
  //       if (!Res)
  //         return Res;
  //     }
  //   }
  //   return Res;
  // }

  /// Returns the pre-header VPBasicBlock of the loop region.
  TPBasicBlock *getPreheaderTPBB() {
    assert(!isReplicator() && "should only get pre-header of loop regions");
    return getSinglePredecessor()->getExitingBasicBlock();
  }

  const DenseMap<Loop *, TPBlockBase *> getLoop2HeaderTPB() const {
    return Loop2HeaderTPB;
  }
  DenseMap<Loop *, TPBlockBase *> getLoop2HeaderTPB() { return Loop2HeaderTPB; }

  const DenseMap<TPBlockBase *, Loop *> getHeaderTPB2Loop() const {
    return HeaderTPB2Loop;
  }
  DenseMap<TPBlockBase *, Loop *> getHeaderTPB2Loop() { return HeaderTPB2Loop; }

  const DenseMap<Loop *, TPBlockBase *> getLoop2LatchTPB() const {
    return Loop2LatchTPB;
  }
  DenseMap<Loop *, TPBlockBase *> getLoop2LatchTPB() { return Loop2LatchTPB; }

  const DenseMap<TPBlockBase *, Loop *> getLatchTPB2Loop() const {
    return LatchTPB2Loop;
  }
  DenseMap<TPBlockBase *, Loop *> getLatchTPB2Loop() { return LatchTPB2Loop; }

  /// An indicator whether this region is to generate multiple replicated
  /// instances of output IR corresponding to its VPBlockBases.
  bool isReplicator() const { return IsReplicator; }

  /// The method which generates the output IR instructions that correspond to
  /// this VPRegionBlock, thereby "executing" the VPlan.
  void execute(TPTransformState *State) override;

  // Return the cost of this region.
  InstructionCost cost(ElementCount VF, TPCostContext &Ctx) override;

  void dropAllReferences(TPValue *NewValue) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print this VPRegionBlock to \p O (recursively), prefixing all lines with
  /// \p Indent. \p SlotTracker is used to print unnamed VPValue's using
  /// consequtive numbers.
  ///
  /// Note that the numbering is applied to the whole VPlan, so printing
  /// individual regions is consistent with the whole VPlan printing.
  void print(raw_ostream &O, const Twine &Indent,
             TPSlotTracker &SlotTracker) const override;
  using TPBlockBase::print; // Get the print(raw_stream &O) version.
#endif

  /// Clone all blocks in the single-entry single-exit region of the block and
  /// their recipes without updating the operands of the cloned recipes.
  TPRegionBlock *clone() override;
};

class TPlan { // yuxin.an: L3253
  // TODO(yuxin.an)
  friend class TPlanPrinter;
  friend class TPSlotTracker;

  /// Hold the single entry to the Hierarchical CFG of the VPlan, i.e. the
  /// preheader of the vector loop.
  TPBasicBlock *Entry;

  /// VPBasicBlock corresponding to the original preheader. Used to place
  /// VPExpandSCEV recipes for expressions used during skeleton creation and the
  /// rest of VPlan execution.
  TPBasicBlock *Preheader;

  // Pattern
  std::shared_ptr<TensorizePattern> Pattern;

  /// Holds the VFs applicable to this VPlan.
  // SmallSetVector<TFTy, 2> TFs;

  /// Holds the UFs applicable to this VPlan. If empty, the VPlan is valid for
  /// any UF.
  // SmallSetVector<TUFTy, 2> UFs;

  /// Holds the name of the VPlan, for printing.
  std::string Name;

  /// Represents the backedge taken count of the original loop, for folding
  /// the tail. It equals TripCount - 1.
  MapVector<Loop *, TPValue *> BackedgeTakenCount;

  /// Represents the vector trip count.
  MapVector<Loop *, TPValue *> TensorTripCount;

  MapVector<Loop *, SCEV *> TripCount;

  /// Represents the loop-invariant VF * UF of the vector loop region.
  MapVector<Loop *, TPValue *> TFxUF;
  
  /// Represents the TPRegionBLock per loop-depth
  SmallVector<TPRegionBlock *, 4> Regions;

  /// Holds a mapping between Values and their corresponding VPValue inside
  /// VPlan.
  Value2TPValueTy Value2TPValue;

  /// Contains all the external definitions created for this VPlan. External
  /// definitions are VPValues that hold a pointer to their underlying IR.
  SmallVector<TPValue *, 16> TPLiveInsToFree;

  // TODO(yuxin.an)
  /// Values used outside the plan. It contains live-outs that need fixing. Any
  /// live-out that is fixed outside VPlan needs to be removed. The remaining
  /// live-outs are fixed via VPLiveOut::fixPhi.
  MapVector<PHINode *, TPLiveOut *> LiveOuts;

  /// PF[0]...PF[Depth-1]
  SmallVector<std::unique_ptr<TPSymbolicValue>> DimPFs;

  /// Total depth of nested-loop
  unsigned Depth = 0;

  /// Dims not in any store IndexExpr.
  SmallBitVector ReductionDims;

  /// dim index -> parallel factor.
  /// (e.g., dim 0 -> 4, dim 1 -> 8 for a 4*8 tile)
  DenseMap<unsigned, unsigned> DimPFMap;

  /// Mapping from SCEVs to the VPValues representing their expansions.
  /// NOTE: This mapping is temporary and will be removed once all users have
  /// been modeled in VPlan directly.
  DenseMap<const SCEV *, TPValue *> SCEVToExpansion;
public:
  /// Construct a VPlan with original preheader \p Preheader, trip count \p TC
  /// and \p Entry to the plan. At the moment, \p Preheader and \p Entry need
  /// to be disconnected, as the bypass blocks between them are not yet
  /// modeled in VPlan.
  TPlan(TPBasicBlock *Preheader, MapVector<Loop *, SCEV *> TC,
        TPBasicBlock *Entry, std::shared_ptr<TensorizePattern> Pattern)
      : TPlan(Preheader, Entry, Pattern) {
    TripCount = TC;
    for (Loop *L : Pattern->Info.Loops) {
      BackedgeTakenCount.insert({L, new TPValue()});
      TensorTripCount.insert({L, new TPValue()});
    }
  }

  /// Construct a VPlan with original preheader \p Preheader and \p Entry to
  /// the plan. At the moment, \p Preheader and \p Entry need to be
  /// disconnected, as the bypass blocks between them are not yet modeled in
  /// VPlan.
  TPlan(TPBasicBlock *Preheader, TPBasicBlock *Entry,
        std::shared_ptr<TensorizePattern> Pattern)
      : Entry(Entry), Preheader(Preheader), Pattern(Pattern) {
    Entry->setPlan(this);
    Preheader->setPlan(this);
    assert(Preheader->getNumSuccessors() == 0 &&
           Preheader->getNumPredecessors() == 0 &&
           "preheader must be disconnected");
    for (Loop *L : Pattern->Info.Loops) {
      BackedgeTakenCount.insert({L, new TPValue()});
      TensorTripCount.insert({L, new TPValue()});
    }
  }

  TPlan(TPBasicBlock *Preheader, TPBasicBlock *Entry)
      : Entry(Entry), Preheader(Preheader) {
    Entry->setPlan(this);
    Preheader->setPlan(this);
    assert(Preheader->getNumSuccessors() == 0 &&
           Preheader->getNumPredecessors() == 0 &&
           "preheader must be disconnected");
  }

  ~TPlan();

  /// The LoopIdx is loop index which represents 0 
  /// for inner-most loop and N-1 for outer-most loop.
  /// DenseMap cannot gaurantee the order of inserting elements.
  MapVector<unsigned, Loop *> LoopIdx2Loop;
  MapVector<Loop *, unsigned> Loop2LoopIdx;
  MapVector<unsigned, TPRegionBlock *> LoopIdx2TPRB;
  MapVector<TPBasicBlock *, unsigned> PreHeaderTPBB2LoopIdx;
  MapVector<unsigned, TPBasicBlock *> LoopIdx2PreHeaderTPBB;
  MapVector<unsigned, TPIRBasicBlock *> LoopIdx2HeaderTPBB;
  MapVector<TPIRBasicBlock *, unsigned> HeaderTPBB2LoopIdx;
  MapVector<unsigned, TPBasicBlock *> LoopIdx2LatchTPBB;
  MapVector<TPBasicBlock *, unsigned> LatchTPBB2LoopIdx;
  MapVector<unsigned, TPBasicBlock *> LoopIdx2ExitingTPBB;
  MapVector<TPBasicBlock *, unsigned> ExitingTPBB2LoopIdx;

  const SCEV *getTCForDim(unsigned Dim) const {
    auto LoopIt = LoopIdx2Loop.find(Dim);
    if (LoopIt == LoopIdx2Loop.end())
      return nullptr;
    Loop *L = LoopIt->second;
  
    auto TCIt = TripCount.find(L);
    if (TCIt == TripCount.end())
      return nullptr;
    
    return TCIt->second;
  }

  std::shared_ptr<TensorizePattern> getPattern() { return Pattern; }

  /// Returns the dense (packed) stride for dimension \p Dim.
  /// Dense stride(D) = product of getPFForDim(d) for all d < D.
  /// Dim 0 (innermost) always returns 1.
  /// \p Dim uses DimIdx convention (innermost=0, outermost=Depth-1).
  uint64_t getDenseStrideForDim(unsigned Dim) const {
    uint64_t Stride = 1;
    for (unsigned D = 0; D < Dim; ++D)
      Stride *= static_cast<uint64_t>(getPFForDim(D));
    return Stride;
  }

  // Set all the bit as false with size Depth + 1 (conservative)
  void InitialReductionDims() { ReductionDims.resize(Depth + 1, false); }
  void setReductionDims(SmallBitVector NewDimset) { ReductionDims = NewDimset; }
  const SmallBitVector &getReductionDims() const { return ReductionDims; }

  /// Returns the parallel factor for dimension \p Dim. Default: 1 (scalar).
  /// TODO(yg0412.yun) Set by LoopTensorize via setDimPF() before lowering.
  unsigned getPFForDim(unsigned Dim) const {
    auto It = DimPFMap.find(Dim);
    return It != DimPFMap.end() ? It->second : 1u;
  }
  void setDimPF(unsigned Dim, unsigned PF) { DimPFMap[Dim] = PF; }

  void setDepth(unsigned NestLoopDepth) { 
    Depth = NestLoopDepth; 
    InitialReductionDims();
  }
  unsigned getDepth() { return Depth; }

  /// Returns the per-dimension parallel-factor symbolic value for dim \p D.
  TPSymbolicValue *getDimPF(unsigned D) const {
    assert(D < DimPFs.size() && "Dim out of range");
    return DimPFs[D].get();
  }

  /// Create initial VPlan, having an "entry" VPBasicBlock (wrapping
  /// original scalar pre-header ) which contains SCEV expansions that need
  /// to happen before the CFG is modified; a VPBasicBlock for the vector
  /// pre-header, followed by a region for the vector loop, followed by the
  /// middle VPBasicBlock. If a check is needed to guard executing the scalar
  /// epilogue loop, it will be added to the middle block, together with
  /// VPBasicBlocks for the scalar preheader and exit blocks.
  static TPlanPtr createInitialTPlan(MapVector<Loop *, SCEV *> TripCount,
                                     ScalarEvolution &SE,
                                     bool RequiresScalarEpilogueCheck,
                                     bool TailFolded, std::shared_ptr<TensorizePattern> Pattern);

  /// Prepare the plan for execution, setting up the required live-in values.
  void prepareToExecute(MapVector<Loop *, Value *> TensorTripCountV,
                        MapVector<Loop *, Value *> CanonicalIVStartValue,
                        TPTransformState &State);

  /// Generate the IR code for this VPlan.
  void execute(TPTransformState *State);

  /// Return the cost of this plan.
  InstructionCost cost(ElementCount VF, TPCostContext &Ctx);

  TPBasicBlock *getEntry() { return Entry; }
  const TPBasicBlock *getEntry() const { return Entry; }

  /// The trip count of the original loop.
  MapVector<Loop *, SCEV *> getTripCount() const {
    assert(!TripCount.empty() &&
           "trip count needs to be set before accessing it");
    return TripCount;
  }

  /// Resets the trip count for the VPlan. The caller must make sure all uses of
  /// the original trip count have been replaced.
  void resetTripCount(MapVector<Loop *, SCEV *> NewTripCount) {
    // assert(!TripCount.empty() && !NewTripCount.empty() &&
    //        "TripCount always must be set");
    // for (auto TCElem : TripCount)
    //   assert(TCElem.second->getNumUsers() && "TripCount always must be set");
    TripCount = NewTripCount;
  }

  /// The backedge taken count of the original loop.
  MapVector<Loop *, TPValue *> getOrCreateBackedgeTakenCount() {
    return BackedgeTakenCount;
  }

  /// The vector trip count.
  MapVector<Loop *, TPValue *> &getTensorTripCount() { return TensorTripCount; }

  /// Returns VF * UF of the vector loop region.
  MapVector<Loop *, TPValue *> &getTFxUF() { return TFxUF; }

  // void addTF(TFTy TF) { TFs.insert(TF); }

  // void setTF(TFTy TF) {
  //   assert(hasTF(TF) && "Cannot set TF not already in plan");
  //   TFs.clear();
  //   TFs.insert(TF);
  // }

  // bool hasTF(TFTy TF) { return TFs.count(TF); }
  bool hasScalableTF() { llvm_unreachable(""); }

  /// Returns an iterator range over all VFs of the plan.
  iterator_range<SmallSetVector<ElementCount, 2>::iterator>
  tensorFactors() const {
    llvm_unreachable("");
  }

  bool hasScalarTFOnly() const {
    //  !FIXME(yuxin.an)
    dbgs() << "[Warning] Please handle `TPlan::hasScalarTFOnly()` \n";
    return false;
  }

  // bool hasUF(TUFTy UF) const { return UFs.empty() || UFs.contains(UF); }

  // void setUF(TUFTy UF) {
  //   assert(hasUF(UF) && "Cannot set the UF not already in plan");
  //   UFs.clear();
  //   UFs.insert(UF);
  // }

  /// Return a string with the name of the plan and the applicable VFs and UFs.
  std::string getName() const;

  void setName(const Twine &newName) { Name = newName.str(); }

  /// Gets the live-in VPValue for \p V or adds a new live-in (if none exists
  ///  yet) for \p V.
  TPValue *getOrAddLiveIn(Value *V) {
    
    // YYG:REMOVE
    errs() << "[getOrAddLiveIn]\n";
    assert(V && "Trying to get or add the VPValue of a null Value");
    if (!Value2TPValue.count(V)) {
      TPValue *TPV = new TPValue(V);
      TPLiveInsToFree.push_back(TPV);
      assert(TPV->isLiveIn() && "TPV must be a live-in.");
      assert(!Value2TPValue.count(V) && "Value already exists in TPlan");
      Value2TPValue[V] = TPV;
    }

    assert(Value2TPValue.count(V) && "Value does not exist in TPlan");
    assert(Value2TPValue[V]->isLiveIn() &&
           "Only live-ins should be in mapping");
    return Value2TPValue[V];
  }

  /// Return the live-in VPValue for \p V, if there is one or nullptr otherwise.
  TPValue *getLiveIn(Value *V) const { return Value2TPValue.lookup(V); }

  Value2TPValueTy getValue2TPValue() { return Value2TPValue; }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the live-ins of this VPlan to \p O.
  void printLiveIns(raw_ostream &O) const;

  /// Print this VPlan to \p O.
  void print(raw_ostream &O) const;

  /// Print this VPlan in DOT format to \p O.
  void printDOT(raw_ostream &O) const;

  /// Dump the plan to stderr (for debugging).
  LLVM_DUMP_METHOD void dump() const;
#endif

  /// Returns the TPRegionBlock of the vector loop (outer‑most region).
  TPRegionBlock *getTensorLoopRegion() {
    return cast<TPRegionBlock>(getEntry()->getSingleSuccessor());
  }
  const TPRegionBlock *getTensorLoopRegion() const {
    return cast<TPRegionBlock>(getEntry()->getSingleSuccessor());
  }

  // TODO(yuxin.an)
  /// Returns the canonical induction recipe of the vector loop.
  TPCanonicalIVPHIRecipe *getCanonicalIV() {
    TPBasicBlock *EntryVPBB = getTensorLoopRegion()->getEntryBasicBlock();
    if (EntryVPBB->empty()) {
      // VPlan native path.
      EntryVPBB = cast<TPBasicBlock>(EntryVPBB->getSingleSuccessor());
    }
    return cast<TPCanonicalIVPHIRecipe>(&*EntryVPBB->begin());
  }

  void addLiveOut(PHINode *PN, TPValue *V);

  void removeLiveOut(PHINode *PN) {
    delete LiveOuts[PN];
    LiveOuts.erase(PN);
  }

  const MapVector<PHINode *, TPLiveOut *> &getLiveOuts() const {
    return LiveOuts;
  }

  TPValue *getSCEVExpansion(const SCEV *S) const {
    return SCEVToExpansion.lookup(S);
  }

  void addSCEVExpansion(const SCEV *S, TPValue *V) {
    assert(!SCEVToExpansion.contains(S) && "SCEV already expanded");
    SCEVToExpansion[S] = V;
  }

  /// \return The block corresponding to the original preheader.
  TPBasicBlock *getPreheader() { return Preheader; }
  const TPBasicBlock *getPreheader() const { return Preheader; }

  /// Clone the current VPlan, update all VPValues of the new VPlan and cloned
  /// recipes to refer to the clones, and return it.
  TPlan *duplicate();
};

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
/// VPlanPrinter prints a given VPlan to a given output stream. The printing is
/// indented and follows the dot format.
class TPlanPrinter {
  // TODO(yuxin.an)
  raw_ostream &OS;
  const TPlan &Plan;
  unsigned Depth = 0;
  unsigned TabWidth = 2;
  std::string Indent;
  unsigned BID = 0;
  SmallDenseMap<const TPBlockBase *, unsigned> BlockID;

  TPSlotTracker SlotTracker;

  /// Handle indentation.
  void bumpIndent(int b) { Indent = std::string((Depth += b) * TabWidth, ' '); }

  /// Print a given \p Block of the Plan.
  void dumpBlock(const TPBlockBase *Block);

  /// Print the information related to the CFG edges going out of a given
  /// \p Block, followed by printing the successor blocks themselves.
  void dumpEdges(const TPBlockBase *Block);

  /// Print a given \p BasicBlock, including its VPRecipes, followed by printing
  /// its successor blocks.
  void dumpBasicBlock(const TPBasicBlock *BasicBlock);

  /// Print a given \p Region of the Plan.
  void dumpRegion(const TPRegionBlock *Region);

  unsigned getOrCreateBID(const TPBlockBase *Block) {
    return BlockID.count(Block) ? BlockID[Block] : BlockID[Block] = BID++;
  }

  Twine getOrCreateName(const TPBlockBase *Block);

  Twine getUID(const TPBlockBase *Block);

  /// Print the information related to a CFG edge between two VPBlockBases.
  void drawEdge(const TPBlockBase *From, const TPBlockBase *To, bool Hidden,
                const Twine &Label);

public:
  TPlanPrinter(raw_ostream &O, const TPlan &P)
      : OS(O), Plan(P), SlotTracker(&P) {}

  LLVM_DUMP_METHOD void dump();
};

struct TPlanIngredient {
  const Value *V;

  TPlanIngredient(const Value *V) : V(V) {}

  void print(raw_ostream &O) const;
};

inline raw_ostream &operator<<(raw_ostream &OS, const TPlanIngredient &I) {
  I.print(OS);
  return OS;
}

inline raw_ostream &operator<<(raw_ostream &OS, const TPlan &Plan) {
  Plan.print(OS);
  return OS;
}
#endif

//===----------------------------------------------------------------------===//
// TPTilingRegion — tiling loop structure replacing innermost TPRegionBlock
//===----------------------------------------------------------------------===//
class TPTilingRegion : public TPBlockBase {
  unsigned TilingDim;
  unsigned TilingPF;
  DimEmitMode Mode;
  TPBasicBlock *Body;
  TPBasicBlock *ScalarEpilogue;
  PHINode *OrigKIVPhi;

public:
  TPTilingRegion(unsigned Dim, unsigned PF, DimEmitMode Mode,
                 TPBasicBlock *Body, TPBasicBlock *ScalarEpilogue,
                 PHINode *OrigKIVPhi)
      : TPBlockBase(TPTilingRegionSC, "tiling-region"), TilingDim(Dim),
        TilingPF(PF), Mode(Mode), Body(Body), ScalarEpilogue(ScalarEpilogue),
        OrigKIVPhi(OrigKIVPhi) {}

  unsigned      getDim()            const { return TilingDim; }
  unsigned      getPF()             const { return TilingPF; }
  DimEmitMode   getMode()           const { return Mode; }
  TPBasicBlock *getBody()           const { return Body; }
  TPBasicBlock *getScalarEpilogue() const { return ScalarEpilogue; }
  PHINode      *getOrigKIVPhi()     const { return OrigKIVPhi; }

  void execute(TPTransformState *State) override;
  void print(raw_ostream &OS, const Twine &Indent,
             TPSlotTracker &Tracker) const override;

  static bool classof(const TPBlockBase *B) {
    return B->getTPBlockID() == TPTilingRegionSC;
  }
};

//===----------------------------------------------------------------------===//
// TPlan Utilities
//===----------------------------------------------------------------------===//

/// Class that provides utilities for VPBlockBases in VPlan.

class TPBlockUtils {
public:
  TPBlockUtils() = delete;

  /// Insert disconnected VPBlockBase \p NewBlock after \p BlockPtr. Add \p
  /// NewBlock as successor of \p BlockPtr and \p BlockPtr as predecessor of \p
  /// NewBlock, and propagate \p BlockPtr parent to \p NewBlock. \p BlockPtr's
  /// successors are moved from \p BlockPtr to \p NewBlock. \p NewBlock must
  /// have neither successors nor predecessors.
  static void insertBlockAfter(TPBlockBase *NewBlock, TPBlockBase *BlockPtr) {
    // assert(NewBlock->getSuccessors().empty() &&
    //        NewBlock->getPredecessors().empty() &&
    //        "Can't insert new block with predecessors or successors.");
    NewBlock->setParent(BlockPtr->getParent());
    SmallVector<TPBlockBase *> Succs(BlockPtr->successors());
    for (TPBlockBase *Succ : Succs) {
      disconnectBlocks(BlockPtr, Succ);
      connectBlocks(NewBlock, Succ);
    }
    connectBlocks(BlockPtr, NewBlock);
  }

  /// Insert disconnected VPBlockBases \p IfTrue and \p IfFalse after \p
  /// BlockPtr. Add \p IfTrue and \p IfFalse as succesors of \p BlockPtr and \p
  /// BlockPtr as predecessor of \p IfTrue and \p IfFalse. Propagate \p BlockPtr
  /// parent to \p IfTrue and \p IfFalse. \p BlockPtr must have no successors
  /// and \p IfTrue and \p IfFalse must have neither successors nor
  /// predecessors.
  static void insertTwoBlocksAfter(TPBlockBase *IfTrue, TPBlockBase *IfFalse,
                                   TPBlockBase *BlockPtr) {
    assert(IfTrue->getSuccessors().empty() &&
           "Can't insert IfTrue with successors.");
    assert(IfFalse->getSuccessors().empty() &&
           "Can't insert IfFalse with successors.");
    BlockPtr->setTwoSuccessors(IfTrue, IfFalse);
    IfTrue->setPredecessors({BlockPtr});
    IfFalse->setPredecessors({BlockPtr});
    IfTrue->setParent(BlockPtr->getParent());
    IfFalse->setParent(BlockPtr->getParent());
  }

  /// Connect VPBlockBases \p From and \p To bi-directionally. Append \p To to
  /// the successors of \p From and \p From to the predecessors of \p To. Both
  /// VPBlockBases must have the same parent, which can be null. Both
  /// VPBlockBases can be already connected to other VPBlockBases.
  static void connectBlocks(TPBlockBase *From, TPBlockBase *To) {
    // assert((From->getParent() == To->getParent()) &&
    //        "Can't connect two block with different parents");
    // assert(From->getNumSuccessors() < 2 &&
    //        "Blocks can't have more than two successors.");
    From->appendSuccessor(To);
    To->appendPredecessor(From);
  }

  /// Disconnect VPBlockBases \p From and \p To bi-directionally. Remove \p To
  /// from the successors of \p From and \p From from the predecessors of \p To.
  static void disconnectBlocks(TPBlockBase *From, TPBlockBase *To) {
    assert(To && "Successor to disconnect is null.");
    From->removeSuccessor(To);
    To->removePredecessor(From);
  }

  /// Return an iterator range over \p Range which only includes \p BlockTy
  /// blocks. The accesses are casted to \p BlockTy.
  template <typename BlockTy, typename T>
  static auto blocksOnly(const T &Range) {
    // Create BaseTy with correct const-ness based on BlockTy.
    using BaseTy = std::conditional_t<std::is_const<BlockTy>::value,
                                      const TPBlockBase, TPBlockBase>;

    // We need to first create an iterator range over (const) BlocktTy & instead
    // of (const) BlockTy * for filter_range to work properly.
    auto Mapped =
        map_range(Range, [](BaseTy *Block) -> BaseTy & { return *Block; });
    auto Filter = make_filter_range(
        Mapped, [](BaseTy &Block) { return isa<BlockTy>(&Block); });
    return map_range(Filter, [](BaseTy &Block) -> BlockTy * {
      return cast<BlockTy>(&Block);
    });
  }
};

namespace tputils {

/// Returns true if only the first lane of \p Def is used.
bool onlyFirstLaneUsed(const TPValue *Def);

/// Returns true if only the first part of \p Def is used.
bool onlyFirstPartUsed(const TPValue *Def);

/// Get or create a VPValue that corresponds to the expansion of \p Expr. If \p
/// Expr is a SCEVConstant or SCEVUnknown, return a VPValue wrapping the live-in
/// value. Otherwise return a VPExpandSCEVRecipe to expand \p Expr. If \p Plan's
/// pre-header already contains a recipe expanding \p Expr, return it. If not,
/// create a new one.
TPValue *getOrCreateTPValueForSCEVExpr(TPlan &Plan, const SCEV *Expr,
                                       ScalarEvolution &SE);

/// Returns true if \p VPV is uniform after vectorization.
inline bool isUniformAfterTensorization(TPValue *TPV) {
  // TODO(yuxin.an)
  llvm_unreachable("");
}

/// Get or create a TPValue that corresponds to the expansion of \p Expr. If \p
/// Expr is a SCEVConstant or SCEVUnknown, return a TPValue wrapping the live-in
/// value. Otherwise return a TPExpandSCEVRecipe to expand \p Expr. If \p Plan's
/// pre-header already contains a recipe expanding \p Expr, return it. If not,
/// create a new one.
TPValue *getOrCreateTPValueForSCEVExpr(TPlan &Plan, const SCEV *Expr,
                                       ScalarEvolution &SE);

/// Returns true if \p TPV is uniform after vectorization.
inline bool isUniformAfterVectorization(TPValue *TPV) {
  // A value defined outside the vector region must be uniform after
  // vectorization inside a vector region.
  if (TPV->isDefinedOutsideVectorRegions())
    return true;
  TPRecipeBase *Def = TPV->getDefiningRecipe();
  assert(Def && "Must have definition for value defined inside vector region");
  if (auto Rep = dyn_cast<TPReplicateRecipe>(Def))
    return Rep->isUniform();
  if (auto *GEP = dyn_cast<TPWidenGEPRecipe>(Def))
    return all_of(GEP->operands(), isUniformAfterVectorization);
  if (auto *VPI = dyn_cast<TPInstruction>(Def))
    return VPI->isSingleScalar() || VPI->isTensorToScalar();
  return false;
}

/// Return true if \p V is a header mask in \p Plan.
bool isHeaderMask(TPValue *V, TPlan &Plan);
} // namespace tputils

} // end of namespace llvm

#endif // LLVM_TRANSFORMS_TENSORIZE_TPLAN_H
