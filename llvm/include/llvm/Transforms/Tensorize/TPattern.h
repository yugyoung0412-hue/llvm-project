#ifndef LLVM_TRANSFORMS_TENSORIZE_TPATTERN_H
#define LLVM_TRANSFORMS_TENSORIZE_TPATTERN_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Transforms/Tensorize/TensorizeCommon.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include <memory>

namespace llvm {
class TPlan;
class TPRecipeBase;
class TPBasicBlock;
class TPSingleDefRecipe;
class TPRecipeBuilder;
using TPlanPtr = std::unique_ptr<TPlan>;

struct PatternStatus {
  bool CanTensorize = true;
};

struct PatternInfo {
  SmallVector<Loop *> Loops; // from inner-most loop
  SmallVector<Loop *> LoopsR; // from outer-most loop
  SmallPtrSet<Loop *, 8> VisitedLoops;
  unsigned Depth; // Loop-Depth
  MapVector<Loop *, PHINode *> Loop2PHI_tmp;
  MapVector<Loop *, SmallPtrSet<PHINode *, 8>> Loop2PHI;
  MapVector<PHINode *, Loop *> PHI2Loop;
  Type *ElementTy = nullptr;

  /// Represents the loop-induction variables
  DenseMap<PHINode *, TPRecipeBase *> Induction2TPRecipe;

  void insertLoop(Loop *L, PHINode *PHI) {
    // insert inner-most loop
    if (!VisitedLoops.count(L)) {
      Loops.push_back(L);
      VisitedLoops.insert(L);
    }
    Loop2PHI_tmp.insert({L, PHI});
    Loop2PHI[L].insert(PHI);
    PHI2Loop.insert({PHI, L});
  }

  void insertInduction2TPRecipe(PHINode * Phi, TPRecipeBase * TPRecipe) {  
    Induction2TPRecipe.insert({Phi, TPRecipe});
  }
  
};

enum class PatternKind {
  AutoPattern,
  GEMMPattern,
  ElementWisePattern,
};

class TensorizePattern {
public:
  // const unsigned char SubclassID; ///< Subclass identifier (for
  // isa/dyn_cast).

  SmallVector<Loop *> Loops;

  TFTy *MaxTF;

  virtual bool isSemanticMatch() = 0;

  // using PatternID = enum { AutoPattern, GEMMPattern };

  /// \return an ID for the concrete type of this object.
  /// This is used to implement the classof checks. This should not be used
  /// for any other purpose, as the values may change as LLVM evolves.
  // unsigned getPatternID() const { return SubclassID; }

  virtual PatternKind getKind() const = 0;

  TensorizePattern(SmallVector<Loop *> &NestedLoops, unsigned ExpectedLoopDepth=0) {
    Loops = NestedLoops;
    if (ExpectedLoopDepth && NestedLoops.size() != ExpectedLoopDepth)
      Status.CanTensorize = false;
    for (Loop *L : NestedLoops) {
      auto header = L->getHeader();
      // PHINode *PHI = cast<PHINode>(&L->getBlocks().front()->front());
      for (PHINode &phi : header->phis()) {
        errs() << "[Auto] Loop: \n";
        L->dump();
        errs() << "phi: " << phi << "\n";

        // from inner-most loop to outer-most loop
        Info.insertLoop(L, &phi);
      }
    }
    Info.LoopsR = SmallVector<Loop *>(Info.Loops.rbegin(), Info.Loops.rend());
    // Index must start 0 and end N-1.
    // YYG::REMOVE
    errs() << "Info.Loops.size(): " << Info.Loops.size() << "\n";
    Info.Depth = Info.Loops.size();
  }

  unsigned getDepth() { return Info.Depth; }

  virtual bool tryToBuildTPlanWithTPRecipes(TPlanPtr &tplan,
                                            TPRecipeBuilder *RecipeBuilder,
                                            TPBasicBlock *TPBB,
                                            bool UseTensorType) = 0;

  virtual ~TensorizePattern() = default;

  PatternInfo Info;
  PatternStatus Status;
};

struct GEMMPatternInfo {
  bool TransposeA = false;
  bool TransposeB = false;
  LoadInst *LoadA = nullptr;
  LoadInst *LoadB = nullptr;
  SmallVector<GetElementPtrInst *> GEPs;
};

class TargetAutoPattern : public TensorizePattern {
public:
  // static constexpr unsigned char ID = PatternKind::AutoPattern;

  TargetAutoPattern(SmallVector<Loop *> &NestedLoops)
      : TensorizePattern(NestedLoops) {}

  PatternKind getKind() const override { return PatternKind::AutoPattern; }

  bool tryToBuildTPlanWithTPRecipes(TPlanPtr &tplan,
                                    TPRecipeBuilder *RecipeBuilder,
                                    TPBasicBlock *TPBB, bool UseTensorType);
  // -----------------------------------------------------------------
  static inline bool classof(const TensorizePattern *P) {
    return P && P->getKind() == PatternKind::AutoPattern;
  }

  static inline bool classof(const TensorizePattern &P) {
    return P.getKind() == PatternKind::AutoPattern;
  }

  // /// Method to support type inquiry through isa, cast, and dyn_cast.
  // static inline bool classof(const TensorizePattern *Pattern) {
  //   return Pattern->getPatternID() == TensorizePattern::AutoPattern;
  // }

  bool isSemanticMatch() override;
  virtual ~TargetAutoPattern();
};

class GEMMPattern : public TensorizePattern {
public:
  // static constexpr unsigned char ID = TensorizePattern::GEMMPattern;

  GEMMPattern(SmallVector<Loop *> &NestedLoops)
      : TensorizePattern(NestedLoops) {
    if (!isSemanticMatch())
      Status.CanTensorize = false;
  }

  bool isSemanticMatch() override;

  bool tryToBuildTPlanWithTPRecipes(TPlanPtr &tplan,
                                    TPRecipeBuilder *RecipeBuilder,
                                    TPBasicBlock *TPBB, bool UseTensorType);

  bool setUserTF(std::vector<unsigned> TensorizationFactors_);

  // -----------------------------------------------------------------
  static inline bool classof(const TensorizePattern *P) {
    return P && P->getKind() == PatternKind::GEMMPattern;
  }

  static inline bool classof(const TensorizePattern &P) {
    return P.getKind() == PatternKind::GEMMPattern;
  }
  PatternKind getKind() const override { return PatternKind::GEMMPattern; }

  // /// Method to support type inquiry through isa, cast, and dyn_cast.
  // static inline bool classof(const TensorizePattern *Pattern) {
  //   return Pattern->getPatternID() == TensorizePattern::GEMMPattern;
  // }

  GEMMPatternInfo GEMMInfo;
};

struct ElementWisePatternInfo {
  LoadInst *LoadA = nullptr;
  LoadInst *LoadB = nullptr;
  SmallVector<GetElementPtrInst *> GEPs;
};

class ElementWiseTensorizePattern : public TensorizePattern {
public:
  ElementWiseTensorizePattern(SmallVector<Loop *> &NestedLoops)
      : TensorizePattern(NestedLoops, 2) {
    if (!isSemanticMatch())
      Status.CanTensorize = false;
  }

  bool isSemanticMatch() override;

  bool tryToBuildTPlanWithTPRecipes(TPlanPtr &tplan,
                                    TPRecipeBuilder *RecipeBuilder,
                                    TPBasicBlock *TPBB, bool UseTensorType);

  bool setUserTF(std::vector<unsigned> TensorizationFactors_);

  // -----------------------------------------------------------------
  static inline bool classof(const TensorizePattern *P) {
    return P && P->getKind() == PatternKind::ElementWisePattern;
  }

  static inline bool classof(const TensorizePattern &P) {
    return P.getKind() == PatternKind::ElementWisePattern;
  }
  PatternKind getKind() const override {
    return PatternKind::ElementWisePattern;
  }

  ElementWisePatternInfo eleWiseInfo;
};

struct ConvInfo {
  // Element type can be (float, double, …)
  Type *ElementTy = nullptr;

  // C(OFM)   = A(IFM) * B(Kernel)
  //   GEPs[0] = GEP for input   (A)
  //   GEPs[1] = GEP for kernel  (B)
  //   GEPs[2] = GEP for output  (C)
  SmallVector<GetElementPtrInst *, 3> GEPs;

  LoadInst *LoadInput  = nullptr;   // A
  LoadInst *LoadWeight = nullptr;   // B
  // If using out-lined version ggml_conv_2d,
  Value *Accumulator = nullptr;  // C
  // If not, use this for output tensor, C.
  //LoadInst *Accumulator = nullptr;  // C

  Instruction *FMulInst = nullptr;
  StoreInst *InstStore = nullptr;

  // loop-idx PHI (innermost, middle, outermost)
  PHINode *InnermostPHI = nullptr;
  PHINode *MiddlePHI    = nullptr;
  PHINode *OutermostPHI = nullptr;

  // Option – if the kernel is transposed (im2col ↔ transposed‑kernel)
  bool TransposeKernel = false;

  Value *StrideH, *StrideW;
  Value *PadH, *PadW;
  Value *DilationH, *DilationW;
};

class ConvolutionTensorizePattern : public TensorizePattern {
public:
  ConvolutionTensorizePattern(SmallVector<Loop *> &NestedLoops)
      : TensorizePattern(NestedLoops, 7) {
    if (!isSemanticMatch()) {
      // YYG::REMOVE
      errs() << "!isSemanticMatch() \n";
      
      Status.CanTensorize = false;
    }
    else {
      // Function containing loop
      Function *F = Loops.front()->getHeader()->getParent();
      // Below is for GGML-specific case,
      std::vector<Value*> IntArgs;
      for (Argument &Arg : F->args()) {
        Type *Ty = Arg.getType();
        if (Ty->isIntegerTy(32))
          IntArgs.push_back(&Arg);
      }
      assert(IntArgs.size() == 6 && "Expected exactly 6 int32 arguments");
      CI.StrideH = IntArgs[0];
      CI.StrideW = IntArgs[1];
      CI.PadH = IntArgs[2];
      CI.PadW = IntArgs[3];
      CI.DilationH = IntArgs[4];
      CI.DilationW = IntArgs[5];
    }
    // YYG:REMOVE
    errs() << "status.CanTensorize: " << Status.CanTensorize << "\n";
  }

  bool isSemanticMatch() override;

  bool tryToBuildTPlanWithTPRecipes(TPlanPtr &tplan,
                                    TPRecipeBuilder *RecipeBuilder,
                                    TPBasicBlock *TPBB, bool UseTensorType);
  
  static PHINode *GetPHIFromLoop(Loop *L) {
    // The very first instruction of the first block of a loop is the PHI
    // that carries the induction variable.
    return cast<PHINode>(&L->getBlocks().front()->front());
  }

  static PHINode *pickPHI(Value *A, Value *B) {
    return isa<PHINode>(A) ? cast<PHINode>(A) : cast<PHINode>(B);
  }

  bool setUserTF(std::vector<unsigned> TensorizationFactors_);

  // -----------------------------------------------------------------
  static inline bool classof(const TensorizePattern *P) {
    return P && P->getKind() == PatternKind::ElementWisePattern;
  }

  static inline bool classof(const TensorizePattern &P) {
    return P.getKind() == PatternKind::ElementWisePattern;
  }
  PatternKind getKind() const override {
    return PatternKind::ElementWisePattern;
  }
  bool extractPaddingFromICmp() {
    // TODO(yg0412.yun)
    return true;
  }

  ElementWisePatternInfo eleWiseInfo;
  ConvInfo CI;
};


SmallVector<unsigned> getTPValueShape(const TPSingleDefRecipe &V,
                                      const TPlan &Plan);
} // namespace llvm

#endif // LLVM_TRANSFORMS_TENSORIZE_TPATTERN_H
