//===- TPlan.h - Tensor plan IR for LoopTensorize ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines TPlan and TPRecipe classes that form a VPlan-inspired
// intermediate representation for the LoopTensorize pass. Each TPlan holds
// typed recipe nodes (induction, memory, compute) and per-IV parallel factors.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_VECTORIZE_TPLAN_H
#define LLVM_TRANSFORMS_VECTORIZE_TPLAN_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Vectorize/LoopNestAnalyzer.h"
#include "llvm/Transforms/Vectorize/TensorPatternClassifier.h"
#include <cstdint>
#include <limits>

namespace llvm {

class TPlan;

/// Base class for all TPlan recipe nodes. Uses LLVM's intrusive list.
class TPRecipeBase : public ilist_node<TPRecipeBase> {
public:
  enum RecipeKind { Induction, Mem, Compute };

private:
  RecipeKind Kind;
  TPlan *Parent = nullptr;

public:
  TPRecipeBase(RecipeKind K) : Kind(K) {}
  virtual ~TPRecipeBase() = default;

  RecipeKind getKind() const { return Kind; }
  TPlan *getParent() const { return Parent; }
  void setParent(TPlan *P) { Parent = P; }

  virtual void print(raw_ostream &OS) const = 0;
  virtual TPRecipeBase *clone() const = 0;
};

/// Recipe representing a loop induction variable with a parallel factor.
class TPInductionRecipe : public TPRecipeBase {
public:
  InductionDesc Desc;
  uint32_t PF = 1;
  unsigned DimIndex = 0;

  TPInductionRecipe() : TPRecipeBase(Induction) {}
  TPInductionRecipe(const InductionDesc &D, unsigned Dim)
      : TPRecipeBase(Induction), Desc(D), DimIndex(Dim) {}

  void print(raw_ostream &OS) const override {
    OS << "[Induction dim=" << DimIndex << " PF=" << PF << "]";
  }

  TPRecipeBase *clone() const override {
    auto *R = new TPInductionRecipe(Desc, DimIndex);
    R->PF = PF;
    return R;
  }

  static bool classof(const TPRecipeBase *R) {
    return R->getKind() == Induction;
  }
};

/// Recipe representing a memory access (load or store).
class TPMemRecipe : public TPRecipeBase {
public:
  MemAccess MA;
  bool IsWrite = false;

  TPMemRecipe() : TPRecipeBase(Mem) {}
  TPMemRecipe(const MemAccess &A, bool Write)
      : TPRecipeBase(Mem), MA(A), IsWrite(Write) {}

  void print(raw_ostream &OS) const override {
    OS << "[Mem " << (IsWrite ? "Write" : "Read")
       << " base=" << MA.BasePtr << "]";
  }

  TPRecipeBase *clone() const override {
    auto *R = new TPMemRecipe(MA, IsWrite);
    return R;
  }

  static bool classof(const TPRecipeBase *R) {
    return R->getKind() == Mem;
  }
};

/// Recipe representing a compute operation.
class TPComputeRecipe : public TPRecipeBase {
public:
  enum ComputeKind { Elementwise, Reduction, MatMul, Conv };

  ComputeKind Kind = Elementwise;
  PatternKind Pattern = PatternKind::Generic;
  bool UseIm2Col = false;

  TPComputeRecipe() : TPRecipeBase(Compute) {}
  explicit TPComputeRecipe(ComputeKind K)
      : TPRecipeBase(Compute), Kind(K) {}

  void print(raw_ostream &OS) const override {
    OS << "[Compute "
       << (Kind == MatMul ? "MatMul"
           : Kind == Conv ? "Conv"
           : Kind == Reduction ? "Reduction"
           : "Elementwise")
       << "]";
  }

  TPRecipeBase *clone() const override {
    auto *R = new TPComputeRecipe(Kind);
    R->Pattern = Pattern;
    R->UseIm2Col = UseIm2Col;
    return R;
  }

  static bool classof(const TPRecipeBase *R) {
    return R->getKind() == Compute;
  }
};

/// A tensor plan: holds loop nest info, per-IV parallel factors, and a
/// list of typed recipe nodes.
class TPlan {
  LoopNestInfo NestInfo;
  SmallVector<uint32_t> PFs;
  iplist<TPRecipeBase> Recipes;
  float Cost = std::numeric_limits<float>::infinity();

public:
  TPlan() = default;
  ~TPlan();

  TPlan(const TPlan &) = delete;
  TPlan &operator=(const TPlan &) = delete;
  TPlan(TPlan &&Other);
  TPlan &operator=(TPlan &&Other);

  /// Build an initial TPlan from a LoopNestInfo with all PFs = 1.
  static TPlan buildInitial(const LoopNestInfo &Info);

  /// Create a new TPlan with different parallel factors.
  TPlan withPFs(ArrayRef<uint32_t> NewPFs) const;

  uint32_t getPF(unsigned Dim) const {
    return Dim < PFs.size() ? PFs[Dim] : 1;
  }
  ArrayRef<uint32_t> getAllPFs() const { return PFs; }
  float getCost() const { return Cost; }
  void setCost(float C) { Cost = C; }
  const LoopNestInfo &getNestInfo() const { return NestInfo; }

  using recipe_iterator = iplist<TPRecipeBase>::iterator;
  using const_recipe_iterator = iplist<TPRecipeBase>::const_iterator;
  iterator_range<recipe_iterator> recipes() {
    return make_range(Recipes.begin(), Recipes.end());
  }
  iterator_range<const_recipe_iterator> recipes() const {
    return make_range(Recipes.begin(), Recipes.end());
  }

  void addRecipe(TPRecipeBase *R) {
    R->setParent(this);
    Recipes.push_back(R);
  }

  void print(raw_ostream &OS) const;
};

} // namespace llvm

#endif
