//===- TPlan.cpp - Tensor plan IR for LoopTensorize -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TPlan.h"

using namespace llvm;

TPlan::~TPlan() {
  Recipes.clear();
}

TPlan::TPlan(TPlan &&Other)
    : NestInfo(std::move(Other.NestInfo)),
      PFs(std::move(Other.PFs)),
      Recipes(std::move(Other.Recipes)),
      Cost(Other.Cost) {
  for (auto &R : Recipes)
    R.setParent(this);
}

TPlan &TPlan::operator=(TPlan &&Other) {
  if (this != &Other) {
    Recipes.clear();
    NestInfo = std::move(Other.NestInfo);
    PFs = std::move(Other.PFs);
    Recipes = std::move(Other.Recipes);
    Cost = Other.Cost;
    for (auto &R : Recipes)
      R.setParent(this);
  }
  return *this;
}

TPlan TPlan::buildInitial(const LoopNestInfo &Info) {
  TPlan Plan;
  Plan.NestInfo = Info;
  Plan.PFs.assign(Info.IVs.size(), 1);
  Plan.Cost = std::numeric_limits<float>::infinity();

  // 1. Induction recipes — one per IV.
  for (unsigned I = 0, E = Info.IVs.size(); I < E; ++I) {
    auto *R = new TPInductionRecipe(Info.IVs[I], I);
    R->PF = 1;
    Plan.addRecipe(R);
  }

  // 2. Count reads/writes.
  unsigned ReadCount = 0, WriteCount = 0;
  for (const auto &MA : Info.Accesses) {
    if (MA.Kind == AccessKind::Read)
      ++ReadCount;
    else
      ++WriteCount; // Write or ReadWrite
  }

  // 3. Mem read recipes (reads first).
  for (const auto &MA : Info.Accesses) {
    if (MA.Kind == AccessKind::Read)
      Plan.addRecipe(new TPMemRecipe(MA, /*IsWrite=*/false));
  }

  // 4. Compute recipe — infer kind from reads/writes/depth.
  TPComputeRecipe::ComputeKind CKind;
  if (ReadCount >= 2 && WriteCount == 1 && Info.Depth >= 4)
    CKind = TPComputeRecipe::Conv;
  else if (ReadCount >= 2 && WriteCount == 1 && Info.Depth >= 3)
    CKind = TPComputeRecipe::MatMul;
  else
    CKind = TPComputeRecipe::Elementwise;

  Plan.addRecipe(new TPComputeRecipe(CKind));

  // 5. Mem write recipes (writes last).
  for (const auto &MA : Info.Accesses) {
    if (MA.Kind == AccessKind::Write || MA.Kind == AccessKind::ReadWrite)
      Plan.addRecipe(new TPMemRecipe(MA, /*IsWrite=*/true));
  }

  return Plan;
}

TPlan TPlan::withPFs(ArrayRef<uint32_t> NewPFs) const {
  TPlan New;
  New.NestInfo = NestInfo;
  New.PFs.assign(NewPFs.begin(), NewPFs.end());
  New.Cost = Cost;

  // Deep-copy each recipe, updating PFs for induction recipes.
  unsigned DimIdx = 0;
  for (const auto &R : Recipes) {
    TPRecipeBase *Copy = R.clone();
    if (auto *IR = dyn_cast<TPInductionRecipe>(Copy)) {
      if (DimIdx < NewPFs.size())
        IR->PF = NewPFs[DimIdx++];
    }
    New.addRecipe(Copy);
  }

  return New;
}

void TPlan::print(raw_ostream &OS) const {
  OS << "TPlan [cost=" << Cost << " depth=" << NestInfo.Depth << "]\n";
  OS << "  PFs: [";
  for (unsigned I = 0; I < PFs.size(); ++I) {
    if (I) OS << ", ";
    OS << PFs[I];
  }
  OS << "]\n";
  for (const auto &R : Recipes) {
    OS << "  ";
    R.print(OS);
    OS << "\n";
  }
}
