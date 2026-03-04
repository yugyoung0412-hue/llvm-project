//===- TPlan.cpp - TPlan CFG IR implementation ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TPlan.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// TPTransformState
//===----------------------------------------------------------------------===//

unsigned TPTransformState::getPF(unsigned Dim) const {
  return Plan ? Plan->getPF(Dim) : 1u;
}

//===----------------------------------------------------------------------===//
// TPBasicBlock
//===----------------------------------------------------------------------===//

void TPBasicBlock::execute(TPTransformState &State) {
  for (TPRecipeBase &R : Recipes)
    R.execute(State);
}

//===----------------------------------------------------------------------===//
// TPlan
//===----------------------------------------------------------------------===//

void TPlan::resolvePF(const TensorOpDesc &Op, ArrayRef<unsigned> ExplicitPF,
                       ArrayRef<Transform> BeamTiles) {
  // Apply explicit per-dimension parallel factors first.
  for (unsigned Dim = 0; Dim < ExplicitPF.size(); ++Dim)
    if (ExplicitPF[Dim] > 0)
      setPF(Dim, ExplicitPF[Dim]);

  // Apply tile sizes from beam-search transforms.
  for (const Transform &T : BeamTiles) {
    if (T.Kind == TransformKind::LoopTile && T.Size > 0)
      setPF(T.Dim, T.Size);
  }

  // Apply ISA op constraints if they set fixed tile dims.
  if (Op.M > 0)
    setPF(0, Op.M);
  if (Op.N > 0)
    setPF(1, Op.N);
  if (Op.K > 0)
    setPF(2, Op.K);
}

void TPlan::print(raw_ostream &O) const {
  O << "TPlan (pattern=";
  switch (Pattern) {
  case PatternKind::GEMM:        O << "GEMM"; break;
  case PatternKind::Conv2D:      O << "Conv2D"; break;
  case PatternKind::Elementwise: O << "Elementwise"; break;
  case PatternKind::Reduction:   O << "Reduction"; break;
  case PatternKind::Generic:     O << "Generic"; break;
  }
  O << ")\n";
}

bool TPlan::verify() const {
  // Minimal verification: all structural blocks must exist.
  if (!Entry || !VectorBody || !MiddleBlock)
    return false;
  return true;
}
