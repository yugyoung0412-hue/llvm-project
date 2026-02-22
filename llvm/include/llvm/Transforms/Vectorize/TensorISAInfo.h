//===- TensorISAInfo.h - Tensor ISA descriptors ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_VECTORIZE_TENSORISAINFO_H
#define LLVM_TRANSFORMS_VECTORIZE_TENSORISAINFO_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"

namespace llvm {

struct TensorOpDesc {
  enum class Kind { MatMul, Conv2D, OuterProduct, Elementwise };

  Kind           OpKind;
  unsigned       M = 0, N = 0, K = 0; // tile dims; 0 = flexible
  Type          *InputTypeA  = nullptr;
  Type          *InputTypeB  = nullptr;
  Type          *AccumType   = nullptr;
  Intrinsic::ID  IntrinsicID = Intrinsic::not_intrinsic;
};

} // namespace llvm
#endif
