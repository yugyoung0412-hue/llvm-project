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

/// Tile sizing policy returned by TTI::getTensorContractTileInfo().
/// Describes how emitContraction() should tile a tensor.contract call
/// when the trip count is a runtime (dynamic) value.
struct TensorContractTileInfo {
  /// Primary tile size along the contraction (K) dimension.
  /// Each main-loop iteration calls tensor.contract with exactly this K.
  unsigned PrimaryK = 0;

  /// Ordered list of epilogue K sizes (largest first).
  /// Each tier handles remaining elements when rem >= EpilogueKSizes[i].
  /// Empty → fall straight to scalar.block after the main loop.
  SmallVector<unsigned, 4> EpilogueKSizes;

  // Note: RequiresAlignedK and SupportsMasking are reserved for future use.
  // The current emitter always uses main_limit = (TC/PF)*PF (aligned) and
  // always emits scalar.block for the remainder, so these flags are not yet
  // consumed. They will be added when predicated/masked epilogue emission is
  // implemented.
};

} // namespace llvm
#endif
