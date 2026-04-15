#ifndef LLVM_TRANSFORMS_TENSORIZE_TENSORIZECOMMON_H
#define LLVM_TRANSFORMS_TENSORIZE_TENSORIZECOMMON_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/TypeSize.h"
#include <utility>

namespace llvm {

// Tensor Factor Type
using TFTy = MapVector<Loop *, ElementCount>;

// Tensor Unroll Factor Type
using TUFTy = MapVector<Loop *, unsigned>;

} // namespace llvm

#endif // LLVM_TRANSFORMS_TENSORIZE_TENSORIZECOMMON_H
