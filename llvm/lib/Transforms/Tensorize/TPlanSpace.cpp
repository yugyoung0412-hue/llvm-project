//===- TPlanSpace.cpp - A Loop Vectorizer ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tensorize/TPlanSpace.h"
#include "llvm/Transforms/Tensorize/TPlanner.h"

using namespace llvm;

namespace llvm {
// extern cl::opt<bool> EnableVPlanNativePath;
class instruction;
class TPlan;

#define DEBUG_TYPE "tplan"
} // namespace llvm
