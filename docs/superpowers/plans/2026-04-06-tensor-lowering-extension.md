# Tensor Lowering Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `TPWidenRecipe::execute()` in `TPlanLowering.cpp` to emit tensor intrinsics for all ElementWise opcodes/types, BroadcastBinary, and PlainReduction — replacing scalar fallbacks.

**Architecture:** Add two shared helpers (`getTypeSuffix`, `getOpcodeStr`) and two new intrinsic-declaration helpers (`getTensorBroadcastFn`, `getTensorReduceFn`) in `TPlanLowering.cpp`. Rewrite the three affected `switch` cases in `TPWidenRecipe::execute()` to call `tryVectorize()`/`tryReduce()` lambdas that emit the appropriate intrinsic, falling back to scalar clone on failure. All changes are confined to a single file.

**Tech Stack:** C++17, LLVM IR Builder, SCEV, LLVM lit test framework (`opt -passes=loop-tensorize -S | FileCheck`)

---

## File Map

| File | Role |
|------|------|
| `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` | All implementation — new helpers + rewritten execute() cases |
| `llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-int.ll` | New: ElementWise integer add test |
| `llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-f16.ll` | New: ElementWise f16 fadd test |
| `llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-icmp.ll` | New: ElementWise icmp_slt test |
| `llvm/test/Transforms/LoopTensorize/basic/tensor-broadcast-2d.ll` | New: BroadcastBinary 2d fadd test |
| `llvm/test/Transforms/LoopTensorize/basic/tensor-reduce-fadd.ll` | New: PlainReduction 2d fadd test |
| `llvm/test/Transforms/LoopTensorize/basic/tensor-reduce-partial.ll` | New: PlainReduction partial (row sum) test |

---

## Task 1: Add `getTypeSuffix()` and `getOpcodeStr()` helpers

These replace the hardcoded `isFloatTy() ? "f32" : "f64"` in `getTensorElementwiseFn()` and centralize opcode-to-string mapping for all intrinsic families.

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp:60-81`

- [ ] **Step 1: Add `getTypeSuffix()` after the `#define DEBUG_TYPE` line (line ~36)**

  Insert immediately before `getTensorMatmulFn()`:

  ```cpp
  /// Maps an LLVM element type to the suffix used in tensor intrinsic names.
  /// Returns "" for unsupported types — callers must fall back to scalar.
  static StringRef getTypeSuffix(Type *Ty) {
    if (Ty->isHalfTy())         return "f16";
    if (Ty->isFloatTy())        return "f32";
    if (Ty->isDoubleTy())       return "f64";
    if (Ty->isIntegerTy(1))     return "i1";
    if (Ty->isIntegerTy(8))     return "i8";
    if (Ty->isIntegerTy(16))    return "i16";
    if (Ty->isIntegerTy(32))    return "i32";
    if (Ty->isIntegerTy(64))    return "i64";
    return "";
  }
  ```

- [ ] **Step 2: Add `getOpcodeStr()` immediately after `getTypeSuffix()`**

  ```cpp
  /// Maps a BinaryOperator or CmpInst to the opcode string used in tensor
  /// intrinsic names. Returns "" for unsupported instructions.
  static std::string getOpcodeStr(const Instruction *I) {
    if (const auto *BO = dyn_cast<BinaryOperator>(I)) {
      switch (BO->getOpcode()) {
      case Instruction::FAdd: return "fadd";
      case Instruction::FSub: return "fsub";
      case Instruction::FMul: return "fmul";
      case Instruction::FDiv: return "fdiv";
      case Instruction::FRem: return "frem";
      case Instruction::Add:  return "add";
      case Instruction::Sub:  return "sub";
      case Instruction::Mul:  return "mul";
      case Instruction::SDiv: return "sdiv";
      case Instruction::UDiv: return "udiv";
      case Instruction::SRem: return "srem";
      case Instruction::URem: return "urem";
      case Instruction::And:  return "and";
      case Instruction::Or:   return "or";
      case Instruction::Xor:  return "xor";
      case Instruction::Shl:  return "shl";
      case Instruction::LShr: return "lshr";
      case Instruction::AShr: return "ashr";
      default: return "";
      }
    }
    if (const auto *CI = dyn_cast<FCmpInst>(I)) {
      switch (CI->getPredicate()) {
      case CmpInst::FCMP_OEQ: return "fcmp_oeq";
      case CmpInst::FCMP_OLT: return "fcmp_olt";
      case CmpInst::FCMP_OLE: return "fcmp_ole";
      case CmpInst::FCMP_OGT: return "fcmp_ogt";
      case CmpInst::FCMP_OGE: return "fcmp_oge";
      case CmpInst::FCMP_ONE: return "fcmp_one";
      case CmpInst::FCMP_ORD: return "fcmp_ord";
      case CmpInst::FCMP_UEQ: return "fcmp_ueq";
      case CmpInst::FCMP_ULT: return "fcmp_ult";
      case CmpInst::FCMP_ULE: return "fcmp_ule";
      case CmpInst::FCMP_UGT: return "fcmp_ugt";
      case CmpInst::FCMP_UGE: return "fcmp_uge";
      case CmpInst::FCMP_UNE: return "fcmp_une";
      case CmpInst::FCMP_UNO: return "fcmp_uno";
      default: return "";
      }
    }
    if (const auto *CI = dyn_cast<ICmpInst>(I)) {
      switch (CI->getPredicate()) {
      case CmpInst::ICMP_EQ:  return "icmp_eq";
      case CmpInst::ICMP_NE:  return "icmp_ne";
      case CmpInst::ICMP_SLT: return "icmp_slt";
      case CmpInst::ICMP_SLE: return "icmp_sle";
      case CmpInst::ICMP_SGT: return "icmp_sgt";
      case CmpInst::ICMP_SGE: return "icmp_sge";
      case CmpInst::ICMP_ULT: return "icmp_ult";
      case CmpInst::ICMP_ULE: return "icmp_ule";
      case CmpInst::ICMP_UGT: return "icmp_ugt";
      case CmpInst::ICMP_UGE: return "icmp_uge";
      default: return "";
      }
    }
    return "";
  }
  ```

- [ ] **Step 3: Update `getTensorElementwiseFn()` to use the new helpers**

  Replace lines 65-68 in `getTensorElementwiseFn()`:
  ```cpp
  // Before
  std::string Name = "llvm.tensor.elementwise.";
  Name += OpName.str();
  Name += "." + std::to_string(Rank) + "d.";
  Name += ElemTy->isFloatTy() ? "f32" : "f64";
  ```
  With:
  ```cpp
  // After
  StringRef TypeSuffix = getTypeSuffix(ElemTy);
  assert(!TypeSuffix.empty() && "unsupported element type");
  std::string Name = "llvm.tensor.elementwise.";
  Name += OpName.str();
  Name += "." + std::to_string(Rank) + "d.";
  Name += TypeSuffix;
  ```

- [ ] **Step 4: Build to verify no compile errors**

  ```bash
  ninja -C build LLVMVectorize 2>&1 | tail -20
  ```
  Expected: no errors.

- [ ] **Step 5: Commit**

  ```bash
  git add llvm/lib/Transforms/Vectorize/TPlanLowering.cpp
  git commit -m "tplan-lower: add getTypeSuffix/getOpcodeStr helpers; generalize getTensorElementwiseFn type suffix"
  ```

---

## Task 2: Extend ElementWise `execute()` to support all opcodes/types

Replace the hardcoded `FAdd/FSub/FMul`-only `switch` inside `tryVectorize()` with `getOpcodeStr()`-based dispatch. Handle FCmp/ICmp element type (operand type, not result `i1`).

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` — ElementWise case (~line 387-503)
- Create: `llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-int.ll`
- Create: `llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-f16.ll`
- Create: `llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-icmp.ll`

- [ ] **Step 1: Write the integer elementwise lit test (failing)**

  Create `llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-int.ll`:

  ```llvm
  ; RUN: opt -passes=loop-tensorize --disable-verify -S < %s | FileCheck %s
  ;
  ; 2D elementwise integer add: C[i][j] = A[i][j] + B[i][j]
  ; CHECK: call void @llvm.tensor.elementwise.add.2d.i32
  ; CHECK-SAME: i64 1, i64 256
  ; CHECK-SAME: i64 1, i64 256
  ; CHECK-SAME: i64 1, i64 256
  ; CHECK-SAME: i64 256, i64 256

  target datalayout = "e-m:e-i64:64-n32:64"
  target triple = "aarch64"

  define void @eltwise_add_2d_i32(ptr %A, ptr %B, ptr %C) {
  entry:
    br label %outer
  outer:
    %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
    br label %inner
  inner:
    %j = phi i64 [ 0, %outer ], [ %j.next, %inner.latch ]
    %ij = add i64 %i, %j
    %aptr = getelementptr i32, ptr %A, i64 %ij
    %bptr = getelementptr i32, ptr %B, i64 %ij
    %cptr = getelementptr i32, ptr %C, i64 %ij
    %av = load i32, ptr %aptr
    %bv = load i32, ptr %bptr
    %cv = add i32 %av, %bv
    store i32 %cv, ptr %cptr
    br label %inner.latch
  inner.latch:
    %j.next = add i64 %j, 1
    %j.done = icmp eq i64 %j.next, 16
    br i1 %j.done, label %outer.latch, label %inner
  outer.latch:
    %i.next = add i64 %i, 1
    %i.done = icmp eq i64 %i.next, 8
    br i1 %i.done, label %exit, label %outer
  exit:
    ret void
  }
  ```

- [ ] **Step 2: Write the f16 elementwise lit test (failing)**

  Create `llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-f16.ll`:

  ```llvm
  ; RUN: opt -passes=loop-tensorize --disable-verify -S < %s | FileCheck %s
  ;
  ; 2D elementwise f16 fadd: C[i][j] = A[i][j] + B[i][j]
  ; CHECK: call void @llvm.tensor.elementwise.fadd.2d.f16
  ; CHECK-SAME: i64 1, i64 256
  ; CHECK-SAME: i64 1, i64 256
  ; CHECK-SAME: i64 1, i64 256
  ; CHECK-SAME: i64 256, i64 256

  target datalayout = "e-m:e-i64:64-n32:64"
  target triple = "aarch64"

  define void @eltwise_fadd_2d_f16(ptr %A, ptr %B, ptr %C) {
  entry:
    br label %outer
  outer:
    %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
    br label %inner
  inner:
    %j = phi i64 [ 0, %outer ], [ %j.next, %inner.latch ]
    %ij = add i64 %i, %j
    %aptr = getelementptr half, ptr %A, i64 %ij
    %bptr = getelementptr half, ptr %B, i64 %ij
    %cptr = getelementptr half, ptr %C, i64 %ij
    %av = load half, ptr %aptr
    %bv = load half, ptr %bptr
    %cv = fadd half %av, %bv
    store half %cv, ptr %cptr
    br label %inner.latch
  inner.latch:
    %j.next = add i64 %j, 1
    %j.done = icmp eq i64 %j.next, 16
    br i1 %j.done, label %outer.latch, label %inner
  outer.latch:
    %i.next = add i64 %i, 1
    %i.done = icmp eq i64 %i.next, 8
    br i1 %i.done, label %exit, label %outer
  exit:
    ret void
  }
  ```

- [ ] **Step 3: Write the ICmp elementwise lit test (failing)**

  Create `llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-icmp.ll`:

  ```llvm
  ; RUN: opt -passes=loop-tensorize --disable-verify -S < %s | FileCheck %s
  ;
  ; 2D elementwise icmp slt: C[i][j] = A[i][j] < B[i][j]
  ; Type suffix uses operand type (i32), not result type (i1).
  ; CHECK: call void @llvm.tensor.elementwise.icmp_slt.2d.i32
  ; CHECK-SAME: i64 256, i64 256

  target datalayout = "e-m:e-i64:64-n32:64"
  target triple = "aarch64"

  define void @eltwise_icmp_slt_2d(ptr %A, ptr %B, ptr %C) {
  entry:
    br label %outer
  outer:
    %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
    br label %inner
  inner:
    %j = phi i64 [ 0, %outer ], [ %j.next, %inner.latch ]
    %ij = add i64 %i, %j
    %aptr = getelementptr i32, ptr %A, i64 %ij
    %bptr = getelementptr i32, ptr %B, i64 %ij
    %cptr = getelementptr i1,  ptr %C, i64 %ij
    %av = load i32, ptr %aptr
    %bv = load i32, ptr %bptr
    %cv = icmp slt i32 %av, %bv
    store i1 %cv, ptr %cptr
    br label %inner.latch
  inner.latch:
    %j.next = add i64 %j, 1
    %j.done = icmp eq i64 %j.next, 16
    br i1 %j.done, label %outer.latch, label %inner
  outer.latch:
    %i.next = add i64 %i, 1
    %i.done = icmp eq i64 %i.next, 8
    br i1 %i.done, label %exit, label %outer
  exit:
    ret void
  }
  ```

- [ ] **Step 4: Run tests to confirm they currently fail**

  ```bash
  llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-int.ll \
              llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-f16.ll \
              llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-icmp.ll
  ```
  Expected: FAIL — scalar clone emitted instead of the expected intrinsic call.

- [ ] **Step 5: Replace the opcode switch in `tryVectorize()` (ElementWise case)**

  In `TPlanLowering.cpp`, find the `tryVectorize` lambda inside `case TensorOpKind::ElementWise`. Replace:

  ```cpp
  // Determine op name from the BinaryOperator opcode.
  StringRef OpName;
  if (auto *BO = dyn_cast<BinaryOperator>(Inst)) {
    switch (BO->getOpcode()) {
    case Instruction::FAdd: OpName = "fadd"; break;
    case Instruction::FSub: OpName = "fsub"; break;
    case Instruction::FMul: OpName = "fmul"; break;
    default: return false;
    }
  } else {
    return false;
  }
  ```

  With:

  ```cpp
  // Determine op name and element type from the instruction.
  std::string OpName = getOpcodeStr(Inst);
  if (OpName.empty()) return false;

  // For CmpInst the result type is i1 — use operand type for the suffix.
  Type *ElemTyForSuffix = Inst->getType()->getScalarType();
  if (isa<CmpInst>(Inst) && Inst->getNumOperands() >= 1)
    ElemTyForSuffix = Inst->getOperand(0)->getType()->getScalarType();
  if (getTypeSuffix(ElemTyForSuffix).empty()) return false;
  ```

  Then replace the later line:
  ```cpp
  Type *ElemTy = ALoad->getInstruction()->getType()->getScalarType();
  if (!ElemTy->isFloatTy() && !ElemTy->isDoubleTy()) return false;
  ```
  With:
  ```cpp
  Type *ElemTy = ElemTyForSuffix;
  ```

  And in the `getTensorElementwiseFn` call, replace `OpName` (StringRef) with `StringRef(OpName)`:
  ```cpp
  auto EltFn = getTensorElementwiseFn(*Mod, StringRef(OpName), Rank, ElemTy);
  ```

- [ ] **Step 6: Build**

  ```bash
  ninja -C build LLVMVectorize 2>&1 | tail -20
  ```
  Expected: no errors.

- [ ] **Step 7: Run tests — confirm they now pass**

  ```bash
  llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-int.ll \
              llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-f16.ll \
              llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-icmp.ll
  ```
  Expected: PASS.

- [ ] **Step 8: Verify existing elementwise tests still pass**

  ```bash
  llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-stride-2d.ll \
              llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-stride-3d.ll
  ```
  Expected: PASS.

- [ ] **Step 9: Commit**

  ```bash
  git add llvm/lib/Transforms/Vectorize/TPlanLowering.cpp \
          llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-int.ll \
          llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-f16.ll \
          llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-icmp.ll
  git commit -m "tplan-lower: extend ElementWise execute() to all opcodes/types via getOpcodeStr"
  ```

---

## Task 3: Add `getTensorBroadcastFn()` and implement BroadcastBinary `execute()`

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` — add helper + rewrite BroadcastBinary case
- Create: `llvm/test/Transforms/LoopTensorize/basic/tensor-broadcast-2d.ll`

- [ ] **Step 1: Write the BroadcastBinary lit test (failing)**

  Create `llvm/test/Transforms/LoopTensorize/basic/tensor-broadcast-2d.ll`:

  ```llvm
  ; RUN: opt -passes=loop-tensorize --disable-verify -S < %s | FileCheck %s
  ;
  ; 2D broadcast fadd: C[i][j] = A[j] + B[i][j]   (A is 1D, B is 2D)
  ; A's dim=1 (i-loop) is missing → stride_A for i-dim = 0.
  ; CHECK: call void @llvm.tensor.broadcast.fadd.2d.f32
  ; CHECK-SAME: i64 0{{.*}}i64 256

  target datalayout = "e-m:e-i64:64-n32:64"
  target triple = "aarch64"

  define void @broadcast_fadd_2d(ptr %A, ptr %B, ptr %C) {
  entry:
    br label %outer
  outer:
    %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
    br label %inner
  inner:
    %j = phi i64 [ 0, %outer ], [ %j.next, %inner.latch ]
    %ij = add i64 %i, %j
    %aptr = getelementptr float, ptr %A, i64 %j
    %bptr = getelementptr float, ptr %B, i64 %ij
    %cptr = getelementptr float, ptr %C, i64 %ij
    %av = load float, ptr %aptr
    %bv = load float, ptr %bptr
    %cv = fadd float %av, %bv
    store float %cv, ptr %cptr
    br label %inner.latch
  inner.latch:
    %j.next = add i64 %j, 1
    %j.done = icmp eq i64 %j.next, 16
    br i1 %j.done, label %outer.latch, label %inner
  outer.latch:
    %i.next = add i64 %i, 1
    %i.done = icmp eq i64 %i.next, 8
    br i1 %i.done, label %exit, label %outer
  exit:
    ret void
  }
  ```

- [ ] **Step 2: Run test to confirm it fails**

  ```bash
  llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tensor-broadcast-2d.ll
  ```
  Expected: FAIL — scalar clone emitted.

- [ ] **Step 3: Add `getTensorBroadcastFn()` after `getTensorElementwiseFn()`**

  ```cpp
  /// Returns (creating if needed) @llvm.tensor.broadcast.<op>.<rank_out>d.<type>.
  /// Signature identical to getTensorElementwiseFn — stride=0 on broadcast dims
  /// is conveyed through the stride arguments, not the function type.
  static FunctionCallee getTensorBroadcastFn(Module &M, StringRef OpName,
                                              unsigned RankOut, Type *ElemTy) {
    StringRef TypeSuffix = getTypeSuffix(ElemTy);
    assert(!TypeSuffix.empty() && "unsupported element type");
    std::string Name = "llvm.tensor.broadcast.";
    Name += OpName.str();
    Name += "." + std::to_string(RankOut) + "d.";
    Name += TypeSuffix;
    LLVMContext &Ctx = M.getContext();
    Type *PtrTy = PointerType::getUnqual(Ctx);
    Type *I64Ty = Type::getInt64Ty(Ctx);
    SmallVector<Type *> Params;
    for (unsigned T = 0; T < 3; ++T) {
      Params.push_back(PtrTy);
      for (unsigned R = 0; R < RankOut; ++R)
        Params.push_back(I64Ty);
    }
    for (unsigned R = 0; R < RankOut; ++R)
      Params.push_back(I64Ty);
    FunctionType *FT = FunctionType::get(Type::getVoidTy(Ctx), Params, false);
    return M.getOrInsertFunction(Name, FT);
  }
  ```

- [ ] **Step 4: Replace the BroadcastBinary scalar fallback in `execute()`**

  Find the `case TensorOpKind::BroadcastBinary:` block (~line 516). Replace the entire case with:

  ```cpp
  case TensorOpKind::BroadcastBinary: {
    auto tryVectorize = [&]() -> bool {
      auto *ADR = dyn_cast<TPSingleDefRecipe>(getOperand(0));
      auto *BDR = dyn_cast<TPSingleDefRecipe>(getOperand(1));
      if (!ADR || !BDR) return false;

      std::string OpName = getOpcodeStr(Inst);
      if (OpName.empty()) return false;

      Type *ElemTyForSuffix = Inst->getType()->getScalarType();
      if (isa<CmpInst>(Inst) && Inst->getNumOperands() >= 1)
        ElemTyForSuffix = Inst->getOperand(0)->getType()->getScalarType();
      if (getTypeSuffix(ElemTyForSuffix).empty()) return false;

      // rank_out = rank of the larger DimSet operand.
      unsigned RankOut = std::max(ADR->DimSet.count(), BDR->DimSet.count());
      if (RankOut < 1 || RankOut > 3) return false;

      auto *ALoad = dyn_cast<TPWidenLoadRecipe>(ADR);
      auto *BLoad = dyn_cast<TPWidenLoadRecipe>(BDR);
      if (!ALoad || !BLoad) return false;

      auto *APtrDR = dyn_cast<TPSingleDefRecipe>(ALoad->getOperand(0));
      auto *BPtrDR = dyn_cast<TPSingleDefRecipe>(BLoad->getOperand(0));
      if (!APtrDR || !BPtrDR) return false;
      Value *APtr = State.getValue(APtrDR);
      Value *BPtr = State.getValue(BPtrDR);
      if (!APtr || !BPtr) return false;

      // Find C pointer from store user.
      Value *CPtr = nullptr;
      TPWidenStoreRecipe *CStoreRecipe = nullptr;
      if (auto *DefVal = this->getDefinedValue()) {
        for (TPUser *U : DefVal->users()) {
          auto *RB = dyn_cast<TPRecipeBase>(U);
          if (!RB) continue;
          if (auto *SR = dyn_cast<TPWidenStoreRecipe>(RB)) {
            CStoreRecipe = SR;
            auto *PtrDR = dyn_cast<TPSingleDefRecipe>(SR->getOperand(0));
            if (PtrDR) CPtr = State.getValue(PtrDR);
            break;
          }
        }
      }
      if (!CPtr) return false;

      IRBuilder<> &B = State.Builder;
      Type *ElemTy = ElemTyForSuffix;
      Module *Mod = B.GetInsertBlock()->getModule();
      auto BcastFn = getTensorBroadcastFn(*Mod, StringRef(OpName), RankOut, ElemTy);

      // Build stride args: for each operand, iterate dims 0..RankOut-1.
      // If the dim is in the operand's DimSet → getMemStride; else → 0 (broadcast).
      auto I64 = [&](uint64_t V) -> Value * { return B.getInt64(V); };
      auto expandStride = [&](const SCEV *S, unsigned Dim) -> Value * {
        if (State.Expander && State.Expander->isSafeToExpand(S))
          return State.Expander->expandCodeFor(S, B.getInt64Ty(),
                                               &*B.GetInsertPoint());
        return B.getInt64(State.Plan.getDenseStrideForDim(Dim));
      };

      SmallVector<Value *> Args;
      // C strides (rank_out dims from store recipe).
      Args.push_back(CPtr);
      for (unsigned D = 0; D < RankOut; ++D) {
        if (CStoreRecipe && CStoreRecipe->DimSet.test(D))
          Args.push_back(expandStride(
              CStoreRecipe->getMemStride(D, State.Plan, *State.SE), D));
        else
          Args.push_back(I64(State.Plan.getDenseStrideForDim(D)));
      }
      // A strides: missing dims → 0.
      Args.push_back(APtr);
      for (unsigned D = 0; D < RankOut; ++D) {
        if (ADR->DimSet.test(D))
          Args.push_back(expandStride(ADR->getMemStride(D, State.Plan, *State.SE), D));
        else
          Args.push_back(I64(0));
      }
      // B strides: missing dims → 0.
      Args.push_back(BPtr);
      for (unsigned D = 0; D < RankOut; ++D) {
        if (BDR->DimSet.test(D))
          Args.push_back(expandStride(BDR->getMemStride(D, State.Plan, *State.SE), D));
        else
          Args.push_back(I64(0));
      }
      // Shape: use the larger operand's shape for each dim.
      const TPSingleDefRecipe *LargerDR =
          ADR->DimSet.count() >= BDR->DimSet.count() ? ADR : BDR;
      SmallVector<unsigned> Shape = getTPValueShape(*LargerDR, State.Plan);
      for (unsigned D : Shape) Args.push_back(I64(D));

      B.CreateCall(BcastFn, Args);
      return true;
    };

    if (tryVectorize()) return;

    // Scalar fallback.
    auto *Clone = Inst->clone();
    State.remapClone(Clone);
    Value *Result = State.Builder.Insert(Clone);
    applyFlags(*cast<Instruction>(Result));
    State.EmittedMap[Inst] = Result;
    State.setValue(this, Result);
    return;
  }
  ```

- [ ] **Step 5: Build**

  ```bash
  ninja -C build LLVMVectorize 2>&1 | tail -20
  ```
  Expected: no errors.

- [ ] **Step 6: Run the broadcast test — confirm it passes**

  ```bash
  llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tensor-broadcast-2d.ll
  ```
  Expected: PASS.

- [ ] **Step 7: Run full LoopTensorize test suite**

  ```bash
  llvm-lit -v llvm/test/Transforms/LoopTensorize/
  ```
  Expected: all tests PASS.

- [ ] **Step 8: Commit**

  ```bash
  git add llvm/lib/Transforms/Vectorize/TPlanLowering.cpp \
          llvm/test/Transforms/LoopTensorize/basic/tensor-broadcast-2d.ll
  git commit -m "tplan-lower: add getTensorBroadcastFn; implement BroadcastBinary execute() with stride=0 broadcast convention"
  ```

---

## Task 4: Add `getTensorReduceFn()` and implement PlainReduction `execute()`

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` — add helper + rewrite PlainReduction case
- Create: `llvm/test/Transforms/LoopTensorize/basic/tensor-reduce-fadd.ll`
- Create: `llvm/test/Transforms/LoopTensorize/basic/tensor-reduce-partial.ll`

- [ ] **Step 1: Write the full-reduction lit test (failing)**

  Create `llvm/test/Transforms/LoopTensorize/basic/tensor-reduce-fadd.ll`:

  ```llvm
  ; RUN: opt -passes=loop-tensorize --disable-verify -S < %s | FileCheck %s
  ;
  ; 2D plain reduction: acc = sum of A[i][j] over all i, j  (no fmul producer)
  ; CHECK: call void @llvm.tensor.reduce.fadd.2d.f32
  ; CHECK-SAME: i64 0, i64 0

  target datalayout = "e-m:e-i64:64-n32:64"
  target triple = "aarch64"

  define float @plain_reduce_2d(ptr %A) {
  entry:
    br label %outer
  outer:
    %i     = phi i64   [ 0,   %entry ],  [ %i.next,   %outer.latch ]
    %acc.o = phi float [ 0.0, %entry ],  [ %acc.next, %outer.latch ]
    br label %inner
  inner:
    %j     = phi i64   [ 0,   %outer ],  [ %j.next,   %inner.latch ]
    %acc   = phi float [ %acc.o, %outer ], [ %acc.next, %inner.latch ]
    %ij    = add i64 %i, %j
    %aptr  = getelementptr float, ptr %A, i64 %ij
    %av    = load float, ptr %aptr
    %acc.next = fadd float %acc, %av
    br label %inner.latch
  inner.latch:
    %j.next = add i64 %j, 1
    %j.done = icmp eq i64 %j.next, 16
    br i1 %j.done, label %outer.latch, label %inner
  outer.latch:
    %i.next = add i64 %i, 1
    %i.done = icmp eq i64 %i.next, 8
    br i1 %i.done, label %exit, label %outer
  exit:
    ret float %acc.next
  }
  ```

- [ ] **Step 2: Write the partial-reduction (row sum) lit test (failing)**

  Create `llvm/test/Transforms/LoopTensorize/basic/tensor-reduce-partial.ll`:

  ```llvm
  ; RUN: opt -passes=loop-tensorize --disable-verify -S < %s | FileCheck %s
  ;
  ; Partial reduction: row_sum[i] += A[i][j]  (j is reduction dim, i is not)
  ; Acc advances along i-dim (stride != 0) but not j-dim (stride = 0).
  ; CHECK: call void @llvm.tensor.reduce.fadd.2d.f32
  ; CHECK-SAME: i64 256, i64 0

  target datalayout = "e-m:e-i64:64-n32:64"
  target triple = "aarch64"

  define void @partial_reduce_row_sum(ptr %A, ptr %RowSum) {
  entry:
    br label %outer
  outer:
    %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
    br label %inner
  inner:
    %j   = phi i64   [ 0, %outer ],   [ %j.next,  %inner.latch ]
    %acc = phi float [ 0.0, %outer ], [ %acc.next, %inner.latch ]
    %ij  = add i64 %i, %j
    %aptr = getelementptr float, ptr %A, i64 %ij
    %av   = load float, ptr %aptr
    %acc.next = fadd float %acc, %av
    br label %inner.latch
  inner.latch:
    %j.next = add i64 %j, 1
    %j.done = icmp eq i64 %j.next, 16
    br i1 %j.done, label %outer.latch, label %inner
  outer.latch:
    %rptr = getelementptr float, ptr %RowSum, i64 %i
    store float %acc.next, ptr %rptr
    %i.next = add i64 %i, 1
    %i.done = icmp eq i64 %i.next, 8
    br i1 %i.done, label %exit, label %outer
  exit:
    ret void
  }
  ```

- [ ] **Step 3: Run tests to confirm they fail**

  ```bash
  llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tensor-reduce-fadd.ll \
              llvm/test/Transforms/LoopTensorize/basic/tensor-reduce-partial.ll
  ```
  Expected: FAIL — existing test checks for `fadd` scalar instruction.

- [ ] **Step 4: Add `getTensorReduceFn()` after `getTensorBroadcastFn()`**

  ```cpp
  /// Returns (creating if needed) @llvm.tensor.reduce.<op>.<rank_in>d.<type>.
  /// Signature: void(ptr Acc, i64×rank_in Acc_strides,
  ///                 ptr A,   i64×rank_in A_strides,
  ///                 i64×rank_in dims)
  /// Reduction dims are encoded as Acc_stride=0.
  static FunctionCallee getTensorReduceFn(Module &M, StringRef OpName,
                                           unsigned RankIn, Type *ElemTy) {
    StringRef TypeSuffix = getTypeSuffix(ElemTy);
    assert(!TypeSuffix.empty() && "unsupported element type");
    std::string Name = "llvm.tensor.reduce.";
    Name += OpName.str();
    Name += "." + std::to_string(RankIn) + "d.";
    Name += TypeSuffix;
    LLVMContext &Ctx = M.getContext();
    Type *PtrTy = PointerType::getUnqual(Ctx);
    Type *I64Ty = Type::getInt64Ty(Ctx);
    SmallVector<Type *> Params;
    // Acc: ptr + rank_in strides
    Params.push_back(PtrTy);
    for (unsigned R = 0; R < RankIn; ++R) Params.push_back(I64Ty);
    // A: ptr + rank_in strides
    Params.push_back(PtrTy);
    for (unsigned R = 0; R < RankIn; ++R) Params.push_back(I64Ty);
    // dims
    for (unsigned R = 0; R < RankIn; ++R) Params.push_back(I64Ty);
    FunctionType *FT = FunctionType::get(Type::getVoidTy(Ctx), Params, false);
    return M.getOrInsertFunction(Name, FT);
  }
  ```

- [ ] **Step 5: Replace the PlainReduction scalar fallback in `execute()`**

  Find the `case TensorOpKind::PlainReduction:` block (~line 625). Replace with:

  ```cpp
  case TensorOpKind::PlainReduction: {
    auto tryReduce = [&]() -> bool {
      std::string OpName = getOpcodeStr(Inst);
      if (OpName.empty()) return false;

      Type *ElemTy = Inst->getType()->getScalarType();
      if (getTypeSuffix(ElemTy).empty()) return false;

      // Get the non-PHI input operand.
      TPValue *Input = nullptr;
      for (TPValue *Op : operands()) {
        auto *RV = dyn_cast<TPRecipeValue>(Op);
        if (!RV || !isa<TPReductionPHIRecipe>(RV->getDefiningRecipe()))
          Input = Op;
      }
      if (!Input) return false;

      auto *InputDR = dyn_cast<TPSingleDefRecipe>(Input);
      if (!InputDR) return false;

      unsigned RankIn = InputDR->DimSet.count();
      if (RankIn < 1 || RankIn > 3) return false;

      auto *InputLoad = dyn_cast<TPWidenLoadRecipe>(InputDR);
      if (!InputLoad) return false;
      auto *APtrDR = dyn_cast<TPSingleDefRecipe>(InputLoad->getOperand(0));
      if (!APtrDR) return false;
      Value *APtr = State.getValue(APtrDR);
      if (!APtr) return false;

      // Get Acc pointer from the TPReductionPHI recipe.
      Value *AccPtr = nullptr;
      for (TPValue *Op : operands()) {
        auto *RV = dyn_cast<TPRecipeValue>(Op);
        if (!RV) continue;
        auto *RedPHI = dyn_cast<TPReductionPHIRecipe>(RV->getDefiningRecipe());
        if (!RedPHI) continue;
        // The IR phi's incoming value from preheader is the initial accumulator.
        // For pointer extraction: use alloca if present, else create one.
        PHINode *Phi = RedPHI->getReductionPhi();
        if (!Phi) return false;
        // Allocate a slot for the accumulator on the stack.
        IRBuilder<> AllocaBuilder(
            &Phi->getParent()->getParent()->getEntryBlock().front());
        AccPtr = AllocaBuilder.CreateAlloca(ElemTy, nullptr, "reduce.acc");
        // Initialize with the PHI's preheader value.
        Value *InitVal = Phi->getIncomingValueForBlock(
            Phi->getParent()->getSinglePredecessor()
                ? Phi->getParent()->getSinglePredecessor()
                : Phi->getIncomingBlock(0));
        State.Builder.CreateStore(InitVal, AccPtr);
        break;
      }
      if (!AccPtr) return false;

      IRBuilder<> &B = State.Builder;
      Module *Mod = B.GetInsertBlock()->getModule();
      auto ReduceFn = getTensorReduceFn(*Mod, StringRef(OpName), RankIn, ElemTy);

      SmallBitVector ReductionDims = State.Plan.getReductionDims();
      ReductionDims.resize(RankIn);

      auto I64 = [&](uint64_t V) -> Value * { return B.getInt64(V); };
      auto expandStride = [&](const SCEV *S, unsigned Dim) -> Value * {
        if (State.Expander && State.Expander->isSafeToExpand(S))
          return State.Expander->expandCodeFor(S, B.getInt64Ty(),
                                               &*B.GetInsertPoint());
        return B.getInt64(State.Plan.getDenseStrideForDim(Dim));
      };

      SmallVector<Value *> Args;
      // Acc strides: reduction dims → 0, others → dense stride.
      Args.push_back(AccPtr);
      for (unsigned D = 0; D < RankIn; ++D) {
        if (D < ReductionDims.size() && ReductionDims.test(D))
          Args.push_back(I64(0));
        else
          Args.push_back(I64(State.Plan.getDenseStrideForDim(D)));
      }
      // A strides.
      Args.push_back(APtr);
      SmallVector<const SCEV *> AStrides =
          getTPValueStrides(*InputDR, State.Plan, *State.SE);
      int Didx = InputDR->DimSet.find_first();
      for (const SCEV *S : AStrides) {
        Args.push_back(expandStride(S, Didx >= 0 ? (unsigned)Didx : 0));
        if (Didx >= 0) Didx = InputDR->DimSet.find_next(Didx);
      }
      // Shape dims.
      SmallVector<unsigned> Shape = getTPValueShape(*InputDR, State.Plan);
      for (unsigned D : Shape) Args.push_back(I64(D));

      B.CreateCall(ReduceFn, Args);

      // Write result back: load from AccPtr and register as this recipe's value.
      Value *Result = B.CreateLoad(ElemTy, AccPtr, "reduce.result");
      State.setValue(this, Result);
      return true;
    };

    if (tryReduce()) return;

    // Scalar fallback.
    LLVM_DEBUG({
      dbgs() << "PlainReduction: tryReduce failed, scalar fallback\n";
    });
    auto *Clone = Inst->clone();
    State.remapClone(Clone);
    Value *Result = State.Builder.Insert(Clone);
    applyFlags(*cast<Instruction>(Result));
    State.EmittedMap[Inst] = Result;
    State.setValue(this, Result);
    return;
  }
  ```

- [ ] **Step 6: Add the `getReductionPhi()` accessor to `TPReductionPHIRecipe` if missing**

  Check `llvm/include/llvm/Transforms/Vectorize/TPlan.h` for `TPReductionPHIRecipe`. If `getReductionPhi()` does not exist, add:

  ```cpp
  PHINode *getReductionPhi() const { return RedPhi; }
  ```

  inside `TPReductionPHIRecipe`'s public section.

- [ ] **Step 7: Build**

  ```bash
  ninja -C build LLVMVectorize 2>&1 | tail -20
  ```
  Expected: no errors.

- [ ] **Step 8: Run the reduction tests**

  ```bash
  llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tensor-reduce-fadd.ll \
              llvm/test/Transforms/LoopTensorize/basic/tensor-reduce-partial.ll
  ```
  Expected: PASS.

- [ ] **Step 9: Run the full test suite**

  ```bash
  llvm-lit -v llvm/test/Transforms/LoopTensorize/
  ```
  Expected: all tests PASS.

- [ ] **Step 10: Commit**

  ```bash
  git add llvm/lib/Transforms/Vectorize/TPlanLowering.cpp \
          llvm/include/llvm/Transforms/Vectorize/TPlan.h \
          llvm/test/Transforms/LoopTensorize/basic/tensor-reduce-fadd.ll \
          llvm/test/Transforms/LoopTensorize/basic/tensor-reduce-partial.ll
  git commit -m "tplan-lower: add getTensorReduceFn; implement PlainReduction execute() with stride=0 reduction convention"
  ```

---

## Task 5: Final verification and push

- [ ] **Step 1: Run the complete LoopTensorize suite one final time**

  ```bash
  llvm-lit -v llvm/test/Transforms/LoopTensorize/
  ```
  Expected: all tests PASS.

- [ ] **Step 2: Push to remote**

  ```bash
  git push yg LoopTensorizebyClaude
  ```

---

## Self-Review Notes

- **Spec §2.2 (FCmp/ICmp predicate list):** All 14 FCmp predicates and 10 ICmp predicates are covered in `getOpcodeStr()`.
- **Spec §3 (stride=0 convention):** Applied in both BroadcastBinary (missing DimSet dims) and PlainReduction (ReductionDims).
- **Spec §4.3 (FMax/FMin):** These are not `BinaryOperator` subclasses in LLVM IR — they are `llvm.maxnum`/`llvm.minnum` intrinsic calls, not regular instructions. `getOpcodeStr()` returns `""` for them, so PlainReduction falls back to scalar for FMax/FMin. This is acceptable for this iteration; a follow-up task can handle intrinsic-call reduction ops.
- **Spec §5.2 (`getOpcodeStr` signature):** Takes `const Instruction *` (not `StringRef`) — consistent across all tasks.
- **Type consistency:** `getTypeSuffix` / `getOpcodeStr` / `getTensorBroadcastFn` / `getTensorReduceFn` names used uniformly in Tasks 1-4.
