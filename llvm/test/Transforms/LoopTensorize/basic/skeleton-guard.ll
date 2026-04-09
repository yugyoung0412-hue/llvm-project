; RUN: opt -passes=loop-tensorize -loop-tensorize-pf=16 -S < %s | FileCheck %s
;
; GEMM with static M=N=16 and DYNAMIC K, with an explicit gemm.ph preheader
; so that the outermost M loop (i.loop) has a single predecessor (entry) for
; its preheader.  The pass should create a tensor.guard block before i.loop
; that checks K >= 16 once — mirroring VPlan's createVectorizedLoopSkeleton().
;
; cloneLoopWithPreheader() inserts the scalar clone BEFORE the original
; preheader in the function layout.  Expected block order in output:
;
;   entry → tensor.guard (via gemm.ph.scalar scalar clone first in layout)
;   gemm.ph.scalar, i/j/k.loop.scalar  (scalar clone — before tensor.guard)
;   tensor.guard                        (guard: icmp uge k.tc.guard, 16)
;   gemm.ph, i/j/k.loop                (tensor path)
;   tensor.body.*, scalar.block         (dynamic K tiling inside tensor path)
;
; CHECK-LABEL: @gemm_skeleton(
;
; Scalar fallback: cloned M/N/K loops come first in the layout.
; CHECK:       gemm.ph.scalar:
; CHECK:       i.loop.scalar:
; CHECK:       j.loop.scalar:
;
; Guard block: emitted before the outermost M loop (tensor path).
; CHECK:       tensor.guard:
; CHECK-NEXT:    %tensor.profitable = icmp uge i64 %k.tc.guard, 16
; CHECK-NEXT:    br i1 %tensor.profitable, label %gemm.ph, label %gemm.ph.scalar
;
; Tensor path entry.
; CHECK:       gemm.ph:
; CHECK:       i.loop:
;
; Both paths converge at %exit.
; CHECK:       exit:
; CHECK-SAME:    preds = %j.latch.scalar, %j.latch
;
; Dynamic K tiling inside the tensor path.
; CHECK:       tensor.body.header:
; CHECK:       tensor.body.body:
; CHECK:         call void @llvm.tensor.contract.2d.2d.2d.f32(
; CHECK:       scalar.block:
; CHECK:         fmul float
; CHECK:         fadd float

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @gemm_skeleton(ptr %A, ptr %B, ptr %C, i64 %K) {
entry:
  br label %gemm.ph
gemm.ph:
  br label %i.loop
i.loop:
  %i = phi i64 [ 0, %gemm.ph ], [ %i.next, %j.latch ]
  br label %j.loop
j.loop:
  %j = phi i64 [ 0, %i.loop ], [ %j.next, %k.latch ]
  br label %k.loop
k.loop:
  %k   = phi i64   [ 0,   %j.loop ], [ %k.next, %k.loop ]
  %acc = phi float [ 0.0, %j.loop ], [ %sum,    %k.loop ]
  %ai  = mul i64 %i, 256
  %ak  = add i64 %ai, %k
  %aptr = getelementptr float, ptr %A, i64 %ak
  %bk   = mul i64 %k, 16
  %bj   = add i64 %bk, %j
  %bptr = getelementptr float, ptr %B, i64 %bj
  %av   = load float, ptr %aptr
  %bv   = load float, ptr %bptr
  %mul  = fmul float %av, %bv
  %sum  = fadd float %acc, %mul
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, %K
  br i1 %k.done, label %k.latch, label %k.loop
k.latch:
  %ij   = add i64 %i, %j
  %cptr = getelementptr float, ptr %C, i64 %ij
  store float %sum, ptr %cptr
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 16
  br i1 %j.done, label %j.latch, label %j.loop
j.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 16
  br i1 %i.done, label %exit, label %i.loop
exit:
  ret void
}
