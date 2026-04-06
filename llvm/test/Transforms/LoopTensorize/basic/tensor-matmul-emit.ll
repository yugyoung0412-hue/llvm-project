; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
;
; 3-level GEMM (16x16x16, static trip counts) using reduction-PHI form.
; Contraction must emit @llvm.tensor.contract.2d.2d.f32.
;
; CHECK: call void @llvm.tensor.contract.2d.2d.f32(
; CHECK-SAME: i64 0
; CHECK-SAME: i64 256, i64 256

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @gemm_16x16x16(ptr %A, ptr %B, ptr %C) {
entry:
  br label %i.loop
i.loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %j.latch ]
  br label %j.loop
j.loop:
  %j = phi i64 [ 0, %i.loop ], [ %j.next, %k.latch ]
  br label %k.loop
k.loop:
  %k   = phi i64   [ 0,   %j.loop ], [ %k.next, %k.loop ]
  %acc = phi float [ 0.0, %j.loop ], [ %sum,    %k.loop ]
  %ik = add i64 %i, %k
  %kj = add i64 %k, %j
  %aptr = getelementptr float, ptr %A, i64 %ik
  %bptr = getelementptr float, ptr %B, i64 %kj
  %av   = load float, ptr %aptr
  %bv   = load float, ptr %bptr
  %mul  = fmul float %av, %bv
  %sum  = fadd float %acc, %mul
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 16
  br i1 %k.done, label %k.latch, label %k.loop
k.latch:
  %ij = add i64 %i, %j
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
