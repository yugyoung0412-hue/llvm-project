; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
;
; GEMM 64x64x64 with default PF=256. All dimensions fit in one tile.
; Expected: single tensor.contract call, no tile.d* blocks.
;
; CHECK-LABEL: @gemm_64(
; CHECK: call void @llvm.tensor.contract.2d.2d.2d.f32(
; CHECK-NOT: tile.d

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @gemm_64(ptr %A, ptr %B, ptr %C) {
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
  %ai   = mul i64 %i, 64
  %ak   = add i64 %ai, %k
  %aptr = getelementptr float, ptr %A, i64 %ak
  %bk   = mul i64 %k, 64
  %bj   = add i64 %bk, %j
  %bptr = getelementptr float, ptr %B, i64 %bj
  %av   = load float, ptr %aptr
  %bv   = load float, ptr %bptr
  %mul  = fmul float %av, %bv
  %sum  = fadd float %acc, %mul
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 64
  br i1 %k.done, label %k.latch, label %k.loop
k.latch:
  %ci   = mul i64 %i, 64
  %cj   = add i64 %ci, %j
  %cptr = getelementptr float, ptr %C, i64 %cj
  store float %sum, ptr %cptr
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 64
  br i1 %j.done, label %j.latch, label %j.loop
j.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 64
  br i1 %i.done, label %exit, label %i.loop
exit:
  ret void
}
