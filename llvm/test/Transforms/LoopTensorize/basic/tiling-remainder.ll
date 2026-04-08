; RUN: opt -passes=loop-tensorize -loop-tensorize-pf=8 -S < %s | FileCheck %s
;
; GEMM 17x17x17 with PF=8. 17 is not divisible by 8 (2 full tiles + 1 rem).
; Expected: umin(8, 17-IV) produces correct remainder tile of size 1.
;
; CHECK-LABEL: @gemm_17_remainder(
; CHECK: tile.d{{[0-9]+}}.body:
; CHECK: %{{.*}} = call i64 @llvm.umin.i64(i64 8,
; CHECK: call void @llvm.tensor.contract.2d.2d.2d.f32(
; CHECK: tile.d{{[0-9]+}}.exit:

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @gemm_17_remainder(ptr %A, ptr %B, ptr %C) {
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
  %ai   = mul i64 %i, 17
  %ak   = add i64 %ai, %k
  %aptr = getelementptr float, ptr %A, i64 %ak
  %bk   = mul i64 %k, 17
  %bj   = add i64 %bk, %j
  %bptr = getelementptr float, ptr %B, i64 %bj
  %av   = load float, ptr %aptr
  %bv   = load float, ptr %bptr
  %mul  = fmul float %av, %bv
  %sum  = fadd float %acc, %mul
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 17
  br i1 %k.done, label %k.latch, label %k.loop
k.latch:
  %ci   = mul i64 %i, 17
  %cj   = add i64 %ci, %j
  %cptr = getelementptr float, ptr %C, i64 %cj
  store float %sum, ptr %cptr
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 17
  br i1 %j.done, label %j.latch, label %j.loop
j.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 17
  br i1 %i.done, label %exit, label %i.loop
exit:
  ret void
}
