; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
;
; 1D vector + 2D matrix broadcast fadd: C[i*N+j] = A[j] + B[i*N+j].
; A: DimSet={j} (1D), B: DimSet={i,j} (2D).
; OutputDimSet={i,j}, RankC=2.  A_strides[i]=0 (broadcast).
; Emits binary.fadd.1d.2d.2d.f32.
;
; CHECK: call void @llvm.tensor.binary.fadd.1d.2d.2d.f32(
; CHECK-SAME: i64 0

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @add_broadcast(ptr %A, ptr %B, ptr %C, i64 %N) {
entry:
  br label %i.loop
i.loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %j.latch ]
  br label %j.loop
j.loop:
  %j = phi i64 [ 0, %i.loop ], [ %j.next, %j.loop ]
  %aptr = getelementptr float, ptr %A, i64 %j
  %bi   = mul i64 %i, %N
  %bij  = add i64 %bi, %j
  %bptr = getelementptr float, ptr %B, i64 %bij
  %av   = load float, ptr %aptr
  %bv   = load float, ptr %bptr
  %res  = fadd float %av, %bv
  %cij  = add i64 %bi, %j
  %cptr = getelementptr float, ptr %C, i64 %cij
  store float %res, ptr %cptr
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
