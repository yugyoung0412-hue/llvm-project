; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
;
; GEMV: y[j] += A[k*N+j] * x[k].  x is 1D (DimSet={k}), A is 2D (DimSet={k,j}).
; OutputDimSet={j}, RankC=1.  Emits contract.1d.2d.1d.f32.
;
; CHECK: call void @llvm.tensor.contract.1d.2d.1d.f32(
; CHECK-SAME: i64 0
; CHECK-SAME: i64 %N

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @gemv(ptr %x, ptr %A, ptr %y, i64 %N) {
entry:
  br label %j.loop

j.loop:
  %j = phi i64 [ 0, %entry ], [ %j.next, %k.latch ]
  br label %k.loop

k.loop:
  %k   = phi i64   [ 0,   %j.loop ], [ %k.next, %k.loop ]
  %acc = phi float [ 0.0, %j.loop ], [ %sum,    %k.loop ]
  ; x[k] — 1D vector
  %xptr = getelementptr float, ptr %x, i64 %k
  %xv   = load float, ptr %xptr
  ; A[k*N + j] — 2D matrix, row-major
  %ak   = mul i64 %k, %N
  %aj   = add i64 %ak, %j
  %aptr = getelementptr float, ptr %A, i64 %aj
  %av   = load float, ptr %aptr
  %prod = fmul float %xv, %av
  %sum  = fadd float %acc, %prod
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 16
  br i1 %k.done, label %k.latch, label %k.loop

k.latch:
  %yptr = getelementptr float, ptr %y, i64 %j
  store float %sum, ptr %yptr
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 16
  br i1 %j.done, label %exit, label %j.loop

exit:
  ret void
}
