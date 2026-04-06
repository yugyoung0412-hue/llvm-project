; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
;
; Dot product: acc += A[k] * B[k].
; A: DimSet={k} (1D), B: DimSet={k} (1D).
; OutputDimSet={} (empty), RankC=0.  Emits contract.1d.1d.0d.f32.
; Signature: void(ptr C, ptr A, i64 A_stride, ptr B, i64 B_stride, i64 K)
; — no stride arrays since RankC=0.
;
; CHECK: call void @llvm.tensor.contract.1d.1d.0d.f32(
; CHECK-NOT: i64 0, i64 0
; CHECK-SAME: i64 1
; CHECK-SAME: i64 1

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @dot_product(ptr %A, ptr %B, ptr %acc) {
entry:
  br label %k.loop

k.loop:
  %k   = phi i64   [ 0,   %entry  ], [ %k.next, %k.loop ]
  %sum = phi float [ 0.0, %entry  ], [ %res,    %k.loop ]
  %aptr = getelementptr float, ptr %A, i64 %k
  %bptr = getelementptr float, ptr %B, i64 %k
  %av   = load float, ptr %aptr
  %bv   = load float, ptr %bptr
  %prod = fmul float %av, %bv
  %res  = fadd float %sum, %prod
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 16
  br i1 %k.done, label %exit, label %k.loop

exit:
  store float %res, ptr %acc
  ret void
}
