; RUN: opt -passes=loop-tensorize -S --disable-verify < %s | FileCheck %s
; REQUIRES: asserts
;
; Verify that a 2-level element-wise fdiv loop is classified BinaryOp
; and the fdiv instruction is present in the output (no reshape).
; D[i][j] = A[i][j] / B[i][j]
;
; CHECK: fdiv

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

define void @eltwise_div(ptr %A, ptr %B, ptr %D, i64 %M, i64 %N) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner

inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner.latch ]
  %idx = mul i64 %i, %N
  %ij = add i64 %idx, %j
  %aptr = getelementptr float, ptr %A, i64 %ij
  %bptr = getelementptr float, ptr %B, i64 %ij
  %dptr = getelementptr float, ptr %D, i64 %ij
  %av = load float, ptr %aptr, align 4
  %bv = load float, ptr %bptr, align 4
  %dv = fdiv float %av, %bv
  store float %dv, ptr %dptr, align 4
  br label %inner.latch

inner.latch:
  %j.next = add nuw nsw i64 %j, 1
  %j.done = icmp eq i64 %j.next, %N
  br i1 %j.done, label %outer.latch, label %inner

outer.latch:
  %i.next = add nuw nsw i64 %i, 1
  %i.done = icmp eq i64 %i.next, %M
  br i1 %i.done, label %exit, label %outer

exit:
  ret void
}
