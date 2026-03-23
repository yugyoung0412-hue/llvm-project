; RUN: opt -passes=loop-tensorize -S --disable-verify < %s | FileCheck %s
; REQUIRES: asserts
;
; Verify that a 1-level reduction (sum += A[i]) is classified PlainReduction
; and the fadd instruction is present in the output.
;
; CHECK: fadd

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

define float @plain_reduction(ptr %A, i64 %N) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop.latch ]
  %acc = phi float [ 0.0, %entry ], [ %acc.next, %loop.latch ]
  %aptr = getelementptr float, ptr %A, i64 %i
  %av = load float, ptr %aptr, align 4
  %acc.next = fadd float %acc, %av
  br label %loop.latch

loop.latch:
  %i.next = add nuw nsw i64 %i, 1
  %i.done = icmp eq i64 %i.next, %N
  br i1 %i.done, label %exit, label %loop

exit:
  ret float %acc.next
}
