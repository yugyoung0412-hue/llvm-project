; RUN: opt -passes=loop-tensorize --disable-verify -debug-only=loop-tensorize %s -o /dev/null 2>&1 | FileCheck %s
; REQUIRES: asserts
;
; Verify that LoopTensorize builds and prints an initial TPlan for a simple
; 3-deep loop nest (GEMM shape: 2 reads + 1 write).

; CHECK: TPlan 'gemm' (depth=3) {
; CHECK: Live-in {{.*}} = PF[0]
; CHECK: Live-in {{.*}} = PF[1]
; CHECK: Live-in {{.*}} = PF[2]
; CHECK: Live-in
; CHECK: loop[0]
; CHECK: CANONICAL-INDUCTION
; CHECK: WIDEN-INDUCTION
; CHECK: loop[1]
; CHECK: CANONICAL-INDUCTION
; CHECK: WIDEN-INDUCTION
; CHECK: loop[2]
; CHECK: CANONICAL-INDUCTION
; CHECK: WIDEN-INDUCTION
; CHECK: WIDEN{{.*}} = fmul
; CHECK: WIDEN store
; CHECK: CANONICAL-INDUCTION-INC
; CHECK: CANONICAL-INDUCTION-CMP

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

; Simple 3-deep GEMM loop: C[i][j] += A[i][k] * B[k][j]
define void @gemm(ptr %A, ptr %B, ptr %C, i64 %M, i64 %N, i64 %K) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %middle

middle:
  %j = phi i64 [ 0, %outer ], [ %j.next, %middle.latch ]
  br label %inner

inner:
  %k = phi i64 [ 0, %middle ], [ %k.next, %inner.latch ]
  %ai = mul i64 %i, %K
  %ak = add i64 %ai, %k
  %aptr = getelementptr float, ptr %A, i64 %ak
  %av = load float, ptr %aptr, align 4
  %bk = mul i64 %k, %N
  %bj = add i64 %bk, %j
  %bptr = getelementptr float, ptr %B, i64 %bj
  %bv = load float, ptr %bptr, align 4
  %prod = fmul float %av, %bv
  %ci = mul i64 %i, %N
  %cj = add i64 %ci, %j
  %cptr = getelementptr float, ptr %C, i64 %cj
  %cv = load float, ptr %cptr, align 4
  %sum = fadd float %cv, %prod
  store float %sum, ptr %cptr, align 4
  br label %inner.latch

inner.latch:
  %k.next = add nuw nsw i64 %k, 1
  %k.done = icmp eq i64 %k.next, %K
  br i1 %k.done, label %middle.latch, label %inner

middle.latch:
  %j.next = add nuw nsw i64 %j, 1
  %j.done = icmp eq i64 %j.next, %N
  br i1 %j.done, label %outer.latch, label %middle

outer.latch:
  %i.next = add nuw nsw i64 %i, 1
  %i.done = icmp eq i64 %i.next, %M
  br i1 %i.done, label %exit, label %outer

exit:
  ret void
}
