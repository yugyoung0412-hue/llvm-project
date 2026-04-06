; RUN: opt -passes=loop-tensorize -S %s | FileCheck %s
;
; Verify that a GEMM with non-dense leading dimensions (%lda, %ldb) produces
; a llvm.tensor.matmul call with runtime stride arguments, not constant
; dense defaults.

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

; CHECK-LABEL: @gemm_strided
; CHECK: call void @llvm.tensor.contract.2d.2d.2d.f32(
; CHECK-SAME: i64 %lda
; CHECK-SAME: i64 %ldb

; CHECK-LABEL: @gemm_dense
; CHECK: call void @llvm.tensor.contract.2d.2d.2d.f32(
; CHECK-SAME: i64 %K
; CHECK-SAME: i64 %N

; Strided GEMM: A[i][k] with leading dim lda, B[k][j] with leading dim ldb.
; PHI-based k-reduction: accumulator initialized to 0.0, stored after k-loop.
define void @gemm_strided(ptr %A, ptr %B, ptr %C,
                           i64 %M, i64 %N, i64 %K,
                           i64 %lda, i64 %ldb) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %middle

middle:
  %j = phi i64 [ 0, %outer ], [ %j.next, %middle.latch ]
  br label %inner

inner:
  %k   = phi i64   [ 0,   %middle ], [ %k.next, %inner.latch ]
  %acc = phi float [ 0.0, %middle ], [ %sum,    %inner.latch ]
  ; A[i*lda + k] — outer stride is %lda (non-dense when lda > K)
  %ai   = mul i64 %i, %lda
  %ak   = add i64 %ai, %k
  %aptr = getelementptr float, ptr %A, i64 %ak
  %av   = load float, ptr %aptr, align 4
  ; B[k*ldb + j] — outer stride is %ldb (non-dense when ldb > N)
  %bk   = mul i64 %k, %ldb
  %bj   = add i64 %bk, %j
  %bptr = getelementptr float, ptr %B, i64 %bj
  %bv   = load float, ptr %bptr, align 4
  %prod = fmul float %av, %bv
  %sum  = fadd float %acc, %prod
  br label %inner.latch

inner.latch:
  %k.next = add nuw nsw i64 %k, 1
  %k.done = icmp eq i64 %k.next, %K
  br i1 %k.done, label %middle.latch, label %inner

middle.latch:
  ; Store accumulated sum to C[i*N + j]
  %ci   = mul i64 %i, %N
  %cj   = add i64 %ci, %j
  %cptr = getelementptr float, ptr %C, i64 %cj
  store float %sum, ptr %cptr, align 4
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

; Dense GEMM: A[i*K+k], B[k*N+j], C[i*N+j]. PHI-based k-reduction.
define void @gemm_dense(ptr %A, ptr %B, ptr %C,
                         i64 %M, i64 %N, i64 %K) {
entry:
  br label %outer
outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %middle
middle:
  %j = phi i64 [ 0, %outer ], [ %j.next, %middle.latch ]
  br label %inner
inner:
  %k   = phi i64   [ 0,   %middle ], [ %k.next, %inner.latch ]
  %acc = phi float [ 0.0, %middle ], [ %sum,    %inner.latch ]
  %ai   = mul i64 %i, %K
  %ak   = add i64 %ai, %k
  %aptr = getelementptr float, ptr %A, i64 %ak
  %av   = load float, ptr %aptr, align 4
  %bk   = mul i64 %k, %N
  %bj   = add i64 %bk, %j
  %bptr = getelementptr float, ptr %B, i64 %bj
  %bv   = load float, ptr %bptr, align 4
  %prod = fmul float %av, %bv
  %sum  = fadd float %acc, %prod
  br label %inner.latch
inner.latch:
  %k.next = add nuw nsw i64 %k, 1
  %k.done = icmp eq i64 %k.next, %K
  br i1 %k.done, label %middle.latch, label %inner
middle.latch:
  %ci   = mul i64 %i, %N
  %cj   = add i64 %ci, %j
  %cptr = getelementptr float, ptr %C, i64 %cj
  store float %sum, ptr %cptr, align 4
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
