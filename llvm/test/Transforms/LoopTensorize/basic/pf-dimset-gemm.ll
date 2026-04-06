; RUN: opt -passes=loop-tensorize -S --disable-verify < %s | FileCheck %s
; REQUIRES: asserts
;
; Verify that a 3-level GEMM loop nest emits @llvm.tensor.contract.2d.2d.2d.f32
; via the PF DimSet system (Contraction classification).
; Uses reduction-PHI form: acc = sum_k(A[i][k] * B[k][j])
; Loop order: i (dim0) -> j (dim1) -> k (dim2, reduction)
;
; CHECK: call void @llvm.tensor.contract.2d.2d.2d.f32
; CHECK-NOT: llvm.matrix.multiply

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

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
  %acc = phi float [ 0.0, %middle ], [ %sum, %inner.latch ]
  %ai = mul i64 %i, %K
  %ak = add i64 %ai, %k
  %aptr = getelementptr float, ptr %A, i64 %ak
  %av = load float, ptr %aptr, align 4
  %bk = mul i64 %k, %N
  %bj = add i64 %bk, %j
  %bptr = getelementptr float, ptr %B, i64 %bj
  %bv = load float, ptr %bptr, align 4
  %prod = fmul float %av, %bv
  %sum = fadd float %acc, %prod
  br label %inner.latch

inner.latch:
  %k.next = add nuw nsw i64 %k, 1
  %k.done = icmp eq i64 %k.next, %K
  br i1 %k.done, label %middle.latch, label %inner

middle.latch:
  %ci = mul i64 %i, %N
  %cij = add i64 %ci, %j
  %cptr = getelementptr float, ptr %C, i64 %cij
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
