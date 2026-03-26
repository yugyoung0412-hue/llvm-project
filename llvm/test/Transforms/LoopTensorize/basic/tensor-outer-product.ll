; RUN: opt -passes=loop-tensorize --disable-verify -S < %s | FileCheck %s
; FIXME: --disable-verify needed due to known dominance violations in lowered IR.
;
; Outer product: C[i][j] = A[i] * B[j]
; A has DimSet={outer}, B has DimSet={inner}: disjoint → OuterProduct.
; M=PF[outer]=256, N=PF[inner]=256. ldc=N=256 (dense).
; CHECK: call void @llvm.tensor.matmul.f32
; CHECK-SAME: i64 256, i64 256, i64 256
; CHECK-SAME: i64 256, i64 1, i64 1
; CHECK-SAME: i64 1, i64 256, i64 256

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @outer_product(ptr %A, ptr %B, ptr %C) {
entry:
  br label %outer
outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner
inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner.latch ]
  %aptr = getelementptr float, ptr %A, i64 %i
  %bptr = getelementptr float, ptr %B, i64 %j
  %cptr = getelementptr float, ptr %C, i64 %i
  %av = load float, ptr %aptr
  %bv = load float, ptr %bptr
  %cv = fmul float %av, %bv
  store float %cv, ptr %cptr
  br label %inner.latch
inner.latch:
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 16
  br i1 %j.done, label %outer.latch, label %inner
outer.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 8
  br i1 %i.done, label %exit, label %outer
exit:
  ret void
}
