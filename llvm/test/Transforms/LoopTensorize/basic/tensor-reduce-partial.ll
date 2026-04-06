; RUN: opt -passes=loop-tensorize --disable-verify -S < %s | FileCheck %s
;
; Partial (row) reduction: row_sum[i] += A[i][j]  (j is reduction dim)
; Acc advances along i-dim (stride != 0) but not j-dim (stride = 0).
; CHECK: call void @llvm.tensor.reduce.fadd.2d.f32
; CHECK-SAME: i64 0

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @partial_reduce_row_sum(ptr %A, ptr %RowSum) {
entry:
  br label %outer
outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner
inner:
  %j   = phi i64   [ 0,   %outer ],   [ %j.next,   %inner.latch ]
  %acc = phi float [ 0.0, %outer ],   [ %acc.next, %inner.latch ]
  %ij  = add i64 %i, %j
  %aptr = getelementptr float, ptr %A, i64 %ij
  %av   = load float, ptr %aptr
  %acc.next = fadd float %acc, %av
  br label %inner.latch
inner.latch:
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 16
  br i1 %j.done, label %outer.latch, label %inner
outer.latch:
  %rptr = getelementptr float, ptr %RowSum, i64 %i
  store float %acc.next, ptr %rptr
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 8
  br i1 %i.done, label %exit, label %outer
exit:
  ret void
}
