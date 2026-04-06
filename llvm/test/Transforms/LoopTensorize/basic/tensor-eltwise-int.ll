; RUN: opt -passes=loop-tensorize --disable-verify -S < %s | FileCheck %s
;
; 2D elementwise integer add: C[i][j] = A[i][j] + B[i][j]
; Strides are currently i64 1, i64 1 (SCEV analysis limitation — pre-existing stride issue).
; CHECK: call void @llvm.tensor.binary.add.2d.2d.2d.i32
; CHECK-SAME: i64 256, i64 256

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @eltwise_add_2d_i32(ptr %A, ptr %B, ptr %C) {
entry:
  br label %outer
outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner
inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner.latch ]
  %ij = add i64 %i, %j
  %aptr = getelementptr i32, ptr %A, i64 %ij
  %bptr = getelementptr i32, ptr %B, i64 %ij
  %cptr = getelementptr i32, ptr %C, i64 %ij
  %av = load i32, ptr %aptr
  %bv = load i32, ptr %bptr
  %cv = add i32 %av, %bv
  store i32 %cv, ptr %cptr
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
