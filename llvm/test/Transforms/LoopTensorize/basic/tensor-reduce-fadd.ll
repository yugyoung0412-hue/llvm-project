; RUN: opt -passes=loop-tensorize --disable-verify -S < %s | FileCheck %s
;
; 2D plain reduction: scalar sum of all A[i][j]
; No fmul producer — classified as PlainReduction.
; CHECK: call void @llvm.tensor.reduce.fadd.2d.f32

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define float @plain_reduce_2d(ptr %A) {
entry:
  br label %outer
outer:
  %i     = phi i64   [ 0,   %entry ],  [ %i.next,   %outer.latch ]
  %acc.o = phi float [ 0.0, %entry ],  [ %acc.next, %outer.latch ]
  br label %inner
inner:
  %j     = phi i64   [ 0,      %outer ],  [ %j.next,   %inner.latch ]
  %acc   = phi float [ %acc.o, %outer ],  [ %acc.next, %inner.latch ]
  %ij    = add i64 %i, %j
  %aptr  = getelementptr float, ptr %A, i64 %ij
  %av    = load float, ptr %aptr
  %acc.next = fadd float %acc, %av
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
  ret float %acc.next
}
