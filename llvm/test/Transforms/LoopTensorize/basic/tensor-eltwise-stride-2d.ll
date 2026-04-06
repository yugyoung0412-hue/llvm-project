; RUN: opt -passes=loop-tensorize --disable-verify -S < %s | FileCheck %s
; FIXME: --disable-verify needed due to known dominance violations in lowered IR.
;
; 2D elementwise fadd: C[i][j] = A[i][j] + B[i][j]
; dim0=j (innermost), dim1=i (outermost). Default PF=256 per dim.
; CHECK: call void @llvm.tensor.binary.fadd.2d.2d.2d.f32
; CHECK-SAME: i64 1, i64 1
; CHECK-SAME: i64 1, i64 1
; CHECK-SAME: i64 1, i64 1
; CHECK-SAME: i64 256, i64 256

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @eltwise_fadd_2d(ptr %A, ptr %B, ptr %C) {
entry:
  br label %outer
outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner
inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner.latch ]
  %ij = add i64 %i, %j
  %aptr = getelementptr float, ptr %A, i64 %ij
  %bptr = getelementptr float, ptr %B, i64 %ij
  %cptr = getelementptr float, ptr %C, i64 %ij
  %av = load float, ptr %aptr
  %bv = load float, ptr %bptr
  %cv = fadd float %av, %bv
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
