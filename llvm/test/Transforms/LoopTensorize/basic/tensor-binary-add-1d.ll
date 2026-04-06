; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
;
; 1D + 1D element-wise fadd: C[i] = A[i] + B[i].
; A: DimSet={i}, B: DimSet={i}.  OutputDimSet={i}, RankC=1.
; Emits binary.fadd.1d.1d.1d.f32.
;
; CHECK: call void @llvm.tensor.binary.fadd.1d.1d.1d.f32(

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @add_1d(ptr %A, ptr %B, ptr %C) {
entry:
  br label %i.loop
i.loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %i.loop ]
  %aptr = getelementptr float, ptr %A, i64 %i
  %bptr = getelementptr float, ptr %B, i64 %i
  %av   = load float, ptr %aptr
  %bv   = load float, ptr %bptr
  %res  = fadd float %av, %bv
  %cptr = getelementptr float, ptr %C, i64 %i
  store float %res, ptr %cptr
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 16
  br i1 %i.done, label %exit, label %i.loop
exit:
  ret void
}
