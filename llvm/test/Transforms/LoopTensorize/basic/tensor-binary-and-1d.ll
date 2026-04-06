; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
;
; 1D integer AND: C[i] = A[i] & B[i].
; A: DimSet={i} (i32), B: DimSet={i} (i32).  OutputDimSet={i}, RankC=1.
; Emits binary.and.1d.1d.1d.i32.
;
; CHECK: call void @llvm.tensor.binary.and.1d.1d.1d.i32(

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @and_1d(ptr %A, ptr %B, ptr %C) {
entry:
  br label %i.loop
i.loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %i.loop ]
  %aptr = getelementptr i32, ptr %A, i64 %i
  %bptr = getelementptr i32, ptr %B, i64 %i
  %av   = load i32, ptr %aptr
  %bv   = load i32, ptr %bptr
  %res  = and i32 %av, %bv
  %cptr = getelementptr i32, ptr %C, i64 %i
  store i32 %res, ptr %cptr
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 16
  br i1 %i.done, label %exit, label %i.loop
exit:
  ret void
}
