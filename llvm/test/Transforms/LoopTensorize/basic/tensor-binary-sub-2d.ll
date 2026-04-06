; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
;
; 2D element-wise fsub: C[i*N+j] = A[i*N+j] - B[i*N+j].
; A: DimSet={i,j}, B: DimSet={i,j}.  OutputDimSet={i,j}, RankC=2.
; Emits binary.fsub.2d.2d.2d.f32.
;
; CHECK: call void @llvm.tensor.binary.fsub.2d.2d.2d.f32(

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @sub_2d(ptr %A, ptr %B, ptr %C, i64 %N) {
entry:
  br label %i.loop
i.loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %j.latch ]
  br label %j.loop
j.loop:
  %j = phi i64 [ 0, %i.loop ], [ %j.next, %j.loop ]
  %ij   = mul i64 %i, %N
  %idx  = add i64 %ij, %j
  %aptr = getelementptr float, ptr %A, i64 %idx
  %bptr = getelementptr float, ptr %B, i64 %idx
  %av   = load float, ptr %aptr
  %bv   = load float, ptr %bptr
  %res  = fsub float %av, %bv
  %cptr = getelementptr float, ptr %C, i64 %idx
  store float %res, ptr %cptr
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 16
  br i1 %j.done, label %j.latch, label %j.loop
j.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 16
  br i1 %i.done, label %exit, label %i.loop
exit:
  ret void
}
