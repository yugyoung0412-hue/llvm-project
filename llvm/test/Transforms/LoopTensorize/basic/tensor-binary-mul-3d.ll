; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
;
; 3D element-wise fmul: C[b*IN+i*N+j] = A[b*IN+i*N+j] * B[b*IN+i*N+j].
; A,B,C: DimSet={b,i,j}.  OutputDimSet={b,i,j}, RankC=3.
; Emits binary.fmul.3d.3d.3d.f32.
;
; CHECK: call void @llvm.tensor.binary.fmul.3d.3d.3d.f32(

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @mul_3d(ptr %A, ptr %B, ptr %C, i64 %IN, i64 %N) {
entry:
  br label %b.loop
b.loop:
  %b = phi i64 [ 0, %entry ], [ %b.next, %i.latch ]
  br label %i.loop
i.loop:
  %i = phi i64 [ 0, %b.loop ], [ %i.next, %j.latch ]
  br label %j.loop
j.loop:
  %j = phi i64 [ 0, %i.loop ], [ %j.next, %j.loop ]
  %bi   = mul i64 %b, %IN
  %ii   = mul i64 %i, %N
  %idx  = add i64 %bi, %ii
  %idx2 = add i64 %idx, %j
  %aptr = getelementptr float, ptr %A, i64 %idx2
  %bptr = getelementptr float, ptr %B, i64 %idx2
  %av   = load float, ptr %aptr
  %bv   = load float, ptr %bptr
  %res  = fmul float %av, %bv
  %cptr = getelementptr float, ptr %C, i64 %idx2
  store float %res, ptr %cptr
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 16
  br i1 %j.done, label %j.latch, label %j.loop
j.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 16
  br i1 %i.done, label %i.latch, label %i.loop
i.latch:
  %b.next = add i64 %b, 1
  %b.done = icmp eq i64 %b.next, 16
  br i1 %b.done, label %exit, label %b.loop
exit:
  ret void
}
