; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
;
; Batched GEMM: C[b,i,j] += A[b,i,k] * B[b,k,j].
; A: DimSet={k,i,b} (3D), B: DimSet={k,j,b} (3D).
; OutputDimSet={j,i,b}, RankC=3.  Emits contract.3d.3d.3d.f32.
; A_strides[j]=0 (A∌j), B_strides[i]=0 (B∌i) — stride-0 broadcast convention.
;
; CHECK: call void @llvm.tensor.contract.3d.3d.3d.f32(
; CHECK-SAME: i64 0
; CHECK-SAME: i64 %IK
; CHECK-SAME: i64 %KN

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

; A[b*IK + i*K + k], B[b*KN + k*N + j], C[b*IN + i*N + j]
; IK = I*K (elements per batch slice of A)
; KN = K*N (elements per batch slice of B)
; IN = I*N (elements per batch slice of C)
define void @batched_gemm(ptr %A, ptr %B, ptr %C,
                           i64 %IK, i64 %KN, i64 %IN,
                           i64 %K,  i64 %N) {
entry:
  br label %b.loop

b.loop:
  %b = phi i64 [ 0, %entry ], [ %b.next, %i.latch ]
  br label %i.loop

i.loop:
  %i = phi i64 [ 0, %b.loop ], [ %i.next, %j.latch ]
  br label %j.loop

j.loop:
  %j = phi i64 [ 0, %i.loop ], [ %j.next, %k.latch ]
  br label %k.loop

k.loop:
  %k   = phi i64   [ 0,   %j.loop ], [ %k.next, %k.loop ]
  %acc = phi float [ 0.0, %j.loop ], [ %sum,    %k.loop ]
  ; A[b*IK + i*K + k]
  %ab   = mul i64 %b, %IK
  %ai   = mul i64 %i, %K
  %aik  = add i64 %ab, %ai
  %aijk = add i64 %aik, %k
  %aptr = getelementptr float, ptr %A, i64 %aijk
  %av   = load float, ptr %aptr
  ; B[b*KN + k*N + j]
  %bb   = mul i64 %b, %KN
  %bk   = mul i64 %k, %N
  %bbk  = add i64 %bb, %bk
  %bbkj = add i64 %bbk, %j
  %bptr = getelementptr float, ptr %B, i64 %bbkj
  %bv   = load float, ptr %bptr
  %prod = fmul float %av, %bv
  %sum  = fadd float %acc, %prod
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 16
  br i1 %k.done, label %k.latch, label %k.loop

k.latch:
  ; C[b*IN + i*N + j]
  %cb   = mul i64 %b, %IN
  %ci   = mul i64 %i, %N
  %cbi  = add i64 %cb, %ci
  %cbij = add i64 %cbi, %j
  %cptr = getelementptr float, ptr %C, i64 %cbij
  store float %sum, ptr %cptr
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
