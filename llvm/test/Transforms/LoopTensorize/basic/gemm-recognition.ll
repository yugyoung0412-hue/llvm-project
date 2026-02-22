; RUN: opt -passes=loop-tensorize -debug-only=loop-tensorize -S < %s 2>&1 \
; RUN:   | FileCheck %s
; REQUIRES: asserts

; A scalar GEMM: C[i][j] += A[i][k] * B[k][j]
; CHECK: PatternHint: GEMM

define void @gemm(ptr %A, ptr %B, ptr %C, i32 %N) {
entry:
  br label %i.loop
i.loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %i.latch ]
  br label %j.loop
j.loop:
  %j = phi i32 [ 0, %i.loop ], [ %j.next, %j.latch ]
  br label %k.loop
k.loop:
  %k = phi i32 [ 0, %j.loop ], [ %k.next, %k.latch ]
  %a.idx = add i32 %i, %k
  %b.idx = add i32 %k, %j
  %c.idx = add i32 %i, %j
  %a.ptr = getelementptr float, ptr %A, i32 %a.idx
  %b.ptr = getelementptr float, ptr %B, i32 %b.idx
  %c.ptr = getelementptr float, ptr %C, i32 %c.idx
  %a.val = load float, ptr %a.ptr
  %b.val = load float, ptr %b.ptr
  %c.val = load float, ptr %c.ptr
  %mul   = fmul float %a.val, %b.val
  %add   = fadd float %c.val, %mul
  store float %add, ptr %c.ptr
  %k.next = add i32 %k, 1
  %k.cond = icmp slt i32 %k.next, %N
  br i1 %k.cond, label %k.loop, label %k.latch
k.latch:
  %j.next = add i32 %j, 1
  %j.cond = icmp slt i32 %j.next, %N
  br i1 %j.cond, label %j.loop, label %j.latch
j.latch:
  %i.next = add i32 %i, 1
  %i.cond = icmp slt i32 %i.next, %N
  br i1 %i.cond, label %i.loop, label %exit
exit:
  ret void
}
