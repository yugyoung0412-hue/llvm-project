; RUN: opt -passes=loop-tensorize -debug-only=loop-tensorize -S \
; RUN:   -mtriple=x86_64-- -mattr=+amx-bf16 < %s 2>&1 | FileCheck %s
; REQUIRES: x86-registered-target, asserts
;
; Verify that AMX-BF16 target reports a GEMM pattern for a scalar loop nest.
; CHECK: PatternHint: GEMM

define void @gemm(ptr %A, ptr %B, ptr %C) {
entry:
  br label %i.loop

i.loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %j.latch ]
  br label %j.loop

j.loop:
  %j = phi i32 [ 0, %i.loop ], [ %j.next, %k.latch ]
  br label %k.loop

k.loop:
  %k = phi i32 [ 0, %j.loop ], [ %k.next, %k.loop ]
  %a.idx = add i32 %i, %k
  %b.idx = add i32 %k, %j
  %c.idx = add i32 %i, %j
  %a.ptr = getelementptr inbounds float, ptr %A, i32 %a.idx
  %b.ptr = getelementptr inbounds float, ptr %B, i32 %b.idx
  %c.ptr = getelementptr inbounds float, ptr %C, i32 %c.idx
  %a.val = load float, ptr %a.ptr
  %b.val = load float, ptr %b.ptr
  %mul   = fmul float %a.val, %b.val
  store float %mul, ptr %c.ptr
  %k.next = add i32 %k, 1
  %k.cond = icmp slt i32 %k.next, 16
  br i1 %k.cond, label %k.loop, label %k.latch

k.latch:
  %j.next = add i32 %j, 1
  %j.cond = icmp slt i32 %j.next, 16
  br i1 %j.cond, label %j.loop, label %j.latch

j.latch:
  %i.next = add i32 %i, 1
  %i.cond = icmp slt i32 %i.next, 16
  br i1 %i.cond, label %i.loop, label %exit

exit:
  ret void
}
