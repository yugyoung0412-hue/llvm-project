; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
;
; 3-nested-loop 4x4x4 GEMM exercising the TPlan path.
; When TPlanBuilder successfully builds a plan and classifyPattern detects GEMM,
; applyPlan() dispatches through TPlanWidener + TPlanLowering which emits
; llvm.matrix.multiply.
;
; CHECK: @llvm.matrix.multiply

define void @tplan_gemm_4x4(ptr %A, ptr %B, ptr %C) {
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
  %c.val = load float, ptr %c.ptr
  %acc   = fadd float %c.val, %mul
  store float %acc, ptr %c.ptr
  %k.next = add i32 %k, 1
  %k.done = icmp slt i32 %k.next, 4
  br i1 %k.done, label %k.loop, label %k.latch

k.latch:
  %j.next = add i32 %j, 1
  %j.done = icmp slt i32 %j.next, 4
  br i1 %j.done, label %j.loop, label %j.latch

j.latch:
  %i.next = add i32 %i, 1
  %i.done = icmp slt i32 %i.next, 4
  br i1 %i.done, label %i.loop, label %exit

exit:
  ret void
}
