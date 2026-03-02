; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
;
; A 4-deep loop with indirect (non-affine) indexing must not be transformed.
; The pass should be a no-op: no matrix intrinsics emitted.
;
; CHECK-NOT: @llvm.matrix.multiply
; CHECK-NOT: col_matrix

define void @non_affine(ptr %ptrs, ptr %kernel, ptr %output) {
entry:
  br label %oh.loop
oh.loop:
  %oh = phi i32 [ 0, %entry ], [ %oh.next, %ow.latch ]
  br label %ow.loop
ow.loop:
  %ow = phi i32 [ 0, %oh.loop ], [ %ow.next, %kh.latch ]
  br label %kh.loop
kh.loop:
  %kh = phi i32 [ 0, %ow.loop ], [ %kh.next, %kw.latch ]
  br label %kw.loop
kw.loop:
  %kw = phi i32 [ 0, %kh.loop ], [ %kw.next, %kw.loop ]
  ; Indirect pointer load — non-affine, analyzeLoopNest rejects this nest.
  %pp  = getelementptr ptr, ptr %ptrs, i32 %oh
  %p   = load ptr, ptr %pp
  %val = load float, ptr %p
  %ki  = mul i32 %kh, 3
  %ki2 = add i32 %ki, %kw
  %kp  = getelementptr inbounds float, ptr %kernel, i32 %ki2
  %kv  = load float, ptr %kp
  %oi  = mul i32 %oh, 6
  %oi2 = add i32 %oi, %ow
  %op  = getelementptr inbounds float, ptr %output, i32 %oi2
  %ov  = load float, ptr %op
  %m   = fmul float %val, %kv
  %a   = fadd float %ov, %m
  store float %a, ptr %op
  %kw.next = add i32 %kw, 1
  %kw.cond = icmp slt i32 %kw.next, 3
  br i1 %kw.cond, label %kw.loop, label %kw.latch
kw.latch:
  %kh.next = add i32 %kh, 1
  %kh.cond = icmp slt i32 %kh.next, 3
  br i1 %kh.cond, label %kh.loop, label %kh.latch
kh.latch:
  %ow.next = add i32 %ow, 1
  %ow.cond = icmp slt i32 %ow.next, 6
  br i1 %ow.cond, label %ow.loop, label %ow.latch
ow.latch:
  %oh.next = add i32 %oh, 1
  %oh.cond = icmp slt i32 %oh.next, 6
  br i1 %oh.cond, label %oh.loop, label %exit
exit:
  ret void
}
