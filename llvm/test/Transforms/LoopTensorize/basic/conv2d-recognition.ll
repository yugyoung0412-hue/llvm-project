; RUN: opt -passes=loop-tensorize -debug-only=loop-tensorize -S < %s 2>&1 \
; RUN:   | FileCheck %s
; REQUIRES: asserts
;
; Conv2D: output[oh*6+ow] += input[(oh+kh)*8+(ow+kw)] * kernel[kh*3+kw]
; The input pointer SCEV contains two AddRecs with step 8 (oh,kh) and two
; with step 1 (ow,kw).  isConv2D() fires → PatternHint: Conv2D.
;
; CHECK: PatternHint: Conv2D

define void @conv2d(ptr %input, ptr %kernel, ptr %output) {
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
  %ih     = add i32 %oh, %kh
  %row    = mul i32 %ih, 8
  %iw     = add i32 %ow, %kw
  %idx_in = add i32 %row, %iw
  %in.ptr = getelementptr float, ptr %input, i32 %idx_in
  %in.val = load float, ptr %in.ptr
  %ki     = mul i32 %kh, 3
  %kidx   = add i32 %ki, %kw
  %k.ptr  = getelementptr float, ptr %kernel, i32 %kidx
  %k.val  = load float, ptr %k.ptr
  %oi     = mul i32 %oh, 6
  %oidx   = add i32 %oi, %ow
  %o.ptr  = getelementptr float, ptr %output, i32 %oidx
  %o.old  = load float, ptr %o.ptr
  %mul    = fmul float %in.val, %k.val
  %acc    = fadd float %o.old, %mul
  store float %acc, ptr %o.ptr
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
