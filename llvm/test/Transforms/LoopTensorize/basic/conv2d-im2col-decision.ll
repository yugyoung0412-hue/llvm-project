; RUN: opt -passes=loop-tensorize -debug-only=loop-tensorize -S < %s 2>&1 \
; RUN:   | FileCheck %s
; REQUIRES: asserts
;
; 5-deep nest: N=1, OH=8, OW=8, KH=3, KW=3.
; col_matrix_bytes = 1*8*8*3*3*4 = 2304 bytes << default 256KB L2.
; Expect: Conv2D detected AND use_im2col=1.
;
; CHECK: PatternHint: Conv2D
; CHECK: Conv2D: col_matrix_bytes={{[0-9]+}} L2={{[0-9]+}} use_im2col=1

define void @conv2d_small(ptr %input, ptr %kernel, ptr %output) {
entry:
  br label %n.loop
n.loop:
  %n = phi i32 [ 0, %entry ], [ %n.next, %oh.latch ]
  br label %oh.loop
oh.loop:
  %oh = phi i32 [ 0, %n.loop ], [ %oh.next, %ow.latch ]
  br label %ow.loop
ow.loop:
  %ow = phi i32 [ 0, %oh.loop ], [ %ow.next, %kh.latch ]
  br label %kh.loop
kh.loop:
  %kh = phi i32 [ 0, %ow.loop ], [ %kh.next, %kw.latch ]
  br label %kw.loop
kw.loop:
  %kw = phi i32 [ 0, %kh.loop ], [ %kw.next, %kw.loop ]
  ; input[(oh+kh)*8 + (ow+kw)]
  %ih     = add i32 %oh, %kh
  %row    = mul i32 %ih, 8
  %iw     = add i32 %ow, %kw
  %idx_in = add i32 %row, %iw
  %in.ptr = getelementptr inbounds float, ptr %input, i32 %idx_in
  %in.val = load float, ptr %in.ptr
  ; kernel[kh*3+kw]
  %ki     = mul i32 %kh, 3
  %kidx   = add i32 %ki, %kw
  %k.ptr  = getelementptr inbounds float, ptr %kernel, i32 %kidx
  %k.val  = load float, ptr %k.ptr
  ; output[oh*8+ow]
  %oi     = mul i32 %oh, 8
  %oidx   = add i32 %oi, %ow
  %o.ptr  = getelementptr inbounds float, ptr %output, i32 %oidx
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
  %ow.cond = icmp slt i32 %ow.next, 8
  br i1 %ow.cond, label %ow.loop, label %ow.latch
ow.latch:
  %oh.next = add i32 %oh, 1
  %oh.cond = icmp slt i32 %oh.next, 8
  br i1 %oh.cond, label %oh.loop, label %oh.latch
oh.latch:
  %n.next = add i32 %n, 1
  %n.cond = icmp slt i32 %n.next, 1
  br i1 %n.cond, label %n.loop, label %exit
exit:
  ret void
}
