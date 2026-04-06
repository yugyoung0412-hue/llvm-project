; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
;
; A constant-trip-count GEMM (16x16x16) using reduction-PHI form.
; Contraction must emit @llvm.tensor.contract.2d.2d.f32.
; CHECK: call void @llvm.tensor.contract.2d.2d.f32
; CHECK-NOT: @llvm.matrix.multiply

define void @gemm_16x16x16(ptr %A, ptr %B, ptr %C, i64 %K, i64 %N) {
entry:
  br label %i.loop

i.loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %j.latch ]
  br label %j.loop

j.loop:
  %j = phi i64 [ 0, %i.loop ], [ %j.next, %k.latch ]
  br label %k.loop

k.loop:
  %k   = phi i64   [ 0,   %j.loop ], [ %k.next, %k.loop ]
  %acc = phi float [ 0.0, %j.loop ], [ %sum,    %k.loop ]
  %ai   = mul i64 %i, %K
  %ak   = add i64 %ai, %k
  %bk   = mul i64 %k, %N
  %bj   = add i64 %bk, %j
  %a.ptr = getelementptr inbounds float, ptr %A, i64 %ak
  %b.ptr = getelementptr inbounds float, ptr %B, i64 %bj
  %a.val = load float, ptr %a.ptr
  %b.val = load float, ptr %b.ptr
  %mul   = fmul float %a.val, %b.val
  %sum   = fadd float %acc, %mul
  %k.next = add i64 %k, 1
  %k.cond = icmp slt i64 %k.next, 16
  br i1 %k.cond, label %k.loop, label %k.latch

k.latch:
  %ci   = mul i64 %i, %N
  %cj   = add i64 %ci, %j
  %c.ptr = getelementptr inbounds float, ptr %C, i64 %cj
  store float %sum, ptr %c.ptr
  %j.next = add i64 %j, 1
  %j.cond = icmp slt i64 %j.next, 16
  br i1 %j.cond, label %j.loop, label %j.latch

j.latch:
  %i.next = add i64 %i, 1
  %i.cond = icmp slt i64 %i.next, 16
  br i1 %i.cond, label %i.loop, label %exit

exit:
  ret void
}
