; llvm/test/Transforms/LoopTensorize/basic/outer-dim-preheader-gep.ll
; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
;
; 3-loop GEMM (M=N=K=16) where A's row pointer is computed in a separate
; i.body block between the M-loop header and the N-loop. This triggers the
; buildInitial() ordering bug: without the fix, i.body is emitted AFTER
; BuildRegion(j.loop) recurses, so %a.row is seen as ir<> (DimSet={}) in
; the k-loop body. After the fix, i.body is emitted first and %a.row
; gets DimSet={2} (M-dim), propagating to A-load DimSet={0,2}.
;
; CHECK: call void @llvm.tensor.contract.2d.2d.2d.f32(
; CHECK-NOT: call void @llvm.tensor.contract.1d.

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @gemm_preheader_gep(ptr %A, ptr %B, ptr %C) {
entry:
  br label %i.loop

i.loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %j.latch ]
  br label %i.body

i.body:
  ; A's row pointer computed in a SEPARATE block — between M-loop header
  ; and N-loop. This is the pattern that triggers the buildInitial() bug.
  %a.row.off = mul i64 %i, 16
  %a.row = getelementptr float, ptr %A, i64 %a.row.off
  br label %j.loop

j.loop:
  %j = phi i64 [ 0, %i.body ], [ %j.next, %k.latch ]
  br label %k.loop

k.loop:
  %k   = phi i64   [ 0,   %j.loop ], [ %k.next, %k.loop ]
  %acc = phi float [ 0.0, %j.loop ], [ %sum,    %k.loop ]
  ; A[i][k]: uses %a.row from i.body (cross-block outer-loop body reference)
  %a.ptr = getelementptr float, ptr %a.row, i64 %k
  ; B[k][j]: flat 2-D layout, K×N stride
  %bk    = mul i64 %k, 16
  %bj    = add i64 %bk, %j
  %b.ptr = getelementptr float, ptr %B, i64 %bj
  %av    = load float, ptr %a.ptr
  %bv    = load float, ptr %b.ptr
  %mul   = fmul float %av, %bv
  %sum   = fadd float %acc, %mul
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 16
  br i1 %k.done, label %k.latch, label %k.loop

k.latch:
  %ci    = mul i64 %i, 16
  %cj    = add i64 %ci, %j
  %c.ptr = getelementptr float, ptr %C, i64 %cj
  store float %sum, ptr %c.ptr
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
