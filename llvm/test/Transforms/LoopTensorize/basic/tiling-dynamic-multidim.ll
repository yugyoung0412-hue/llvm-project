; RUN: opt -passes=loop-tensorize -loop-tensorize-pf=8 -S < %s | FileCheck %s
;
; GEMM with DYNAMIC M (runtime argument), static N=8, and STATIC K=32.
; M is an output dim with a dynamic trip count.
;
; Before the Problem-B fix, checkDim() would silently skip M (treating it as
; "not supported") and cause an assertion failure or miss tiling entirely.
; After the fix, M is routed to the static tiling path (emitTilingLoop with
; umin) so the tensor.contract receives min(PF=8, remaining-M) for that dim.
;
; CHECK-LABEL: @gemm_dynamic_m(
;
; Dynamic-M tiling: a tile.d2 loop with umin-bounded tile size.
; CHECK:       tile.d2.header:
; CHECK:       tile.d2.body:
; CHECK:         %tile.d2.actual = call i64 @llvm.umin.i64(i64 8
; CHECK:       tile.d0.header:
; CHECK:         call void @llvm.tensor.contract.2d.2d.2d.f32(
;
; No tensor.body / scalar.block (K is static, no dynamic-K tiling here).
; CHECK-NOT:   tensor.body.header:
; CHECK-NOT:   scalar.block:

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @gemm_dynamic_m(ptr %A, ptr %B, ptr %C, i64 %M) {
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
  %ai  = mul i64 %i, 32
  %ak  = add i64 %ai, %k
  %aptr = getelementptr float, ptr %A, i64 %ak
  %bk   = mul i64 %k, 8
  %bj   = add i64 %bk, %j
  %bptr = getelementptr float, ptr %B, i64 %bj
  %av   = load float, ptr %aptr
  %bv   = load float, ptr %bptr
  %mul  = fmul float %av, %bv
  %sum  = fadd float %acc, %mul
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 32
  br i1 %k.done, label %k.latch, label %k.loop
k.latch:
  %ij   = add i64 %i, %j
  %cptr = getelementptr float, ptr %C, i64 %ij
  store float %sum, ptr %cptr
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 8
  br i1 %j.done, label %j.latch, label %j.loop
j.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, %M
  br i1 %i.done, label %exit, label %i.loop
exit:
  ret void
}
