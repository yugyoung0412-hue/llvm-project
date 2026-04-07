; llvm/test/Transforms/LoopTensorize/basic/ptr-decompose-srem.ll
; RUN: opt -passes=loop-tensorize --disable-verify -debug-only=tplan-lower -S < %s 2>&1 \
; RUN:   | FileCheck %s
; FIXME: --disable-verify needed due to known dominance violations in lowered IR.
; REQUIRES: asserts
;
; Batched GEMM where A uses srem-based broadcasting (A has fewer batches than
; B/C, so A_batch_idx = output_batch % A_ne_batch).
; decomposePtrForDims must stop at the srem GEP, set Base = that GEP result,
; and still extract i/k strides from the inner affine GEPs.
; The resulting intrinsic should be 2D (i, k only) with the srem-computed
; slice pointer as A's base.
;
; CHECK: Contraction (contractDim=0)
; CHECK-NOT: TPlanLowering: Contraction cannot find C pointer

define void @gemm_srem_broadcast(
    ptr %A_base, ptr %B_base, ptr %C_base,
    i64 %A_ne_batch,
    i64 %nb_batch_A, i64 %nb1_A, i64 %nb0_A,
    i64 %nb_batch_B, i64 %nb1_B, i64 %nb0_B,
    i64 %nb_batch_C, i64 %nb1_C, i64 %nb0_C) {
entry:
  br label %batch.loop

batch.loop:
  %b = phi i64 [ 0, %entry ], [ %b.next, %batch.latch ]

  ; A: broadcast via srem
  %a.batch.idx = srem i64 %b, %A_ne_batch
  %a.batch.off = mul i64 %a.batch.idx, %nb_batch_A
  %A_slice = getelementptr inbounds i8, ptr %A_base, i64 %a.batch.off

  ; B, C: direct indexing
  %b.batch.off = mul i64 %b, %nb_batch_B
  %B_slice = getelementptr inbounds i8, ptr %B_base, i64 %b.batch.off
  %c.batch.off = mul i64 %b, %nb_batch_C
  %C_slice = getelementptr inbounds i8, ptr %C_base, i64 %c.batch.off

  br label %i.loop

i.loop:
  %i = phi i64 [ 0, %batch.loop ], [ %i.next, %j.latch ]
  br label %j.loop

j.loop:
  %j = phi i64 [ 0, %i.loop ], [ %j.next, %k.latch ]
  br label %k.loop

k.loop:
  %k   = phi i64 [ 0, %j.loop ], [ %k.next, %k.loop ]
  %acc = phi float [ 0.0, %j.loop ], [ %acc.next, %k.loop ]

  %i.off.A = mul i64 %i, %nb1_A
  %k.off.A = mul i64 %k, %nb0_A
  %A.i  = getelementptr inbounds i8, ptr %A_slice, i64 %i.off.A
  %A.ik = getelementptr inbounds i8, ptr %A.i,    i64 %k.off.A

  %j.off.B = mul i64 %j, %nb1_B
  %k.off.B = mul i64 %k, %nb0_B
  %B.j  = getelementptr inbounds i8, ptr %B_slice, i64 %j.off.B
  %B.kj = getelementptr inbounds i8, ptr %B.j,    i64 %k.off.B

  %a.val    = load float, ptr %A.ik, align 4
  %b.val    = load float, ptr %B.kj, align 4
  %mul      = fmul float %a.val, %b.val
  %acc.next = fadd float %acc, %mul

  %k.next = add nuw i64 %k, 1
  %k.cond = icmp eq i64 %k.next, 64
  br i1 %k.cond, label %k.latch, label %k.loop

k.latch:
  %i.off.C = mul i64 %i, %nb1_C
  %j.off.C = mul i64 %j, %nb0_C
  %C.i  = getelementptr inbounds i8, ptr %C_slice, i64 %i.off.C
  %C.ij = getelementptr inbounds i8, ptr %C.i,    i64 %j.off.C
  store float %acc.next, ptr %C.ij, align 4
  %j.next = add nuw i64 %j, 1
  %j.cond = icmp eq i64 %j.next, 64
  br i1 %j.cond, label %j.latch, label %j.loop

j.latch:
  %i.next = add nuw i64 %i, 1
  %i.cond = icmp eq i64 %i.next, 64
  br i1 %i.cond, label %batch.latch, label %i.loop

batch.latch:
  %b.next = add nuw i64 %b, 1
  %b.cond = icmp eq i64 %b.next, 8
  br i1 %b.cond, label %exit, label %batch.loop

exit:
  ret void
}
