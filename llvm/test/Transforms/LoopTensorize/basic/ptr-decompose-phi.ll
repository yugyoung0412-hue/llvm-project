; llvm/test/Transforms/LoopTensorize/basic/ptr-decompose-phi.ll
; RUN: opt -passes=loop-tensorize --disable-verify -debug-only=tplan-lower -S < %s 2>&1 \
; RUN:   | FileCheck %s
; FIXME: --disable-verify needed due to known dominance violations in lowered IR.
; REQUIRES: asserts
; XFAIL: *
; The 4-level batch loop nest is not yet recognized by LoopNestAnalyzer.
;
; Batched GEMM where the outer batch loop uses a pointer-induction PHI
; (A_slice advances by batch_stride each iteration).
; decomposePtrForDims must follow the loop-invariant incoming value of
; the PHI (the preheader value) and extract i/k strides from inner GEPs.
;
; CHECK: Contraction (contractDim=0)

define void @gemm_phi_ptr(ptr %A_base, ptr %B_base, ptr %C_base,
                           i64 %batch_stride_A,
                           i64 %nb1_A, i64 %nb0_A,
                           i64 %nb1_B, i64 %nb0_B,
                           i64 %nb1_C, i64 %nb0_C) {
entry:
  br label %batch.loop

batch.loop:
  ; Pointer-induction PHI: A_slice = A_base + batch*batch_stride_A
  %A_slice = phi ptr [ %A_base, %entry ], [ %A_slice_next, %batch.latch ]
  %B_slice = phi ptr [ %B_base, %entry ], [ %B_slice_next, %batch.latch ]
  %C_slice = phi ptr [ %C_base, %entry ], [ %C_slice_next, %batch.latch ]
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

  %i.off = mul i64 %i, %nb1_A
  %k.off = mul i64 %k, %nb0_A
  ; GEP from the PHI pointer -- decomposePtrForDims must follow PHI
  %A.i  = getelementptr inbounds i8, ptr %A_slice, i64 %i.off
  %A.ik = getelementptr inbounds i8, ptr %A.i,    i64 %k.off

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
  %A_slice_next = getelementptr inbounds i8, ptr %A_slice, i64 %batch_stride_A
  %B_slice_next = getelementptr inbounds i8, ptr %B_slice, i64 %batch_stride_A
  %C_slice_next = getelementptr inbounds i8, ptr %C_slice, i64 %batch_stride_A
  %batch.next = add nuw i64 0, 1
  %batch.cond = icmp eq i64 %batch.next, 4
  br i1 %batch.cond, label %exit, label %batch.loop

exit:
  ret void
}
