; llvm/test/Transforms/LoopTensorize/basic/ptr-decompose-bitcast.ll
; RUN: opt -passes=loop-tensorize -debug-only=tplan-lower -S < %s 2>&1 \
; RUN:   | FileCheck %s
; REQUIRES: asserts
;
; A 2D GEMM where the load pointer has a bitcast between the GEP and the load.
; decomposePtrForDims must skip the bitcast and still extract stride %nb0.
;
; CHECK: Contraction (contractDim=0)

define void @gemm_bitcast(ptr %A, ptr %B, ptr %C,
                           i64 %nb0_A, i64 %nb1_A,
                           i64 %nb0_B, i64 %nb1_B,
                           i64 %nb0_C, i64 %nb1_C) {
entry:
  br label %i.loop

i.loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %j.latch ]
  br label %j.loop

j.loop:
  %j = phi i64 [ 0, %i.loop ], [ %j.next, %k.latch ]
  %j.off.B = mul i64 %j, %nb1_B
  %j.off.C = mul i64 %j, %nb1_C
  %B.j = getelementptr inbounds i8, ptr %B, i64 %j.off.B
  %C.j = getelementptr inbounds i8, ptr %C, i64 %j.off.C
  br label %k.loop

k.loop:
  %k = phi i64 [ 0, %j.loop ], [ %k.next, %k.loop ]
  %acc = phi float [ 0.0, %j.loop ], [ %acc.next, %k.loop ]

  ; A[i][k]: GEP then bitcast
  %i.off.A = mul i64 %i, %nb1_A
  %k.off.A = mul i64 %k, %nb0_A
  %A.i = getelementptr inbounds i8, ptr %A, i64 %i.off.A
  %A.ik.i8 = getelementptr inbounds i8, ptr %A.i, i64 %k.off.A
  %A.ik = bitcast ptr %A.ik.i8 to ptr   ; <- bitcast to skip

  ; B[k][j]
  %k.off.B = mul i64 %k, %nb0_B
  %B.kj.i8 = getelementptr inbounds i8, ptr %B.j, i64 %k.off.B
  %B.kj = bitcast ptr %B.kj.i8 to ptr

  %a.val = load float, ptr %A.ik, align 4
  %b.val = load float, ptr %B.kj, align 4
  %mul   = fmul float %a.val, %b.val
  %acc.next = fadd float %acc, %mul

  %k.next = add nuw i64 %k, 1
  %k.cond = icmp eq i64 %k.next, 64
  br i1 %k.cond, label %k.latch, label %k.loop

k.latch:
  ; C[i][j]
  %i.off.C = mul i64 %i, %nb1_C
  %j.off.C2 = mul i64 %j, %nb0_C
  %C.ij.i8 = getelementptr inbounds i8, ptr %C, i64 %i.off.C
  %C.ij = getelementptr inbounds i8, ptr %C.ij.i8, i64 %j.off.C2
  store float %acc.next, ptr %C.ij, align 4
  %j.next = add nuw i64 %j, 1
  %j.cond = icmp eq i64 %j.next, 64
  br i1 %j.cond, label %j.latch, label %j.loop

j.latch:
  %i.next = add nuw i64 %i, 1
  %i.cond = icmp eq i64 %i.next, 64
  br i1 %i.cond, label %exit, label %i.loop

exit:
  ret void
}
