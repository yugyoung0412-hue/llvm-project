; Test: fixHeaderPhis correctness for 2-level nested loop
;
; Structure:
;   outer loop (i: 0..M)
;     inner loop (k: 0..K)  <-- single-block self-loop, same as ggml.ll innermost
;       sum += A[i*K + k]    <-- reduction PHI here
;     C[i] = sum             <-- store after inner loop exits (in outer.latch)
;
; fixHeaderPhis must connect the backedge of the reduction PHI (%sum) to the
; fadd result from the INNER loop's latch, not the outer loop's latch.
;
; Expected in 2>&1 output:
;   WIDEN-REDUCTION-PHI tp<%N> = phi ir<0.0>, ir<%sum.next_instruction>
;   (exactly 2 operands — initial value + backedge from inner latch)
;
; RUN: /Users/yun-yugyeong/Dev/llvm/build/bin/opt -passes=loop-tensorize -S %s 2>&1 | FileCheck %s

; fixHeaderPhis adds the backedge operand to the REDUCTION-PHI recipe AFTER all
; recipes are built. Before fixHeaderPhis: 1 operand (initial value only).
; After fixHeaderPhis: 2 operands (initial + backedge from inner latch).
;
; If fixHeaderPhis used the outer loop's latch instead of the inner loop's latch,
; getRecipe() would fail with an assertion because the outer latch has no
; "recording" entry for %sum.next (which lives in the inner loop body).

; CHECK:     WIDEN-REDUCTION-PHI ir<%sum> = phi ir<0.000000e+00>, ir<%sum.next>

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

define void @fixHeaderPhis_2level(
    i8* nocapture readonly %A_raw,
    i8* nocapture          %C_raw,
    i64 %M, i64 %K,
    i64 %stride_a_bytes) #0 {

entry:
  %has_outer = icmp sgt i64 %M, 0
  %has_inner = icmp sgt i64 %K, 0
  br i1 %has_outer, label %outer.preheader, label %exit

outer.preheader:
  br label %outer.header

; outer loop header: only holds the IV phi
outer.header:
  %i = phi i64 [ 0, %outer.preheader ], [ %i.next, %outer.latch ]
  %row_offset = mul i64 %i, %stride_a_bytes
  %row_ptr = getelementptr i8, i8* %A_raw, i64 %row_offset
  br i1 %has_inner, label %inner.preheader, label %outer.latch

inner.preheader:
  br label %inner.body

; inner loop: single-block self-loop (header == latch), mirrors ggml.ll %102
; The reduction PHI %sum lives here.
; fixHeaderPhis must pick inner.body (the inner latch) for the backedge operand.
inner.body:
  %k        = phi i64   [ 0,   %inner.preheader ], [ %k.next,    %inner.body ]
  %sum      = phi float [ 0.0, %inner.preheader ], [ %sum.next,  %inner.body ]
  %k_bytes  = mul i64 %k, 4
  %a_ptr8   = getelementptr i8, i8* %row_ptr, i64 %k_bytes
  %a_ptr    = bitcast i8* %a_ptr8 to float*
  %a_val    = load float, float* %a_ptr, align 4, !tbaa !0
  %sum.next = fadd float %sum, %a_val
  %k.next   = add nuw nsw i64 %k, 1
  %k.done   = icmp eq i64 %k.next, %K
  br i1 %k.done, label %outer.latch, label %inner.body

; outer loop latch: collects final sum, stores C[i], increments i
outer.latch:
  ; %sum.final comes from: 0.0 if inner was skipped, %sum.next if inner ran
  %sum.final = phi float [ 0.0, %outer.header ], [ %sum.next, %inner.body ]
  %c_offset  = mul i64 %i, 4
  %c_ptr8    = getelementptr i8, i8* %C_raw, i64 %c_offset
  %c_ptr     = bitcast i8* %c_ptr8 to float*
  store float %sum.final, float* %c_ptr, align 4, !tbaa !0
  %i.next    = add nuw nsw i64 %i, 1
  %i.done    = icmp eq i64 %i.next, %M
  br i1 %i.done, label %exit, label %outer.header

exit:
  ret void
}

attributes #0 = { nofree noinline norecurse nounwind
    "correctly-rounded-divide-sqrt-fp-math"="false"
    "disable-tail-calls"="false"
    "less-precise-fpmad"="false"
    "no-infs-fp-math"="false"
    "no-nans-fp-math"="false"
    "no-signed-zeros-fp-math"="false"
    "no-trapping-math"="false"
    "stack-protector-buffer-size"="8"
    "target-cpu"="generic"
    "target-features"="+neon"
    "unsafe-fp-math"="false"
    "use-soft-float"="false"
}

!0 = !{!1, !1, i64 0}
!1 = !{!"float", !2, i64 0}
!2 = !{!"tbaa root"}
