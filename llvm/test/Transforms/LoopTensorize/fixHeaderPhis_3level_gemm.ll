; Test: fixHeaderPhis correctness for 3-level nested GEMM loop
;
; Structure (mirrors ggml.ll's innermost 3 loop levels):
;   i-loop (outer,  0..M)
;     j-loop (middle, 0..N)
;       k-loop (inner,  0..K)  -- single-block self-loop
;         sum += A[i*stride_a + k*4] * B[k*stride_b + j*4]
;       C[i*stride_c + j*4] = sum   -- stored when k-loop exits
;
; fixHeaderPhis must connect the backedge of the reduction PHI (%sum) to the
; fadd result from the K-LOOP's latch (k.body itself), NOT from j-latch or i-latch.
;
; Key verification:
;   WIDEN-REDUCTION-PHI ir<%sum> = phi ir<0.0>, ir<%sum.next>
;                                                  ^^^ from k.body latch only
;
; There should be exactly 2 operands. If fixHeaderPhis used the wrong loop's
; latch, the getRecipe() call would fail (instruction not in Ingredient2Recipe)
; causing an assertion, or the operand would point to a wrong value.
;
; NestedOrigLoops = [k_loop, j_loop, i_loop]
; For %sum: phi->getParent() = k.body = k_loop->getHeader()
;           → fixHeaderPhis picks k_loop, latch = k.body (self-loop)
;           → incoming = %sum.next (fadd) ✓
;
; RUN: /Users/yun-yugyeong/Dev/llvm/build/bin/opt -passes=loop-tensorize -S %s 2>&1 | FileCheck %s

; Verifies fixHeaderPhis for 3-level nesting:
;   NestedOrigLoops = [k_loop, j_loop, i_loop]
;   %sum->getParent() = k.body = k_loop->getHeader()
;   → fixHeaderPhis picks k_loop → latch = k.body (self-loop)
;   → PN->getIncomingValueForBlock(k.body) = %sum.next (fadd)
;   → getRecipe(%sum.next) added as operand[1] ✓
;
; If a wrong loop's latch were used (e.g. j-latch or i-latch), the incoming value
; would not be recorded in Ingredient2Recipe, causing an assertion failure.

; The Final TPlan dump shows k.body inside "tensor loop0":
; CHECK:     ir-bb<k.body>:
; CHECK:     WIDEN-REDUCTION-PHI ir<%sum> = phi ir<0.000000e+00>, ir<%sum.next>

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

; void gemm_f32(i8* A, i8* B, i8* C,
;               i64 M, i64 N, i64 K,
;               i64 stride_a, i64 stride_b, i64 stride_c)
define void @fixHeaderPhis_3level_gemm(
    i8* nocapture readonly %A,
    i8* nocapture readonly %B,
    i8* nocapture          %C,
    i64 %M, i64 %N, i64 %K,
    i64 %stride_a, i64 %stride_b, i64 %stride_c) #0 {

entry:
  %has_m = icmp sgt i64 %M, 0
  %has_n = icmp sgt i64 %N, 0
  %has_k = icmp sgt i64 %K, 0
  br i1 %has_m, label %i.preheader, label %exit

; ===== i-loop (outer) =====
i.preheader:
  br label %i.header

i.header:
  %i = phi i64 [ 0, %i.preheader ], [ %i.next, %i.latch ]
  ; Row pointer for A: A + i * stride_a
  %a_row = getelementptr inbounds i8, i8* %A, i64 %i
  ; Row pointer for C: C + i * stride_c
  %c_row = getelementptr inbounds i8, i8* %C, i64 %i
  br i1 %has_n, label %j.preheader, label %i.latch

; ===== j-loop (middle) =====
j.preheader:
  br label %j.header

j.header:
  %j = phi i64 [ 0, %j.preheader ], [ %j.next, %j.latch ]
  ; Column pointer for B: B + j * 4
  %j4    = mul i64 %j, 4
  %b_col = getelementptr inbounds i8, i8* %B, i64 %j4
  br i1 %has_k, label %k.preheader, label %j.latch

; ===== k-loop preheader =====
k.preheader:
  br label %k.body

; ===== k-loop: single-block self-loop (header == latch) =====
; Mirrors ggml.ll block %102: <header><latch><exiting>
; %sum is the reduction PHI.
; fixHeaderPhis must connect %sum.next (fadd) as operand[1] from THIS latch.
k.body:
  %k     = phi i64   [ 0,   %k.preheader ], [ %k.next,    %k.body ]
  %sum   = phi float [ 0.0, %k.preheader ], [ %sum.next,  %k.body ]
  ; Load A[i*stride_a + k*4]
  %k4      = mul i64 %k, 4
  %a_off   = mul i64 %i, %stride_a
  %a_ptr8  = getelementptr inbounds i8, i8* %A, i64 %a_off
  %a_k8    = getelementptr inbounds i8, i8* %a_ptr8, i64 %k4
  %a_ptr   = bitcast i8* %a_k8 to float*
  %a_val   = load float, float* %a_ptr, align 4, !tbaa !0
  ; Load B[k*stride_b + j*4]
  %b_off   = mul i64 %k, %stride_b
  %b_ptr8  = getelementptr inbounds i8, i8* %B, i64 %b_off
  %b_j8    = getelementptr inbounds i8, i8* %b_ptr8, i64 %j4
  %b_ptr   = bitcast i8* %b_j8 to float*
  %b_val   = load float, float* %b_ptr, align 4, !tbaa !0
  ; Accumulate
  %prod     = fmul float %a_val, %b_val
  %sum.next = fadd float %sum, %prod
  %k.next   = add nuw nsw i64 %k, 1
  %k.done   = icmp eq i64 %k.next, %K
  br i1 %k.done, label %j.latch, label %k.body

; ===== j-loop latch: store result, advance j =====
j.latch:
  ; Capture final sum (0.0 if k was skipped, %sum.next if ran)
  %sum.out = phi float [ 0.0, %j.header ], [ %sum.next, %k.body ]
  ; Store to C[i*stride_c + j*4]
  %c_off   = mul i64 %i, %stride_c
  %c_ptr8  = getelementptr inbounds i8, i8* %C, i64 %c_off
  %c_j8    = getelementptr inbounds i8, i8* %c_ptr8, i64 %j4
  %c_ptr   = bitcast i8* %c_j8 to float*
  store float %sum.out, float* %c_ptr, align 4, !tbaa !0
  %j.next  = add nuw nsw i64 %j, 1
  %j.done  = icmp eq i64 %j.next, %N
  br i1 %j.done, label %i.latch, label %j.header

; ===== i-loop latch =====
i.latch:
  %i.next = add nuw nsw i64 %i, 1
  %i.done = icmp eq i64 %i.next, %M
  br i1 %i.done, label %exit, label %i.header

exit:
  ret void
}

attributes #0 = { nofree noinline norecurse nounwind
    "correctly-rounded-divide-sqrt-fp-math"="false"
    "disable-tail-calls"="false"
    "frame-pointer"="non-leaf"
    "less-precise-fpmad"="false"
    "min-legal-vector-width"="0"
    "no-infs-fp-math"="false"
    "no-jump-tables"="false"
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
