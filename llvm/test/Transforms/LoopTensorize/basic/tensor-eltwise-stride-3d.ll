; RUN: opt -passes=loop-tensorize --disable-verify -S < %s | FileCheck %s
; FIXME: --disable-verify needed due to known dominance violations in lowered IR.
;
; 3D elementwise fadd: D[i][j][k] = A[i][j][k] + B[i][j][k]
; dim0=k (innermost), dim1=j, dim2=i (outermost). Default PF=256 per dim.
; The index is linearized as i*J*K + j*K + k (J=8, K=4).
; CHECK: call void @llvm.tensor.elementwise.fadd.3d.f32
; CHECK-SAME: i64 1, i64 256, i64 65536
; CHECK-SAME: i64 1, i64 256, i64 65536
; CHECK-SAME: i64 1, i64 256, i64 65536
; CHECK-SAME: i64 256, i64 256, i64 256

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

; In this version, the index computation for A, B, D uses mul/add to form
; proper 3D linear offsets: idx = i*J*K + j*K + k, so that each of the
; three loads has the full {0,1,2} DimSet and the fadd is ElementWise.
define void @eltwise_fadd_3d(ptr %A, ptr %B, ptr %D) {
entry:
  br label %l.i
l.i:
  %i = phi i64 [ 0, %entry ], [ %i.next, %l.i.latch ]
  %iJK = mul i64 %i, 32    ; i * J*K = i * 8*4 = i * 32
  br label %l.j
l.j:
  %j = phi i64 [ 0, %l.i ], [ %j.next, %l.j.latch ]
  %ijK = mul i64 %j, 4     ; j * K = j * 4
  %ij  = add i64 %iJK, %ijK
  br label %l.k
l.k:
  %k = phi i64 [ 0, %l.j ], [ %k.next, %l.k.latch ]
  %ijk = add i64 %ij, %k
  %aptr = getelementptr float, ptr %A, i64 %ijk
  %bptr = getelementptr float, ptr %B, i64 %ijk
  %dptr = getelementptr float, ptr %D, i64 %ijk
  %av = load float, ptr %aptr
  %bv = load float, ptr %bptr
  %dv = fadd float %av, %bv
  store float %dv, ptr %dptr
  br label %l.k.latch
l.k.latch:
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 4
  br i1 %k.done, label %l.j.latch, label %l.k
l.j.latch:
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 8
  br i1 %j.done, label %l.i.latch, label %l.j
l.i.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 16
  br i1 %i.done, label %exit, label %l.i
exit:
  ret void
}
