; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
;
; Non-affine loop with indirect memory access (pointer-chasing).
; The pass should NOT emit llvm.matrix.multiply.
; CHECK-NOT: @llvm.matrix.multiply

define void @indirect_access(ptr %Ptrs, ptr %Out, i32 %N) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  ; Load a pointer from the pointer array (pointer-chasing, non-affine)
  %ptr.addr = getelementptr ptr, ptr %Ptrs, i32 %i
  %inner.ptr = load ptr, ptr %ptr.addr
  %val = load float, ptr %inner.ptr
  %out.ptr = getelementptr float, ptr %Out, i32 %i
  store float %val, ptr %out.ptr
  %i.next = add i32 %i, 1
  %cond = icmp slt i32 %i.next, %N
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}
