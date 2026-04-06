//===- IntrinsicsTPlan.td - TPlan tensor intrinsics --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Declares the tensor intrinsic families emitted by TPlanLowering:
///
///   llvm.tensor.contract.<Ra>d.<Rb>d.<Rc>d.<type>
///   llvm.tensor.binary.<op>.<Ra>d.<Rb>d.<Rc>d.<type>
///
/// Ra, Rb in {1,2,3,4}   Rc in {0,1,2,3,4} (contract) or {1,2,3,4} (binary)
/// type  in {f16, f32, f64, i8, i16, i32, i64}
///
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// llvm.tensor.contract.<Ra>d.<Rb>d.<Rc>d.<type>
//
// void(ptr C, i64xRc C_strides,
//      ptr A, i64xRc A_strides, i64 A_contract_stride,
//      ptr B, i64xRc B_strides, i64 B_contract_stride,
//      i64 K, i64xRc output_dims)
//
// stride=0 means the operand does not span that output dim (broadcast).
// Rc=0 is the dot-product case -- no stride arrays, 6 total parameters.
//===----------------------------------------------------------------------===//

foreach Ra = [1, 2, 3, 4] in {
  foreach Rb = [1, 2, 3, 4] in {
    foreach Rc = [0, 1, 2, 3, 4] in {
      foreach type = ["f16", "f32", "f64", "i8", "i16", "i32", "i64"] in {
        def int_tensor_contract_ # Ra # d_ # Rb # d_ # Rc # d_ # type : Intrinsic<[],
          !listconcat([llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                      [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc), [llvm_i64_ty],
                      [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc), [llvm_i64_ty],
                      [llvm_i64_ty], !listsplat(llvm_i64_ty, Rc)),
          [IntrArgMemOnly, IntrWillReturn]>;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// llvm.tensor.binary.<op>.<Ra>d.<Rb>d.<Rc>d.<type>
//
// void(ptr C, i64xRc C_strides,
//      ptr A, i64xRc A_strides,   ; 0 if A does not span that dim (broadcast)
//      ptr B, i64xRc B_strides,   ; 0 if B does not span that dim (broadcast)
//      i64xRc output_dims)
//
// Rc = |(A.DimSet union B.DimSet)| -- no contraction dim removed.
// Ops: fadd fsub fmul fdiv frem add sub mul sdiv udiv and or xor shl lshr ashr
//===----------------------------------------------------------------------===//

foreach Op = ["fadd", "fsub", "fmul", "fdiv", "frem",
              "add",  "sub",  "mul",  "sdiv", "udiv",
              "and",  "or",   "xor",  "shl",  "lshr", "ashr"] in {
  foreach Ra = [1, 2, 3, 4] in {
    foreach Rb = [1, 2, 3, 4] in {
      foreach Rc = [1, 2, 3, 4] in {
        foreach type = ["f16", "f32", "f64", "i8", "i16", "i32", "i64"] in {
          def int_tensor_binary_ # Op # _ # Ra # d_ # Rb # d_ # Rc # d_ # type : Intrinsic<[],
            !listconcat([llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                        [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                        [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                        !listsplat(llvm_i64_ty, Rc)),
            [IntrArgMemOnly, IntrWillReturn]>;
        }
      }
    }
  }
}
