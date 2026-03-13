; ModuleID = 'do_conv2d.ll'
source_filename = "do_conv2d.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @_Z9do_conv2diiiiiiPKfS0_lllPf(i32 noundef %s0, i32 noundef %s1, i32 noundef %p0, i32 noundef %p1, i32 noundef %d0, i32 noundef %d1, ptr noundef readonly captures(none) %X, ptr noundef readonly captures(none) %W, i64 noundef %OC, i64 noundef %OH, i64 noundef %OW, ptr noundef writeonly captures(none) %o_data) local_unnamed_addr #0 {
entry:
  %cmp2156 = icmp sgt i64 %OC, 0
  %cmp6154 = icmp sgt i64 %OH, 0
  %cmp10152 = icmp sgt i64 %OW, 0
  %conv = sext i32 %s0 to i64
  %conv25 = sext i32 %p0 to i64
  %conv26 = sext i32 %d0 to i64
  %conv28 = sext i32 %s1 to i64
  %conv30 = sext i32 %p1 to i64
  %conv32 = sext i32 %d1 to i64
  %0 = sext i32 %p0 to i34
  %1 = mul i34 %0, -12
  %2 = zext i34 %1 to i64
  %3 = sext i32 %s0 to i34
  %4 = mul i34 %3, 12
  %5 = zext i34 %4 to i64
  %6 = sext i32 %d0 to i34
  %7 = mul i34 %6, 12
  %8 = zext i34 %7 to i64
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond.cleanup3, %entry
  %n.0158 = phi i64 [ 0, %entry ], [ %inc100, %for.cond.cleanup3 ]
  br i1 %cmp2156, label %for.cond5.preheader.lr.ph, label %for.cond.cleanup3

for.cond5.preheader.lr.ph:                        ; preds = %for.cond1.preheader
  %.idx139 = shl nuw nsw i64 %n.0158, 14
  %invariant.gep149 = getelementptr inbounds nuw i8, ptr %X, i64 %.idx139
  %mul82 = mul i64 %n.0158, %OC
  br label %for.cond5.preheader

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3
  ret void

for.cond5.preheader:                              ; preds = %for.cond.cleanup7, %for.cond5.preheader.lr.ph
  %oc.0157 = phi i64 [ 0, %for.cond5.preheader.lr.ph ], [ %inc96, %for.cond.cleanup7 ]
  %9 = mul i64 %oc.0157, 576
  %10 = add i64 %2, %9
  br i1 %cmp6154, label %for.cond9.preheader.lr.ph, label %for.cond.cleanup7

for.cond9.preheader.lr.ph:                        ; preds = %for.cond5.preheader
  %mul48 = mul nuw nsw i64 %oc.0157, 144
  %reass.add141 = add i64 %oc.0157, %mul82
  %reass.mul142 = mul i64 %reass.add141, %OH
  br label %for.cond9.preheader

for.cond.cleanup3.loopexit:                       ; preds = %for.cond.cleanup7
  br label %for.cond.cleanup3

for.cond.cleanup3:                                ; preds = %for.cond.cleanup3.loopexit, %for.cond1.preheader
  %inc100 = add nuw nsw i64 %n.0158, 1
  %exitcond164.not = icmp eq i64 %inc100, 16
  br i1 %exitcond164.not, label %for.cond.cleanup, label %for.cond1.preheader, !llvm.loop !9

for.cond9.preheader:                              ; preds = %for.cond.cleanup11, %for.cond9.preheader.lr.ph
  %oh.0155 = phi i64 [ 0, %for.cond9.preheader.lr.ph ], [ %inc92, %for.cond.cleanup11 ]
  %11 = mul i64 %5, %oh.0155
  %12 = add i64 %10, %11
  br i1 %cmp10152, label %for.cond13.preheader.lr.ph, label %for.cond.cleanup11

for.cond13.preheader.lr.ph:                       ; preds = %for.cond9.preheader
  %mul = mul nsw i64 %oh.0155, %conv
  %sub = sub nsw i64 %mul, %conv25
  %reass.add140 = add i64 %reass.mul142, %oh.0155
  %reass.mul = mul i64 %reass.add140, %OW
  br label %for.cond13.preheader

for.cond.cleanup7.loopexit:                       ; preds = %for.cond.cleanup11
  br label %for.cond.cleanup7

for.cond.cleanup7:                                ; preds = %for.cond.cleanup7.loopexit, %for.cond5.preheader
  %inc96 = add nuw nsw i64 %oc.0157, 1
  %exitcond163.not = icmp eq i64 %inc96, %OC
  br i1 %exitcond163.not, label %for.cond.cleanup3.loopexit, label %for.cond5.preheader, !llvm.loop !12

for.cond13.preheader:                             ; preds = %for.cond.cleanup15, %for.cond13.preheader.lr.ph
  %ow.0153 = phi i64 [ 0, %for.cond13.preheader.lr.ph ], [ %inc88, %for.cond.cleanup15 ]
  %mul29 = mul nsw i64 %ow.0153, %conv28
  %sub31 = sub nsw i64 %mul29, %conv30
  br label %for.cond17.preheader

for.cond.cleanup11.loopexit:                      ; preds = %for.cond.cleanup15
  br label %for.cond.cleanup11

for.cond.cleanup11:                               ; preds = %for.cond.cleanup11.loopexit, %for.cond9.preheader
  %inc92 = add nuw nsw i64 %oh.0155, 1
  %exitcond162.not = icmp eq i64 %inc92, %OH
  br i1 %exitcond162.not, label %for.cond.cleanup7.loopexit, label %for.cond9.preheader, !llvm.loop !13

for.cond17.preheader:                             ; preds = %for.cond.cleanup19, %for.cond13.preheader
  %ic.0151 = phi i64 [ 0, %for.cond13.preheader ], [ %inc72, %for.cond.cleanup19 ]
  %sum.0150 = phi float [ 0.000000e+00, %for.cond13.preheader ], [ %sum.3.lcssa.lcssa, %for.cond.cleanup19 ]
  %13 = mul nuw nsw i64 %ic.0151, 36
  %14 = add i64 %12, %13
  %mul44 = mul nuw nsw i64 %ic.0151, 9
  %add42 = add nuw nsw i64 %mul44, %mul48
  %.idx138 = shl nuw nsw i64 %ic.0151, 10
  %gep = getelementptr inbounds nuw i8, ptr %invariant.gep149, i64 %.idx138
  br label %for.cond21.preheader

for.cond.cleanup15:                               ; preds = %for.cond.cleanup19
  %sum.3.lcssa.lcssa.lcssa = phi float [ %sum.3.lcssa.lcssa, %for.cond.cleanup19 ]
  %add83 = add i64 %ow.0153, %reass.mul
  %sext = shl i64 %add83, 32
  %15 = ashr exact i64 %sext, 30
  %arrayidx86 = getelementptr inbounds i8, ptr %o_data, i64 %15
  store float %sum.3.lcssa.lcssa.lcssa, ptr %arrayidx86, align 4, !tbaa !14
  %inc88 = add nuw nsw i64 %ow.0153, 1
  %exitcond161.not = icmp eq i64 %inc88, %OW
  br i1 %exitcond161.not, label %for.cond.cleanup11.loopexit, label %for.cond13.preheader, !llvm.loop !16

for.cond21.preheader:                             ; preds = %for.cond.cleanup23, %for.cond17.preheader
  %kh.0148 = phi i64 [ 0, %for.cond17.preheader ], [ %inc68, %for.cond.cleanup23 ]
  %sum.1147 = phi float [ %sum.0150, %for.cond17.preheader ], [ %sum.3.lcssa, %for.cond.cleanup23 ]
  %16 = mul i64 %8, %kh.0148
  %17 = add i64 %14, %16
  %18 = trunc i64 %17 to i34
  %mul27 = mul nsw i64 %kh.0148, %conv26
  %add = add nsw i64 %mul27, %sub
  %19 = icmp ugt i64 %add, 15
  %mul41 = mul nuw nsw i64 %add, 3
  %add45 = add nuw nsw i64 %add42, %mul41
  %.idx = shl nuw nsw i64 %add, 6
  %gep146 = getelementptr inbounds nuw i8, ptr %gep, i64 %.idx
  br label %vector.scevcheck

vector.scevcheck:                                 ; preds = %for.cond21.preheader
  %mul1 = call { i34, i1 } @llvm.umul.with.overflow.i34(i34 4, i34 1023)
  %mul.result = extractvalue { i34, i1 } %mul1, 0
  %mul.overflow = extractvalue { i34, i1 } %mul1, 1
  %20 = add i34 %18, %mul.result
  %21 = icmp slt i34 %20, %18
  %22 = or i1 %21, %mul.overflow
  %ident.check = icmp ne i32 %d1, 1
  %23 = or i1 %22, %ident.check
  br i1 %23, label %scalar.ph, label %vector.ph

vector.ph:                                        ; preds = %vector.scevcheck
  %24 = insertelement <4 x float> splat (float -0.000000e+00), float %sum.1147, i32 0
  %broadcast.splatinsert = insertelement <4 x i1> poison, i1 %19, i64 0
  %broadcast.splat = shufflevector <4 x i1> %broadcast.splatinsert, <4 x i1> poison, <4 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %pred.load.continue7, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %pred.load.continue7 ]
  %vec.phi = phi <4 x float> [ %24, %vector.ph ], [ %predphi, %pred.load.continue7 ]
  %25 = add i64 %index, 0
  %26 = add i64 %index, 1
  %27 = add i64 %index, 2
  %28 = add i64 %index, 3
  %29 = add nsw i64 %25, %sub31
  %30 = add nsw i64 %26, %sub31
  %31 = add nsw i64 %27, %sub31
  %32 = add nsw i64 %28, %sub31
  %33 = insertelement <4 x i64> poison, i64 %29, i32 0
  %34 = insertelement <4 x i64> %33, i64 %30, i32 1
  %35 = insertelement <4 x i64> %34, i64 %31, i32 2
  %36 = insertelement <4 x i64> %35, i64 %32, i32 3
  %37 = icmp ugt <4 x i64> %36, splat (i64 15)
  %38 = select <4 x i1> %37, <4 x i1> splat (i1 true), <4 x i1> %broadcast.splat
  %39 = xor <4 x i1> %38, splat (i1 true)
  %40 = extractelement <4 x i1> %39, i32 0
  br i1 %40, label %pred.load.if, label %pred.load.continue

pred.load.if:                                     ; preds = %vector.body
  %41 = add nuw nsw i64 %add45, %25
  %42 = shl i64 %41, 32
  %43 = ashr exact i64 %42, 30
  %44 = getelementptr inbounds i8, ptr %W, i64 %43
  %45 = load float, ptr %44, align 4, !tbaa !14
  %46 = insertelement <4 x float> poison, float %45, i32 0
  %47 = getelementptr inbounds nuw float, ptr %gep146, i64 %29
  %48 = load float, ptr %47, align 4, !tbaa !14
  %49 = insertelement <4 x float> poison, float %48, i32 0
  br label %pred.load.continue

pred.load.continue:                               ; preds = %pred.load.if, %vector.body
  %50 = phi <4 x float> [ poison, %vector.body ], [ %46, %pred.load.if ]
  %51 = phi <4 x float> [ poison, %vector.body ], [ %49, %pred.load.if ]
  %52 = extractelement <4 x i1> %39, i32 1
  br i1 %52, label %pred.load.if2, label %pred.load.continue3

pred.load.if2:                                    ; preds = %pred.load.continue
  %53 = add nuw nsw i64 %add45, %26
  %54 = shl i64 %53, 32
  %55 = ashr exact i64 %54, 30
  %56 = getelementptr inbounds i8, ptr %W, i64 %55
  %57 = load float, ptr %56, align 4, !tbaa !14
  %58 = insertelement <4 x float> %50, float %57, i32 1
  %59 = getelementptr inbounds nuw float, ptr %gep146, i64 %30
  %60 = load float, ptr %59, align 4, !tbaa !14
  %61 = insertelement <4 x float> %51, float %60, i32 1
  br label %pred.load.continue3

pred.load.continue3:                              ; preds = %pred.load.if2, %pred.load.continue
  %62 = phi <4 x float> [ %50, %pred.load.continue ], [ %58, %pred.load.if2 ]
  %63 = phi <4 x float> [ %51, %pred.load.continue ], [ %61, %pred.load.if2 ]
  %64 = extractelement <4 x i1> %39, i32 2
  br i1 %64, label %pred.load.if4, label %pred.load.continue5

pred.load.if4:                                    ; preds = %pred.load.continue3
  %65 = add nuw nsw i64 %add45, %27
  %66 = shl i64 %65, 32
  %67 = ashr exact i64 %66, 30
  %68 = getelementptr inbounds i8, ptr %W, i64 %67
  %69 = load float, ptr %68, align 4, !tbaa !14
  %70 = insertelement <4 x float> %62, float %69, i32 2
  %71 = getelementptr inbounds nuw float, ptr %gep146, i64 %31
  %72 = load float, ptr %71, align 4, !tbaa !14
  %73 = insertelement <4 x float> %63, float %72, i32 2
  br label %pred.load.continue5

pred.load.continue5:                              ; preds = %pred.load.if4, %pred.load.continue3
  %74 = phi <4 x float> [ %62, %pred.load.continue3 ], [ %70, %pred.load.if4 ]
  %75 = phi <4 x float> [ %63, %pred.load.continue3 ], [ %73, %pred.load.if4 ]
  %76 = extractelement <4 x i1> %39, i32 3
  br i1 %76, label %pred.load.if6, label %pred.load.continue7

pred.load.if6:                                    ; preds = %pred.load.continue5
  %77 = add nuw nsw i64 %add45, %28
  %78 = shl i64 %77, 32
  %79 = ashr exact i64 %78, 30
  %80 = getelementptr inbounds i8, ptr %W, i64 %79
  %81 = load float, ptr %80, align 4, !tbaa !14
  %82 = insertelement <4 x float> %74, float %81, i32 3
  %83 = getelementptr inbounds nuw float, ptr %gep146, i64 %32
  %84 = load float, ptr %83, align 4, !tbaa !14
  %85 = insertelement <4 x float> %75, float %84, i32 3
  br label %pred.load.continue7

pred.load.continue7:                              ; preds = %pred.load.if6, %pred.load.continue5
  %86 = phi <4 x float> [ %74, %pred.load.continue5 ], [ %82, %pred.load.if6 ]
  %87 = phi <4 x float> [ %75, %pred.load.continue5 ], [ %85, %pred.load.if6 ]
  %88 = fmul <4 x float> %86, %87
  %89 = fadd <4 x float> %vec.phi, %88
  %predphi = select <4 x i1> %38, <4 x float> %vec.phi, <4 x float> %89
  %index.next = add nuw i64 %index, 4
  %90 = icmp eq i64 %index.next, 1024
  br i1 %90, label %middle.block, label %vector.body, !llvm.loop !17

middle.block:                                     ; preds = %pred.load.continue7
  %91 = call float @llvm.vector.reduce.fadd.v4f32(float -0.000000e+00, <4 x float> %predphi)
  br label %for.cond.cleanup23

scalar.ph:                                        ; preds = %vector.scevcheck
  br label %for.body24

for.cond.cleanup19:                               ; preds = %for.cond.cleanup23
  %sum.3.lcssa.lcssa = phi float [ %sum.3.lcssa, %for.cond.cleanup23 ]
  %inc72 = add nuw nsw i64 %ic.0151, 1
  %exitcond160.not = icmp eq i64 %inc72, 16
  br i1 %exitcond160.not, label %for.cond.cleanup15, label %for.cond17.preheader, !llvm.loop !20

for.cond.cleanup23:                               ; preds = %middle.block, %cleanup
  %sum.3.lcssa = phi float [ %sum.3, %cleanup ], [ %91, %middle.block ]
  %inc68 = add nuw nsw i64 %kh.0148, 1
  %exitcond159.not = icmp eq i64 %inc68, 3
  br i1 %exitcond159.not, label %for.cond.cleanup19, label %for.cond21.preheader, !llvm.loop !21

for.body24:                                       ; preds = %scalar.ph, %cleanup
  %kw.0144 = phi i64 [ 0, %scalar.ph ], [ %inc, %cleanup ]
  %sum.2143 = phi float [ %sum.1147, %scalar.ph ], [ %sum.3, %cleanup ]
  %mul33 = mul nsw i64 %kw.0144, %conv32
  %add34 = add nsw i64 %mul33, %sub31
  %or.cond = icmp ugt i64 %add34, 15
  %or.cond104 = select i1 %or.cond, i1 true, i1 %19
  br i1 %or.cond104, label %cleanup, label %if.end

if.end:                                           ; preds = %for.body24
  %add49 = add nuw nsw i64 %add45, %kw.0144
  %sext137 = shl i64 %add49, 32
  %92 = ashr exact i64 %sext137, 30
  %arrayidx = getelementptr inbounds i8, ptr %W, i64 %92
  %93 = load float, ptr %arrayidx, align 4, !tbaa !14
  %arrayidx62 = getelementptr inbounds nuw float, ptr %gep146, i64 %add34
  %94 = load float, ptr %arrayidx62, align 4, !tbaa !14
  %mul63 = fmul float %93, %94
  %add64 = fadd float %sum.2143, %mul63
  br label %cleanup

cleanup:                                          ; preds = %if.end, %for.body24
  %sum.3 = phi float [ %add64, %if.end ], [ %sum.2143, %for.body24 ]
  %inc = add nuw nsw i64 %kw.0144, 1
  %exitcond.not = icmp eq i64 %inc, 1024
  br i1 %exitcond.not, label %for.cond.cleanup23, label %for.body24, !llvm.loop !22
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare { i34, i1 } @llvm.umul.with.overflow.i34(i34, i34) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.vector.reduce.fadd.v4f32(float, <4 x float>) #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}
!llvm.errno.tbaa = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"clang version 23.0.0git (https://github.samsungds.net/yg0412-yun/llvm.git 2466d2683075f2d8e80477192d1f3be191ef3c85)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = distinct !{!9, !10, !11}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!"llvm.loop.unroll.disable"}
!12 = distinct !{!12, !10, !11}
!13 = distinct !{!13, !10, !11}
!14 = !{!15, !15, i64 0}
!15 = !{!"float", !7, i64 0}
!16 = distinct !{!16, !10, !11}
!17 = distinct !{!17, !10, !11, !18, !19}
!18 = !{!"llvm.loop.isvectorized", i32 1}
!19 = !{!"llvm.loop.unroll.runtime.disable"}
!20 = distinct !{!20, !10, !11}
!21 = distinct !{!21, !10, !11}
!22 = distinct !{!22, !10, !11, !18}
