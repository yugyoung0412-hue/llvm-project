; ModuleID = 'do_conv2d.cpp'
source_filename = "do_conv2d.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

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
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %entry, %for.cond.cleanup3
  %n.0158 = phi i64 [ 0, %entry ], [ %inc100, %for.cond.cleanup3 ]
  br i1 %cmp2156, label %for.cond5.preheader.lr.ph, label %for.cond.cleanup3

for.cond5.preheader.lr.ph:                        ; preds = %for.cond1.preheader
  %.idx139 = shl nuw nsw i64 %n.0158, 14
  %invariant.gep149 = getelementptr inbounds nuw i8, ptr %X, i64 %.idx139
  %mul82 = mul i64 %n.0158, %OC
  br label %for.cond5.preheader

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3
  ret void

for.cond5.preheader:                              ; preds = %for.cond5.preheader.lr.ph, %for.cond.cleanup7
  %oc.0157 = phi i64 [ 0, %for.cond5.preheader.lr.ph ], [ %inc96, %for.cond.cleanup7 ]
  br i1 %cmp6154, label %for.cond9.preheader.lr.ph, label %for.cond.cleanup7

for.cond9.preheader.lr.ph:                        ; preds = %for.cond5.preheader
  %mul48 = mul nuw nsw i64 %oc.0157, 144
  %reass.add141 = add i64 %oc.0157, %mul82
  %reass.mul142 = mul i64 %reass.add141, %OH
  br label %for.cond9.preheader

for.cond.cleanup3:                                ; preds = %for.cond.cleanup7, %for.cond1.preheader
  %inc100 = add nuw nsw i64 %n.0158, 1
  %exitcond164.not = icmp eq i64 %inc100, 16
  br i1 %exitcond164.not, label %for.cond.cleanup, label %for.cond1.preheader, !llvm.loop !9

for.cond9.preheader:                              ; preds = %for.cond9.preheader.lr.ph, %for.cond.cleanup11
  %oh.0155 = phi i64 [ 0, %for.cond9.preheader.lr.ph ], [ %inc92, %for.cond.cleanup11 ]
  br i1 %cmp10152, label %for.cond13.preheader.lr.ph, label %for.cond.cleanup11

for.cond13.preheader.lr.ph:                       ; preds = %for.cond9.preheader
  %mul = mul nsw i64 %oh.0155, %conv
  %sub = sub nsw i64 %mul, %conv25
  %reass.add140 = add i64 %reass.mul142, %oh.0155
  %reass.mul = mul i64 %reass.add140, %OW
  br label %for.cond13.preheader

for.cond.cleanup7:                                ; preds = %for.cond.cleanup11, %for.cond5.preheader
  %inc96 = add nuw nsw i64 %oc.0157, 1
  %exitcond163.not = icmp eq i64 %inc96, %OC
  br i1 %exitcond163.not, label %for.cond.cleanup3, label %for.cond5.preheader, !llvm.loop !12

for.cond13.preheader:                             ; preds = %for.cond13.preheader.lr.ph, %for.cond.cleanup15
  %ow.0153 = phi i64 [ 0, %for.cond13.preheader.lr.ph ], [ %inc88, %for.cond.cleanup15 ]
  %mul29 = mul nsw i64 %ow.0153, %conv28
  %sub31 = sub nsw i64 %mul29, %conv30
  br label %for.cond17.preheader

for.cond.cleanup11:                               ; preds = %for.cond.cleanup15, %for.cond9.preheader
  %inc92 = add nuw nsw i64 %oh.0155, 1
  %exitcond162.not = icmp eq i64 %inc92, %OH
  br i1 %exitcond162.not, label %for.cond.cleanup7, label %for.cond9.preheader, !llvm.loop !13

for.cond17.preheader:                             ; preds = %for.cond13.preheader, %for.cond.cleanup19
  %ic.0151 = phi i64 [ 0, %for.cond13.preheader ], [ %inc72, %for.cond.cleanup19 ]
  %sum.0150 = phi float [ 0.000000e+00, %for.cond13.preheader ], [ %sum.3, %for.cond.cleanup19 ]
  %mul44 = mul nuw nsw i64 %ic.0151, 9
  %add42 = add nuw nsw i64 %mul44, %mul48
  %.idx138 = shl nuw nsw i64 %ic.0151, 10
  %gep = getelementptr inbounds nuw i8, ptr %invariant.gep149, i64 %.idx138
  br label %for.cond21.preheader

for.cond.cleanup15:                               ; preds = %for.cond.cleanup19
  %add83 = add i64 %ow.0153, %reass.mul
  %sext = shl i64 %add83, 32
  %0 = ashr exact i64 %sext, 30
  %arrayidx86 = getelementptr inbounds i8, ptr %o_data, i64 %0
  store float %sum.3, ptr %arrayidx86, align 4, !tbaa !14
  %inc88 = add nuw nsw i64 %ow.0153, 1
  %exitcond161.not = icmp eq i64 %inc88, %OW
  br i1 %exitcond161.not, label %for.cond.cleanup11, label %for.cond13.preheader, !llvm.loop !16

for.cond21.preheader:                             ; preds = %for.cond17.preheader, %for.cond.cleanup23
  %kh.0148 = phi i64 [ 0, %for.cond17.preheader ], [ %inc68, %for.cond.cleanup23 ]
  %sum.1147 = phi float [ %sum.0150, %for.cond17.preheader ], [ %sum.3, %for.cond.cleanup23 ]
  %mul27 = mul nsw i64 %kh.0148, %conv26
  %add = add nsw i64 %mul27, %sub
  %1 = icmp ugt i64 %add, 15
  %mul41 = mul nuw nsw i64 %add, 3
  %add45 = add nuw nsw i64 %add42, %mul41
  %.idx = shl nuw nsw i64 %add, 6
  %gep146 = getelementptr inbounds nuw i8, ptr %gep, i64 %.idx
  br label %for.body24

for.cond.cleanup19:                               ; preds = %for.cond.cleanup23
  %inc72 = add nuw nsw i64 %ic.0151, 1
  %exitcond160.not = icmp eq i64 %inc72, 16
  br i1 %exitcond160.not, label %for.cond.cleanup15, label %for.cond17.preheader, !llvm.loop !17

for.cond.cleanup23:                               ; preds = %cleanup
  %inc68 = add nuw nsw i64 %kh.0148, 1
  %exitcond159.not = icmp eq i64 %inc68, 3
  br i1 %exitcond159.not, label %for.cond.cleanup19, label %for.cond21.preheader, !llvm.loop !18

for.body24:                                       ; preds = %for.cond21.preheader, %cleanup
  %kw.0144 = phi i64 [ 0, %for.cond21.preheader ], [ %inc, %cleanup ]
  %sum.2143 = phi float [ %sum.1147, %for.cond21.preheader ], [ %sum.3, %cleanup ]
  %mul33 = mul nsw i64 %kw.0144, %conv32
  %add34 = add nsw i64 %mul33, %sub31
  %or.cond = icmp ugt i64 %add34, 15
  %or.cond104 = select i1 %or.cond, i1 true, i1 %1
  br i1 %or.cond104, label %cleanup, label %if.end

if.end:                                           ; preds = %for.body24
  %add49 = add nuw nsw i64 %add45, %kw.0144
  %sext137 = shl i64 %add49, 32
  %2 = ashr exact i64 %sext137, 30
  %arrayidx = getelementptr inbounds i8, ptr %W, i64 %2
  %3 = load float, ptr %arrayidx, align 4, !tbaa !14
  %arrayidx62 = getelementptr inbounds nuw float, ptr %gep146, i64 %add34
  %4 = load float, ptr %arrayidx62, align 4, !tbaa !14
  %mul63 = fmul float %3, %4
  %add64 = fadd float %sum.2143, %mul63
  br label %cleanup

cleanup:                                          ; preds = %for.body24, %if.end
  %sum.3 = phi float [ %add64, %if.end ], [ %sum.2143, %for.body24 ]
  %inc = add nuw nsw i64 %kw.0144, 1
  %exitcond.not = icmp eq i64 %inc, 3
  br i1 %exitcond.not, label %for.cond.cleanup23, label %for.body24, !llvm.loop !19
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

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
!17 = distinct !{!17, !10, !11}
!18 = distinct !{!18, !10, !11}
!19 = distinct !{!19, !10, !11}
