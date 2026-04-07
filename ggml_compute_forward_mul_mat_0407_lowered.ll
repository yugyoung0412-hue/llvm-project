; ModuleID = '/Users/yun-yugyeong/Dev/llvm/ggml_compute_forward_mul_mat.ll'
source_filename = "ggml_compute_forward_mul_mat.cpp"
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

%struct.ggml_tensor = type { i32, ptr, [4 x i64], [4 x i64], i32, [16 x i32], i32, [10 x ptr], ptr, i64, ptr, [64 x i8], ptr, [8 x i8] }

@llvm.used = appending global [1 x ptr] [ptr @_ZL32ggml_compute_forward_mul_mat_f32PK11ggml_tensorS1_PS_], section "llvm.metadata"

; Function Attrs: nofree noinline norecurse nounwind
define internal void @_ZL32ggml_compute_forward_mul_mat_f32PK11ggml_tensorS1_PS_(ptr readonly captures(none) %0, ptr readonly captures(none) %1, ptr readonly captures(none) %2) #0 {
  %4 = getelementptr inbounds %struct.ggml_tensor, ptr %0, i64 0, i32 2, i64 2
  %5 = load i64, ptr %4, align 8, !tbaa !2
  %6 = getelementptr inbounds %struct.ggml_tensor, ptr %0, i64 0, i32 2, i64 3
  %7 = load i64, ptr %6, align 8, !tbaa !2
  %8 = getelementptr inbounds %struct.ggml_tensor, ptr %1, i64 0, i32 2, i64 0
  %9 = load i64, ptr %8, align 8, !tbaa !2
  %10 = getelementptr inbounds %struct.ggml_tensor, ptr %2, i64 0, i32 2, i64 0
  %11 = load i64, ptr %10, align 8, !tbaa !2
  %12 = getelementptr inbounds %struct.ggml_tensor, ptr %2, i64 0, i32 2, i64 1
  %13 = load i64, ptr %12, align 8, !tbaa !2
  %14 = getelementptr inbounds %struct.ggml_tensor, ptr %2, i64 0, i32 2, i64 2
  %15 = load i64, ptr %14, align 8, !tbaa !2
  %16 = getelementptr inbounds %struct.ggml_tensor, ptr %2, i64 0, i32 2, i64 3
  %17 = load i64, ptr %16, align 8, !tbaa !2
  %18 = getelementptr inbounds %struct.ggml_tensor, ptr %0, i64 0, i32 3, i64 0
  %19 = load i64, ptr %18, align 8, !tbaa !6
  %20 = getelementptr inbounds %struct.ggml_tensor, ptr %0, i64 0, i32 3, i64 1
  %21 = load i64, ptr %20, align 8, !tbaa !6
  %22 = getelementptr inbounds %struct.ggml_tensor, ptr %0, i64 0, i32 3, i64 2
  %23 = load i64, ptr %22, align 8, !tbaa !6
  %24 = getelementptr inbounds %struct.ggml_tensor, ptr %0, i64 0, i32 3, i64 3
  %25 = load i64, ptr %24, align 8, !tbaa !6
  %26 = getelementptr inbounds %struct.ggml_tensor, ptr %1, i64 0, i32 3, i64 0
  %27 = load i64, ptr %26, align 8, !tbaa !6
  %28 = getelementptr inbounds %struct.ggml_tensor, ptr %1, i64 0, i32 3, i64 1
  %29 = load i64, ptr %28, align 8, !tbaa !6
  %30 = getelementptr inbounds %struct.ggml_tensor, ptr %1, i64 0, i32 3, i64 2
  %31 = load i64, ptr %30, align 8, !tbaa !6
  %32 = getelementptr inbounds %struct.ggml_tensor, ptr %1, i64 0, i32 3, i64 3
  %33 = load i64, ptr %32, align 8, !tbaa !6
  %34 = getelementptr inbounds %struct.ggml_tensor, ptr %2, i64 0, i32 3, i64 0
  %35 = load i64, ptr %34, align 8, !tbaa !6
  %36 = getelementptr inbounds %struct.ggml_tensor, ptr %2, i64 0, i32 3, i64 1
  %37 = load i64, ptr %36, align 8, !tbaa !6
  %38 = getelementptr inbounds %struct.ggml_tensor, ptr %2, i64 0, i32 3, i64 2
  %39 = load i64, ptr %38, align 8, !tbaa !6
  %40 = getelementptr inbounds %struct.ggml_tensor, ptr %2, i64 0, i32 3, i64 3
  %41 = load i64, ptr %40, align 8, !tbaa !6
  %42 = getelementptr inbounds %struct.ggml_tensor, ptr %0, i64 0, i32 10
  %43 = load ptr, ptr %42, align 8, !tbaa !8
  %44 = getelementptr inbounds %struct.ggml_tensor, ptr %1, i64 0, i32 10
  %45 = load ptr, ptr %44, align 8, !tbaa !8
  %46 = getelementptr inbounds %struct.ggml_tensor, ptr %2, i64 0, i32 10
  %47 = load ptr, ptr %46, align 8, !tbaa !8
  %48 = icmp sgt i64 %17, 0
  br i1 %48, label %49, label %71

49:                                               ; preds = %3
  %50 = icmp sgt i64 %15, 0
  %51 = icmp sgt i64 %13, 0
  %52 = icmp sgt i64 %11, 0
  %53 = icmp sgt i64 %9, 0
  br label %54

54:                                               ; preds = %72, %49
  %55 = phi i64 [ 0, %49 ], [ %75, %72 ]
  %56 = srem i64 %55, %7
  %57 = mul i64 %56, %25
  %58 = getelementptr inbounds i8, ptr %43, i64 %57
  %59 = mul i64 %55, %33
  %60 = getelementptr inbounds i8, ptr %45, i64 %59
  %61 = mul i64 %55, %41
  %62 = getelementptr inbounds i8, ptr %47, i64 %61
  br i1 %50, label %63, label %72

63:                                               ; preds = %54
  %64 = srem i64 %55, %7
  %65 = mul i64 %64, %25
  %66 = getelementptr inbounds i8, ptr %43, i64 %65
  %67 = mul i64 %55, %33
  %68 = getelementptr inbounds i8, ptr %45, i64 %67
  %69 = mul i64 %55, %41
  %70 = getelementptr inbounds i8, ptr %47, i64 %69
  br label %77

71:                                               ; preds = %72, %3
  ret void

72:                                               ; preds = %104, %54
  %73 = add nuw nsw i64 %55, 1
  %74 = icmp eq i64 %73, %17
  %75 = add nuw nsw i64 %55, 1
  %76 = icmp eq i64 %75, %17
  br i1 %76, label %71, label %54

77:                                               ; preds = %104, %63
  %78 = phi i64 [ 0, %63 ], [ %107, %104 ]
  %79 = srem i64 %78, %5
  %80 = mul i64 %79, %23
  %81 = getelementptr inbounds i8, ptr %58, i64 %80
  %82 = mul i64 %78, %31
  %83 = getelementptr inbounds i8, ptr %60, i64 %82
  %84 = mul i64 %78, %39
  %85 = getelementptr inbounds i8, ptr %62, i64 %84
  %86 = srem i64 %78, %5
  %87 = mul i64 %86, %23
  %88 = getelementptr inbounds i8, ptr %66, i64 %87
  %89 = mul i64 %78, %31
  %90 = getelementptr inbounds i8, ptr %68, i64 %89
  %91 = mul i64 %78, %39
  %92 = getelementptr inbounds i8, ptr %70, i64 %91
  br i1 %51, label %93, label %104

93:                                               ; preds = %116, %77
  %94 = phi i64 [ %119, %116 ], [ 0, %77 ]
  %95 = mul i64 %94, %29
  %96 = mul i64 %94, %37
  %97 = getelementptr inbounds i8, ptr %83, i64 %95
  %98 = getelementptr inbounds i8, ptr %85, i64 %96
  br i1 %52, label %99, label %116

99:                                               ; preds = %93
  %100 = mul i64 %94, %29
  %101 = mul i64 %94, %37
  %102 = getelementptr inbounds i8, ptr %90, i64 %100
  %103 = getelementptr inbounds i8, ptr %92, i64 %101
  br label %109

104:                                              ; preds = %116, %77
  %105 = add nuw nsw i64 %78, 1
  %106 = icmp eq i64 %105, %15
  %107 = add nuw nsw i64 %78, 1
  %108 = icmp eq i64 %107, %15
  br i1 %108, label %72, label %77

109:                                              ; preds = %121, %99
  %110 = phi i64 [ 0, %99 ], [ %131, %121 ]
  %111 = mul i64 %110, %21
  %112 = getelementptr inbounds i8, ptr %81, i64 %111
  br i1 %53, label %113, label %121

113:                                              ; preds = %109
  %114 = mul i64 %110, %21
  %115 = getelementptr inbounds i8, ptr %88, i64 %114
  br label %133

116:                                              ; preds = %121, %93
  %117 = add nuw nsw i64 %94, 1
  %118 = icmp eq i64 %117, %13
  %119 = add nuw nsw i64 %94, 1
  %120 = icmp eq i64 %119, %13
  br i1 %120, label %104, label %93

121:                                              ; preds = %133, %109
  %122 = phi float [ 0.000000e+00, %109 ], [ %165, %133 ]
  %123 = mul i64 %110, %35
  %124 = getelementptr inbounds i8, ptr %98, i64 %123
  %125 = bitcast ptr %124 to ptr
  store float %122, ptr %125, align 4, !tbaa !14
  %126 = add nuw nsw i64 %110, 1
  %127 = icmp eq i64 %126, %11
  %128 = mul i64 %110, %35
  %129 = getelementptr inbounds i8, ptr %103, i64 %128
  %130 = bitcast ptr %129 to ptr
  store float %122, ptr %130, align 4, !tbaa !14
  %131 = add nuw nsw i64 %110, 1
  %132 = icmp eq i64 %131, %11
  br i1 %132, label %116, label %109

133:                                              ; preds = %133, %113
  %134 = phi i64 [ 0, %113 ], [ %166, %133 ]
  %135 = phi float [ 0.000000e+00, %113 ], [ %165, %133 ]
  %136 = mul i64 %134, %19
  %137 = getelementptr inbounds i8, ptr %112, i64 %136
  %138 = bitcast ptr %137 to ptr
  %139 = load float, ptr %138, align 4, !tbaa !14
  %140 = mul i64 %134, %27
  %141 = getelementptr inbounds i8, ptr %97, i64 %140
  %142 = bitcast ptr %141 to ptr
  %143 = load float, ptr %142, align 4, !tbaa !14
  %144 = add nuw nsw i64 %134, 1
  %145 = icmp eq i64 %144, %9
  %146 = mul i64 %134, %19
  %147 = getelementptr inbounds i8, ptr %112, i64 %146
  %148 = bitcast ptr %147 to ptr
  %149 = load float, ptr %148, align 4, !tbaa !14
  %150 = mul i64 %134, %27
  %151 = getelementptr inbounds i8, ptr %97, i64 %150
  %152 = bitcast ptr %151 to ptr
  %153 = load float, ptr %152, align 4, !tbaa !14
  %154 = add nuw nsw i64 %134, 1
  %155 = icmp eq i64 %154, %9
  %156 = mul i64 %134, %19
  %157 = getelementptr inbounds i8, ptr %115, i64 %156
  %158 = bitcast ptr %157 to ptr
  %159 = load float, ptr %158, align 4, !tbaa !14
  %160 = mul i64 %134, %27
  %161 = getelementptr inbounds i8, ptr %102, i64 %160
  %162 = bitcast ptr %161 to ptr
  %163 = load float, ptr %162, align 4, !tbaa !14
  %164 = fmul float %159, %163
  %165 = fadd float %135, %164
  %166 = add nuw nsw i64 %134, 1
  %167 = icmp eq i64 %166, %9
  br i1 %167, label %121, label %133
}

attributes #0 = { nofree noinline norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="non-leaf" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0-4ubuntu1 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"long long", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"long", !4, i64 0}
!8 = !{!9, !11, i64 248}
!9 = !{!"_ZTS11ggml_tensor", !10, i64 0, !11, i64 8, !4, i64 16, !4, i64 48, !12, i64 80, !4, i64 84, !13, i64 148, !4, i64 152, !11, i64 232, !7, i64 240, !11, i64 248, !4, i64 256, !11, i64 320, !4, i64 328}
!10 = !{!"_ZTS9ggml_type", !4, i64 0}
!11 = !{!"any pointer", !4, i64 0}
!12 = !{!"_ZTS7ggml_op", !4, i64 0}
!13 = !{!"int", !4, i64 0}
!14 = !{!15, !15, i64 0}
!15 = !{!"float", !4, i64 0}
