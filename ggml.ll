; ModuleID = 'ggml_compute_forward_mul_mat.cpp'
source_filename = "ggml_compute_forward_mul_mat.cpp"
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

%struct.ggml_tensor = type { i32, %struct.ggml_backend_buffer*, [4 x i64], [4 x i64], i32, [16 x i32], i32, [10 x %struct.ggml_tensor*], %struct.ggml_tensor*, i64, i8*, [64 x i8], i8*, [8 x i8] }
%struct.ggml_backend_buffer = type opaque

@llvm.used = appending global [1 x i8*] [i8* bitcast (void (%struct.ggml_tensor*, %struct.ggml_tensor*, %struct.ggml_tensor*)* @_ZL32ggml_compute_forward_mul_mat_f32PK11ggml_tensorS1_PS_ to i8*)], section "llvm.metadata"

; Function Attrs: nofree noinline norecurse nounwind
define internal void @_ZL32ggml_compute_forward_mul_mat_f32PK11ggml_tensorS1_PS_(%struct.ggml_tensor* nocapture readonly %0, %struct.ggml_tensor* nocapture readonly %1, %struct.ggml_tensor* nocapture readonly %2) #0 {
  %4 = getelementptr inbounds %struct.ggml_tensor, %struct.ggml_tensor* %0, i64 0, i32 2, i64 2
  %5 = load i64, i64* %4, align 8, !tbaa !2
  %6 = getelementptr inbounds %struct.ggml_tensor, %struct.ggml_tensor* %0, i64 0, i32 2, i64 3
  %7 = load i64, i64* %6, align 8, !tbaa !2
  %8 = getelementptr inbounds %struct.ggml_tensor, %struct.ggml_tensor* %1, i64 0, i32 2, i64 0
  %9 = load i64, i64* %8, align 8, !tbaa !2
  %10 = getelementptr inbounds %struct.ggml_tensor, %struct.ggml_tensor* %2, i64 0, i32 2, i64 0
  %11 = load i64, i64* %10, align 8, !tbaa !2
  %12 = getelementptr inbounds %struct.ggml_tensor, %struct.ggml_tensor* %2, i64 0, i32 2, i64 1
  %13 = load i64, i64* %12, align 8, !tbaa !2
  %14 = getelementptr inbounds %struct.ggml_tensor, %struct.ggml_tensor* %2, i64 0, i32 2, i64 2
  %15 = load i64, i64* %14, align 8, !tbaa !2
  %16 = getelementptr inbounds %struct.ggml_tensor, %struct.ggml_tensor* %2, i64 0, i32 2, i64 3
  %17 = load i64, i64* %16, align 8, !tbaa !2
  %18 = getelementptr inbounds %struct.ggml_tensor, %struct.ggml_tensor* %0, i64 0, i32 3, i64 0
  %19 = load i64, i64* %18, align 8, !tbaa !6
  %20 = getelementptr inbounds %struct.ggml_tensor, %struct.ggml_tensor* %0, i64 0, i32 3, i64 1
  %21 = load i64, i64* %20, align 8, !tbaa !6
  %22 = getelementptr inbounds %struct.ggml_tensor, %struct.ggml_tensor* %0, i64 0, i32 3, i64 2
  %23 = load i64, i64* %22, align 8, !tbaa !6
  %24 = getelementptr inbounds %struct.ggml_tensor, %struct.ggml_tensor* %0, i64 0, i32 3, i64 3
  %25 = load i64, i64* %24, align 8, !tbaa !6
  %26 = getelementptr inbounds %struct.ggml_tensor, %struct.ggml_tensor* %1, i64 0, i32 3, i64 0
  %27 = load i64, i64* %26, align 8, !tbaa !6
  %28 = getelementptr inbounds %struct.ggml_tensor, %struct.ggml_tensor* %1, i64 0, i32 3, i64 1
  %29 = load i64, i64* %28, align 8, !tbaa !6
  %30 = getelementptr inbounds %struct.ggml_tensor, %struct.ggml_tensor* %1, i64 0, i32 3, i64 2
  %31 = load i64, i64* %30, align 8, !tbaa !6
  %32 = getelementptr inbounds %struct.ggml_tensor, %struct.ggml_tensor* %1, i64 0, i32 3, i64 3
  %33 = load i64, i64* %32, align 8, !tbaa !6
  %34 = getelementptr inbounds %struct.ggml_tensor, %struct.ggml_tensor* %2, i64 0, i32 3, i64 0
  %35 = load i64, i64* %34, align 8, !tbaa !6
  %36 = getelementptr inbounds %struct.ggml_tensor, %struct.ggml_tensor* %2, i64 0, i32 3, i64 1
  %37 = load i64, i64* %36, align 8, !tbaa !6
  %38 = getelementptr inbounds %struct.ggml_tensor, %struct.ggml_tensor* %2, i64 0, i32 3, i64 2
  %39 = load i64, i64* %38, align 8, !tbaa !6
  %40 = getelementptr inbounds %struct.ggml_tensor, %struct.ggml_tensor* %2, i64 0, i32 3, i64 3
  %41 = load i64, i64* %40, align 8, !tbaa !6
  %42 = getelementptr inbounds %struct.ggml_tensor, %struct.ggml_tensor* %0, i64 0, i32 10
  %43 = load i8*, i8** %42, align 8, !tbaa !8
  %44 = getelementptr inbounds %struct.ggml_tensor, %struct.ggml_tensor* %1, i64 0, i32 10
  %45 = load i8*, i8** %44, align 8, !tbaa !8
  %46 = getelementptr inbounds %struct.ggml_tensor, %struct.ggml_tensor* %2, i64 0, i32 10
  %47 = load i8*, i8** %46, align 8, !tbaa !8
  %48 = icmp sgt i64 %17, 0
  br i1 %48, label %49, label %64

49:                                               ; preds = %3
  %50 = icmp sgt i64 %15, 0
  %51 = icmp sgt i64 %13, 0
  %52 = icmp sgt i64 %11, 0
  %53 = icmp sgt i64 %9, 0
  br label %54

54:                                               ; preds = %65, %49
  %55 = phi i64 [ 0, %49 ], [ %66, %65 ]
  br i1 %50, label %56, label %65

56:                                               ; preds = %54
  %57 = srem i64 %55, %7
  %58 = mul i64 %57, %25
  %59 = getelementptr inbounds i8, i8* %43, i64 %58
  %60 = mul i64 %55, %33
  %61 = getelementptr inbounds i8, i8* %45, i64 %60
  %62 = mul i64 %55, %41
  %63 = getelementptr inbounds i8, i8* %47, i64 %62
  br label %68

64:                                               ; preds = %65, %3
  ret void

65:                                               ; preds = %84, %54
  %66 = add nuw nsw i64 %55, 1
  %67 = icmp eq i64 %66, %17
  br i1 %67, label %64, label %54

68:                                               ; preds = %84, %56
  %69 = phi i64 [ 0, %56 ], [ %85, %84 ]
  %70 = srem i64 %69, %5
  %71 = mul i64 %70, %23
  %72 = getelementptr inbounds i8, i8* %59, i64 %71
  %73 = mul i64 %69, %31
  %74 = getelementptr inbounds i8, i8* %61, i64 %73
  %75 = mul i64 %69, %39
  %76 = getelementptr inbounds i8, i8* %63, i64 %75
  br i1 %51, label %77, label %84

77:                                               ; preds = %68, %92
  %78 = phi i64 [ %93, %92 ], [ 0, %68 ]
  br i1 %52, label %79, label %92

79:                                               ; preds = %77
  %80 = mul i64 %78, %29
  %81 = mul i64 %78, %37
  %82 = getelementptr inbounds i8, i8* %74, i64 %80
  %83 = getelementptr inbounds i8, i8* %76, i64 %81
  br label %87

84:                                               ; preds = %92, %68
  %85 = add nuw nsw i64 %69, 1
  %86 = icmp eq i64 %85, %15
  br i1 %86, label %65, label %68

87:                                               ; preds = %95, %79
  %88 = phi i64 [ 0, %79 ], [ %100, %95 ]
  br i1 %53, label %89, label %95

89:                                               ; preds = %87
  %90 = mul i64 %88, %21
  %91 = getelementptr inbounds i8, i8* %72, i64 %90
  br label %102

92:                                               ; preds = %95, %77
  %93 = add nuw nsw i64 %78, 1
  %94 = icmp eq i64 %93, %13
  br i1 %94, label %84, label %77

95:                                               ; preds = %102, %87
  %96 = phi float [ 0.000000e+00, %87 ], [ %114, %102 ]
  %97 = mul i64 %88, %35
  %98 = getelementptr inbounds i8, i8* %83, i64 %97
  %99 = bitcast i8* %98 to float*
  store float %96, float* %99, align 4, !tbaa !14
  %100 = add nuw nsw i64 %88, 1
  %101 = icmp eq i64 %100, %11
  br i1 %101, label %92, label %87

102:                                              ; preds = %102, %89
  %103 = phi i64 [ 0, %89 ], [ %115, %102 ]
  %104 = phi float [ 0.000000e+00, %89 ], [ %114, %102 ]
  %105 = mul i64 %103, %19
  %106 = getelementptr inbounds i8, i8* %91, i64 %105
  %107 = bitcast i8* %106 to float*
  %108 = load float, float* %107, align 4, !tbaa !14
  %109 = mul i64 %103, %27
  %110 = getelementptr inbounds i8, i8* %82, i64 %109
  %111 = bitcast i8* %110 to float*
  %112 = load float, float* %111, align 4, !tbaa !14
  %113 = fmul float %108, %112
  %114 = fadd float %104, %113
  %115 = add nuw nsw i64 %103, 1
  %116 = icmp eq i64 %115, %9
  br i1 %116, label %95, label %102
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
