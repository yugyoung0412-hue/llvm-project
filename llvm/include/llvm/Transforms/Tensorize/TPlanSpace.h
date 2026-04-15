#ifndef LLVM_TRANSFORMS_TENSORIZE_TPLANSPACE_H
#define LLVM_TRANSFORMS_TENSORIZE_TPLANSPACE_H

#include "TPattern.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {

enum class PatternKind;
// --------------------------------------------------------
// Search Space 1) Loop-Restructuring Space
// --------------------------------------------------------
enum class LoopInterchangeMode {
  None,
  InterchangeAB,
  InterchangeBC,
  InterchangeAC
};
enum class UnrollFactor {
  None = 0,
  Factor2 = 2,
  Factor4 = 4,
  Factor8 = 8,
  Factor16 = 16,
  Factor32 = 32
};

struct LoopRestructuringOption {
  LoopInterchangeMode interchange = LoopInterchangeMode::None;
  UnrollFactor unroll = UnrollFactor::None;

  bool operator==(const LoopRestructuringOption &) const = default;
};

// ---------------------------------------------------------
// Search Space 2) Data-flow (operand vector width)
// ---------------------------------------------------------
// enum Dataflow {
//   single_reduction,
//   r1_fir_order_parallel,
//   r1_sec_order_parallel,
//   r2_fir_order_parallel,
//   r2_sec_order_parallel,
// };

enum class OperandKind { Scalar, Vector, Matrix, Tensor };

struct DataFlowOption {
  uint32_t lhs_width = 0; // 0 → scalar
  uint32_t rhs_width = 0; // 0 → scalar
  OperandKind result_kind = OperandKind::Scalar;

  bool operator==(const DataFlowOption &) const = default;
};

// ---------------------------------------------------------
// Search Space 3) Architecture-traits (TTI-based Info)
// ---------------------------------------------------------
struct ArchTraitOption {
  uint32_t num_vector_registers = 0;   // e.g. 64
  uint32_t num_matrix_registers = 0;   // e.g. 32
  uint32_t num_tensor_registers = 0;   // e.g. 8
  uint32_t max_vector_width_bytes = 0; // e.g. 256 (bytes)

  bool operator==(const ArchTraitOption &) const = default;
};

// ---------------------------------------------------------
// Search Space 4) Computation-aglorithm
// ---------------------------------------------------------
enum class ComputationAlgorithm {
  InnerProduct, // dot‑product style
  OuterProduct, // A·Bᵀ style
  GEMM,         // general matrix‑multiply
  Convolution,
  Custom // user-defined algorithm
};

class TPlanSpace {
public:
  // Search Space
  std::optional<LoopRestructuringOption> loopOption;
  std::optional<DataFlowOption> dataFlow;
  std::optional<ArchTraitOption> archTrait;
  std::optional<ComputationAlgorithm> alg;

  bool setHeuristic;
  std::shared_ptr<TensorizePattern> Pattern;
  std::vector<unsigned> TensorizationFactors;
  Triple::ArchType target;

  TPlanSpace(bool useHeuristic, Triple::ArchType target,
             std::shared_ptr<TensorizePattern> Pattern = nullptr)
      : setHeuristic(useHeuristic), target(target), Pattern(Pattern) {}
};

} // end of namespace llvm

#endif // LLVM_TRANSFORMS_TENSORIZE_TPLANSPACE_H
