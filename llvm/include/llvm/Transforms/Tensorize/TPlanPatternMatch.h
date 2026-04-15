#ifndef LLVM_TRANSFORM_TENSORIZE_TPLANPATTERNMATCH_H
#define LLVM_TRANSFORM_TENSORIZE_TPLANPATTERNMATCH_H

#include "TPlan.h"

namespace llvm {
namespace TPlanPatternMatch {

template <typename Val, typename Pattern> bool match(Val *V, const Pattern &P) {
  return const_cast<Pattern &>(P).match(V);
}

template <typename Class> struct class_match {
  template <typename ITy> bool match(ITy *V) { return isa<Class>(V); }
};

/// Match an arbitrary TPValue and ignore it.
inline class_match<TPValue> m_TPValue() { return class_match<TPValue>(); }

template <typename Class> struct bind_ty {
  Class *&VR;

  bind_ty(Class *&V) : VR(V) {}

  template <typename ITy> bool match(ITy *V) {
    if (auto *CV = dyn_cast<Class>(V)) {
      VR = CV;
      return true;
    }
    return false;
  }
};

/// Match a specified integer value or vector of all elements of that
/// value. \p BitWidth optionally specifies the bitwidth the matched constant
/// must have. If it is 0, the matched constant can have any bitwidth.
template <unsigned BitWidth = 0> struct specific_intval {
  APInt Val;

  specific_intval(APInt V) : Val(std::move(V)) {}

  bool match(TPValue *TPV) {
    if (!TPV->isLiveIn())
      return false;
    Value *V = TPV->getLiveInIRValue();
    const auto *CI = dyn_cast<ConstantInt>(V);
    if (!CI && V->getType()->isVectorTy())
      if (const auto *C = dyn_cast<Constant>(V))
        CI = dyn_cast_or_null<ConstantInt>(
            C->getSplatValue(/*AllowPoison=*/false));
    if (!CI)
      return false;

    assert((BitWidth == 0 || CI->getBitWidth() == BitWidth) &&
           "Trying the match constant with unexpected bitwidth.");
    return APInt::isSameValue(CI->getValue(), Val);
  }
};

inline specific_intval<0> m_SpecificInt(uint64_t V) {
  return specific_intval<0>(APInt(64, V));
}

inline specific_intval<1> m_False() { return specific_intval<1>(APInt(64, 0)); }

/// Matching combinators
template <typename LTy, typename RTy> struct match_combine_or {
  LTy L;
  RTy R;

  match_combine_or(const LTy &Left, const RTy &Right) : L(Left), R(Right) {}

  template <typename ITy> bool match(ITy *V) {
    if (L.match(V))
      return true;
    if (R.match(V))
      return true;
    return false;
  }
};

template <typename LTy, typename RTy>
inline match_combine_or<LTy, RTy> m_CombineOr(const LTy &L, const RTy &R) {
  return match_combine_or<LTy, RTy>(L, R);
}

/// Match a TPValue, capturing it if we match.
inline bind_ty<TPValue> m_TPValue(TPValue *&V) { return V; }

namespace detail {

/// A helper to match an opcode against multiple recipe types.
template <unsigned Opcode, typename...> struct MatchRecipeAndOpcode {};

template <unsigned Opcode, typename RecipeTy>
struct MatchRecipeAndOpcode<Opcode, RecipeTy> {
  static bool match(const TPRecipeBase *R) {
    auto *DefR = dyn_cast<RecipeTy>(R);
    return DefR && DefR->getOpcode() == Opcode;
  }
};

template <unsigned Opcode, typename RecipeTy, typename... RecipeTys>
struct MatchRecipeAndOpcode<Opcode, RecipeTy, RecipeTys...> {
  static bool match(const TPRecipeBase *R) {
    return MatchRecipeAndOpcode<Opcode, RecipeTy>::match(R) ||
           MatchRecipeAndOpcode<Opcode, RecipeTys...>::match(R);
  }
};
} // namespace detail

template <typename Op0_t, unsigned Opcode, typename... RecipeTys>
struct UnaryRecipe_match {
  Op0_t Op0;

  UnaryRecipe_match(Op0_t Op0) : Op0(Op0) {}

  bool match(const TPValue *V) {
    auto *DefR = V->getDefiningRecipe();
    return DefR && match(DefR);
  }

  bool match(const TPRecipeBase *R) {
    if (!detail::MatchRecipeAndOpcode<Opcode, RecipeTys...>::match(R))
      return false;
    assert(R->getNumOperands() == 1 &&
           "recipe with matched opcode does not have 1 operands");
    return Op0.match(R->getOperand(0));
  }
};

template <typename Op0_t, unsigned Opcode>
using UnaryTPInstruction_match =
    UnaryRecipe_match<Op0_t, Opcode, TPInstruction>;

template <typename Op0_t, unsigned Opcode>
using AllUnaryRecipe_match =
    UnaryRecipe_match<Op0_t, Opcode, TPWidenRecipe, TPReplicateRecipe,
                      TPWidenCastRecipe, TPInstruction>;

template <typename Op0_t, typename Op1_t, unsigned Opcode, bool Commutative,
          typename... RecipeTys>
struct BinaryRecipe_match {
  Op0_t Op0;
  Op1_t Op1;

  BinaryRecipe_match(Op0_t Op0, Op1_t Op1) : Op0(Op0), Op1(Op1) {}

  bool match(const TPValue *V) {
    auto *DefR = V->getDefiningRecipe();
    return DefR && match(DefR);
  }

  bool match(const TPSingleDefRecipe *R) {
    return match(static_cast<const TPRecipeBase *>(R));
  }

  bool match(const TPRecipeBase *R) {
    if (!detail::MatchRecipeAndOpcode<Opcode, RecipeTys...>::match(R))
      return false;
    assert(R->getNumOperands() == 2 &&
           "recipe with matched opcode does not have 2 operands");
    if (Op0.match(R->getOperand(0)) && Op1.match(R->getOperand(1)))
      return true;
    return Commutative && Op0.match(R->getOperand(1)) &&
           Op1.match(R->getOperand(0));
  }
};

template <typename Op0_t, typename Op1_t, unsigned Opcode>
using BinaryTPInstruction_match =
    BinaryRecipe_match<Op0_t, Op1_t, Opcode, /*Commutative*/ false,
                       TPInstruction>;

template <typename Op0_t, typename Op1_t, unsigned Opcode,
          bool Commutative = false>
using AllBinaryRecipe_match =
    BinaryRecipe_match<Op0_t, Op1_t, Opcode, Commutative, TPWidenRecipe,
                       TPReplicateRecipe, TPWidenCastRecipe, TPInstruction>;

template <unsigned Opcode, typename Op0_t>
inline UnaryTPInstruction_match<Op0_t, Opcode>
m_TPInstruction(const Op0_t &Op0) {
  return UnaryTPInstruction_match<Op0_t, Opcode>(Op0);
}

template <unsigned Opcode, typename Op0_t, typename Op1_t>
inline BinaryTPInstruction_match<Op0_t, Op1_t, Opcode>
m_TPInstruction(const Op0_t &Op0, const Op1_t &Op1) {
  return BinaryTPInstruction_match<Op0_t, Op1_t, Opcode>(Op0, Op1);
}

template <typename Op0_t>
inline UnaryTPInstruction_match<Op0_t, TPInstruction::Not>
m_Not(const Op0_t &Op0) {
  return m_TPInstruction<TPInstruction::Not>(Op0);
}

template <typename Op0_t>
inline UnaryTPInstruction_match<Op0_t, TPInstruction::BranchOnCond>
m_BranchOnCond(const Op0_t &Op0) {
  return m_TPInstruction<TPInstruction::BranchOnCond>(Op0);
}

template <typename Op0_t, typename Op1_t>
inline BinaryTPInstruction_match<Op0_t, Op1_t, TPInstruction::ActiveLaneMask>
m_ActiveLaneMask(const Op0_t &Op0, const Op1_t &Op1) {
  return m_TPInstruction<TPInstruction::ActiveLaneMask>(Op0, Op1);
}

template <typename Op0_t, typename Op1_t>
inline BinaryTPInstruction_match<Op0_t, Op1_t, TPInstruction::BranchOnCount>
m_BranchOnCount(const Op0_t &Op0, const Op1_t &Op1) {
  return m_TPInstruction<TPInstruction::BranchOnCount>(Op0, Op1);
}

template <unsigned Opcode, typename Op0_t>
inline AllUnaryRecipe_match<Op0_t, Opcode> m_Unary(const Op0_t &Op0) {
  return AllUnaryRecipe_match<Op0_t, Opcode>(Op0);
}

template <typename Op0_t>
inline AllUnaryRecipe_match<Op0_t, Instruction::Trunc>
m_Trunc(const Op0_t &Op0) {
  return m_Unary<Instruction::Trunc, Op0_t>(Op0);
}

template <typename Op0_t>
inline AllUnaryRecipe_match<Op0_t, Instruction::ZExt> m_ZExt(const Op0_t &Op0) {
  return m_Unary<Instruction::ZExt, Op0_t>(Op0);
}

template <typename Op0_t>
inline AllUnaryRecipe_match<Op0_t, Instruction::SExt> m_SExt(const Op0_t &Op0) {
  return m_Unary<Instruction::SExt, Op0_t>(Op0);
}

template <typename Op0_t>
inline match_combine_or<AllUnaryRecipe_match<Op0_t, Instruction::ZExt>,
                        AllUnaryRecipe_match<Op0_t, Instruction::SExt>>
m_ZExtOrSExt(const Op0_t &Op0) {
  return m_CombineOr(m_ZExt(Op0), m_SExt(Op0));
}

template <unsigned Opcode, typename Op0_t, typename Op1_t,
          bool Commutative = false>
inline AllBinaryRecipe_match<Op0_t, Op1_t, Opcode, Commutative>
m_Binary(const Op0_t &Op0, const Op1_t &Op1) {
  return AllBinaryRecipe_match<Op0_t, Op1_t, Opcode, Commutative>(Op0, Op1);
}

template <typename Op0_t, typename Op1_t>
inline AllBinaryRecipe_match<Op0_t, Op1_t, Instruction::Mul>
m_Mul(const Op0_t &Op0, const Op1_t &Op1) {
  return m_Binary<Instruction::Mul, Op0_t, Op1_t>(Op0, Op1);
}

template <typename Op0_t, typename Op1_t>
inline AllBinaryRecipe_match<Op0_t, Op1_t, Instruction::Mul,
                             /* Commutative =*/true>
m_c_Mul(const Op0_t &Op0, const Op1_t &Op1) {
  return m_Binary<Instruction::Mul, Op0_t, Op1_t, true>(Op0, Op1);
}

/// Match a binary OR operation. Note that while conceptually the operands can
/// be matched commutatively, \p Commutative defaults to false in line with the
/// IR-based pattern matching infrastructure. Use m_c_BinaryOr for a commutative
/// version of the matcher.
template <typename Op0_t, typename Op1_t, bool Commutative = false>
inline AllBinaryRecipe_match<Op0_t, Op1_t, Instruction::Or, Commutative>
m_BinaryOr(const Op0_t &Op0, const Op1_t &Op1) {
  return m_Binary<Instruction::Or, Op0_t, Op1_t, Commutative>(Op0, Op1);
}

template <typename Op0_t, typename Op1_t>
inline AllBinaryRecipe_match<Op0_t, Op1_t, Instruction::Or,
                             /*Commutative*/ true>
m_c_BinaryOr(const Op0_t &Op0, const Op1_t &Op1) {
  return m_BinaryOr<Op0_t, Op1_t, /*Commutative*/ true>(Op0, Op1);
}

template <typename Op0_t, typename Op1_t>
inline BinaryTPInstruction_match<Op0_t, Op1_t, TPInstruction::LogicalAnd>
m_LogicalAnd(const Op0_t &Op0, const Op1_t &Op1) {
  return m_TPInstruction<TPInstruction::LogicalAnd, Op0_t, Op1_t>(Op0, Op1);
}

struct TPCanonicalIVPHI_match {
  bool match(const TPValue *V) {
    auto *DefR = V->getDefiningRecipe();
    return DefR && match(DefR);
  }

  bool match(const TPRecipeBase *R) { return isa<TPCanonicalIVPHIRecipe>(R); }
};

inline TPCanonicalIVPHI_match m_CanonicalIV() {
  return TPCanonicalIVPHI_match();
}

template <typename Op0_t, typename Op1_t> struct TPScalarIVSteps_match {
  Op0_t Op0;
  Op1_t Op1;

  TPScalarIVSteps_match(Op0_t Op0, Op1_t Op1) : Op0(Op0), Op1(Op1) {}

  bool match(const TPValue *V) {
    auto *DefR = V->getDefiningRecipe();
    return DefR && match(DefR);
  }

  bool match(const TPRecipeBase *R) {
    if (!isa<TPScalarIVStepsRecipe>(R))
      return false;
    assert(R->getNumOperands() == 2 &&
           "TPScalarIVSteps must have exactly 2 operands");
    return Op0.match(R->getOperand(0)) && Op1.match(R->getOperand(1));
  }
};

template <typename Op0_t, typename Op1_t>
inline TPScalarIVSteps_match<Op0_t, Op1_t> m_ScalarIVSteps(const Op0_t &Op0,
                                                           const Op1_t &Op1) {
  return TPScalarIVSteps_match<Op0_t, Op1_t>(Op0, Op1);
}

} // namespace TPlanPatternMatch
} // namespace llvm

#endif // LLVM_TRANSFORM_TENSORIZE_TPLANPATTERNMATCH_H
