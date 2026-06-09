//==============================================================================
// Auto Generated Code for TMANOpPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

#include "hvx_funcs.h"

BEGIN_PKG_OP_DEFINITION(PKG_TMANPrecompute);

static Qnn_Scalar_t sg_opDefaultT_Group_SizeScalar = {
    .dataType = Qnn_DataType_t::QNN_DATATYPE_INT_32,
    .int32Value = 64};
static Qnn_Param_t sg_opDefaultT_Group_Size = {
    .paramType = QNN_PARAMTYPE_SCALAR,
    .scalarParam = sg_opDefaultT_Group_SizeScalar};
static Qnn_Scalar_t sg_opDefaultT_BitsScalar = {
    .dataType = Qnn_DataType_t::QNN_DATATYPE_INT_32,
    .int32Value = 2};
static Qnn_Param_t sg_opDefaultT_Bits = {
    .paramType = QNN_PARAMTYPE_SCALAR,
    .scalarParam = sg_opDefaultT_BitsScalar};
static Qnn_Scalar_t sg_opDefaultT_SymmetricScalar = {
    .dataType = Qnn_DataType_t::QNN_DATATYPE_INT_32,
    .int32Value = 0};
static Qnn_Param_t sg_opDefaultT_Symmetric = {
    .paramType = QNN_PARAMTYPE_SCALAR,
    .scalarParam = sg_opDefaultT_SymmetricScalar};

template <typename TensorType>
GraphStatus tmanprecomputeImpl(
    TensorType& l,
    const TensorType& x,
    const Int32Tensor& t_group_size,
    const Int32Tensor& t_bits,
    const Int32Tensor& t_symmetric);

static float tmanprecomputeCostFunc(const Op* op);

DEF_PACKAGE_OP((tmanprecomputeImpl<Tensor>), "TMANPrecompute")

DEF_TENSOR_PROPERTIES(
    Op("TMANPrecompute", "x", "t_group_size", "t_bits", "t_symmetric"),
    Flat("*", "x"),
    MainMemory("t_group_size", "t_bits", "t_symmetric"),
    Tcm("*", "x"))

DEF_PACKAGE_PARAM_ORDER(
    "TMANPrecompute",
    "t_group_size",
    false,
    &sg_opDefaultT_Group_Size,
    "t_bits",
    false,
    &sg_opDefaultT_Bits,
    "t_symmetric",
    false,
    &sg_opDefaultT_Symmetric)

template <typename TensorType>
GraphStatus tmanprecomputeImpl(
    TensorType& l,
    const TensorType& x,
    const Int32Tensor& t_group_size,
    const Int32Tensor& t_bits,
    const Int32Tensor& t_symmetric) {
  using LType = int16_t; // LUT entries
  using XType = __fp16; // activation

  // activations per quant scale chunk
  constexpr int32_t ACT_GROUP_SIZE = 256;
  // weights per LUT group
  constexpr int32_t LUT_G = 4;
  // 2^LUT_G possible binary patterns
  constexpr int32_t LUT_SIZE = 16;

  // [N, H, n, k]
  const int32_t gemm_k = x.dims()[3];
  const int32_t gemm_n = x.dims()[2];

  const int32_t group_size = ((const int32_t*)t_group_size.raw_data_const())[0];
  // (uses) zero_point = true -> asymmetric
  const bool zero_point =
      ((const int32_t*)t_symmetric.raw_data_const())[0] == 0;

  // no. of LUT entries
  const int32_t l_size = gemm_k / LUT_G * LUT_SIZE;
  // activation scale group size
  const int32_t real_act_group_size = (group_size == 0) ? -1 : ACT_GROUP_SIZE;
  // no. of scale values (-1 means global scale)
  const int32_t ls_size =
      (real_act_group_size == -1) ? 1 : (gemm_k / real_act_group_size);

  const XType* x_ptr = (const XType*)x.raw_data_const();
  // output partition: LUT entries | scales | bias
  // output buffer l
  LType* l_ptr = (LType*)l.raw_data();
  // output buffer ls, offset after l
  float* ls_ptr = (float*)(l_ptr + l_size);
  // output buffer lb, offset after ls
  float* lb_ptr = ls_ptr + MAX(ls_size, 128 / sizeof(float));

  if (zero_point && group_size == 64) {
    // w2g64, symmetric=False
    hvx_lut_ctor<LType, XType, ACT_GROUP_SIZE, 64, true, LUT_G>(
        gemm_k, gemm_n, x_ptr, l_ptr, ls_ptr, lb_ptr);
  } else if (!zero_point && group_size == 128) {
    // w4g128, symmetric=True
    hvx_lut_ctor<LType, XType, ACT_GROUP_SIZE, 128, false, LUT_G>(
        gemm_k, gemm_n, x_ptr, l_ptr, ls_ptr, lb_ptr);
  } else if (zero_point && group_size == 128) {
    // w4g128, symmetric=False
    hvx_lut_ctor<LType, XType, ACT_GROUP_SIZE, 128, true, LUT_G>(
        gemm_k, gemm_n, x_ptr, l_ptr, ls_ptr, lb_ptr);
  } else if (!zero_point && group_size == 0) {
    // bitnet
    hvx_lut_ctor<LType, XType, -1, 0, false, LUT_G>(
        gemm_k, gemm_n, x_ptr, l_ptr, ls_ptr, lb_ptr);
  } else {
    return GraphStatus::ErrorDimensions;
  }

  return GraphStatus::Success;
}

__attribute__((unused)) static float tmanprecomputeCostFunc(const Op* op) {
  float cost = 0.0;
  return cost;
}

END_PKG_OP_DEFINITION(PKG_TMANPrecompute);