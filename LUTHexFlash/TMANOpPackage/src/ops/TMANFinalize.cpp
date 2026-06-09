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

BEGIN_PKG_OP_DEFINITION(PKG_TMANFinalize);

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
GraphStatus tmanfinalizeImpl(
    TensorType& y,
    const TensorType& c,
    const Int32Tensor& t_group_size,
    const Int32Tensor& t_bits,
    const Int32Tensor& t_symmetric);

static float tmanfinalizeCostFunc(const Op* op);

DEF_PACKAGE_OP((tmanfinalizeImpl<Tensor>), "TMANFinalize")

DEF_TENSOR_PROPERTIES(
    Op("TMANFinalize", "c", "t_group_size", "t_bits", "t_symmetric"),
    Flat("*", "c"),
    MainMemory("*", "t_group_size", "t_bits", "t_symmetric"),
    Tcm("c"))

DEF_PACKAGE_PARAM_ORDER(
    "TMANFinalize",
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
GraphStatus tmanfinalizeImpl(
    TensorType& y,
    const TensorType& c,
    const Int32Tensor& t_group_size,
    const Int32Tensor& t_bits,
    const Int32Tensor& t_symmetric) {
  using XType = __fp16;
  using CType = float;

  const int32_t gemm_m = y.dims()[3];
  const int32_t gemm_n = y.dims()[2];

  const int32_t bits = ((const int32_t*)t_bits.raw_data_const())[0];

  const CType* c_ptr = (const CType*)c.raw_data_const();
  XType* y_ptr = (XType*)y.raw_data();

  if (bits == 2) {
    hvx_bit_serial<XType, CType, 2>(gemm_m, gemm_n, c_ptr, y_ptr);
  } else if (bits == 4) {
    hvx_bit_serial<XType, CType, 4>(gemm_m, gemm_n, c_ptr, y_ptr);
  } else {
    return GraphStatus::ErrorDimensions;
  }

  return GraphStatus::Success;
}

__attribute__((unused)) static float tmanfinalizeCostFunc(const Op* op) {
  float cost = 0.0;
  return cost;
}

END_PKG_OP_DEFINITION(PKG_TMANFinalize);