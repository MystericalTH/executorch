//==============================================================================
// Auto Generated Code for HexFlashAttention
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

BEGIN_PKG_OP_DEFINITION(PKG_FlashAttention);

static Qnn_Scalar_t sg_opDefaultScaleScalar = {
    .dataType = Qnn_DataType_t::QNN_DATATYPE_FLOAT_32,
    .floatValue = 0};
static Qnn_Param_t sg_opDefaultScale = {
    .paramType = QNN_PARAMTYPE_SCALAR,
    .scalarParam = sg_opDefaultScaleScalar};

template <typename TensorType>
GraphStatus flashattentionImpl(
    TensorType& out_0,
    const TensorType& query,
    const TensorType& key,
    const TensorType& value,
    const TensorType& attn_mask,
    const PlainFloatTensor& scale);

static float flashattentionCostFunc(const Op* op);

DEF_PACKAGE_OP((flashattentionImpl<Tensor>), "FlashAttention")

DEF_TENSOR_PROPERTIES(
    Op("FlashAttention", "query", "key", "value", "attn_mask", "scale"),
    Flat("*", "query", "key", "value", "attn_mask"),
    MainMemory("query", "key", "value", "attn_mask", "scale"))

DEF_PACKAGE_PARAM_ORDER("FlashAttention", "scale", false, &sg_opDefaultScale)

template <typename TensorType>
GraphStatus flashattentionImpl(
    TensorType& out_0,
    const TensorType& query,
    const TensorType& key,
    const TensorType& value,
    const TensorType& attn_mask,
    const PlainFloatTensor& scale) {
  return GraphStatus::Success;
}

__attribute__((unused)) static float flashattentionCostFunc(const Op* op) {
  float cost = 0.0;
  return cost;
}

END_PKG_OP_DEFINITION(PKG_FlashAttention);