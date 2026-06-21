//==============================================================================
// Auto Generated Code for HexFlashAttentionV1
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

#include "constant.h"
#include "package_optimization.h"

BEGIN_PKG_OP_DEFINITION(PKG_HexFlashAttentionV1);

static Qnn_Scalar_t sg_opDefaultIs_CausalScalar = {
    .dataType = Qnn_DataType_t::QNN_DATATYPE_INT_32,
    .int32Value = 0};
static Qnn_Param_t sg_opDefaultIs_Causal = {
    .paramType = QNN_PARAMTYPE_SCALAR,
    .scalarParam = sg_opDefaultIs_CausalScalar};
static Qnn_Scalar_t sg_opDefaultEnable_GqaScalar = {
    .dataType = Qnn_DataType_t::QNN_DATATYPE_INT_32,
    .int32Value = 0};
static Qnn_Param_t sg_opDefaultEnable_Gqa = {
    .paramType = QNN_PARAMTYPE_SCALAR,
    .scalarParam = sg_opDefaultEnable_GqaScalar};

template <typename TensorType>
GraphStatus hexflashattentionv1Impl(
    PlainFloatTensor& out_0,
    const PlainFloatTensor& query,
    const PlainFloatTensor& key,
    const PlainFloatTensor& value,
    const PlainFloatTensor& attn_mask,
    const Int32Tensor& is_causal,
    const PlainFloatTensor& scale,
    const Int32Tensor& enable_gqa);

DEF_PACKAGE_OP(
    (hexflashattentionv1Impl<PlainFloatTensor>),
    "HexFlashAttentionV1")

#define ORIGINAL_OP         \
  Op("HexFlashAttentionV1", \
     "query",               \
     "key",                 \
     "value",               \
     "attn_mask",           \
     "is_causal",           \
     "scale",               \
     "enable_gqa")

DEF_PACKAGE_OPTIMIZATION(
    EARLY,
    ORIGINAL_OP,
    EQ(DIM_WIDTH("query"), 1),
    AUTOSPLIT(
        1,
        "I",
        HFAQ_INIT_HEAD_TILE,
        Op("HexFlashAttentionQ",
           TYPICAL_SLICE(CAST_FP16("query"), "I"),
           TYPICAL_SLICE(CAST_FP16("key"), "I"),
           TYPICAL_SLICE(CAST_FP16("value"), "I"),
           CAST_FP16("attn_mask"),
           "scale",
           "enable_gqa")))

DEF_PACKAGE_PARAM_ORDER(
    "HexFlashAttentionV1",
    "is_causal",
    false,
    &sg_opDefaultIs_Causal,
    "scale",
    true,
    nullptr,
    "enable_gqa",
    false,
    &sg_opDefaultEnable_Gqa)

template <typename TensorType>
GraphStatus hexflashattentionv1Impl(
    PlainFloatTensor& out_0,
    const PlainFloatTensor& query,
    const PlainFloatTensor& key,
    const PlainFloatTensor& value,
    const PlainFloatTensor& attn_mask,
    const Int32Tensor& is_causal,
    const PlainFloatTensor& scale,
    const Int32Tensor& enable_gqa) {
  return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_HexFlashAttentionV1);