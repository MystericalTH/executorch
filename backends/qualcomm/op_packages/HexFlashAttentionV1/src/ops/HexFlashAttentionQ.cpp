//==============================================================================
// Auto Generated Code for HexFlashAttentionV1
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

#include "package_optimization.h"

BEGIN_PKG_OP_DEFINITION(PKG_HexFlashAttentionQ);

static Qnn_Scalar_t sg_opDefaultEnable_GqaScalar = {
    .dataType = Qnn_DataType_t::QNN_DATATYPE_INT_32,
    .int32Value = 0};
static Qnn_Param_t sg_opDefaultEnable_Gqa = {
    .paramType = QNN_PARAMTYPE_SCALAR,
    .scalarParam = sg_opDefaultEnable_GqaScalar};

template <typename TensorType>
GraphStatus hexflashattentionqImpl(
    TensorType& out_0,
    const TensorType& query,
    const TensorType& key,
    const TensorType& value,
    const TensorType& attn_mask,
    const PlainFloatTensor& scale,
    const Int32Tensor& enable_gqa);

DEF_PACKAGE_OP((hexflashattentionqImpl<Tensor>), "HexFlashAttentionQ")

#define ORIGINAL_OP        \
  Op("HexFlashAttentionQ", \
     "query",              \
     "key",                \
     "value",              \
     "attn_mask",          \
     "scale",              \
     "enable_gqa")

RELAXED_PRECISION_OP(
    ORIGINAL_OP,
    AND(IS_FLOAT32_ALL("query", "key", "value", "attn_mask", "scale")),
    Op("HexFlashAttentionQ",
       CAST_FP16("query"),
       CAST_FP16("key"),
       CAST_FP16("value"),
       CAST_FP16("attn_mask"),
       "scale",
       "enable_gqa"))

DEF_PACKAGE_OPTIMIZATION(
    EARLY,
    ORIGINAL_OP,
    OK,
    Op("HexFlashAttentionQFinalize",
       WITH_SIZE(
           HFAQ_LOCAL_OUTPUT_SHAPE,
           Op("HexFlashAttentionQLocal",
              "query",
              "key",
              "value",
              "attn_mask",
              "scale"))))

DEF_PACKAGE_PARAM_ORDER(
    "HexFlashAttentionQ",
    "scale",
    false,
    nullptr,
    "enable_gqa",
    false,
    &sg_opDefaultEnable_Gqa)

template <typename TensorType>
GraphStatus hexflashattentionqImpl(
    TensorType& out_0,
    const TensorType& query,
    const TensorType& key,
    const TensorType& value,
    const TensorType& attn_mask,
    const PlainFloatTensor& scale,
    const Int32Tensor& enable_gqa) {
  return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_HexFlashAttentionQ);