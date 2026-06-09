//==============================================================================
// Auto Generated Code for HexFlashAttention
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_hook_base.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

#include "hfa_inv_op.h"
#include "hfa_ops.h"
#include "hfa_utils.h"

BEGIN_PKG_OP_DEFINITION(PKG_FlashAttention);

static Qnn_Scalar_t sg_opDefaultScaleScalar = {
    .dataType = Qnn_DataType_t::QNN_DATATYPE_FLOAT_32,
    .floatValue = 0};
static Qnn_Param_t sg_opDefaultScale = {
    .paramType = QNN_PARAMTYPE_SCALAR,
    .scalarParam = sg_opDefaultScaleScalar};

template <typename TensorType>
GraphStatus flashattentionImpl(
    PlainFloat16Tensor_TCM& out_0,
    const PlainFloat16Tensor_TCM& query,
    const PlainFloat16Tensor& key,
    const PlainFloat16Tensor& value,
    const PlainFloat16Tensor& attn_mask,
    const PlainFloatTensor& scale,
    PlainFloat16Tensor_TCM& scratch);

// Handles one token query inference
template <typename TensorType>
GraphStatus flashattentionOneQImpl(
    PlainFloat16Tensor_TCM& out_0,
    const PlainFloat16Tensor_TCM& query,
    const PlainFloat16Tensor& key,
    const PlainFloat16Tensor& value,
    const PlainFloat16Tensor& attn_mask,
    const PlainFloatTensor& scale,
    PlainFloat16Tensor_TCM& scratch);

DEF_PACKAGE_OP((flashattentionImpl<Tensor>), "FlashAttention")

DEF_PACKAGE_OP((flashattentionOneQImpl<Tensor>), "FlashAttentionOneQ")

DEF_TENSOR_PROPERTIES(
    Op("FlashAttention", "query", "key", "value", "attn_mask", "scale"),
    Flat("*", "query", "key", "value", "attn_mask", "scale"),
    Tcm("*", "query"),
    MainMemory("..."))

DEF_TENSOR_PROPERTIES(
    Op("FlashAttentionOneQ", "query", "key", "value", "attn_mask", "scale"),
    Flat("*", "query", "key", "value", "attn_mask", "scale"),
    Tcm("*", "query"),
    MainMemory("..."))

#define CAST_FP16(ARG)  \
  WITH_SIZE(            \
      ARG,              \
      WITH_OUTPUT_TYPE( \
          DType::Float16, 0, 1.0f, Op(FROM_DEFAULT_PACKAGE("Cast"), ARG)))

// FP32 -> FP16 -> FP32
DEF_PACKAGE_OPTIMIZATION_WITH_FLAGS(
    GRAPH_CLEANUP,
    relaxed_precision_flag,
    Op("FlashAttention", "query", "key", "value", "attn_mask", "scale"),
    AND(EQ(DTYPE_OF("query"), DType::Float32),
        EQ(DTYPE_OF("key"), DType::Float32),
        EQ(DTYPE_OF("value"), DType::Float32),
        EQ(DTYPE_OF("attn_mask"), DType::Float32),
        EQ(DTYPE_OF("scale"), DType::Float32),
        EQ(DTYPE_OF("*"), DType::Float32)),
    WITH_OUTPUT_TYPE(
        DType::Float32,
        0,
        1.0f,
        Op(FROM_DEFAULT_PACKAGE("Cast"),
           WITH_SIZE(
               "*",
               WITH_OUTPUT_TYPE(
                   DType::Float16,
                   0,
                   1.0f,
                   Op("FlashAttention",
                      CAST_FP16("query"),
                      CAST_FP16("key"),
                      CAST_FP16("value"),
                      CAST_FP16("attn_mask"),
                      "scale"))))))

// Batch Tiling
DEF_PACKAGE_OPTIMIZATION(
    EARLY,
    Op("FlashAttention", "query", "key", "value", "attn_mask", "scale"),
    AND(GT(DIM_BATCHES("query"), 1),
        GT(DIM_BATCHES("key"), 1),
        GT(DIM_BATCHES("value"), 1)),
    AUTOSPLIT(
        0,
        "I",
        1,
        Op("FlashAttention",
           TYPICAL_SLICE("query", "I"),
           TYPICAL_SLICE("key", "I"),
           TYPICAL_SLICE("value", "I"),
           "attn_mask",
           "scale")))

// Head Tiling
DEF_PACKAGE_OPTIMIZATION(
    EARLY + 1,
    Op("FlashAttention", "query", "key", "value", "attn_mask", "scale"),
    AND(GT(DIM_HEIGHT("query"), 1),
        GT(DIM_HEIGHT("key"), 1),
        GT(DIM_HEIGHT("value"), 1)),
    AUTOSPLIT(
        1,
        "I",
        1,
        Op("FlashAttention",
           TYPICAL_SLICE("query", "I"),
           TYPICAL_SLICE("key", "I"),
           TYPICAL_SLICE("value", "I"),
           "attn_mask",
           "scale")))

// Q Seq Len Tiling
DEF_PACKAGE_OPTIMIZATION(
    EARLY + 2,
    Op("FlashAttention", "query", "key", "value", "attn_mask", "scale"),
    GT(DIM_WIDTH("query"), 64),
    AUTOSPLIT(
        2,
        "I",
        64,
        Op("FlashAttention",
           TYPICAL_SLICE("query", "I"),
           "key",
           "value",
           TYPICAL_SLICE("attn_mask", "I"),
           "scale")))

// After Q Seq Tiling, len < 8 get split into OneQ case
DEF_PACKAGE_OPTIMIZATION(
    EARLY + 3,
    Op("FlashAttention", "query", "key", "value", "attn_mask", "scale"),
    LT(DIM_WIDTH("query"), 8),
    AUTOSPLIT(
        2,
        "I",
        1,
        Op("FlashAttentionOneQ",
           TYPICAL_SLICE("query", "I"),
           "key",
           "value",
           TYPICAL_SLICE("attn_mask", "I"),
           "scale")))

DEF_PACKAGE_PARAM_ORDER("FlashAttention", "scale", false, &sg_opDefaultScale)

DEF_PACKAGE_PARAM_ORDER(
    "FlashAttentionOneQ",
    "scale",
    false,
    &sg_opDefaultScale)

template <typename TensorType>
GraphStatus flashattentionImpl(
    PlainFloat16Tensor_TCM& out_0,
    const PlainFloat16Tensor_TCM& query,
    const PlainFloat16Tensor& key,
    const PlainFloat16Tensor& value,
    const PlainFloat16Tensor& attn_mask,
    const PlainFloatTensor& scale,
    PlainFloat16Tensor_TCM& scratch) {
  auto [qk_emb_len, v_emb_len, kv_blocks, key_size, value_size] =
      get_comp_sizes<KV_BLOCK_LEN>(key, value);

  Float16* query_ptr = query.data_ptr();
  Float16* key_ptr = key.data_ptr();
  Float16* value_ptr = value.data_ptr();
  Float16* out_ptr = out_0.data_ptr();

  auto [out_acc_ptr, kv_ptr, qk_mul_ptr, tmp_ptr] =
      allocate_tcm_scratch_ptrs<Q_BLOCK_LEN, KV_BLOCK_LEN>(
          scratch, qk_emb_len, v_emb_len);

  const Float16 scale_val = get_fp16_adjusted_scale(scale);

  // running max: init to fp16 min value
  HVX_Vector max_vec = Q6_Vh_vsplat_R(0xFBFF);
  // running denominator: init to 0
  HVX_Vector l_vec = Q6_V_vzero();
  // out_acc: init to 0
  for (size_t i = 0; i < v_emb_len; ++i) {
    ((HVX_Vector*)out_acc_ptr)[i] = Q6_V_vzero();
  }

  for (size_t b = 0; b < kv_blocks; ++b) {
    auto [new_max_vec, new_l_vec] = hfa_process_block(
        query_ptr,
        key_ptr,
        value_ptr,
        out_acc_ptr,
        kv_ptr,
        qk_mul_ptr,
        tmp_ptr,
        max_vec,
        l_vec,
        scale_val,
        b,
        qk_emb_len,
        v_emb_len);

    // update running stats
    max_vec = new_max_vec;
    l_vec = new_l_vec;

    // increment ptrs
    key_ptr += key_size;
    value_ptr += value_size;
  }

  // acc *= 1/l
  HVX_Vector inv_l_vec = hvx_Vhf_exp2rsqrt_inv_Vhf(l_vec);
  hvx_Vhf_multimpy_VhfVhf((HVX_Vector*)out_acc_ptr, inv_l_vec, v_emb_len);

  // transpose to final output
  hvx_Vhf_mat_transpose64Nrx64(out_ptr, out_acc_ptr, v_emb_len, tmp_ptr);

  return GraphStatus::Success;
}

template <typename TensorType>
GraphStatus flashattentionOneQImpl(
    PlainFloat16Tensor_TCM& out_0,
    const PlainFloat16Tensor_TCM& query,
    const PlainFloat16Tensor& key,
    const PlainFloat16Tensor& value,
    const PlainFloat16Tensor& attn_mask,
    const PlainFloatTensor& scale,
    PlainFloat16Tensor_TCM& scratch) {
  auto [qk_emb_len, v_emb_len, kv_blocks, key_size, value_size] =
      get_comp_sizes<KV_BLOCK_LEN>(key, value);

  Float16* query_ptr = query.data_ptr();
  Float16* key_ptr = key.data_ptr();
  Float16* value_ptr = value.data_ptr();
  Float16* out_ptr = out_0.data_ptr();

  auto [_, kv_ptr, qk_mul_ptr, tmp_ptr] =
      allocate_tcm_scratch_ptrs<1, KV_BLOCK_LEN>(
          scratch, qk_emb_len, v_emb_len);

  const Float16 scale_val = get_fp16_adjusted_scale(scale);

  HVX_Vector max_vec = Q6_Vh_vsplat_R(0xFBFF);
  HVX_Vector l_vec = Q6_V_vzero();
  const size_t num_vecs = v_emb_len / 64;
  for (size_t i = 0; i < num_vecs; ++i) {
    ((HVX_Vector*)out_ptr)[i] = Q6_V_vzero();
  }

  for (size_t b = 0; b < kv_blocks; ++b) {
    auto [new_max_vec, new_l_vec] = hfa_process_block_oneq(
        query_ptr,
        key_ptr,
        value_ptr,
        out_ptr,
        kv_ptr,
        qk_mul_ptr,
        tmp_ptr,
        max_vec,
        l_vec,
        scale_val,
        b,
        qk_emb_len,
        v_emb_len);

    max_vec = new_max_vec;
    l_vec = new_l_vec;

    key_ptr += key_size;
    value_ptr += value_size;
  }

  HVX_Vector inv_l_vec = hvx_Vhf_exp2rsqrt_inv_Vhf(l_vec);
  hvx_Vhf_multimpy_VhfVhf((HVX_Vector*)out_ptr, inv_l_vec, num_vecs);

  return GraphStatus::Success;
}

#ifndef PREPARE_DISABLED

namespace {
class FlashAttentionImplConstructorHook : public hnnx::OpHookBase {
  // This is called after the output tensors are created, but before allocation.
  virtual GraphStatus pre_allocate(hnnx::OpIoPtrs const& iop, Op& op)
      const override {
    const size_t qk_emb_len = op.get_input(0)->dim(3);
    const size_t v_emb_len = op.get_input(2)->dim(3);
    const size_t max_emb_len = max_i32(qk_emb_len, v_emb_len);

    const size_t block0_size = v_emb_len * Q_BLOCK_LEN;
    const size_t block1_size = KV_BLOCK_LEN * max_emb_len;
    const size_t block2_size = Q_BLOCK_LEN * KV_BLOCK_LEN;
    const size_t block3_size = Q_BLOCK_LEN * max_emb_len;

    size_t new_dims[4] = {
        1, 1, 1, block0_size + block1_size + block2_size + block3_size};
    GraphStatus result =
        hnnx::change_output_tensor_shape(op, 1, iop.graph(), 4, new_dims);
    if (result != GraphStatus::Success) {
      errlog("!! change_output_tensor_shape failed");
    }
    return result;
  }
};

class FlashAttentionOneQImplConstructorHook : public hnnx::OpHookBase {
  // This is called after the output tensors are created, but before allocation.
  virtual GraphStatus pre_allocate(hnnx::OpIoPtrs const& iop, Op& op)
      const override {
    const size_t qk_emb_len = op.get_input(0)->dim(3);
    const size_t v_emb_len = op.get_input(2)->dim(3);
    const size_t max_emb_len = max_i32(qk_emb_len, v_emb_len);

    const size_t block0_size = 0; // block 0 is not needed
    const size_t block1_size = KV_BLOCK_LEN * max_emb_len;
    const size_t block2_size = 1 * KV_BLOCK_LEN;
    const size_t block3_size = 1 * max_emb_len;

    size_t new_dims[4] = {
        1, 1, 1, block0_size + block1_size + block2_size + block3_size};
    GraphStatus result =
        hnnx::change_output_tensor_shape(op, 1, iop.graph(), 4, new_dims);
    if (result != GraphStatus::Success) {
      errlog("!! change_output_tensor_shape failed");
    }
    return result;
  }
};
} // namespace

CTOR_OPHOOK((flashattentionImpl<Tensor>), FlashAttentionImplConstructorHook)
CTOR_OPHOOK(
    (flashattentionOneQImpl<Tensor>),
    FlashAttentionOneQImplConstructorHook)
#endif

END_PKG_OP_DEFINITION(PKG_FlashAttention);