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

static float flashattentionCostFunc(const Op* op);

DEF_PACKAGE_OP((flashattentionImpl<Tensor>), "FlashAttention")

DEF_TENSOR_PROPERTIES(
    Op("FlashAttention", "query", "key", "value", "attn_mask", "scale"),
    Flat("*", "query", "key", "value", "attn_mask", "scale"),
    Tcm("*", "query"),
    MainMemory("..."))

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

DEF_PACKAGE_PARAM_ORDER("FlashAttention", "scale", false, &sg_opDefaultScale)

template <typename TensorType>
GraphStatus flashattentionImpl(
    PlainFloat16Tensor_TCM& out_0,
    const PlainFloat16Tensor_TCM& query,
    const PlainFloat16Tensor& key,
    const PlainFloat16Tensor& value,
    const PlainFloat16Tensor& attn_mask,
    const PlainFloatTensor& scale,
    PlainFloat16Tensor_TCM& scratch) {
  // log_hfa_input(out_0, query, key, value, attn_mask, scale, scratch);

  auto [qk_emb_len, v_emb_len, kv_blocks, key_size, value_size, qkmul_size] =
      get_comp_sizes<64, 64>(key, value);

  Float16* query_ptr = query.data_ptr();
  Float16* key_ptr = key.data_ptr();
  Float16* value_ptr = value.data_ptr();
  Float16* out_ptr = out_0.data_ptr();

  auto [out_acc_ptr, kv_ptr, qk_mul_ptr, tmp_ptr] =
      allocate_tcm_scratch_ptrs<64, 64>(scratch, qk_emb_len, v_emb_len);

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
        v_emb_len,
        key_size,
        value_size,
        qkmul_size);

    // update running stats
    max_vec = new_max_vec;
    l_vec = new_l_vec;

    // increment ptrs
    key_ptr += key_size;
    value_ptr += value_size;
  }

  // acc *= 1/l
  HVX_Vector inv_l_vec = hvx_Vhf_exp2rsqrt_inv_Vhf(l_vec);
  hvx_mpy_multiple((HVX_Vector*)out_acc_ptr, inv_l_vec, v_emb_len);

  // transpose to final output
  mat_transpose_64Nx64_fp16(out_ptr, out_acc_ptr, v_emb_len / 64, tmp_ptr);

  return GraphStatus::Success;
}

#ifndef PREPARE_DISABLED

namespace {
class FlashAttentionImplConstructorHook : public hnnx::OpHookBase {
  // This is called after the output tensors are created, but before allocation.
  virtual GraphStatus pre_allocate(hnnx::OpIoPtrs const& iop, Op& op)
      const override {
    const size_t v_emb_len = op.get_input(2)->dim(3);
    const size_t max_emb_len = max_i32(op.get_input(0)->dim(3), v_emb_len);

    const size_t block0_size = v_emb_len * 64;
    const size_t block1_size = 64 * max_emb_len;
    const size_t block2_size = 64 * 64;
    const size_t block3_size = 64 * max_emb_len;

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
#endif

__attribute__((unused)) static float flashattentionCostFunc(const Op* op) {
  float cost = 0.0;
  return cost;
}

END_PKG_OP_DEFINITION(PKG_FlashAttention);