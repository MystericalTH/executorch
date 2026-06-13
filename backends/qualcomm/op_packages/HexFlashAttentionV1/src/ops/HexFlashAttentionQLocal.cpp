//==============================================================================
// Auto Generated Code for HexFlashAttentionV1
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_hook_base.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

#include "HFAQ/allocate.h"
#include "HFAQ/debug.h"
#include "HFAQ/local_ops.h"

#include "constant.h"
#include "hvx/hvx_matmul_ops.h"
#include "hvx/hvx_transpose_ops.h"
#include "package_optimization.h"

BEGIN_PKG_OP_DEFINITION(PKG_HexFlashAttentionQLocal);

// Compute HFAQ local block
// In (FP16):
//    - Q(1, H, 1, D)
//    - K(1, H, pS, D)
//    - V(1, H, pS, Dv)
//
// Out (FP16, Transposed):
//    - Local(1, Dv + 2, 1, H)
template <typename TensorType>
GraphStatus hexflashattentionqlocalImpl(
    PlainFloatTensor_TCM& out_0,
    const PlainFloat16Tensor_TCM& query,
    const PlainFloat16Tensor_TCM& key,
    const PlainFloat16Tensor_TCM& value,
    const PlainFloat16Tensor_TCM& attn_mask,
    const PlainFloatTensor_TCM& scale,
    PlainFloat16Tensor_TCM& scratch);

DEF_PACKAGE_OP(
    (hexflashattentionqlocalImpl<PlainFloatTensor_TCM>),
    "HexFlashAttentionQLocal")

#define ORIGINAL_OP \
  Op("HexFlashAttentionQLocal", "query", "key", "value", "attn_mask", "scale")

DEF_TENSOR_PROPERTIES(
    ORIGINAL_OP,
    Flat("*", "query", "key", "value", "attn_mask", "scale"),
    Tcm("*", "query", "key", "value", "attn_mask", "scale"))

RELAXED_PRECISION_OP(
    ORIGINAL_OP,
    AND(IS_FLOAT32_ALL("query", "key", "value", "attn_mask", "scale")),
    Op("HexFlashAttentionQLocal",
       CAST_FP16("query"),
       CAST_FP16("key"),
       CAST_FP16("value"),
       CAST_FP16("attn_mask"),
       "scale"))

DEF_PACKAGE_OPTIMIZATION(
    EARLY,
    ORIGINAL_OP,
    HFAQ_SHOULD_TILE_KV,
    Op("HexFlashAttentionQMerge",
       HFAQ_LOCAL_RECURSIVE_OP(0),
       HFAQ_LOCAL_RECURSIVE_OP(1)))

DEF_PACKAGE_PARAM_ORDER("HexFlashAttentionQLocal", "scale", true, nullptr)

template <typename TensorType>
GraphStatus hexflashattentionqlocalImpl(
    PlainFloatTensor_TCM& out_0,
    const PlainFloat16Tensor_TCM& query,
    const PlainFloat16Tensor_TCM& key,
    const PlainFloat16Tensor_TCM& value,
    const PlainFloat16Tensor_TCM& attn_mask,
    const PlainFloatTensor_TCM& scale,
    PlainFloat16Tensor_TCM& scratch) {
  log_hfaq_local_memfp(out_0, query, key, value, attn_mask, scale, scratch);

  const auto qk_emb = query.dim(3);
  const auto v_emb = value.dim(3);
  const auto q_head_stride = 1 * qk_emb;
  const auto k_head_stride = qk_emb * HFAQ_KV_SEQ_PROC_TILE;
  auto q_ptr = query.data_ptr();
  auto k_ptr = key.data_ptr();
  auto v_ptr = value.data_ptr();

  const auto out_acc_size = HFAQ_ACC_HEAD_TILE * v_emb;
  auto out_ptr = out_0.data_ptr();
  auto out_max_ptr = out_ptr + out_acc_size;
  auto out_l_ptr = out_max_ptr + HFAQ_ACC_HEAD_TILE;

  const auto [scr_block_0, scr_block_1] = hfaq_local_scratch_blocks(qk_emb);
  auto scr_block_0_ptr = scratch.data_ptr();
  auto scr_block_1_ptr = scr_block_0_ptr + scr_block_0;

  // case HFAQ_KV_SEQ_PROC_TILE = HFAQ_KV_SEQ_TILE = 64
  static_assert(
      HFAQ_KV_SEQ_PROC_TILE == HFAQ_KV_SEQ_TILE,
      "this implementation requires proc tile == graph tile");

  const auto rscale = *(uint32_t*)scale.data_ptr();
  auto attn_mask_vec = *(HVX_Vector*)attn_mask.data_ptr();

  // for each head do: scale*matmul(Q, K.T) + attn_mask
  auto k_t_ptr = scr_block_0_ptr;
  auto att_row_ptr = scr_block_1_ptr;

  for (uint16_t h = 0; h < HFAQ_ACC_HEAD_TILE; ++h) {
    hvx_mat_transposeMxN_Vhf(k_t_ptr, k_ptr, HFAQ_KV_SEQ_PROC_TILE, qk_emb);
    hvx_Vhf_matmul1x64N_Vhf(
        att_row_ptr, q_ptr, k_t_ptr, qk_emb, HFAQ_KV_SEQ_PROC_TILE, rscale);
    *(HVX_Vector*)att_row_ptr =
        Q6_Vhf_vadd_VhfVhf(*(HVX_Vector*)att_row_ptr, attn_mask_vec);

    q_ptr += q_head_stride; // move to next q head
    k_ptr += k_head_stride; // move to next k head
    att_row_ptr += HFAQ_KV_SEQ_PROC_TILE; // move to next head row
  }

  auto att_t_ptr = scr_block_1_ptr;

  // transpose (32, 64) -> (64, 32)
  hvx_mat_transpose64x64_Vhf(att_t_ptr, att_t_ptr);

  auto att_vec_ptr = (HVX_Vector*)att_t_ptr;
  auto local_max_vec = Q6_V_vsplat_R(SF_NEG_INF);
  auto local_l_vec = Q6_V_vzero();

  upcast_scale_max_att_t(att_vec_ptr, &local_max_vec);

  norm_exp_l_att_t(att_vec_ptr, local_max_vec, &local_l_vec);

  auto att_ptr = (float*)scr_block_0_ptr;

  // transpose back sf(64, 32) -> sf(32, 64)
  hvx_mat_transposeMxN_Vsf(
      att_ptr, (float*)att_t_ptr, HFAQ_KV_SEQ_PROC_TILE, HFAQ_ACC_HEAD_TILE);

  auto att_x_v_ptr = (float*)scr_block_1_ptr;

  // for each head do: matmul(ATT, V)
  att_heads_matmul_v(att_x_v_ptr, att_ptr, v_ptr, v_emb);

  // transpose for merging
  hvx_mat_transposeMxN_Vsf(out_ptr, att_x_v_ptr, HFAQ_ACC_HEAD_TILE, v_emb);

  *(HVX_Vector*)out_max_ptr = local_max_vec;
  *(HVX_Vector*)out_l_ptr = local_l_vec;

  return GraphStatus::Success;
}

#ifndef PREPARE_DISABLED
namespace {
class FlashAttentionQLocalImplConstructorHook : public hnnx::OpHookBase {
  // This is called after the output tensors are created, but before
  // allocation.
  virtual GraphStatus pre_allocate(hnnx::OpIoPtrs const& iop, Op& op)
      const override {
    const size_t qk_emb = op.get_input(1)->dim(3);
    auto [block_0, block_1] = hfaq_local_scratch_blocks(qk_emb);
    size_t new_dims[4] = {1, 1, 1, block_0 + block_1};
    GraphStatus result =
        hnnx::change_output_tensor_shape(op, 1, iop.graph(), 4, new_dims);
    if (result != GraphStatus::Success) {
      errlog("!! change_output_tensor_shape failed");
    }
    return result;
  }
};
} // namespace

CTOR_OPHOOK(
    (hexflashattentionqlocalImpl<PlainFloatTensor_TCM>),
    FlashAttentionQLocalImplConstructorHook)
#endif

END_PKG_OP_DEFINITION(PKG_HexFlashAttentionQLocal);