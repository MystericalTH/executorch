#pragma once

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

#include "hfa_exp_op.h"
#include "hfa_hvx_ops.h"
#include "hfa_mat_ops.h"
#include "hfa_matmul.op.h"

constexpr uint16_t Q_BLOCK_LEN = 64;
constexpr uint16_t KV_BLOCK_LEN = 64;

/**
 * @returns
 * new_max_vec, new_l_vec
 */
static inline std::array<HVX_Vector, 2> hfa_process_block(
    Float16* query_ptr,
    const Float16* mm_key_ptr,
    const Float16* mm_value_ptr,
    Float16* out_acc_ptr,
    Float16* kv_ptr,
    Float16* qk_mul_ptr,
    Float16* tmp_ptr,
    const HVX_Vector max_vec,
    const HVX_Vector l_vec,
    const Float16 scale_val,
    const size_t curr_b,
    const size_t qk_emb_len,
    const size_t v_emb_len) {
  auto acc_vec_ptr = (HVX_Vector*)out_acc_ptr;
  auto qk_mul_vec_ptr = (HVX_Vector*)qk_mul_ptr;

  // load K.T via transpose to TCM directly (no need to load + transpose)
  // kv_ptr(T) <- mm_key_ptr
  hvx_Vhf_mat_transpose64x64Nc(kv_ptr, mm_key_ptr, qk_emb_len / 64);

  // scale K
  hvx_Vhf_multimpy_VhfVhf(
      (HVX_Vector*)kv_ptr, Q6_Vh_vsplat_R(scale_val.raw()), qk_emb_len);

  // Q @ K.t
  // qk_mul_ptr <- query_ptr, kv_ptr
  hvx_Vhf_matmul_VhfVhf(
      qk_mul_ptr, query_ptr, kv_ptr, Q_BLOCK_LEN, qk_emb_len, KV_BLOCK_LEN);

  // need to *=attn_mask here

  // transpose to align for row-wise ops
  hvx_Vhf_mat_transpose64x64(qk_mul_ptr);

  // row-wise max
  HVX_Vector new_max_vec =
      hvx_Vhf_redmax_VhfVhf<KV_BLOCK_LEN>(max_vec, qk_mul_vec_ptr);

  // exp2(x-max)
  hvx_Vhf_multisub_VhfVhf<KV_BLOCK_LEN>(qk_mul_vec_ptr, new_max_vec);
#pragma unroll
  for (uint16_t i = 0; i < KV_BLOCK_LEN; ++i) {
    qk_mul_vec_ptr[i] = hvx_Vhf_horner_exp2_Vhf(qk_mul_vec_ptr[i]);
  }

  HVX_Vector new_l_vec;
  // no need to do adj for first block
  if (curr_b == 0) {
    new_l_vec = hvx_Vhf_redadd_VhfVhf<KV_BLOCK_LEN>(l_vec, qk_mul_vec_ptr);
  } else {
    // adj_sfmx_vec = exp2(old_max - new_max)
    HVX_Vector max_diff = Q6_Vhf_vsub_VhfVhf(max_vec, new_max_vec);
    HVX_Vector adj_sfmx_vec = hvx_Vhf_horner_exp2_Vhf(max_diff);
    // l_vec += l_vec*adj_sfmx_vec + new_l_vec
    HVX_Vector adj_l_vec = Q6_Vhf_vmpy_VhfVhf(l_vec, adj_sfmx_vec);
    new_l_vec = hvx_Vhf_redadd_VhfVhf<KV_BLOCK_LEN>(adj_l_vec, qk_mul_vec_ptr);

    // prev_acc *= adj_scale
    hvx_Vhf_multimpy_VhfVhf(acc_vec_ptr, adj_sfmx_vec, v_emb_len);
  }

  /**
   * Here we compute (ATT @ V).T (HVX friendly)
   * (ATT @ V).T = V.T @ ATT.T
   * Note: ATT.T is already computed in the previous step
   */
  // load V.T to TCM: kv_ptr(T) <- mm_value_ptr
  hvx_Vhf_mat_transpose64x64Nc(kv_ptr, mm_value_ptr, v_emb_len / KV_BLOCK_LEN);
  // V.T @ ATT.T: tmp_ptr(T) <- kv_ptr(T), qk_mul_ptr(T)
  hvx_Vhf_matmul_VhfVhf(
      tmp_ptr, kv_ptr, qk_mul_ptr, v_emb_len, KV_BLOCK_LEN, Q_BLOCK_LEN);

  // add to acc: acc_vec_ptr(T) += tmp_ptr(T)
  hvx_Vhf_accadd_VhfVhf(acc_vec_ptr, (HVX_Vector*)tmp_ptr, v_emb_len);

  return {new_max_vec, new_l_vec};
}