#pragma once

#include "constant.h"
#include "hvx/hvx_exp_ops.h"
#include "hvx/hvx_matmul_ops.h"
#include "hvx/hvx_transpose_ops.h"

// for each head do:
// - scale * Q x K.T
// - += attn_mask
static inline void q_heads_scale_matmul_kt_mask(
    Float16* att_row_ptr,
    Float16* q_ptr,
    Float16* k_ptr,
    const HVX_Vector attn_mask_vec,
    const uint32_t rscale,
    const size_t num_heads,
    const size_t num_kv_heads,
    const size_t qk_emb,
    Float16* scr_block_ptr) {
  const HVX_Vector scaleline =
      Q6_Vqf32_vadd_VsfVsf(Q6_V_vsplat_R(rscale), Q6_V_vzero());
  const size_t group = num_heads / num_kv_heads;
  for (uint16_t h = 0; h < num_heads; ++h) {
    hvx_Vhf_matmul1x64N_transpose_Vhf(
        att_row_ptr,
        q_ptr,
        k_ptr,
        qk_emb,
        HFAQ_KV_SEQ_TILE,
        scaleline,
        scr_block_ptr);

    *(HVX_Vector*)att_row_ptr =
        Q6_Vhf_vadd_VhfVhf(*(HVX_Vector*)att_row_ptr, attn_mask_vec);

    q_ptr += qk_emb;
    if ((h + 1) % group == 0) {
      k_ptr += qk_emb * HFAQ_KV_SEQ_TILE;
    }
    att_row_ptr += HFAQ_KV_SEQ_TILE;
  }
}

// - upcast hf -> sf
// - scale att by 1 / ln(2)
// - find and returns local max
static inline HVX_Vector upcast_scale_max_att_t(HVX_Vector* att_vec) {
  HVX_Vector att_vec_sf;
  HVX_Vector local_max_vec = Q6_V_vsplat_R(SF_NEG_INF);
  HVX_Vector inv_ln2_vec = Q6_V_vsplat_R(SF_INV_LN2);
  for (uint16_t i = 0; i < HFAQ_KV_SEQ_TILE; ++i) {
    att_vec_sf = Q6_V_lo_W(Q6_Wsf_vcvt_Vhf(Q6_Vh_vshuff_Vh(att_vec[i])));
    att_vec_sf = Q6_Vsf_vmpy_VsfVsf(att_vec_sf, inv_ln2_vec);
    local_max_vec = Q6_Vsf_vmax_VsfVsf(local_max_vec, att_vec_sf);
    att_vec[i] = att_vec_sf;
  }
  return local_max_vec;
}

// - norm: x - max
// - u = exp2(x)
// - l: sum(u)
static inline HVX_Vector norm_exp_l_att_t(
    HVX_Vector* att_vec,
    HVX_Vector local_max_vec) {
  HVX_Vector att_vec_sf;
  HVX_Vector local_l_vec = Q6_V_vzero();
  for (uint16_t i = 0; i < HFAQ_KV_SEQ_TILE; ++i) {
    att_vec_sf =
        hvx_Vsf_vexp2_Vsf(Q6_Vsf_vsub_VsfVsf(att_vec[i], local_max_vec));
    local_l_vec = Q6_Vsf_vadd_VsfVsf(local_l_vec, att_vec_sf);
    att_vec[i] = att_vec_sf;
  }
  return local_l_vec;
}

static inline void att_heads_matmul_v(
    float* out_row_ptr,
    float* att_ptr,
    Float16* v_ptr,
    const size_t num_heads,
    const size_t num_kv_heads,
    const size_t v_emb) {
  const size_t group = num_heads / num_kv_heads;
  for (uint16_t h = 0; h < num_heads; ++h) {
    hvx_Vsf_matmul1x64N_VsfVhf(
        out_row_ptr, att_ptr, v_ptr, HFAQ_KV_SEQ_TILE, v_emb);

    att_ptr += HFAQ_KV_SEQ_TILE;
    if ((h + 1) % group == 0) {
      v_ptr += v_emb * HFAQ_KV_SEQ_TILE;
    }
    out_row_ptr += v_emb;
  }
}