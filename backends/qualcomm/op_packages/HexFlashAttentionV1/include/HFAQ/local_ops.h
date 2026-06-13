#pragma once

#include "constant.h"
#include "hvx/hvx_exp_ops.h"
#include "hvx/hvx_matmul_ops.h"

// - upcast hf -> sf
// - scale att by 1 / ln(2)
// - find local max
static inline void upcast_scale_max_att_t(
    HVX_Vector* att_vec,
    HVX_Vector* local_max_vec) {
  HVX_Vector att_vec_sf;
  HVX_Vector inv_ln2_vec = Q6_V_vsplat_R(SF_INV_LN2);
  for (uint16_t i = 0; i < HFAQ_KV_SEQ_PROC_TILE; ++i) {
    att_vec_sf = Q6_V_lo_W(Q6_Wsf_vcvt_Vhf(Q6_Vh_vshuff_Vh(att_vec[i])));
    att_vec_sf = Q6_Vsf_vmpy_VsfVsf(att_vec_sf, inv_ln2_vec);
    *local_max_vec = Q6_Vsf_vmax_VsfVsf(*local_max_vec, att_vec_sf);
    att_vec[i] = att_vec_sf;
  }
}

// - norm: x - max
// - u = exp2(x)
// - l: sum(u)
static inline void norm_exp_l_att_t(
    HVX_Vector* att_vec,
    HVX_Vector local_max_vec,
    HVX_Vector* local_l_vec) {
  HVX_Vector att_vec_sf;
  for (uint16_t i = 0; i < HFAQ_KV_SEQ_PROC_TILE; ++i) {
    att_vec_sf =
        hvx_Vsf_vexp2_Vsf(Q6_Vsf_vsub_VsfVsf(att_vec[i], local_max_vec));
    *local_l_vec = Q6_Vsf_vadd_VsfVsf(*local_l_vec, att_vec_sf);
    att_vec[i] = att_vec_sf;
  }
}

static inline void att_heads_matmul_v(
    float* out_row_ptr,
    float* att_ptr,
    Float16* v_ptr,
    const size_t v_emb) {
  for (uint16_t h = 0; h < HFAQ_ACC_HEAD_TILE; ++h) {
    hvx_Vsf_matmul1x64N_VsfVhf(
        out_row_ptr, att_ptr, v_ptr, HFAQ_KV_SEQ_PROC_TILE, v_emb);

    att_ptr += HFAQ_KV_SEQ_PROC_TILE;
    v_ptr += v_emb * HFAQ_KV_SEQ_PROC_TILE;
    out_row_ptr += v_emb;
  }
}