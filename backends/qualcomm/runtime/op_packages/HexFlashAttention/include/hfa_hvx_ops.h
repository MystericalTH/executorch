#pragma once

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

template <uint16_t num_vecs>
static inline void hvx_Vhf_multisub_VhfVhf(
    HVX_Vector* vec_ptr,
    HVX_Vector opr) {
#pragma unroll
  for (uint16_t i = 0; i < num_vecs; ++i) {
    vec_ptr[i] = Q6_Vhf_vsub_VhfVhf(vec_ptr[i], opr);
  }
}

static inline void hvx_Vhf_multimpy_VhfVhf(
    HVX_Vector* vec_ptr,
    HVX_Vector opr,
    uint16_t num_vecs) {
  for (uint16_t i = 0; i < num_vecs; ++i) {
    vec_ptr[i] = Q6_Vhf_vmpy_VhfVhf(vec_ptr[i], opr);
  }
}

template <uint16_t num_vecs>
static inline HVX_Vector hvx_Vhf_redadd_VhfVhf(
    HVX_Vector curr,
    const HVX_Vector* __restrict__ vec_ptr) {
#pragma unroll
  for (uint16_t i = 0; i < num_vecs; ++i) {
    curr = Q6_Vhf_vadd_VhfVhf(curr, vec_ptr[i]);
  }
  return curr;
}

template <uint16_t num_vecs>
static inline HVX_Vector hvx_Vhf_redmax_VhfVhf(
    HVX_Vector curr,
    const HVX_Vector* vec_ptr) {
#pragma unroll
  for (uint16_t i = 0; i < num_vecs; ++i) {
    curr = Q6_Vhf_vmax_VhfVhf(curr, vec_ptr[i]);
  }
  return curr;
}

static inline void hvx_Vhf_accadd_VhfVhf(
    HVX_Vector* acc_vec,
    const HVX_Vector* opr_vec,
    uint16_t num_vecs) {
  for (uint16_t i = 0; i < num_vecs; ++i) {
    acc_vec[i] = Q6_Vhf_vadd_VhfVhf(acc_vec[i], opr_vec[i]);
  }
}