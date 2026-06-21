#pragma once

#include "HTP/core/float16.h"

static inline void hvx_mat_transpose64x64_Vhf(
    Float16* Out_ptr,
    const Float16* U_ptr,
    const uint32_t rvec_offset = 1) {
  auto U_vec_ptr = (const HVX_Vector*)U_ptr;
  auto out_vec_ptr = (HVX_Vector*)Out_ptr;

  HVX_VectorPair pair;

#pragma unroll
  for (uint8_t i = 0; i < 64; i += 2) {
    pair = Q6_W_vshuff_VVR(
        U_vec_ptr[(i + 1) * rvec_offset], U_vec_ptr[i * rvec_offset], 2);
    out_vec_ptr[i * rvec_offset] = Q6_V_lo_W(pair);
    out_vec_ptr[(i + 1) * rvec_offset] = Q6_V_hi_W(pair);
  }

  constexpr uint8_t groups[5] = {4, 8, 16, 32, 64};
#pragma unroll
  for (uint8_t i = 0; i < 5; ++i) {
    const uint8_t group = groups[i];
#pragma unroll
    for (uint8_t j = 0; j < 64; j += group) {
      const uint8_t stride = group / 2;
#pragma unroll
      for (uint8_t k = 0; k < stride; ++k) {
        pair = Q6_W_vshuff_VVR(
            out_vec_ptr[(j + k + stride) * rvec_offset],
            out_vec_ptr[(j + k) * rvec_offset],
            group);
        out_vec_ptr[(j + k) * rvec_offset] = Q6_V_lo_W(pair);
        out_vec_ptr[(j + k + stride) * rvec_offset] = Q6_V_hi_W(pair);
      }
    }
  }
}

// m, n % 64 == 0
static inline void hvx_mat_transposeMxN_Vhf(
    Float16* Out_ptr,
    const Float16* U_ptr,
    const uint16_t m,
    const uint16_t n) {
  const uint16_t m_vecs = m / 64;
  const uint16_t n_vecs = n / 64;

  size_t inr_offset, outr_offset;
  auto U_vec_ptr = (const HVX_Vector*)U_ptr;
  auto Out_vec_ptr = (HVX_Vector*)Out_ptr;

  for (uint16_t cblk = 0; cblk < n_vecs; ++cblk) {
    for (uint16_t rblk = 0; rblk < m_vecs; ++rblk) {
      inr_offset = 64 * rblk * n_vecs;
      outr_offset = 64 * cblk * m_vecs;
      for (uint16_t r = 0; r < 64; ++r) {
        Out_vec_ptr[outr_offset + r * m_vecs + rblk] =
            U_vec_ptr[inr_offset + r * n_vecs + cblk];
      }
      hvx_mat_transpose64x64_Vhf(Out_ptr, Out_ptr, m_vecs);
      Out_ptr += 64;
    }
    Out_ptr += 63 * 64 * m_vecs;
  }
}

static inline void hvx_mat_transpose32x32_Vsf(
    float* Out_ptr,
    const float* U_ptr,
    const uint32_t rvec_offset = 1) {
  auto U_vec_ptr = (const HVX_Vector*)U_ptr;
  auto out_vec_ptr = (HVX_Vector*)Out_ptr;

  HVX_VectorPair pair;

#pragma unroll
  for (uint8_t i = 0; i < 32; i += 2) {
    pair = Q6_W_vshuff_VVR(
        U_vec_ptr[(i + 1) * rvec_offset], U_vec_ptr[i * rvec_offset], 4);
    out_vec_ptr[i * rvec_offset] = Q6_V_lo_W(pair);
    out_vec_ptr[(i + 1) * rvec_offset] = Q6_V_hi_W(pair);
  }

  constexpr uint8_t groups[4] = {4, 8, 16, 32};
#pragma unroll
  for (uint8_t i = 0; i < 4; ++i) {
    const uint8_t group = groups[i];
#pragma unroll
    for (uint8_t j = 0; j < 32; j += group) {
      const uint8_t stride = group / 2;
#pragma unroll
      for (uint8_t k = 0; k < stride; ++k) {
        pair = Q6_W_vshuff_VVR(
            out_vec_ptr[(j + k + stride) * rvec_offset],
            out_vec_ptr[(j + k) * rvec_offset],
            group * 2);
        out_vec_ptr[(j + k) * rvec_offset] = Q6_V_lo_W(pair);
        out_vec_ptr[(j + k + stride) * rvec_offset] = Q6_V_hi_W(pair);
      }
    }
  }
}

// m, n % 32 == 0
static inline void hvx_mat_transposeMxN_Vsf(
    float* Out_ptr,
    const float* U_ptr,
    const uint16_t m,
    const uint16_t n) {
  const uint16_t m_vecs = m / 32;
  const uint16_t n_vecs = n / 32;

  size_t inr_offset, outr_offset;
  auto U_vec_ptr = (const HVX_Vector*)U_ptr;
  auto Out_vec_ptr = (HVX_Vector*)Out_ptr;

  for (uint16_t cblk = 0; cblk < n_vecs; ++cblk) {
    for (uint16_t rblk = 0; rblk < m_vecs; ++rblk) {
      inr_offset = 32 * rblk * n_vecs;
      outr_offset = 32 * cblk * m_vecs;
      for (uint16_t r = 0; r < 32; ++r) {
        Out_vec_ptr[outr_offset + r * m_vecs + rblk] =
            U_vec_ptr[inr_offset + r * n_vecs + cblk];
      }
      hvx_mat_transpose32x32_Vsf(Out_ptr, Out_ptr, m_vecs);
      Out_ptr += 32;
    }
    Out_ptr += 31 * 32 * m_vecs;
  }
}