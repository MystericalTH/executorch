#pragma once

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

// load (flat) matrix
inline void hvx_Vhf_mat_load_Vhf(
    Float16* Out_ptr,
    const Float16* U_ptr,
    uint16_t U_nrow,
    uint16_t U_ncol) {
  assert(U_ncol % 64 == 0);

  auto U_vec_ptr = (const HVX_Vector*)U_ptr;
  auto Out_vec_ptr = (HVX_Vector*)Out_ptr;

  const uint16_t chunks = U_nrow * (U_ncol / 64);
  for (uint16_t i = 0; i < chunks; ++i) {
    Out_vec_ptr[i] = U_vec_ptr[i];
  }

  return;
}

// input must be 64x64
static inline void hvx_Vhf_mat_transpose64x64(
    Float16* Out_ptr,
    const Float16* U_ptr) {
  auto U_vec_ptr = (const HVX_Vector*)U_ptr;
  auto out_pair_ptr = (HVX_VectorPair*)Out_ptr;
  auto out_vec_ptr = (HVX_Vector*)Out_ptr;

#pragma unroll
  for (uint8_t i = 0; i < 32; ++i) {
    out_pair_ptr[i] =
        Q6_W_vshuff_VVR(U_vec_ptr[2 * i + 1], U_vec_ptr[2 * i], 2);
  }

  HVX_VectorPair pair;
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
            out_vec_ptr[j + k + stride], out_vec_ptr[j + k], group);
        out_vec_ptr[j + k] = Q6_V_lo_W(pair);
        out_vec_ptr[j + k + stride] = Q6_V_hi_W(pair);
      }
    }
  }
}

// inplace transpose
static inline void hvx_Vhf_mat_transpose64x64(Float16* ptr) {
  hvx_Vhf_mat_transpose64x64(ptr, ptr);
}

static inline void hvx_Vhf_mat_transpose64x64Nc(
    Float16* Out_ptr,
    const Float16* U_ptr,
    uint16_t ncol_chunks) {
  auto U_vec_ptr = (const HVX_Vector*)U_ptr;
  auto out_vec_ptr = (HVX_Vector*)Out_ptr;
  for (uint16_t i = 0; i < ncol_chunks * 64; ++i) {
    // i_row + col_offset = (i % 64) * col + floor(i / 64)
    out_vec_ptr[i] = U_vec_ptr[(i & 63) * ncol_chunks + (i >> 6)];
  }
  for (uint16_t i = 0; i < ncol_chunks; ++i) {
    hvx_Vhf_mat_transpose64x64(Out_ptr + i * 64 * 64);
  }
}

// requires a tmp memory
static inline void hvx_Vhf_mat_transpose64Ncx64(
    Float16* Out_ptr,
    const Float16* U_ptr,
    uint16_t nrow_chunks,
    Float16* tmp_ptr) {
  load_mat_fp16(tmp_ptr, U_ptr, nrow_chunks * 64, 64);

  auto out_vec_ptr = (HVX_Vector*)Out_ptr;
  auto tmp_vec_ptr = (HVX_Vector*)tmp_ptr;
  for (uint16_t i = 0; i < nrow_chunks; ++i) {
    hvx_Vhf_mat_transpose64x64(tmp_ptr + i * 64 * 64);
  }
  for (uint16_t i = 0; i < nrow_chunks * 64; ++i) {
    // i_row + row_offset = (i % row) * 64 + floor(i / row)
    out_vec_ptr[i] = tmp_vec_ptr[(i % nrow_chunks) * 64 + int(i / nrow_chunks)];
  }
}
