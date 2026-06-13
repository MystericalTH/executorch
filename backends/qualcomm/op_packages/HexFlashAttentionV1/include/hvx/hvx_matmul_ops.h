#pragma once

#include "HTP/core/float16.h"

// For HFAQ special case
// A must be 1xK, B must be Kx64N, scale must be fp32
// upcast accumulator and downcast final result
static inline void hvx_Vhf_matmul1x64N_Vhf(
    Float16* Out_ptr,
    const Float16* A_ptr,
    Float16* B_ptr,
    const uint32_t k,
    const uint32_t n,
    const uint32_t rscale = SF_ONE) {
  HVX_Vector* iptr1;
  HVX_Vector* optr;
  HVX_Vector sline1, sline2, sout_lo, sout_hi, scaleline;
  HVX_VectorPair temp_w;
  int32_t n_vecs = n / 64;
  Float16* B_row_addr;

  scaleline = Q6_Vqf32_vadd_VsfVsf(Q6_V_vsplat_R(rscale), Q6_V_vzero());

  for (int32_t ncnt = 0; ncnt < n_vecs; ++ncnt) {
    sout_lo = Q6_V_vzero();
    sout_hi = Q6_V_vzero();
    optr = ((HVX_Vector*)Out_ptr) + ncnt;

    for (uint32_t kcnt = 0; kcnt < k; kcnt++) {
      B_row_addr = B_ptr + (kcnt * n);
      iptr1 = ((HVX_Vector*)B_row_addr) + ncnt;
      sline1 = *iptr1++;

      sline2 = Q6_Vh_vsplat_R(A_ptr[kcnt].raw());

      temp_w = Q6_Wqf32_vmpy_VhfVhf(sline1, sline2);
      sout_lo = Q6_Vqf32_vadd_Vqf32Vqf32(sout_lo, Q6_V_lo_W(temp_w));
      sout_hi = Q6_Vqf32_vadd_Vqf32Vqf32(sout_hi, Q6_V_hi_W(temp_w));
    }

    sout_lo = Q6_Vqf32_vmpy_Vqf32Vqf32(sout_lo, scaleline);
    sout_hi = Q6_Vqf32_vmpy_Vqf32Vqf32(sout_hi, scaleline);

    *optr = Q6_Vhf_vcvt_VsfVsf(
        Q6_Vsf_equals_Vqf32(sout_lo), Q6_Vsf_equals_Vqf32(sout_hi));
  }
}

// For HFAQ special case
// A must be 1xK, B must be Kx32N, scale must be fp32
static inline void hvx_Vsf_matmul1x64N_VsfVhf(
    float* Out_ptr,
    const float* A_ptr,
    Float16* B_ptr,
    const uint32_t k,
    const uint32_t n,
    const uint32_t rscale = SF_ONE) {
  HVX_Vector* iptr1;
  HVX_VectorPair* optr;
  HVX_Vector sline1, sline2, sout_lo, sout_hi, scaleline;
  HVX_VectorPair temp_w;
  int32_t n_vecs = n / 64;
  Float16* B_row_addr;

  scaleline = Q6_Vqf32_vadd_VsfVsf(Q6_V_vsplat_R(rscale), Q6_V_vzero());

  for (int32_t ncnt = 0; ncnt < n_vecs; ++ncnt) {
    sout_lo = Q6_V_vzero();
    sout_hi = Q6_V_vzero();
    optr = ((HVX_VectorPair*)Out_ptr) + ncnt;

    for (uint32_t kcnt = 0; kcnt < k; kcnt++) {
      B_row_addr = B_ptr + (kcnt * n);
      iptr1 = ((HVX_Vector*)B_row_addr) + ncnt;
      sline1 = *iptr1++;

      sline2 = Q6_Vh_vsplat_R(Float16(A_ptr[kcnt]).raw());

      temp_w = Q6_Wqf32_vmpy_VhfVhf(sline1, sline2);
      sout_lo = Q6_Vqf32_vadd_Vqf32Vqf32(sout_lo, Q6_V_lo_W(temp_w));
      sout_hi = Q6_Vqf32_vadd_Vqf32Vqf32(sout_hi, Q6_V_hi_W(temp_w));
    }

    sout_lo = Q6_Vqf32_vmpy_Vqf32Vqf32(sout_lo, scaleline);
    sout_hi = Q6_Vqf32_vmpy_Vqf32Vqf32(sout_hi, scaleline);

    *optr = Q6_W_vshuff_VVR(
        Q6_Vsf_equals_Vqf32(sout_hi),
        Q6_Vsf_equals_Vqf32(sout_lo),
        -4); // 4 + 8 + 16 + 32
  }
}