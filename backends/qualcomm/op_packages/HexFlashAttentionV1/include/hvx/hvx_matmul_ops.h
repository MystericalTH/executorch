#pragma once

#include "HFAQ/debug.h"
#include "HTP/core/float16.h"
#include "hvx/hvx_transpose_ops.h"

// For HFAQ special case
// A must be 1xK, B must be Kx64N
static inline void hvx_Vhf_matmul1x64N_Vhf(
    Float16* Out_ptr,
    const Float16* A_ptr,
    Float16* B_ptr,
    const uint32_t k,
    const uint32_t n,
    const HVX_Vector scaleline_qf32) {
  HVX_Vector* optr;
  HVX_Vector sline1, sline2, sout_lo, sout_hi;
  HVX_VectorPair temp_w;
  int32_t n_vecs = n / 64;
  Float16* B_row_addr;

  for (int32_t ncnt = 0; ncnt < n_vecs; ++ncnt) {
    sout_lo = Q6_V_vzero();
    sout_hi = Q6_V_vzero();
    optr = ((HVX_Vector*)Out_ptr) + ncnt;

    for (uint32_t kcnt = 0; kcnt < k; ++kcnt) {
      B_row_addr = B_ptr + (kcnt * n);
      sline1 = *(((HVX_Vector*)B_row_addr) + ncnt);
      sline2 = Q6_Vh_vsplat_R(A_ptr[kcnt].raw());

      temp_w = Q6_Wqf32_vmpy_VhfVhf(sline1, sline2);
      sout_lo = Q6_Vqf32_vadd_Vqf32Vqf32(sout_lo, Q6_V_lo_W(temp_w));
      sout_hi = Q6_Vqf32_vadd_Vqf32Vqf32(sout_hi, Q6_V_hi_W(temp_w));
    }

    sout_lo = Q6_Vqf32_vmpy_Vqf32Vqf32(sout_lo, scaleline_qf32);
    sout_hi = Q6_Vqf32_vmpy_Vqf32Vqf32(sout_hi, scaleline_qf32);

    *optr = Q6_Vhf_vcvt_VsfVsf(
        Q6_Vsf_equals_Vqf32(sout_lo), Q6_Vsf_equals_Vqf32(sout_hi));
  }
}

// For HFAQ special case
// A must be 1xK, B must be 64Nx64K
static inline void hvx_Vhf_matmul1x64N_transpose_Vhf(
    Float16* Out_ptr,
    const Float16* A_ptr,
    Float16* B_ptr,
    const uint32_t k,
    const uint32_t n,
    const HVX_Vector scaleline_qf32,
    Float16* scr_4096_ptr) {
  HVX_Vector* optr;
  HVX_Vector* accptr;
  HVX_Vector sline1, sline2, sout, sout_lo, sout_hi;
  HVX_VectorPair temp_w;
  int32_t n_vecs = n / 64;
  int32_t k_vecs = k / 64;
  Float16* B_row_addr;

  for (int32_t ncnt = 0; ncnt < n_vecs; ++ncnt) {
    optr = ((HVX_Vector*)Out_ptr) + ncnt;

#pragma unroll
    for (int32_t r = 0; r < 64; ++r) {
      sout = Q6_V_vzero();
      accptr = ((HVX_Vector*)scr_4096_ptr) + r;
      B_row_addr = B_ptr + (ncnt * 64 + r) * k;

      for (uint32_t kcnt = 0; kcnt < k_vecs; ++kcnt) {
        sline1 = *(((HVX_Vector*)B_row_addr) + kcnt);
        sline2 = *(((HVX_Vector*)A_ptr) + kcnt);

        temp_w = Q6_Wqf32_vmpy_VhfVhf(sline1, sline2);
        sout = Q6_Vqf32_vadd_Vqf32Vqf32(sout, Q6_V_lo_W(temp_w));
        sout = Q6_Vqf32_vadd_Vqf32Vqf32(sout, Q6_V_hi_W(temp_w));
      }

      sout = Q6_Vqf32_vmpy_Vqf32Vqf32(sout, scaleline_qf32);
      *accptr = Q6_Vhf_vcvt_VsfVsf(Q6_Vsf_equals_Vqf32(sout), Q6_V_vzero());
    }

    hvx_mat_transpose64x64_Vhf(scr_4096_ptr, scr_4096_ptr);

    sout_lo = Q6_V_vzero();
    sout_hi = Q6_V_vzero();
#pragma unroll
    for (int32_t r = 0; r < 64; r += 2) {
      accptr = ((HVX_Vector*)scr_4096_ptr) + r;

      temp_w = Q6_Wsf_vcvt_Vhf(*accptr);
      sout_lo = Q6_Vqf32_vadd_Vqf32Vsf(sout_lo, Q6_V_lo_W(temp_w));
      sout_hi = Q6_Vqf32_vadd_Vqf32Vsf(sout_hi, Q6_V_hi_W(temp_w));
    }

    *optr = Q6_Vhf_vcvt_VsfVsf(
        Q6_Vsf_equals_Vqf32(sout_lo), Q6_Vsf_equals_Vqf32(sout_hi));
  }
}

// For HFAQ special case
// A must be 1xK, B must be Kx32N
static inline void hvx_Vsf_matmul1x64N_VsfVhf(
    float* Out_ptr,
    const float* A_ptr,
    Float16* B_ptr,
    const uint32_t k,
    const uint32_t n) {
  HVX_VectorPair* optr;
  HVX_Vector sline1, sline2, sout_lo, sout_hi;
  HVX_VectorPair temp_w;
  int32_t n_vecs = n / 64;
  Float16* B_row_addr;

  for (int32_t ncnt = 0; ncnt < n_vecs; ++ncnt) {
    sout_lo = Q6_V_vzero();
    sout_hi = Q6_V_vzero();
    optr = ((HVX_VectorPair*)Out_ptr) + ncnt;

    for (uint32_t kcnt = 0; kcnt < k; ++kcnt) {
      B_row_addr = B_ptr + (kcnt * n);
      sline1 = *(((HVX_Vector*)B_row_addr) + ncnt);
      sline2 = Q6_V_vsplat_R(((uint32_t*)A_ptr)[kcnt]);
      sline2 = Q6_Vhf_vcvt_VsfVsf(sline2, sline2);

      temp_w = Q6_Wqf32_vmpy_VhfVhf(sline1, sline2);
      sout_lo = Q6_Vqf32_vadd_Vqf32Vqf32(sout_lo, Q6_V_lo_W(temp_w));
      sout_hi = Q6_Vqf32_vadd_Vqf32Vqf32(sout_hi, Q6_V_hi_W(temp_w));
    }

    *optr = Q6_W_vshuff_VVR(
        Q6_Vsf_equals_Vqf32(sout_hi),
        Q6_Vsf_equals_Vqf32(sout_lo),
        -4); // 4 + 8 + 16 + 32
  }
}