#pragma once

#include "HTP/core/float16.h"

// For HFAQ special case
// A must be 1xK
// B must be Kx64N
static inline void hvx_Vhf_matmul1x64N_VhfVhf(
    Float16* Out_ptr,
    Float16* A_ptr,
    Float16* B_ptr,
    uint32_t k,
    uint32_t n) {
  HVX_Vector* iptr1;
  HVX_Vector* optr;
  HVX_Vector sline1, sline2, temp, sout;
  int32_t n_vecs = n / 64;
  Float16* V_row_addr;

  for (int32_t ncnt = 0; ncnt < n_vecs; ++ncnt) {
    sout = Q6_V_vzero();
    optr = ((HVX_Vector*)Out_ptr) + ncnt;

    for (uint32_t kcnt = 0; kcnt < k; kcnt++) {
      V_row_addr = B_ptr + (kcnt * n);
      iptr1 = ((HVX_Vector*)V_row_addr) + ncnt;
      sline1 = *iptr1++;

      sline2 = Q6_Vh_vsplat_R(A_ptr[kcnt].raw());

      temp = Q6_Vqf16_vmpy_VhfVhf(sline1, sline2);
      sout = Q6_Vqf16_vadd_Vqf16Vqf16(sout, temp);
    }
    *optr = Q6_Vhf_equals_Vqf16(sout);
  }
}