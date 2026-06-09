#pragma once

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

constexpr bool USE_BASELINE = false;

static inline void cpu_Vhf_baseline_matmul_VhfVhf(
    Float16* Out_ptr,
    Float16* U_ptr,
    Float16* V_ptr,
    uint32_t m,
    uint32_t k,
    uint32_t n) {
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      Out_ptr[i * n + j] = 0;
      for (size_t x = 0; x < k; ++x) {
        Out_ptr[i * n + j] = Out_ptr[i * n + j] +
            Float16(float(U_ptr[i * k + x]) * float(V_ptr[x * n + j]));
      }
    }
  }
}

// must: n % 64 == 0
static inline void hvx_Vhf_matmul_VhfVhf(
    Float16* Out_ptr,
    Float16* U_ptr,
    Float16* V_ptr,
    uint32_t m,
    uint32_t k,
    uint32_t n) {
  if (USE_BASELINE) {
    cpu_Vhf_baseline_matmul_VhfVhf(Out_ptr, U_ptr, V_ptr, m, k, n);
    return;
  }

  HVX_Vector* iptr1;
  HVX_Vector* optr;
  HVX_Vector sline1, sline2, temp, sout;
  int32_t vectors_in_rounddown = n / 64;
  Float16* Out_row_addr;
  Float16* V_row_addr;

  for (int32_t ncnt = 0; ncnt < vectors_in_rounddown; ++ncnt) {
    for (uint32_t mcnt = 0; mcnt < m; mcnt++) {
      sout = Q6_V_vzero();
      Out_row_addr = Out_ptr + (mcnt * n);
      optr = ((HVX_Vector*)Out_row_addr) + ncnt;

      if (mcnt + 1 < m) {
        for (uint32_t i = 0; i < k; i += 32) {
          Float16* addr = U_ptr + (mcnt + 1) * k + i;
          Q6_dcfetch_A(addr);
        }
      }

      for (uint32_t kcnt = 0; kcnt < k; kcnt++) {
        V_row_addr = V_ptr + (kcnt * n);
        iptr1 = ((HVX_Vector*)V_row_addr) + ncnt;
        sline1 = *iptr1++;

        sline2 = Q6_Vh_vsplat_R(U_ptr[mcnt * k + kcnt].raw());

        temp = Q6_Vqf16_vmpy_VhfVhf(sline1, sline2);
        sout = Q6_Vqf16_vadd_Vqf16Vqf16(sout, temp);
      }
      *optr = Q6_Vhf_equals_Vqf16(sout);
    }
  }
}