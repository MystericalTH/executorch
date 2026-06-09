#pragma once

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

static inline HVX_Vector cpu_Vhf_baseline_inv_Vhf(
    HVX_Vector vec,
    Float16* scr_ptr) {
  auto scr_vec_ptr = ((HVX_Vector*)scr_ptr);
  scr_vec_ptr[0] = vec;
  for (size_t i = 0; i < 64; ++i) {
    scr_ptr[i] = Float16(1 / float(scr_ptr[i]));
  }
  return scr_vec_ptr[0];
}

static inline HVX_Vector hvx_Vhf_rsqrt_Vhf(HVX_Vector in_vec) {
  //   x2 = input*0.5
  //   y  = * (long *) &input
  //   y  = 0x59bb - (y >> 2)
  //   y  = y * (threehalfs - x2 * y * y)

  HVX_Vector rsqrtconst = Q6_Vh_vsplat_R(0x59bb);
  HVX_Vector onehalf = Q6_Vh_vsplat_R(0x3800);
  HVX_Vector threehalfs = Q6_Vh_vsplat_R(0x600f);

  HVX_Vector x2, y, ypower2, temp;

  x2 = Q6_Vqf16_vmpy_VhfVhf(in_vec, onehalf);
  x2 = Q6_Vqf16_vadd_Vqf16Vhf(x2, Q6_V_vzero());

  y = Q6_Vh_vasr_VhR(in_vec, 1);
  y = Q6_Vh_vsub_VhVh(rsqrtconst, y);

  // 1st iteration
  ypower2 = Q6_Vqf16_vmpy_VhfVhf(y, y);
  ypower2 = Q6_Vqf16_vadd_Vqf16Vhf(ypower2, Q6_V_vzero());
  temp = Q6_Vqf16_vmpy_Vqf16Vqf16(x2, ypower2);
  temp = Q6_Vqf16_vsub_Vqf16Vqf16(threehalfs, temp);
  temp = Q6_Vqf16_vmpy_VhfVhf(y, Q6_Vhf_equals_Vqf16(temp));

  // 2nd iteration
  y = Q6_Vqf16_vadd_Vqf16Vhf(temp, Q6_V_vzero());
  ypower2 = Q6_Vqf16_vmpy_Vqf16Vqf16(y, y);
  ypower2 = Q6_Vqf16_vadd_Vqf16Vhf(ypower2, Q6_V_vzero());
  temp = Q6_Vqf16_vmpy_Vqf16Vqf16(x2, ypower2);
  temp = Q6_Vqf16_vsub_Vqf16Vqf16(threehalfs, temp);
  temp = Q6_Vqf16_vmpy_Vqf16Vqf16(y, temp);

  // 3rd iteration
  y = Q6_Vqf16_vadd_Vqf16Vhf(temp, Q6_V_vzero());
  ypower2 = Q6_Vqf16_vmpy_Vqf16Vqf16(y, y);
  ypower2 = Q6_Vqf16_vadd_Vqf16Vhf(ypower2, Q6_V_vzero());
  temp = Q6_Vqf16_vmpy_Vqf16Vqf16(x2, ypower2);
  temp = Q6_Vqf16_vsub_Vqf16Vqf16(threehalfs, temp);
  temp = Q6_Vqf16_vmpy_Vqf16Vqf16(y, temp);

  return Q6_Vhf_equals_Vqf16(temp);
}

// 1/x = (1/sqrt(x))**2
static inline HVX_Vector hvx_Vhf_exp2rsqrt_inv_Vhf(HVX_Vector in_vec) {
  HVX_Vector rsqrt_vec = hvx_Vhf_rsqrt_Vhf(in_vec);
  return Q6_Vhf_vmpy_VhfVhf(rsqrt_vec, rsqrt_vec);
}