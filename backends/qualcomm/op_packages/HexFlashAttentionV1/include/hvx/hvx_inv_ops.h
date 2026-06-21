#pragma once

#include "constant.h"

static inline HVX_Vector hvx_Vsf_vinv_Vsf(HVX_Vector x) {
  HVX_Vector y, m;
  HVX_Vector magic = Q6_V_vsplat_R(INV_MAGIC_c0);
  HVX_Vector e0 = Q6_V_vsplat_R(INV_e0);
  HVX_Vector e1 = Q6_V_vsplat_R(INV_e1);
  HVX_Vector two = Q6_V_vsplat_R(SF_TWO);

  // int i = *(int*)&x;
  //
  // i = 0x7EB504F3 - i;
  //
  // float y = *(float*)&i;
  //
  // y = y * e0 * (e1 - x * y);
  //
  // y = y * (2 - x * y)

  y = Q6_Vw_vsub_VwVw(magic, x);

  // first iter
  // m = e0 * (e1 - x * y), y = y * m
  m = Q6_Vsf_vmpy_VsfVsf(x, y);
  m = Q6_Vsf_vsub_VsfVsf(e1, m);
  m = Q6_Vsf_vmpy_VsfVsf(e0, m);
  y = Q6_Vsf_vmpy_VsfVsf(y, m);

  // second iter
  // (2 - x * y), y = y * m
  m = Q6_Vsf_vmpy_VsfVsf(x, y);
  m = Q6_Vsf_vsub_VsfVsf(two, m);
  y = Q6_Vsf_vmpy_VsfVsf(y, m);

  return y;
}