#pragma once

#include "constant.h"

static inline HVX_Vector Q6_Vw_vfloor_VsfVsf(HVX_Vector Vu, HVX_Vector* Vd) {
  HVX_Vector round_f;
  HVX_Vector o_i_v;
  HVX_Vector o_f_v;
  HVX_Vector expval_v;
  HVX_Vector mantissa_v;
  HVX_Vector mantissa_shift_v;
  HVX_Vector mask;
  HVX_Vector const_zero_v;
  HVX_Vector const_v;

  const_zero_v = Q6_V_vzero();

  round_f = Vu;

  const_v = Q6_V_vsplat_R(0x7FFFFFFF);
  HVX_VectorPred qpred_negative_vq = Q6_Q_vcmp_gt_VuwVuw(round_f, const_v);

  expval_v = Q6_Vuw_vlsr_VuwR(round_f, 23);
  const_v = Q6_V_vsplat_R(0xFF);
  expval_v = Q6_V_vand_VV(expval_v, const_v);
  const_v = Q6_V_vsplat_R(127);
  expval_v = Q6_Vw_vsub_VwVw(expval_v, const_v);

  HVX_VectorPred qpred_negativexp_vq =
      Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);
  // clip negative exponent to zero
  expval_v = Q6_Vw_vmax_VwVw(expval_v, const_zero_v);

  const_v = Q6_V_vsplat_R(23);
  mantissa_shift_v = Q6_Vw_vsub_VwVw(const_v, expval_v);
  mantissa_shift_v = Q6_Vw_vmax_VwVw(mantissa_shift_v, const_zero_v);

  mantissa_v = round_f;
  const_v = Q6_V_vsplat_R(0x007FFFFF); // ((1 << 23) - 1)
  mantissa_v = Q6_V_vand_VV(mantissa_v, const_v);

  mantissa_v = Q6_Vw_vlsr_VwVw(mantissa_v, mantissa_shift_v);

  const_v = Q6_V_vsplat_R(0x0001);
  o_i_v = Q6_Vw_vasl_VwVw(const_v, expval_v); // 1 << expval
  o_i_v = Q6_V_vmux_QVV(qpred_negativexp_vq, const_zero_v, o_i_v);

  o_i_v = Q6_Vw_vadd_VwVw(o_i_v, mantissa_v);

  // fixing sign of integer value
  HVX_Vector negative_i_v = Q6_Vw_vsub_VwVw(const_zero_v, o_i_v);
  o_i_v = Q6_V_vmux_QVV(qpred_negative_vq, negative_i_v, o_i_v);

  mask = Q6_Vw_vasl_VwVw(const_v, mantissa_shift_v); //(1 << mantissa_shift_v);
  mask = Q6_Vw_vsub_VwVw(const_zero_v, mask);
  round_f = Q6_V_vand_VV(round_f, mask);

  //    o_f_v = round_f;
  o_f_v = Q6_V_vmux_QVV(qpred_negativexp_vq, const_zero_v, round_f);

  *Vd = o_f_v;
  return o_i_v;
}

static inline HVX_Vector hvx_Vsf_vexp2_Vsf(HVX_Vector input) {
  HVX_Vector x_v, k_v, f_v, y_v;
  HVX_Vector x_qf32_v, temp_qf32_v;
  HVX_Vector E_const;
  HVX_Vector one_w = Q6_V_vsplat_R(0x00000001);
  HVX_Vector one_sf = Q6_V_vsplat_R(SF_ONE);
  HVX_Vector half_sf = Q6_V_vsplat_R(SF_HALF);
  HVX_Vector zero_v = Q6_V_vzero();
  HVX_VectorPred qlarger;

  k_v = Q6_Vw_vfloor_VsfVsf(input, &f_v);

  x_qf32_v = Q6_Vqf32_vadd_VsfVsf(input, zero_v);

  x_qf32_v = Q6_Vqf32_vsub_Vqf32Vsf(x_qf32_v, f_v); //    x = x - f;
  //    if (x > 0.5)
  x_v = Q6_Vsf_equals_Vqf32(x_qf32_v);
  qlarger = Q6_Q_vcmp_gt_VsfVsf(x_v, half_sf);
  k_v = Q6_Vw_condacc_QVwVw(qlarger, k_v, one_w); // k += 1;
  temp_qf32_v = Q6_Vqf32_vsub_Vqf32Vsf(x_qf32_v, one_sf); // x -= 1.0;
  x_v = Q6_V_vmux_QVV(
      qlarger, Q6_Vsf_equals_Vqf32(temp_qf32_v), Q6_Vsf_equals_Vqf32(x_qf32_v));

  //    y = E4 + E5 * x;
  E_const = Q6_V_vsplat_R(EXP2_e5);
  y_v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(E_const, x_v));
  E_const = Q6_V_vsplat_R(EXP2_e4);
  y_v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(y_v, E_const));

  //    y = E3 + y * x;
  E_const = Q6_V_vsplat_R(EXP2_e3);
  y_v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(y_v, x_v));
  y_v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(y_v, E_const));

  //    y = E2 + y * x;
  E_const = Q6_V_vsplat_R(EXP2_e2);
  y_v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(y_v, x_v));
  y_v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(y_v, E_const));

  //    y = E1 + y * x;
  E_const = Q6_V_vsplat_R(EXP2_e1);
  y_v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(y_v, x_v));
  y_v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(y_v, E_const));

  //    y = E0 + y * x;
  E_const = Q6_V_vsplat_R(EXP2_e0);
  y_v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(y_v, x_v));
  y_v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(y_v, E_const));

  //    y = 1.0 + y * x;
  y_v = Q6_Vqf32_vmpy_VsfVsf(y_v, x_v);
  E_const = one_sf;
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);

  // insert exponents
  //         y = ldexpf(y, k);
  //     y_v += k_v; // qf32
  //  modify exponent
  y_v = Q6_Vsf_equals_Vqf32(y_v);

  // add k_v to the exponent of y_v
  HVX_Vector y_v_exponent = Q6_Vw_vasl_VwR(y_v, 1);

  y_v_exponent = Q6_Vuw_vlsr_VuwR(y_v_exponent, 24);

  y_v_exponent = Q6_Vw_vadd_VwVw(k_v, y_v_exponent);

  // exponent can't be negative; if overflow is detected, set to zero
  HVX_VectorPred qy_v_negative_exponent =
      Q6_Q_vcmp_gt_VwVw(zero_v, y_v_exponent);

  y_v = Q6_Vw_vaslacc_VwVwR(y_v, k_v, 23);

  return Q6_V_vmux_QVV(qy_v_negative_exponent, zero_v, y_v);
}

static inline HVX_Vector Q6_Vh_vfloor_VhfVhf(HVX_Vector Vu, HVX_Vector* Vd) {
  HVX_Vector round_f;
  HVX_Vector o_i_v;
  HVX_Vector o_f_v;
  HVX_Vector expval_v;
  HVX_Vector mantissa_v;
  HVX_Vector mantissa_shift_v;
  HVX_Vector mask;
  HVX_Vector const_zero_v;
  HVX_Vector const_v;

  const_zero_v = Q6_V_vzero();

  round_f = Vu;

  const_v = Q6_Vh_vsplat_R(0x7FFF);
  HVX_VectorPred qpred_negative_vq = Q6_Q_vcmp_gt_VuhVuh(round_f, const_v);

  // const_v = Q6_V_vsplat_R(23);
  expval_v = Q6_Vuh_vlsr_VuhR(round_f, 10);
  const_v = Q6_Vh_vsplat_R(0x1F);
  expval_v = Q6_V_vand_VV(expval_v, const_v);
  const_v = Q6_Vh_vsplat_R(0x000F); // 15
  expval_v = Q6_Vh_vsub_VhVh(expval_v, const_v); // exponent - offset 15

  HVX_VectorPred qpred_negativexp_vq =
      Q6_Q_vcmp_gt_VhVh(const_zero_v, expval_v);
  // clip negative exponent to zero
  expval_v = Q6_Vh_vmax_VhVh(expval_v, const_zero_v);

  const_v = Q6_Vh_vsplat_R(0x000A); // 10
  mantissa_shift_v = Q6_Vh_vsub_VhVh(const_v, expval_v);
  mantissa_shift_v = Q6_Vh_vmax_VhVh(mantissa_shift_v, const_zero_v);

  mantissa_v = round_f;
  const_v = Q6_Vh_vsplat_R(0x03FF); // ((1 << 10) - 1)
  mantissa_v = Q6_V_vand_VV(mantissa_v, const_v);

  mantissa_v = Q6_Vh_vlsr_VhVh(mantissa_v, mantissa_shift_v);

  const_v = Q6_Vh_vsplat_R(0x0001); // 1
  o_i_v = Q6_Vh_vasl_VhVh(const_v, expval_v); // 1 << expval_v
  o_i_v = Q6_V_vmux_QVV(qpred_negativexp_vq, const_zero_v, o_i_v);

  o_i_v = Q6_Vh_vadd_VhVh(o_i_v, mantissa_v); // o_i_v += mantissa_v;

  // fixing sign of integer value
  HVX_Vector negative_i_v =
      Q6_Vh_vsub_VhVh(const_zero_v, o_i_v); // check this - o_i_v
  o_i_v = Q6_V_vmux_QVV(qpred_negative_vq, negative_i_v, o_i_v);

  mask = Q6_Vh_vasl_VhVh(const_v, mantissa_shift_v); //(1 << mantissa_shift_v);
  mask = Q6_Vh_vsub_VhVh(const_zero_v, mask); // check this - mask
  round_f = Q6_V_vand_VV(round_f, mask);

  //    o_f_v = round_f;
  o_f_v = Q6_V_vmux_QVV(qpred_negativexp_vq, const_zero_v, round_f);

  *Vd = o_f_v;
  return o_i_v;
}

static inline HVX_Vector hvx_Vhf_vexp2_Vhf(HVX_Vector vin) {
  HVX_Vector x_v, f_v;
  HVX_Vector x_qf16_v;
  HVX_Vector y_v;
  HVX_Vector half_hf = Q6_Vh_vsplat_R(HF_ONE_HALF);
  HVX_Vector one_h = Q6_Vh_vsplat_R(0x0001);
  HVX_Vector zero_v = Q6_V_vzero();
  HVX_Vector one_hf = Q6_Vh_vsplat_R(HF_ONE);
  HVX_Vector E_const;

  HVX_Vector k_v = Q6_Vh_vfloor_VhfVhf(vin, &f_v);

  x_qf16_v = Q6_Vqf16_vadd_VhfVhf(vin, zero_v);

  //    x = x - f * LOG20;
  x_qf16_v = Q6_Vqf16_vsub_Vqf16Vhf(x_qf16_v, f_v);

  //    if (x > 0.5)
  x_v = Q6_Vhf_equals_Vqf16(x_qf16_v);
  HVX_VectorPred QLarger = Q6_Q_vcmp_gt_VhfVhf(x_v, half_hf);
  // k += 1;
  k_v = Q6_Vh_condacc_QVhVh(QLarger, k_v, one_h);
  // x -= 1.0;
  HVX_Vector temp_qf16_v = Q6_Vqf16_vsub_Vqf16Vhf(x_qf16_v, one_hf);
  x_v = Q6_V_vmux_QVV(
      QLarger, Q6_Vhf_equals_Vqf16(temp_qf16_v), Q6_Vhf_equals_Vqf16(x_qf16_v));

  //    y = E4 + E5 * x;
  E_const = Q6_Vh_vsplat_R(EXP2_HF_e5);
  y_v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(E_const, x_v));
  E_const = Q6_Vh_vsplat_R(EXP2_HF_e4);
  y_v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(y_v, E_const));

  //    y = E3 + y * x;
  E_const = Q6_Vh_vsplat_R(EXP2_HF_e3);
  y_v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(y_v, x_v));
  y_v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(y_v, E_const));

  //    y = E2 + y * x;
  E_const = Q6_Vh_vsplat_R(EXP2_HF_e2);
  y_v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(y_v, x_v));
  y_v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(y_v, E_const));

  //    y = E1 + y * x;
  E_const = Q6_Vh_vsplat_R(EXP2_HF_e1);
  y_v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(y_v, x_v));
  y_v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(y_v, E_const));

  //    y = E0 + y * x;
  E_const = Q6_Vh_vsplat_R(EXP2_HF_e0);
  y_v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(y_v, x_v));
  y_v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(y_v, E_const));

  //    y = y * x + 1.0;
  y_v = Q6_Vqf16_vmpy_VhfVhf(y_v, x_v);
  E_const = Q6_Vh_vsplat_R(HF_ONE);
  y_v = Q6_Vqf16_vadd_Vqf16Vhf(y_v, E_const);

  // insert exponents
  //         y = ldexpf(y, k);
  //     y_v += k_v; // qf32
  //  modify exponent
  y_v = Q6_Vhf_equals_Vqf16(y_v);

  HVX_Vector ldexpf = k_v;
  ldexpf = Q6_Vh_vasl_VhR(ldexpf, 0x000A); // ldexpf <<= 10;
  y_v = Q6_Vh_vadd_VhVh(y_v, ldexpf); // y_v += ldexpf;

  return y_v;
};