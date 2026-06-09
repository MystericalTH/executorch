#pragma once

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

#define IEEE_VHF_EXPBIAS (15)
#define IEEE_VHF_EXPMASK (0x1F)
#define IEEE_VHF_MANTLEN (10)
#define IEEE_VHF_MANTMASK (0x3FF)
#define IEEE_VHF_MIMPMASK (0x400)

#define ONE_HF 0x3C00
#define ONE_HALF_HF 0x3800

#define e5 0x0908 // 0.000153534
#define e4 0x157D // 0.00133989
#define e3 0x20ED // 0.00961844
#define e2 0x2B1B // 0.0555033
#define e1 0x33B0 // 0.240226
#define e0 0x398C // 0.693147

static inline HVX_Vector cpu_Vhf_baseline_exp2_Vhf(
    HVX_Vector vec,
    Float16* scr_ptr) {
  ((HVX_Vector*)scr_ptr)[0] = vec;
  for (size_t i = 0; i < 64; ++i) {
    scr_ptr[i] = Float16(exp2(double(scr_ptr[i])));
  }
  return ((HVX_Vector*)scr_ptr)[0];
}

static inline HVX_Vector hvx_Vh_vfloor_VhfVhf(HVX_Vector Vu, HVX_Vector* Vd) {
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

#ifdef USE_ROUND_IN_FLOOR
  const_v = Q6_Vh_vsplat_R(ONE_HALF_HF);
  round_f = Q6_V_vadd_VhfVhf(Vu, const_v);
  round_f = Q6_Vhf_equals_V(round_f);
#else
  round_f = Vu;
#endif

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

static inline HVX_Vector hvx_Vhf_horner_exp2_Vhf(HVX_Vector vin) {
  HVX_Vector x_v, f_v;
  HVX_Vector x_qf16_v;
  HVX_Vector y_v;
  HVX_Vector half_hf = Q6_Vh_vsplat_R(ONE_HALF_HF);
  HVX_Vector one_h = Q6_Vh_vsplat_R(0x0001);
  HVX_Vector zero_v = Q6_V_vzero();
  HVX_Vector one_hf = Q6_Vh_vsplat_R(ONE_HF);
  HVX_Vector E_const;

  HVX_Vector k_v = hvx_Vh_vfloor_VhfVhf(vin, &f_v);

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
  E_const = Q6_Vh_vsplat_R(e5);
  y_v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(E_const, x_v));
  E_const = Q6_Vh_vsplat_R(e4);
  y_v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(y_v, E_const));

  //    y = E3 + y * x;
  E_const = Q6_Vh_vsplat_R(e3);
  y_v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(y_v, x_v));
  y_v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(y_v, E_const));

  //    y = E2 + y * x;
  E_const = Q6_Vh_vsplat_R(e2);
  y_v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(y_v, x_v));
  y_v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(y_v, E_const));

  //    y = E1 + y * x;
  E_const = Q6_Vh_vsplat_R(e1);
  y_v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(y_v, x_v));
  y_v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(y_v, E_const));

  //    y = E0 + y * x;
  E_const = Q6_Vh_vsplat_R(e0);
  y_v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(y_v, x_v));
  y_v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(y_v, E_const));

  //    y = y * x + 1.0;
  y_v = Q6_Vqf16_vmpy_VhfVhf(y_v, x_v);
  E_const = Q6_Vh_vsplat_R(ONE_HF);
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
