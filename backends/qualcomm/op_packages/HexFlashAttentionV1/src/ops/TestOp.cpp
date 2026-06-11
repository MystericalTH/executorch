//==============================================================================
// Auto Generated Code for HexFlashAttentionV1
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

#include "hvx/hvx_exp_ops.h"
#include "hvx/hvx_mat_ops.h"

BEGIN_PKG_OP_DEFINITION(PKG_TestOp);

template <typename TensorType>
GraphStatus
testopImpl(Tensor& out_0, const Tensor& in_0, const Int32Tensor& op_id);

DEF_PACKAGE_OP((testopImpl<Tensor>), "TestOp")

DEF_PACKAGE_PARAM_ORDER("TestOp", "op_id", true, nullptr)

template <typename TensorType = Tensor>
GraphStatus
testopImpl(Tensor& out_0, const Tensor& in_0, const Int32Tensor& op_id) {
  const int32_t op_id_val = ((int32_t*)op_id.raw_data_const())[0];
  errlog("[Test] Op ID: %d", op_id_val);

  auto in_0_ptr = in_0.raw_data_const();
  auto out_ptr = out_0.raw_data();

  HVX_Vector in_0_vec, result_vec;

  in_0_vec = ((HVX_Vector*)in_0_ptr)[0];

  bool set_out = true;

  auto start = std::chrono::steady_clock::now();

  switch (op_id_val) {
    case 0:
      result_vec = hvx_Vsf_vexp2_Vsf(in_0_vec);
      break;
    case 1:
      result_vec = hvx_Vhf_vexp2_Vhf(in_0_vec);
      break;
    default:
      return GraphStatus::ErrorBadInput;
  }

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::micro> elapsed = end - start;
  errlog("took: %lf μs", elapsed.count());

  if (set_out) {
    ((HVX_Vector*)out_ptr)[0] = result_vec;
  }

  return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_TestOp);