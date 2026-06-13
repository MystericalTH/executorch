//==============================================================================
// Auto Generated Code for HexFlashAttentionV1
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

#include "constant.h"
#include "hvx/hvx_matmul_ops.h"

BEGIN_PKG_OP_DEFINITION(PKG_TestOp2Input);

template <typename TensorType>
GraphStatus testop2inputImpl(
    Tensor& out_0,
    const Tensor& in_0,
    const Tensor& in_1,
    const Int32Tensor& op_id);

DEF_PACKAGE_OP((testop2inputImpl<Tensor>), "TestOp2Input")

DEF_PACKAGE_PARAM_ORDER("TestOp2Input", "op_id", true, nullptr)

template <typename TensorType = Tensor>
GraphStatus testop2inputImpl(
    Tensor& out_0,
    const Tensor& in_0,
    const Tensor& in_1,
    const Int32Tensor& op_id) {
  const auto op_id_val = *(int32_t*)op_id.raw_data_const();
  errlog("[Test] Op ID: %d", op_id_val);

  auto in_0_ptr = in_0.raw_data_const();
  auto in_1_ptr = in_1.raw_data_const();
  auto out_ptr = out_0.raw_data();

  auto start = std::chrono::steady_clock::now();

  switch (op_id_val) {
    case 50:
      hvx_Vhf_matmul1x64N_Vhf(
          (Float16*)out_ptr,
          (Float16*)in_0_ptr,
          (Float16*)in_1_ptr,
          in_0.dim(3),
          in_1.dim(3));
      break;
    default:
      return GraphStatus::ErrorBadInput;
  }

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::micro> elapsed = end - start;
  errlog("took: %lf μs", elapsed.count());

  return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_TestOp2Input);