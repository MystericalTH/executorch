//==============================================================================
// Auto Generated Code for HexFlashAttentionV1
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_hook_base.h"
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
    const Int32Tensor& op_id,
    PlainFloat16Tensor& scratch_ptr);

DEF_PACKAGE_OP((testop2inputImpl<Tensor>), "TestOp2Input")

DEF_PACKAGE_PARAM_ORDER("TestOp2Input", "op_id", true, nullptr)

template <typename TensorType = Tensor>
GraphStatus testop2inputImpl(
    Tensor& out_0,
    const Tensor& in_0,
    const Tensor& in_1,
    const Int32Tensor& op_id,
    PlainFloat16Tensor& scratch_ptr) {
  const auto op_id_val = *(int32_t*)op_id.raw_data_const();
  errlog("[Test] Op ID: %d", op_id_val);

  auto in_0_ptr = in_0.raw_data_const();
  auto in_1_ptr = in_1.raw_data_const();
  auto out_ptr = out_0.raw_data();

  HVX_Vector scaleline_qf32 =
      Q6_Vqf32_vadd_VsfVsf(Q6_V_vsplat_R(SF_ONE), Q6_V_vzero());

  auto start = std::chrono::steady_clock::now();

  switch (op_id_val) {
    case 50:
      hvx_Vhf_matmul1x64N_Vhf(
          (Float16*)out_ptr,
          (Float16*)in_0_ptr,
          (Float16*)in_1_ptr,
          in_0.dim(3),
          in_1.dim(3),
          scaleline_qf32);
      break;
    case 51:
      hvx_Vhf_matmul1x64N_transpose_Vhf(
          (Float16*)out_ptr,
          (Float16*)in_0_ptr,
          (Float16*)in_1_ptr,
          in_0.dim(3),
          in_1.dim(2),
          scaleline_qf32,
          scratch_ptr.data_ptr());
      break;
    default:
      return GraphStatus::ErrorBadInput;
  }

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::micro> elapsed = end - start;
  errlog("took: %lf μs", elapsed.count());

  return GraphStatus::Success;
}

#ifndef PREPARE_DISABLED
namespace {
class TestOp2InputImplConstructorHook : public hnnx::OpHookBase {
  // This is called after the output tensors are created, but before
  // allocation.
  virtual GraphStatus pre_allocate(hnnx::OpIoPtrs const& iop, Op& op)
      const override {
    size_t new_dims[4] = {1, 1, 1, 4096};
    GraphStatus result =
        hnnx::change_output_tensor_shape(op, 1, iop.graph(), 4, new_dims);
    if (result != GraphStatus::Success) {
      errlog("!! change_output_tensor_shape failed");
    }
    return result;
  }
};
} // namespace

CTOR_OPHOOK((testop2inputImpl<Tensor>), TestOp2InputImplConstructorHook)
#endif

END_PKG_OP_DEFINITION(PKG_TestOp2Input);