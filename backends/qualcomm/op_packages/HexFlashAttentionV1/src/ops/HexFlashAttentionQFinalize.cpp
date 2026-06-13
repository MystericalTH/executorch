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
#include "hvx/hvx_inv_ops.h"
#include "hvx/hvx_transpose_ops.h"

BEGIN_PKG_OP_DEFINITION(PKG_HexFlashAttentionQFinalize);

template <typename TensorType>
GraphStatus hexflashattentionqfinalizeImpl(
    PlainFloatTensor_TCM& out_0,
    const PlainFloatTensor_TCM& in_0,
    PlainFloatTensor_TCM& scratch);

DEF_PACKAGE_OP(
    (hexflashattentionqfinalizeImpl<PlainFloatTensor_TCM>),
    "HexFlashAttentionQFinalize")

DEF_TENSOR_PROPERTIES(
    Op("HexFlashAttentionQFinalize", "in_0"),
    Flat("*", "in_0"),
    Tcm("*", "in_0"))

template <typename TensorType>
GraphStatus hexflashattentionqfinalizeImpl(
    PlainFloatTensor_TCM& out_0,
    const PlainFloatTensor_TCM& in_0,
    PlainFloatTensor_TCM& scratch) {
  const auto v_emb = in_0.dim(1) - 2;
  const auto acc_size = v_emb * HFAQ_ACC_HEAD_TILE;

  auto acc_ptr = in_0.data_ptr();
  auto acc_vec = (HVX_Vector*)acc_ptr;

  auto tmp_ptr = scratch.data_ptr();
  auto tmp_vec = (HVX_Vector*)tmp_ptr;

  auto out_ptr = out_0.data_ptr();

  auto l_vec_ptr = (HVX_Vector*)(acc_ptr + acc_size) + 1;

  HVX_Vector l_inv = hvx_Vsf_vinv_Vsf(*l_vec_ptr);

  for (uint16_t i = 0; i < v_emb; ++i) {
    tmp_vec[i] = Q6_Vsf_vmpy_VsfVsf(l_inv, acc_vec[i]);
  }

  hvx_mat_transposeMxN_Vsf(out_ptr, tmp_ptr, v_emb, HFAQ_ACC_HEAD_TILE);

  return GraphStatus::Success;
}

#ifndef PREPARE_DISABLED
namespace {
class FlashAttentionQFinalizeImplConstructorHook : public hnnx::OpHookBase {
  // This is called after the output tensors are created, but before
  // allocation.
  virtual GraphStatus pre_allocate(hnnx::OpIoPtrs const& iop, Op& op)
      const override {
    const size_t heads = op.get_input(0)->dim(3);
    const size_t v_emb = op.get_input(0)->dim(1) - 2;
    const size_t block_0 = heads * v_emb;
    size_t new_dims[4] = {1, 1, 1, block_0};
    GraphStatus result =
        hnnx::change_output_tensor_shape(op, 1, iop.graph(), 4, new_dims);
    if (result != GraphStatus::Success) {
      errlog("!! change_output_tensor_shape failed");
    }
    return result;
  }
};
} // namespace

CTOR_OPHOOK(
    (hexflashattentionqfinalizeImpl<PlainFloatTensor_TCM>),
    FlashAttentionQFinalizeImplConstructorHook)
#endif

END_PKG_OP_DEFINITION(PKG_HexFlashAttentionQFinalize);