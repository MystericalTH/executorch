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
#include "package_optimization.h"

BEGIN_PKG_OP_DEFINITION(PKG_HexFlashAttentionQLocal);

// Compute HFAQ local block
// In (FP16):
//    - Q(1, H, 1, D)
//    - K(1, H, pS, D)
//    - V(1, H, pS, Dv)
//
// Out (FP16, Transposed):
//    - Local(1, Dv + 2, 1, H)
template <typename TensorType>
GraphStatus hexflashattentionqlocalImpl(
    PlainFloat16Tensor& out_0,
    const PlainFloat16Tensor& query,
    const PlainFloat16Tensor& key,
    const PlainFloat16Tensor& value,
    const PlainFloat16Tensor& attn_mask,
    const PlainFloatTensor& scale);

DEF_PACKAGE_OP(
    (hexflashattentionqlocalImpl<PlainFloat16Tensor>),
    "HexFlashAttentionQLocal")

DEF_PACKAGE_PARAM_ORDER("HexFlashAttentionQLocal", "scale", true, nullptr)

template <typename TensorType>
GraphStatus hexflashattentionqlocalImpl(
    PlainFloat16Tensor& out_0,
    const PlainFloat16Tensor& query,
    const PlainFloat16Tensor& key,
    const PlainFloat16Tensor& value,
    const PlainFloat16Tensor& attn_mask,
    const PlainFloatTensor& scale) {
  return GraphStatus::Success;
}

/*
#ifndef PREPARE_DISABLED
namespace {
class FlashAttentionQLocalImplConstructorHook : public hnnx::OpHookBase {
  // This is called after the output tensors are created, but before allocation.
  virtual GraphStatus pre_allocate(hnnx::OpIoPtrs const& iop, Op& op)
      const override {
    const size_t seq_tile_len = op.get_input(1)->dim(2);

    // (1, H, 1, pS)
    const size_t qk_mat_size = HFAQ_ACC_HEAD_TILE * seq_tile_len;

    size_t new_dims[4] = {
        1, 1, 1, block0_size + block1_size + block2_size + block3_size};
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
    (hexflashattentionqlocalImpl<PlainFloat16Tensor>),
    FlashAttentionQLocalImplConstructorHook)
#endif*/

END_PKG_OP_DEFINITION(PKG_HexFlashAttentionQLocal);