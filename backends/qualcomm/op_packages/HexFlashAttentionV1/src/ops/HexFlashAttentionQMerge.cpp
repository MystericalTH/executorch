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
#include "hvx/hvx_exp_ops.h"

BEGIN_PKG_OP_DEFINITION(PKG_HexFlashAttentionQMerge);

// Merge the results from 2 `HFAQLocal`
// In (FP32, Transposed):
//    - Local_0(1, Dv + 2, 1, H)
//    - Local_1(1, Dv + 2, 1, H)
//
// Out (FP32, Transposed):
//    - Out(1, Dv + 2, 1, H)
template <typename TensorType>
GraphStatus hexflashattentionqmergeImpl(
    PlainFloatTensor_TCM& out_0,
    const PlainFloatTensor_TCM& local_0,
    const PlainFloatTensor_TCM& local_1);

DEF_PACKAGE_OP(
    (hexflashattentionqmergeImpl<PlainFloatTensor_TCM>),
    "HexFlashAttentionQMerge")

DEF_TENSOR_PROPERTIES(
    Op("HexFlashAttentionQMerge", "local_0", "local_1"),
    Flat("*", "local_0", "local_1"),
    Tcm("*", "local_0", "local_1"))

static inline std::array<HVX_Vector*, 3> get_local_hvx_vec_ptrs(
    const PlainFloatTensor_TCM& local,
    const size_t acc_tile_size,
    const size_t max_tile_size) {
  float* local_ptr = local.data_ptr();
  float* local_max_ptr = local_ptr + acc_tile_size;
  float* local_l_ptr = local_max_ptr + max_tile_size;

  HVX_Vector* local_acc = (HVX_Vector*)local_ptr;
  HVX_Vector* local_max = (HVX_Vector*)local_max_ptr;
  HVX_Vector* local_l = (HVX_Vector*)local_l_ptr;

  return {local_acc, local_max, local_l};
}

template <typename TensorType>
GraphStatus hexflashattentionqmergeImpl(
    PlainFloatTensor_TCM& out_0,
    const PlainFloatTensor_TCM& local_0,
    const PlainFloatTensor_TCM& local_1) {
  errlog("HFAQ_Merge");

  const size_t emb_len = local_0.dim(1) - 2;

  const size_t acc_tile_size = emb_len * HFAQ_ACC_HEAD_TILE;
  const size_t max_tile_size = 1 * HFAQ_ACC_HEAD_TILE;

  float* out_ptr = out_0.data_ptr();
  float* out_max_ptr = out_ptr + acc_tile_size;
  float* out_l_ptr = out_max_ptr + max_tile_size;
  HVX_Vector* out_acc = (HVX_Vector*)out_ptr;
  HVX_Vector* out_max = (HVX_Vector*)out_max_ptr;
  HVX_Vector* out_l = (HVX_Vector*)out_l_ptr;

  auto [local_acc_0, local_max_0, local_l_0] =
      get_local_hvx_vec_ptrs(local_0, acc_tile_size, max_tile_size);

  auto [local_acc_1, local_max_1, local_l_1] =
      get_local_hvx_vec_ptrs(local_1, acc_tile_size, max_tile_size);

  HVX_Vector new_max = Q6_Vsf_vmax_VsfVsf(local_max_0[0], local_max_1[0]);

  HVX_Vector adj_0 =
      hvx_Vsf_vexp2_Vsf(Q6_Vsf_vsub_VsfVsf(local_max_0[0], new_max));
  HVX_Vector adj_l_0 = Q6_Vsf_vmpy_VsfVsf(local_l_0[0], adj_0);

  HVX_Vector adj_1 =
      hvx_Vsf_vexp2_Vsf(Q6_Vsf_vsub_VsfVsf(local_max_1[0], new_max));
  HVX_Vector adj_l_1 = Q6_Vsf_vmpy_VsfVsf(local_l_1[0], adj_1);

  for (size_t i = 0; i < emb_len; ++i) {
    HVX_Vector adj_acc_0 = Q6_Vsf_vmpy_VsfVsf(local_acc_0[i], adj_0);
    HVX_Vector adj_acc_1 = Q6_Vsf_vmpy_VsfVsf(local_acc_1[i], adj_1);

    out_acc[i] = Q6_Vsf_vadd_VsfVsf(adj_acc_0, adj_acc_1);
  }

  out_max[0] = new_max;
  out_l[0] = Q6_Vsf_vadd_VsfVsf(adj_l_0, adj_l_1);

  return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_HexFlashAttentionQMerge);