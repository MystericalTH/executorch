#pragma once

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

#ifndef PREPARE_DISABLED

#define CAST_FP16(IN)   \
  WITH_SIZE(            \
      IN,               \
      WITH_OUTPUT_TYPE( \
          DType::Float16, 0, 1.0f, Op(FROM_DEFAULT_PACKAGE("Cast"), IN)))

#define CAST_OUTPUT(OP) Op(FROM_DEFAULT_PACKAGE("Cast"), WITH_SIZE("*", OP))

#define WITH_OUTPUT_FP16(OP) WITH_OUTPUT_TYPE(DType::Float16, 0, 1.0f, OP)

#define WITH_OUTPUT_FP32(OP) WITH_OUTPUT_TYPE(DType::Float32, 0, 1.0f, OP)

#define RELAXED_PRECISION_OP(TARGET, COND, REPL) \
  DEF_PACKAGE_OPTIMIZATION_WITH_FLAGS(           \
      GRAPH_CLEANUP, relaxed_precision_flag, TARGET, COND, REPL)

// HexFlashAttentionQ

#define HFAQ_SHOULD_TILE_KV                   \
  AND(GT(DIM_WIDTH("key"), HFAQ_KV_SEQ_TILE), \
      GT(DIM_WIDTH("value"), HFAQ_KV_SEQ_TILE))

#define HFAQ_LOCAL_OUTPUT_SHAPE   \
  gen_Shape(                      \
      DIM_BATCHES("query"),       \
      ADD(DIM_DEPTH("value"), 2), \
      1,                          \
      DIM_HEIGHT("query"))

// HFAQLocal(KV_slice) -> Out -> (1, Dv + 2, 1, H) -> FP32
#define HFAQ_LOCAL_RECURSIVE_OP(IDX)                             \
  WITH_SIZE(                                                     \
      HFAQ_LOCAL_OUTPUT_SHAPE,                                   \
      Op("HexFlashAttentionQLocal",                              \
         "query",                                                \
         RECURSIVE_SLICE("key", 2, IDX, HFAQ_KV_SEQ_TILE),       \
         RECURSIVE_SLICE("value", 2, IDX, HFAQ_KV_SEQ_TILE),     \
         RECURSIVE_SLICE("attn_mask", 3, IDX, HFAQ_KV_SEQ_TILE), \
         "scale"))

API_EXPORT static inline QuickShape broadcast_split_start(
    Replacement& rpx,
    Split_Context const& splitinfo,
    int dim,
    int num_all,
    int num_broadcast) {
  assert(num_all % num_broadcast == 0);
  size_t dims[4] = {0, 0, 0, 0};
  const size_t group_size = num_all / num_broadcast;
  const size_t group_idx = splitinfo.start / group_size;
  dims[dim] = group_idx;
  return QuickShape(dims[0], dims[1], dims[2], dims[3]);
}

API_EXPORT static inline QuickShape broadcast_split_size(
    Replacement& rpx,
    Split_Context const& splitinfo,
    OpRef const& orig,
    int dim) {
  size_t dims[4] = {
      orig.dim(rpx.graph(), 0),
      orig.dim(rpx.graph(), 1),
      orig.dim(rpx.graph(), 2),
      orig.dim(rpx.graph(), 3)};
  dims[dim] = splitinfo.size;
  return QuickShape(dims[0], dims[1], dims[2], dims[3]);
}

// Broadcast slices from smaller (N_BROADCAST) to larger (N_ALL)
// Example:
//      - A(1,32,S,D), B(1,8,S,D): Broadcast B[,0] -> A[,0:4]
#define BROADCAST_SLICE(ARG, CTX, SLICE_DIM, N_ALL, N_BROADCAST)      \
  AUTOSPLIT_SLICE(                                                    \
      ARG,                                                            \
      AUTOSPLIT_SHAPEFN_APPLY(                                        \
          broadcast_split_start, CTX, SLICE_DIM, N_ALL, N_BROADCAST), \
      AUTOSPLIT_SHAPEFN_APPLY(broadcast_split_size, CTX, ARG))

API_EXPORT static inline QuickShape recursive_2_split_start(
    Replacement& rpx,
    OpRef const& input,
    int idx,
    int dim,
    int unit_op_tile) {
  size_t dims[4] = {0, 0, 0, 0};
  size_t dim_size = input.dim(rpx.graph(), dim);
  size_t split_size = unit_op_tile;
  while ((split_size << 1) < dim_size) {
    split_size = split_size << 1;
  }
  dims[dim] = idx == 0 ? 0 : split_size;
  return QuickShape(dims[0], dims[1], dims[2], dims[3]);
}

API_EXPORT static inline QuickShape recursive_2_split_size(
    Replacement& rpx,
    OpRef const& input,
    int idx,
    int dim,
    int unit_op_tile) {
  assert(idx < 2);
  size_t dims[4] = {
      input.dim(rpx.graph(), 0),
      input.dim(rpx.graph(), 1),
      input.dim(rpx.graph(), 2),
      input.dim(rpx.graph(), 3),
  };
  size_t dim_size = dims[dim];
  size_t split_size = unit_op_tile;
  while ((split_size << 1) < dim_size) {
    split_size = split_size << 1;
  }
  dims[dim] = idx == 0 ? split_size : (dim_size - split_size);
  return QuickShape(dims[0], dims[1], dims[2], dims[3]);
}

#define RECURSIVE_SLICE(IN, DIM, IDX, UNIT_TILE)                       \
  AUTOSPLIT_SLICE(                                                     \
      IN,                                                              \
      SHAPEFN_APPLY(recursive_2_split_start, IN, IDX, DIM, UNIT_TILE), \
      SHAPEFN_APPLY(recursive_2_split_size, IN, IDX, DIM, UNIT_TILE))

#endif
