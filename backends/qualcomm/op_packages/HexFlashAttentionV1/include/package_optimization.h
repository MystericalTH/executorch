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

API_EXPORT QuickShape broadcast_split_start(
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

API_EXPORT QuickShape broadcast_split_size(
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

API_EXPORT QuickShape recursive_2_split_start(
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
  errlog("split start: %d %d %d %d", dims[0], dims[1], dims[2], dims[3]);
  return QuickShape(dims[0], dims[1], dims[2], dims[3]);
}

API_EXPORT QuickShape recursive_2_split_size(
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
  errlog("split size: %d %d %d %d", dims[0], dims[1], dims[2], dims[3]);
  return QuickShape(dims[0], dims[1], dims[2], dims[3]);
}

#define RECURSIVE_SLICE(IN, DIM, IDX, UNIT_TILE)                       \
  AUTOSPLIT_SLICE(                                                     \
      IN,                                                              \
      SHAPEFN_APPLY(recursive_2_split_start, IN, IDX, DIM, UNIT_TILE), \
      SHAPEFN_APPLY(recursive_2_split_size, IN, IDX, DIM, UNIT_TILE))

#endif
