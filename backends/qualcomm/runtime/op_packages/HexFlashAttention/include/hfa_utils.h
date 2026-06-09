#pragma once

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

static inline void log_hfa_input(
    PlainFloat16Tensor_TCM& out_0,
    const PlainFloat16Tensor_TCM& query,
    const PlainFloat16Tensor& key,
    const PlainFloat16Tensor& value,
    const PlainFloat16Tensor& attn_mask,
    const PlainFloatTensor& scale,
    PlainFloat16Tensor_TCM& scratch) {
  auto [qB, qH, qW, qD] = query.dims();
  auto [kB, kH, kW, kD] = key.dims();
  auto [vB, vH, vW, vD] = value.dims();
  auto [amB, amH, amW, amD] = attn_mask.dims();
  auto [oB, oH, oW, oD] = out_0.dims();
  auto [scB, scH, scW, scD] = scratch.dims();
  errlog(
      "[HexFlashAttention]\n"
      "query dim: %d %d %d %d\n"
      "key dim: %d %d %d %d\n"
      "value dim: %d %d %d %d\n"
      "attn mask dim: %d %d %d %d\n"
      "out dim: %d %d %d %d, size: %d\n"
      "scratch dim: %d %d %d %d, blocks: %d, size: %d",
      qB,
      qH,
      qW,
      qD,
      kB,
      kH,
      kW,
      kD,
      vB,
      vH,
      vW,
      vD,
      amB,
      amH,
      amW,
      amD,
      oB,
      oH,
      oW,
      oD,
      out_0.blocktab_blocksize(),
      scB,
      scH,
      scW,
      scD,
      scratch.blocktab_len(),
      scratch.blocktab_blocksize());
}

static inline void log_hvx_vec(Float16* vec) {
  errlog("vector start");
  for (uint8_t i = 0; i < 64; i += 16) {
    errlog(
        "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f",
        float(vec[i]),
        float(vec[i + 1]),
        float(vec[i + 2]),
        float(vec[i + 3]),
        float(vec[i + 4]),
        float(vec[i + 5]),
        float(vec[i + 6]),
        float(vec[i + 7]),
        float(vec[i + 8]),
        float(vec[i + 9]),
        float(vec[i + 10]),
        float(vec[i + 11]),
        float(vec[i + 12]),
        float(vec[i + 13]),
        float(vec[i + 14]),
        float(vec[i + 15]));
  }
  errlog("vector end");
}

// fp16 scale value adjusted by 1/ln(2)
static inline Float16 get_fp16_adjusted_scale(const PlainFloatTensor& scale) {
  constexpr float INV_LN2 = 1.44269504089;
  return Float16(((float*)scale.data_ptr())[0] * INV_LN2);
}

/**
 * get sizes for computation
 *
 * @tparam QUERY_BLOCK_LEN query seq block length
 * @tparam KV_BLOCK_LEN KV seq block length
 *
 * @returns
 * - QK embedding length
 *
 * - V embedding length
 *
 * - num chunks of KV blocks
 *
 * - num elements of key in each block
 *
 * - num elements of value in each block
 *
 * - num elements in Q @ K.t
 */
template <uint16_t Q_BLOCK_LEN, uint16_t KV_BLOCK_LEN>
static inline std::array<const size_t, 6> get_comp_sizes(
    const PlainFloat16Tensor& key,
    const PlainFloat16Tensor& value) {
  const size_t K_SEQ_LEN = key.dim(2);
  const size_t D = key.dim(3);
  const size_t V_D = value.dim(3);
  const size_t kv_blocks = K_SEQ_LEN / KV_BLOCK_LEN;
  // num elements in each key chunk
  const size_t key_size = KV_BLOCK_LEN * D;
  // num elements in each value chunk
  const size_t value_size = KV_BLOCK_LEN * V_D;
  const size_t qkmul_size = Q_BLOCK_LEN * Q_BLOCK_LEN;

  return {D, V_D, kv_blocks, key_size, value_size, qkmul_size};
}

// Allocate TCM scratch ptrs
// - block0: out_acc (v_emb_len, Q_BLOCK_LEN)
// - block1: (key/value) (KV_BLOCK_LEN, MAX(qk_emb_len, v_emb_len))
// - block2: qk_mul (Q_BLOCK_LEN, KV_BLOCK_LEN)
// - block3: tmp (KV_BLOCK_LEN, MAX(qk_emb_len, v_emb_len))
template <uint16_t Q_BLOCK_LEN, uint16_t KV_BLOCK_LEN>
static inline std::array<Float16*, 4> allocate_tcm_scratch_ptrs(
    PlainFloat16Tensor_TCM& scratch,
    const size_t qk_emb_len,
    const size_t v_emb_len) {
  Float16* block0_ptr = scratch.data_ptr();
  const size_t block0_size = v_emb_len * Q_BLOCK_LEN;

  Float16* block1_ptr = block0_ptr + block0_size;
  const size_t block1_size = KV_BLOCK_LEN * max_i32(qk_emb_len, v_emb_len);

  Float16* block2_ptr = block1_ptr + block1_size;
  const size_t block2_size = Q_BLOCK_LEN * KV_BLOCK_LEN;

  Float16* block3_ptr = block2_ptr + block2_size;

  return {block0_ptr, block1_ptr, block2_ptr, block3_ptr};
}