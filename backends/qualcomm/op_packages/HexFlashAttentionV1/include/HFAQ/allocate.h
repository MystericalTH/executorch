#pragma once

#include "HTP/core/simple_reg.h"
#include "constant.h"

#define MAX(a, b) a > b ? a : b

// Scratch blocks usage:
// 1. QKxT:   [b0, b1]
// 2. ATT_T:  [b1]
// 3. ATT:    [b1, b0]
// 4. ATTxV:  [b0, b1]
// 5. ACC_T:  [b1]
static inline std::array<size_t, 2> hfaq_local_scratch_blocks(size_t v_emb) {
  constexpr size_t matmul_block = 4096;
  // sf usecase: *=2 (scratch is declared in fp16)
  constexpr size_t att_block = 2 * HFAQ_ACC_HEAD_TILE * HFAQ_KV_SEQ_TILE;
  const size_t acc_block = 2 * HFAQ_ACC_HEAD_TILE * v_emb;

  const size_t block_0 = MAX(matmul_block, att_block);
  const size_t block_1 = MAX(att_block, acc_block);

  return {block_0, block_1};
}