#pragma once

#include "HTP/core/simple_reg.h"
#include "constant.h"

// Scratch blocks usage:
// (b1) k.T: `(D, SEQ_TILE)`
// (b2) att_hf: `2 * (HEAD_TILE, SEQ_PROC_TILE)`
// (b3) att_sf: `= b2`
// *b3 sf multi-block transpose uses extra space, maybe inplace is possible?
// b1, b3 is not used at the same time, scratch blocks: `2 x max(b1,b2)`
static inline std::array<size_t, 2> hfaq_local_scratch_blocks(size_t qk_emb) {
  const size_t k_t_block = qk_emb * HFAQ_KV_SEQ_PROC_TILE;
  // sf usecase: *=2
  const size_t att_block = 2 * HFAQ_ACC_HEAD_TILE * HFAQ_KV_SEQ_PROC_TILE;

  const size_t max_block = att_block > k_t_block ? att_block : k_t_block;

  return {max_block, max_block};
}