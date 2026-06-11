#pragma once

#include "HTP/core/log.h"

#include "constant.h"

// Calculate HFAQLocal memory allocation
//
// In:
//      - Q(1,  H,  1,      Dqk)
//      - K(1,  H,  STile,  Dqk)
//      - V(1,  H,  STile,  Dv )
//
// Out:
//      - RowMax(1, H, 1, 1 )
//      - L     (1, H, 1, 1 )
//      - ACC   (1, H, 1, Dv)
//
// `num_query_seq_tile` is implied 1
static inline void calc_hfaq_local_scratch_malloc(
    size_t num_head_tile,
    size_t num_seq_tile,
    size_t qk_emb_len,
    size_t v_emb_len) {
  // input sizes
  const size_t query_t_size = num_head_tile * qk_emb_len;
  const size_t key_t_size = num_head_tile * num_seq_tile * qk_emb_len;
  const size_t value_t_size = num_head_tile * num_seq_tile * v_emb_len;

  errlog(
      "[HFAQ_MALLOC]"
      " Q: %d K: %d V: %d",
      query_t_size * HF_BYTES,
      key_t_size * HF_BYTES,
      value_t_size * HF_BYTES);
}

// Calculate HFAQMerge memory allocation
//
// In(2) / Out (1):
//      - RowMax(1, H, 1, 1 )
//      - L     (1, H, 1, 1 )
//      - ACC   (1, H, 1, Dv)
static inline void calc_hfaq_merge_scratch_malloc(
    size_t num_head_tile,
    size_t num_seq_tile,
    size_t qk_emb_len,
    size_t v_emb_len) {}