#pragma once

#include "HTP/core/log.h"
#include "HTP/core/simple_reg.h"

#include "constant.h"

constexpr bool ENABLE_LOG = false;

template <typename DataType>
static inline void log_hvx_vector(DataType* ptr) {
  if (!ENABLE_LOG) {
    return;
  }
  errlog("START");
  for (size_t i = 0; i < 128 / sizeof(DataType); i += 8) {
    errlog(
        "HVX_Vector[%d:%d]"
        "%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f",
        i,
        i + 8,
        float(ptr[i]),
        float(ptr[i + 1]),
        float(ptr[i + 2]),
        float(ptr[i + 3]),
        float(ptr[i + 4]),
        float(ptr[i + 5]),
        float(ptr[i + 6]),
        float(ptr[i + 7]));
  }
  errlog("END");
}

static inline size_t calc_tensor_elems(const Tensor& tensor) {
  auto [t0, t1, t2, t3] = tensor.dims();
  return t0 * t1 * t2 * t3;
}

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
static inline void log_hfaq_local_memfp(
    PlainFloatTensor_TCM& out_0,
    const PlainFloat16Tensor_TCM& query,
    const PlainFloat16Tensor_TCM& key,
    const PlainFloat16Tensor_TCM& value,
    const PlainFloat16Tensor_TCM& attn_mask,
    const PlainFloatTensor_TCM& scale,
    PlainFloat16Tensor_TCM& scratch) {
  if (!ENABLE_LOG) {
    return;
  }
  errlog(
      "[HFAQLocal]\n"
      "[Input] query: %zu bytes, key: %zu bytes, value: %zu bytes, attn_mask: %zu bytes, scale: %zu bytes\n"
      "[Output] out: %zu bytes\n"
      "[Temp] scratch: %zu bytes",
      calc_tensor_elems(query) * HF_BYTES,
      calc_tensor_elems(key) * HF_BYTES,
      calc_tensor_elems(value) * HF_BYTES,
      calc_tensor_elems(attn_mask) * HF_BYTES,
      calc_tensor_elems(scale) * SF_BYTES,
      calc_tensor_elems(out_0) * SF_BYTES,
      calc_tensor_elems(scratch) * HF_BYTES);
}
