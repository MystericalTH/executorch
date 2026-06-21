#pragma once

#include "HTP/core/log.h"
#include "HTP/core/simple_reg.h"

#include "constant.h"

#define ENABLE_LOG true

#ifdef ENABLE_LOG

#define TIMER_START                              \
  auto start = std::chrono::steady_clock::now(); \
  std::chrono::duration<double, std::micro> elapsed;

#define TIMER_RESET start = std::chrono::steady_clock::now();

#define TIMER_END(NAME)                               \
  elapsed = std::chrono::steady_clock::now() - start; \
  errlog(NAME " took: %lf μs", elapsed.count());

#else

#define TIMER_START
#define TIMER_RESET
#define TIMER_END(NAME)

#endif

template <typename DataType>
static inline void log_hvx_vector(DataType* ptr) {
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
  auto [kB, kH, kW, kD] = key.dims();
  auto [oB, oH, oW, oD] = out_0.dims();
  auto [sB, sH, sW, sD] = scratch.dims();
  errlog(
      "[HFAQLocal]\n"
      "[Input] query: %zu bytes, key (%d %d %d %d): %zu bytes, "
      "value: %zu bytes, attn_mask: %zu bytes, scale: %zu bytes\n"
      "[Output] out (%d %d %d %d): %zu bytes\n"
      "[Temp] scratch (%d %d %d %d): %zu bytes",
      calc_tensor_elems(query) * HF_BYTES,
      kB,
      kH,
      kW,
      kD,
      calc_tensor_elems(key) * HF_BYTES,
      calc_tensor_elems(value) * HF_BYTES,
      calc_tensor_elems(attn_mask) * HF_BYTES,
      calc_tensor_elems(scale) * SF_BYTES,
      oB,
      oH,
      oW,
      oD,
      calc_tensor_elems(out_0) * SF_BYTES,
      sB,
      sH,
      sW,
      sD,
      calc_tensor_elems(scratch) * HF_BYTES);
}
