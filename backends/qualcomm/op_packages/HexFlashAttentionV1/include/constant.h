#pragma once

// VTCM size: 8 * 1024 * 1024
#define VTCM_SIZE_BYTES 8388608

#define HF_BYTES 2
#define SF_BYTES 4

#define HFAQ_ACC_HEAD_TILE 32
#define HFAQ_KV_SEQ_TILE 64 // graph tiling
#define HFAQ_KV_SEQ_PROC_TILE 64 // in-kernel tiling

// Math
#define SF_NEG_INF 0xFF800000 // -inf

#define SF_TWO 0x40000000 // 2.0

#define SF_ONE 0x3F800000 // 1.0
#define HF_ONE 0x3C00 // 1.0

#define SF_HALF 0x3F000000 // 0.5
#define HF_ONE_HALF 0x3800 // 0.5

#define SF_INV_LN2 0x3FB8AA3B // 1 / ln(2)

#define EXP2_e5 0x3920FDDE // 0.000153534
#define EXP2_e4 0x3AAF9F29 // 0.00133989
#define EXP2_e3 0x3C1D96A6 // 0.00961844
#define EXP2_e2 0x3D635774 // 0.0555033
#define EXP2_e1 0x3E75FDEE // 0.240226
#define EXP2_e0 0x3F317218 // 0.693147

#define EXP2_HF_e5 0x0908 // 0.000153534
#define EXP2_HF_e4 0x157D // 0.00133989
#define EXP2_HF_e3 0x20ED // 0.00961844
#define EXP2_HF_e2 0x2B1B // 0.0555033
#define EXP2_HF_e1 0x33B0 // 0.240226
#define EXP2_HF_e0 0x398C // 0.693147

#define INV_MAGIC_c0 0x7EB504F3
#define INV_e0 0x3FF86FB4 // 1.94090888923
#define INV_e1 0x3FB7C3B6 // 1.43566017178