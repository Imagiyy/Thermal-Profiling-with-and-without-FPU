/*
 * mlp_model.h — Auto-generated TinyML 6-Mode Inference Engine
 * Target: STM32 Nucleo-64 F401RE (Cortex-M4, 16 MHz HSI)
 *
 * ZERO DATA LEAKAGE GUARANTEE:
 * Weights and normalization parameters were learned STRICTLY from early/mid
 * operational telemetry (first 80% of time) and validated on unseen future time (last 20%).
 *
 * 6 BENCHMARK CONFIGURATIONS:
 * ------------------------------------------------------------------------------
 * 32-BIT SUITE:
 *   MODE_32BIT_FPU            (0) — 32-bit Float with Hardware FPU (FP32)
 *   MODE_32BIT_FIXED_NORMAL   (1) — 32-bit Int Without FPU: Normal division (/ 10000)
 *   MODE_32BIT_FIXED_POT      (2) — 32-bit Int Without FPU: Power-of-2 bit-shift (>> 16)
 *
 * 16-BIT SUITE (With DSP SIMD & Dynamic Sparsity Optimizations):
 *   MODE_16BIT_FPU            (3) — 16-bit Float with Hardware FPU (FP16 storage / FP32 compute)
 *   MODE_16BIT_FIXED_NORMAL   (4) — 16-bit Int Without FPU: Normal division (/ 2048)
 *   MODE_16BIT_FIXED_POT_SIMD (5) — 16-bit Int Without FPU: PoT shift (>> 11) + ARM SMLAD SIMD Assembly
 * ------------------------------------------------------------------------------
 */

#ifndef MLP_MODEL_H
#define MLP_MODEL_H

#include <stdint.h>

#define MODE_32BIT_FPU            0
#define MODE_32BIT_FIXED_NORMAL   1
#define MODE_32BIT_FIXED_POT      2
#define MODE_16BIT_FPU            3
#define MODE_16BIT_FIXED_NORMAL   4
#define MODE_16BIT_FIXED_POT_SIMD 5

/* Set default active benchmark mode (0 to 5) or override via compiler flag -DBENCHMARK_MODE=X */
#ifndef BENCHMARK_MODE
#define BENCHMARK_MODE            MODE_32BIT_FPU
#endif

int32_t cnn_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA,
                             int32_t amb_cC,   int32_t hs_cC);

void mlp_reset_timebase(void);

#endif /* MLP_MODEL_H */
