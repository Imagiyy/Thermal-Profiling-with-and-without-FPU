/*
 * mlp_model_new.h -- Auto-generated TinyML 6-Mode Stateless Thermal Predictor
 * Target: STM32 Nucleo-64 F401RE (Cortex-M4, 16 MHz HSI)
 *
 * Architecture: 4 inputs -> 32 neurons (ReLU) -> 1 output (Linear)
 * Features: [heatsink_temp_C, ambient_temp_C, power_loss_mW, current_mA]
 *
 * WHY 4 inputs (not 2+elapsed_time):
 *   The previous model used elapsed_time (g_sample_count) which always
 *   started at 0 regardless of where in the heating curve prediction began.
 *   Starting mid-heating caused large errors. All 4 inputs here are available
 *   every sample loop with no internal counter -- fully stateless.
 *
 * 6 BENCHMARK CONFIGURATIONS:
 * ----------------------------------------------------------------------------
 * 32-BIT SUITE:
 *   MODE_32BIT_FPU            (0) -- 32-bit Float with Hardware FPU (FP32)
 *   MODE_32BIT_FIXED_NORMAL   (1) -- 32-bit Int: Normal division (/ 10000)
 *   MODE_32BIT_FIXED_POT      (2) -- 32-bit Int: Power-of-2 bit-shift (>> 16)
 *
 * 16-BIT SUITE (With DSP SIMD & Dynamic Sparsity Optimizations):
 *   MODE_16BIT_FPU            (3) -- 16-bit Float with HW FPU (FP16/FP32)
 *   MODE_16BIT_FIXED_NORMAL   (4) -- 16-bit Int: Normal division (/ 2048)
 *   MODE_16BIT_FIXED_POT_SIMD (5) -- 16-bit Int: PoT >> 11 + ARM SMLAD SIMD
 * ----------------------------------------------------------------------------
 */

#ifndef MLP_MODEL_NEW_H
#define MLP_MODEL_NEW_H

#include <stdint.h>

#define MODE_32BIT_FPU            0
#define MODE_32BIT_FIXED_NORMAL   1
#define MODE_32BIT_FIXED_POT      2
#define MODE_16BIT_FPU            3
#define MODE_16BIT_FIXED_NORMAL   4
#define MODE_16BIT_FIXED_POT_SIMD 5

#ifndef BENCHMARK_MODE
#define BENCHMARK_MODE            MODE_32BIT_FPU
#endif

/*
 * Predict heatsink temperature 1 second ahead (centi-degC).
 * All inputs are stateless -- no elapsed_time counter needed.
 *   hs_cC     : current heatsink temp (centi-degC)
 *   amb_cC    : ambient temp          (centi-degC)
 *   ploss_cmW : power dissipation     (centi-mW = mW*100)
 *   i_cmA     : current               (centi-mA = mA*100)
 */
int32_t mlp_new_predict_temp_cC(int32_t hs_cC, int32_t amb_cC,
                                 int32_t ploss_cmW, int32_t i_cmA);

#endif /* MLP_MODEL_NEW_H */
