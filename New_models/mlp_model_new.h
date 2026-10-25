/*
 * mlp_model_new.h -- Auto-generated TinyML 6-Mode Stateless Thermal Predictor
 * Target: STM32 Nucleo-64 F401RE (Cortex-M4, 16 MHz HSI)
 *
 * Architecture: 4 inputs -> 32 neurons (ReLU) -> 1 output (Linear)
 * Features: [heatsink_temp_C, ambient_temp_C, power_loss_mW, current_mA]
 *
 * All 6 modes are compiled into firmware simultaneously. You can call
 * specific mode functions directly or use mlp_new_predict_temp_cC().
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
#define BENCHMARK_MODE            MODE_16BIT_FIXED_POT_SIMD
#endif

/* Individual mode inference entrypoints */
int32_t mlp_new_predict_mode0_fp32(int32_t hs_cC, int32_t amb_cC, int32_t ploss_cmW, int32_t i_cmA);
int32_t mlp_new_predict_mode1_int32_norm(int32_t hs_cC, int32_t amb_cC, int32_t ploss_cmW, int32_t i_cmA);
int32_t mlp_new_predict_mode2_int32_pot(int32_t hs_cC, int32_t amb_cC, int32_t ploss_cmW, int32_t i_cmA);
int32_t mlp_new_predict_mode3_fp16(int32_t hs_cC, int32_t amb_cC, int32_t ploss_cmW, int32_t i_cmA);
int32_t mlp_new_predict_mode4_int16_norm(int32_t hs_cC, int32_t amb_cC, int32_t ploss_cmW, int32_t i_cmA);
int32_t mlp_new_predict_mode5_int16_simd(int32_t hs_cC, int32_t amb_cC, int32_t ploss_cmW, int32_t i_cmA);

/* Master dispatcher entrypoint (dispatches based on BENCHMARK_MODE) */
int32_t mlp_new_predict_temp_cC(int32_t hs_cC, int32_t amb_cC, int32_t ploss_cmW, int32_t i_cmA);

#endif /* MLP_MODEL_NEW_H */
