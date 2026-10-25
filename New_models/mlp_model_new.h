/*
 * mlp_model_new.h -- Auto-generated TinyML 6-Mode Stateless Dual Thermal Predictor
 * Target: STM32 Nucleo-64 F401RE (Cortex-M4, 16 MHz HSI)
 *
 * Architecture: 4 inputs -> 32 neurons (ReLU) -> 2 outputs (Linear: 3s & 10s)
 * Features: [heatsink_temp_C, ambient_temp_C, power_loss_mW, current_mA]
 * Targets:  [+3s lookahead, +10s lookahead]
 *
 * All 6 modes are compiled into firmware simultaneously.
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

typedef struct {
    int32_t pred_3s_cC;
    int32_t pred_10s_cC;
} mlp_prediction_t;

/* Individual mode dual-inference entrypoints */
mlp_prediction_t mlp_new_predict_mode0_fp32(int32_t hs_cC, int32_t amb_cC, int32_t ploss_cmW, int32_t i_cmA);
mlp_prediction_t mlp_new_predict_mode1_int32_norm(int32_t hs_cC, int32_t amb_cC, int32_t ploss_cmW, int32_t i_cmA);
mlp_prediction_t mlp_new_predict_mode2_int32_pot(int32_t hs_cC, int32_t amb_cC, int32_t ploss_cmW, int32_t i_cmA);
mlp_prediction_t mlp_new_predict_mode3_fp16(int32_t hs_cC, int32_t amb_cC, int32_t ploss_cmW, int32_t i_cmA);
mlp_prediction_t mlp_new_predict_mode4_int16_norm(int32_t hs_cC, int32_t amb_cC, int32_t ploss_cmW, int32_t i_cmA);
mlp_prediction_t mlp_new_predict_mode5_int16_simd(int32_t hs_cC, int32_t amb_cC, int32_t ploss_cmW, int32_t i_cmA);

/* Master dispatcher entrypoints */
mlp_prediction_t mlp_new_predict_dual_cC(int32_t hs_cC, int32_t amb_cC, int32_t ploss_cmW, int32_t i_cmA);
int32_t mlp_new_predict_temp_cC(int32_t hs_cC, int32_t amb_cC, int32_t ploss_cmW, int32_t i_cmA);

#endif /* MLP_MODEL_NEW_H */
