/*
 * mlp_model_new.h — Auto-generated TinyML 6-Mode Thermal Lookahead Engine
 * Target: STM32 Nucleo-64 F401RE (Cortex-M4, 16 MHz HSI)
 * Architecture: 4 Inputs -> 16 Neurons (ReLU) -> 1 Output (Linear Delta_T)
 * Forecast Horizon: +10 Seconds Ahead
 *
 * 6 BENCHMARK CONFIGURATIONS:
 * ------------------------------------------------------------------------------
 * Mode 0: MODE_32BIT_FPU            (32-bit Float, Hardware FPU)
 * Mode 1: MODE_32BIT_FIXED_NORMAL   (32-bit Fixed-Point, SDIV Division)
 * Mode 2: MODE_32BIT_FIXED_POT      (32-bit Fixed-Point, Single-Cycle PoT Shift)
 * Mode 3: MODE_16BIT_FPU            (16-bit Float in Flash, FP32 in Registers)
 * Mode 4: MODE_16BIT_FIXED_NORMAL   (16-bit Fixed-Point, Division / 2048)
 * Mode 5: MODE_16BIT_FIXED_POT_SIMD (16-bit Fixed-Point, PoT + ARM SMLAD SIMD)
 * ------------------------------------------------------------------------------
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

/* Set default active benchmark mode (0 to 5) or override via compiler flag -DBENCHMARK_MODE=X */
#ifndef BENCHMARK_MODE
#define BENCHMARK_MODE            MODE_32BIT_FPU
#endif

/*
 * TinyML Thermal Lookahead Inference Entrypoint
 *
 * Inputs:
 *   ploss_cmW — Power dissipation in centi-mW (100 = 1 mW)
 *   i_cmA     — Regulator input current in centi-mA (100 = 1 mA)
 *   amb_cC    — Ambient temperature in centi-°C (100 = 1 °C)
 *   hs_cC     — Current heatsink temperature in centi-°C (100 = 1 °C)
 *
 * Returns:
 *   Predicted heatsink temperature 10s ahead in centi-°C
 */
int32_t mlp_new_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC);

#endif /* MLP_MODEL_NEW_H */
