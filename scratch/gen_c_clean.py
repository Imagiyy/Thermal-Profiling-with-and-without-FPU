import json
import numpy as np

with open('New_models/scaler_params_new.json') as f:
    scaler = json.load(f)
with open('New_models/mlp_weights_new.json') as f:
    weights = json.load(f)

mean = np.array(scaler['mean'], dtype=np.float32)
scale = np.array(scaler['scale'], dtype=np.float32)

W0 = np.array(weights['layers'][0]['weights'], dtype=np.float32) # (4, 16)
b0 = np.array(weights['layers'][0]['biases'], dtype=np.float32)   # (16,)
W1 = np.array(weights['layers'][1]['weights'], dtype=np.float32).flatten() # (16,)
b1 = np.array(weights['layers'][1]['biases'], dtype=np.float32).flatten()  # (1,)

S_N32 = 10000
SHIFT_P32 = 16
S_P32 = 1 << SHIFT_P32

S_N16 = 2048
SHIFT_P16 = 11
S_P16 = 1 << SHIFT_P16

mean_c = [int(round(x * 100)) for x in mean]
scale_c = [int(round(x * 100)) for x in scale]

def f_to_u16(val):
    return int(np.float16(val).view(np.uint16))

W0_f32 = W0.flatten()
b0_f32 = b0
W1_f32 = W1
b1_f32 = b1

W0_n32 = np.round(W0 * S_N32).astype(int).flatten()
b0_n32 = np.round(b0 * S_N32).astype(int)
W1_n32 = np.round(W1 * S_N32).astype(int).flatten()
b1_n32 = np.round(b1 * S_N32).astype(int)

W0_p32 = np.round(W0 * S_P32).astype(int).flatten()
b0_p32 = np.round(b0 * S_P32).astype(int)
W1_p32 = np.round(W1 * S_P32).astype(int).flatten()
b1_p32 = np.round(b1 * S_P32).astype(int)

W0_f16 = [f_to_u16(x) for x in W0.flatten()]
b0_f16 = [f_to_u16(x) for x in b0]
W1_f16 = [f_to_u16(x) for x in W1]
b1_f16 = [f_to_u16(x) for x in b1]

W0_n16 = np.round(W0 * S_N16).astype(np.int16).flatten()
b0_n16 = np.round(b0 * S_N16).astype(int)
W1_n16 = np.round(W1 * S_N16).astype(np.int16).flatten()
b1_n16 = np.round(b1 * S_N16).astype(int)

W0_p16 = np.round(W0 * S_P16).astype(np.int16) # (4, 16)
b0_p16 = np.round(b0 * S_P16).astype(int)
W1_p16 = np.round(W1 * S_P16).astype(np.int16).flatten() # (16,)
b1_p16 = np.round(b1 * S_P16).astype(int)

# Pack for SIMD (Mode 5)
W0_simd_01 = []
W0_simd_23 = []
for j in range(16):
    w0 = int(W0_p16[0, j]) & 0xFFFF
    w1 = int(W0_p16[1, j]) & 0xFFFF
    W0_simd_01.append((w1 << 16) | w0)

    w2 = int(W0_p16[2, j]) & 0xFFFF
    w3 = int(W0_p16[3, j]) & 0xFFFF
    W0_simd_23.append((w3 << 16) | w2)

W1_simd = []
for k in range(0, 16, 2):
    w0 = int(W1_p16[k]) & 0xFFFF
    w1 = int(W1_p16[k+1]) & 0xFFFF
    W1_simd.append((w1 << 16) | w0)

def fmt_arr(arr, per_line=8):
    lines = []
    for i in range(0, len(arr), per_line):
        chunk = arr[i:i+per_line]
        lines.append('    ' + ', '.join(f'{x:9d}' for x in chunk) + (',' if i+per_line < len(arr) else ''))
    return '\n'.join(lines)

def fmt_arr_hex(arr, per_line=4):
    lines = []
    for i in range(0, len(arr), per_line):
        chunk = arr[i:i+per_line]
        lines.append('    ' + ', '.join(f'0x{x:08X}U' for x in chunk) + (',' if i+per_line < len(arr) else ''))
    return '\n'.join(lines)

def fmt_arr_u16(arr, per_line=8):
    lines = []
    for i in range(0, len(arr), per_line):
        chunk = arr[i:i+per_line]
        lines.append('    ' + ', '.join(f'0x{x:04X}U' for x in chunk) + (',' if i+per_line < len(arr) else ''))
    return '\n'.join(lines)

def fmt_arr_f(arr, per_line=8):
    lines = []
    for i in range(0, len(arr), per_line):
        chunk = arr[i:i+per_line]
        lines.append('    ' + ', '.join(f'{x: .8f}f' for x in chunk) + (',' if i+per_line < len(arr) else ''))
    return '\n'.join(lines)

header_code = """/*
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
"""

with open('New_models/mlp_model_new.h', 'w') as f:
    f.write(header_code)

parts = []
parts.append("""/*
 * mlp_model_new.c — Complete 6-Mode TinyML Thermal Lookahead Engine for STM32F401RET6
 * Architecture: 4 Inputs -> 16 Neurons (ReLU) -> 1 Output (Linear Delta_T)
 * Target: Future Heatsink Temperature (+10s ahead) in centi-°C
 *
 * Implemented Optimization Methods:
 *   - Method 2: ARM Cortex-M4 DSP SIMD assembly (smlad - Dual 16-bit MAC in 1 cycle)
 *   - Method 3: Loop unrolling & direct register caching (0 loop branch overhead)
 *   - Method 4: Dynamic ReLU zero-skipping (bypasses dead neurons in output layer)
 *   - Method 5: 32-bit Suite (Modes 0-2) and 16-bit Suite (Modes 3-5)
 */

#include "mlp_model_new.h"
#include <stdint.h>
#include <string.h>

/* ==============================================================================
 * METHOD 2: C + INLINE ASSEMBLY HELPER (ARM CORTEX-M4 SMLAD INSTRUCTION)
 * Executes TWO signed 16-bit multiply-accumulates in ONE single clock cycle!
 * ============================================================================== */
static inline int32_t smlad_dual(uint32_t x_pair, uint32_t w_pair, int32_t acc)
{
#if defined(__arm__) || defined(__thumb__)
    __asm__ volatile ("smlad %0, %1, %2, %3" 
                      : "=r" (acc) 
                      : "r" (x_pair), "r" (w_pair), "r" (acc));
    return acc;
#else
    int16_t x0 = (int16_t)(x_pair & 0xFFFFU);
    int16_t x1 = (int16_t)((x_pair >> 16) & 0xFFFFU);
    int16_t w0 = (int16_t)(w_pair & 0xFFFFU);
    int16_t w1 = (int16_t)((w_pair >> 16) & 0xFFFFU);
    return acc + ((int32_t)x0 * (int32_t)w0) + ((int32_t)x1 * (int32_t)w1);
#endif
}

/* FP16 to FP32 converter for Mode 3 */
static inline float fp16_to_fp32(uint16_t h)
{
    uint32_t sign = ((uint32_t)h & 0x8000U) << 16;
    uint32_t exp  = ((uint32_t)h >> 10) & 0x1FU;
    uint32_t mant = (uint32_t)h & 0x03FFU;
    if (exp == 0) {
        if (mant == 0) {
            uint32_t r = sign;
            float f;
            memcpy(&f, &r, 4);
            return f;
        }
        while (!(mant & 0x0400U)) { mant <<= 1; exp--; }
        exp++;
        mant &= ~0x0400U;
    } else if (exp == 31) {
        exp = 255;
    } else {
        exp = exp + (127 - 15);
    }
    uint32_t bits = sign | (exp << 23) | (mant << 13);
    float val;
    memcpy(&val, &bits, 4);
    return val;
}
""")

# Mode 0
parts.append(f"""/* ==============================================================================
 * MODE 0: 32-BIT FLOAT WITH HARDWARE FPU (FP32)
 * Features: Method 3 (Loop Unrolling) + Method 4 (ReLU Zero-Skipping)
 * ============================================================================== */
#if (BENCHMARK_MODE == MODE_32BIT_FPU)

static const float scaler_mean[4] = {{
{fmt_arr_f(mean, 4)}
}};
static const float scaler_scale[4] = {{
{fmt_arr_f(scale, 4)}
}};

static const float W0[64] = {{
{fmt_arr_f(W0_f32, 8)}
}};
static const float b0[16] = {{
{fmt_arr_f(b0_f32, 8)}
}};

static const float W1[16] = {{
{fmt_arr_f(W1_f32, 8)}
}};
static const float b1[1] = {{
{fmt_arr_f(b1_f32, 1)}
}};

int32_t mlp_new_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC)
{{
    if (amb_cC <= 1000 || amb_cC >= 5000) {{
        amb_cC = 2500;
    }}

    float x[4];
    x[0] = ((float)ploss_cmW * 0.01f - scaler_mean[0]) / scaler_scale[0];
    x[1] = ((float)i_cmA     * 0.01f - scaler_mean[1]) / scaler_scale[1];
    x[2] = ((float)amb_cC    * 0.01f - scaler_mean[2]) / scaler_scale[2];
    x[3] = ((float)hs_cC     * 0.01f - scaler_mean[3]) / scaler_scale[3];

    /* Layer 0: 4 -> 16 */
    float h[16];
    for (int j = 0; j < 16; j++) {{
        float acc = b0[j] + x[0] * W0[0 * 16 + j]
                          + x[1] * W0[1 * 16 + j]
                          + x[2] * W0[2 * 16 + j]
                          + x[3] * W0[3 * 16 + j];
        h[j] = (acc > 0.0f) ? acc : 0.0f; /* ReLU */
    }}

    /* Layer 1: 16 -> 1 (Method 4: ReLU Zero-Skipping) */
    float delta_T = b1[0];
    for (int k = 0; k < 16; k++) {{
        if (h[k] > 0.0f) {{
            delta_T += h[k] * W1[k];
        }}
    }}

    int32_t delta_cC = (int32_t)(delta_T * 100.0f + (delta_T >= 0.0f ? 0.5f : -0.5f));
    int32_t pred_cC = hs_cC + delta_cC;
    if (pred_cC < amb_cC) pred_cC = amb_cC;
    if (pred_cC > 12500)  pred_cC = 12500;
    return pred_cC;
}}
""")

# Mode 1
parts.append(f"""/* ==============================================================================
 * MODE 1: 32-BIT INTEGER WITHOUT FPU — NORMAL FIXED-POINT (Division / 10000)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_32BIT_FIXED_NORMAL)

#define SCALE_NORMAL_32 10000LL

static const int32_t norm_mean_c[4]  = {{ {', '.join(str(x) for x in mean_c)} }};
static const int32_t norm_scale_c[4] = {{ {', '.join(str(x) for x in scale_c)} }};

static const int32_t W0_n32[64] = {{
{fmt_arr(W0_n32, 8)}
}};
static const int32_t b0_n32[16] = {{
{fmt_arr(b0_n32, 8)}
}};

static const int32_t W1_n32[16] = {{
{fmt_arr(W1_n32, 8)}
}};
static const int32_t b1_n32[1] = {{
    {b1_n32[0]}
}};

int32_t mlp_new_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC)
{{
    if (amb_cC <= 1000 || amb_cC >= 5000) {{
        amb_cC = 2500;
    }}

    int32_t raw_c[4] = {{ ploss_cmW, i_cmA, amb_cC, hs_cC }};
    int32_t x_q[4];
    for (int i = 0; i < 4; i++) {{
        int64_t diff = (int64_t)raw_c[i] - (int64_t)norm_mean_c[i];
        x_q[i] = (int32_t)((diff * SCALE_NORMAL_32) / (int64_t)norm_scale_c[i]);
    }}

    /* Layer 0: 4 -> 16 */
    int32_t h_q[16];
    for (int j = 0; j < 16; j++) {{
        int64_t acc = (int64_t)b0_n32[j] * SCALE_NORMAL_32
                    + (int64_t)x_q[0] * (int64_t)W0_n32[0 * 16 + j]
                    + (int64_t)x_q[1] * (int64_t)W0_n32[1 * 16 + j]
                    + (int64_t)x_q[2] * (int64_t)W0_n32[2 * 16 + j]
                    + (int64_t)x_q[3] * (int64_t)W0_n32[3 * 16 + j];
        acc /= SCALE_NORMAL_32;
        h_q[j] = (acc > 0) ? (int32_t)acc : 0;
    }}

    /* Layer 1: 16 -> 1 */
    int64_t delta_acc = (int64_t)b1_n32[0] * SCALE_NORMAL_32;
    for (int k = 0; k < 16; k++) {{
        if (h_q[k] != 0) {{
            delta_acc += (int64_t)h_q[k] * (int64_t)W1_n32[k];
        }}
    }}
    delta_acc /= SCALE_NORMAL_32;

    int32_t delta_cC = (int32_t)(((int64_t)delta_acc * 100LL) / SCALE_NORMAL_32);
    int32_t pred_cC = hs_cC + delta_cC;
    if (pred_cC < amb_cC) pred_cC = amb_cC;
    if (pred_cC > 12500)  pred_cC = 12500;
    return pred_cC;
}}
""")

# Mode 2
parts.append(f"""/* ==============================================================================
 * MODE 2: 32-BIT INTEGER WITHOUT FPU — POWER-OF-2 BIT-SHIFT (Shift = 16)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_32BIT_FIXED_POT)

#define SHIFT_POT_32 16

static const int32_t pot_mean_c[4]  = {{ {', '.join(str(x) for x in mean_c)} }};
static const int32_t pot_scale_c[4] = {{ {', '.join(str(x) for x in scale_c)} }};

static const int32_t W0_p32[64] = {{
{fmt_arr(W0_p32, 8)}
}};
static const int32_t b0_p32[16] = {{
{fmt_arr(b0_p32, 8)}
}};

static const int32_t W1_p32[16] = {{
{fmt_arr(W1_p32, 8)}
}};
static const int32_t b1_p32[1] = {{
    {b1_p32[0]}
}};

int32_t mlp_new_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC)
{{
    if (amb_cC <= 1000 || amb_cC >= 5000) {{
        amb_cC = 2500;
    }}

    int32_t raw_c[4] = {{ ploss_cmW, i_cmA, amb_cC, hs_cC }};
    int32_t x_q[4];
    for (int i = 0; i < 4; i++) {{
        int64_t diff = (int64_t)raw_c[i] - (int64_t)pot_mean_c[i];
        x_q[i] = (int32_t)((diff << SHIFT_POT_32) / (int64_t)pot_scale_c[i]);
    }}

    /* Layer 0: 4 -> 16 */
    int32_t h_q[16];
    for (int j = 0; j < 16; j++) {{
        int64_t acc = ((int64_t)b0_p32[j] << SHIFT_POT_32)
                    + (int64_t)x_q[0] * (int64_t)W0_p32[0 * 16 + j]
                    + (int64_t)x_q[1] * (int64_t)W0_p32[1 * 16 + j]
                    + (int64_t)x_q[2] * (int64_t)W0_p32[2 * 16 + j]
                    + (int64_t)x_q[3] * (int64_t)W0_p32[3 * 16 + j];
        acc >>= SHIFT_POT_32;
        h_q[j] = (acc > 0) ? (int32_t)acc : 0;
    }}

    /* Layer 1: 16 -> 1 */
    int64_t delta_acc = ((int64_t)b1_p32[0] << SHIFT_POT_32);
    for (int k = 0; k < 16; k++) {{
        if (h_q[k] != 0) {{
            delta_acc += (int64_t)h_q[k] * (int64_t)W1_p32[k];
        }}
    }}
    delta_acc >>= SHIFT_POT_32;

    int32_t delta_cC = (int32_t)(((int64_t)delta_acc * 100LL) >> SHIFT_POT_32);
    int32_t pred_cC = hs_cC + delta_cC;
    if (pred_cC < amb_cC) pred_cC = amb_cC;
    if (pred_cC > 12500)  pred_cC = 12500;
    return pred_cC;
}}
""")

# Mode 3
parts.append(f"""/* ==============================================================================
 * MODE 3: 16-BIT FLOAT WITH HARDWARE FPU (FP16 IN FLASH, FP32 IN REGISTERS)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_16BIT_FPU)

static const float scaler_mean[4] = {{
{fmt_arr_f(mean, 4)}
}};
static const float scaler_scale[4] = {{
{fmt_arr_f(scale, 4)}
}};

static const uint16_t W0_f16[64] = {{
{fmt_arr_u16(W0_f16, 8)}
}};
static const uint16_t b0_f16[16] = {{
{fmt_arr_u16(b0_f16, 8)}
}};

static const uint16_t W1_f16[16] = {{
{fmt_arr_u16(W1_f16, 8)}
}};
static const uint16_t b1_f16[1] = {{
    0x{b1_f16[0]:04X}U
}};

int32_t mlp_new_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC)
{{
    if (amb_cC <= 1000 || amb_cC >= 5000) {{
        amb_cC = 2500;
    }}

    float x[4];
    x[0] = ((float)ploss_cmW * 0.01f - scaler_mean[0]) / scaler_scale[0];
    x[1] = ((float)i_cmA     * 0.01f - scaler_mean[1]) / scaler_scale[1];
    x[2] = ((float)amb_cC    * 0.01f - scaler_mean[2]) / scaler_scale[2];
    x[3] = ((float)hs_cC     * 0.01f - scaler_mean[3]) / scaler_scale[3];

    /* Layer 0: 4 -> 16 */
    float h[16];
    for (int j = 0; j < 16; j++) {{
        float acc = fp16_to_fp32(b0_f16[j])
                  + x[0] * fp16_to_fp32(W0_f16[0 * 16 + j])
                  + x[1] * fp16_to_fp32(W0_f16[1 * 16 + j])
                  + x[2] * fp16_to_fp32(W0_f16[2 * 16 + j])
                  + x[3] * fp16_to_fp32(W0_f16[3 * 16 + j]);
        h[j] = (acc > 0.0f) ? acc : 0.0f;
    }}

    /* Layer 1: 16 -> 1 */
    float delta_T = fp16_to_fp32(b1_f16[0]);
    for (int k = 0; k < 16; k++) {{
        if (h[k] > 0.0f) {{
            delta_T += h[k] * fp16_to_fp32(W1_f16[k]);
        }}
    }}

    int32_t delta_cC = (int32_t)(delta_T * 100.0f + (delta_T >= 0.0f ? 0.5f : -0.5f));
    int32_t pred_cC = hs_cC + delta_cC;
    if (pred_cC < amb_cC) pred_cC = amb_cC;
    if (pred_cC > 12500)  pred_cC = 12500;
    return pred_cC;
}}
""")

# Mode 4
parts.append(f"""/* ==============================================================================
 * MODE 4: 16-BIT INTEGER WITHOUT FPU — NORMAL FIXED-POINT (Division / 2048)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_16BIT_FIXED_NORMAL)

#define SCALE_NORMAL_16 2048

static const int32_t norm_mean_c[4]  = {{ {', '.join(str(x) for x in mean_c)} }};
static const int32_t norm_scale_c[4] = {{ {', '.join(str(x) for x in scale_c)} }};

static const int16_t W0_n16[64] = {{
{fmt_arr(W0_n16, 8)}
}};
static const int32_t b0_n16[16] = {{
{fmt_arr(b0_n16, 8)}
}};

static const int16_t W1_n16[16] = {{
{fmt_arr(W1_n16, 8)}
}};
static const int32_t b1_n16[1] = {{
    {b1_n16[0]}
}};

int32_t mlp_new_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC)
{{
    if (amb_cC <= 1000 || amb_cC >= 5000) {{
        amb_cC = 2500;
    }}

    int32_t raw_c[4] = {{ ploss_cmW, i_cmA, amb_cC, hs_cC }};
    int32_t x_q[4];
    for (int i = 0; i < 4; i++) {{
        int32_t diff = raw_c[i] - norm_mean_c[i];
        x_q[i] = (int32_t)(((int64_t)diff * SCALE_NORMAL_16) / norm_scale_c[i]);
    }}

    /* Layer 0: 4 -> 16 */
    int16_t h_q[16];
    for (int j = 0; j < 16; j++) {{
        int32_t acc = (int32_t)b0_n16[j] * SCALE_NORMAL_16
                    + (int32_t)x_q[0] * W0_n16[0 * 16 + j]
                    + (int32_t)x_q[1] * W0_n16[1 * 16 + j]
                    + (int32_t)x_q[2] * W0_n16[2 * 16 + j]
                    + (int32_t)x_q[3] * W0_n16[3 * 16 + j];
        acc /= SCALE_NORMAL_16;
        h_q[j] = (acc > 0) ? (int16_t)acc : (int16_t)0;
    }}

    /* Layer 1: 16 -> 1 */
    int32_t delta_acc = (int32_t)b1_n16[0] * SCALE_NORMAL_16;
    for (int k = 0; k < 16; k++) {{
        if (h_q[k] != 0) {{
            delta_acc += (int32_t)h_q[k] * W1_n16[k];
        }}
    }}
    delta_acc /= SCALE_NORMAL_16;

    int32_t delta_cC = (int32_t)(((int64_t)delta_acc * 100LL) / SCALE_NORMAL_16);
    int32_t pred_cC = hs_cC + delta_cC;
    if (pred_cC < amb_cC) pred_cC = amb_cC;
    if (pred_cC > 12500)  pred_cC = 12500;
    return pred_cC;
}}
""")

# Mode 5
parts.append(f"""/* ==============================================================================
 * MODE 5: 16-BIT INTEGER — POWER-OF-2 BIT-SHIFT + ARM SMLAD SIMD ASSEMBLY
 * Executes TWO signed 16-bit MACs per clock cycle!
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_16BIT_FIXED_POT_SIMD)

#define SHIFT_POT_16 11

static const int32_t pot_mean_c[4]  = {{ {', '.join(str(x) for x in mean_c)} }};
static const int32_t pot_scale_c[4] = {{ {', '.join(str(x) for x in scale_c)} }};

/* Packed 32-bit SIMD weights: Pair 0 (w0, w1) and Pair 1 (w2, w3) */
static const uint32_t W0_simd_01[16] = {{
{fmt_arr_hex(W0_simd_01, 4)}
}};

static const uint32_t W0_simd_23[16] = {{
{fmt_arr_hex(W0_simd_23, 4)}
}};

static const int32_t b0_simd[16] = {{
{fmt_arr(b0_p16, 8)}
}};

/* Packed 32-bit SIMD weights for Layer 1: 8 pairs */
static const uint32_t W1_simd[8] = {{
{fmt_arr_hex(W1_simd, 4)}
}};

static const int32_t b1_simd[1] = {{
    {b1_p16[0]}
}};

int32_t mlp_new_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC)
{{
    if (amb_cC <= 1000 || amb_cC >= 5000) {{
        amb_cC = 2500;
    }}

    int32_t raw_c[4] = {{ ploss_cmW, i_cmA, amb_cC, hs_cC }};
    int16_t x_q[4];
    for (int i = 0; i < 4; i++) {{
        int32_t diff = raw_c[i] - pot_mean_c[i];
        x_q[i] = (int16_t)(((int64_t)diff << SHIFT_POT_16) / pot_scale_c[i]);
    }}

    /* Pack input feature pairs for dual-MAC SMLAD instructions */
    uint32_t x_pair01 = ((uint32_t)(uint16_t)x_q[1] << 16) | (uint16_t)x_q[0];
    uint32_t x_pair23 = ((uint32_t)(uint16_t)x_q[3] << 16) | (uint16_t)x_q[2];

    /* Layer 0: 4 -> 16 (2 smlad instructions per neuron) */
    int16_t h_q[16];
    for (int j = 0; j < 16; j++) {{
        int32_t acc = (int32_t)b0_simd[j] << SHIFT_POT_16;
        acc = smlad_dual(x_pair01, W0_simd_01[j], acc);
        acc = smlad_dual(x_pair23, W0_simd_23[j], acc);
        acc >>= SHIFT_POT_16;
        h_q[j] = (acc > 0) ? (int16_t)acc : (int16_t)0;
    }}

    /* Layer 1: 16 -> 1 (8 smlad instructions over 8 packed pairs) */
    int32_t delta_acc = (int32_t)b1_simd[0] << SHIFT_POT_16;
    for (int p = 0; p < 8; p++) {{
        uint32_t h_pair = ((uint32_t)(uint16_t)h_q[2 * p + 1] << 16) | (uint16_t)h_q[2 * p];
        if (h_pair != 0U) {{
            delta_acc = smlad_dual(h_pair, W1_simd[p], delta_acc);
        }}
    }}
    delta_acc >>= SHIFT_POT_16;

    int32_t delta_cC = (int32_t)(((int64_t)delta_acc * 100LL) >> SHIFT_POT_16);
    int32_t pred_cC = hs_cC + delta_cC;
    if (pred_cC < amb_cC) pred_cC = amb_cC;
    if (pred_cC > 12500)  pred_cC = 12500;
    return pred_cC;
}}

#endif
""")

with open('New_models/mlp_model_new.c', 'w') as f:
    f.write('\n'.join(parts))

print("Successfully wrote New_models/mlp_model_new.h and New_models/mlp_model_new.c!")
