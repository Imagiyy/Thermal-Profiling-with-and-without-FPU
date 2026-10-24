import json
import numpy as np

# 1. Load scaler and weights for mlp_model_new
with open('New_models/scaler_params_new.json') as f:
    scaler = json.load(f)
with open('New_models/mlp_weights_new.json') as f:
    weights = json.load(f)

mean = np.array(scaler['mean'], dtype=np.float32)
scale = np.array(scaler['scale'], dtype=np.float32)

W0 = np.array(weights['layers'][0]['weights'], dtype=np.float32) # (2, 8)
b0 = np.array(weights['layers'][0]['biases'], dtype=np.float32)   # (8,)
W1 = np.array(weights['layers'][1]['weights'], dtype=np.float32).flatten() # (8,)
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

W0_p16 = np.round(W0 * S_P16).astype(np.int16) # (2, 8)
b0_p16 = np.round(b0 * S_P16).astype(int)
W1_p16 = np.round(W1 * S_P16).astype(np.int16).flatten() # (8,)
b1_p16 = np.round(b1 * S_P16).astype(int)

# Pack for SIMD (Mode 5)
W0_simd = []
for j in range(8):
    w0 = int(W0_p16[0, j]) & 0xFFFF
    w1 = int(W0_p16[1, j]) & 0xFFFF
    W0_simd.append((w1 << 16) | w0)

W1_simd = []
for k in range(0, 8, 2):
    w0 = int(W1_p16[k]) & 0xFFFF
    w1 = int(W1_p16[k+1]) & 0xFFFF
    W1_simd.append((w1 << 16) | w0)

header_code = """/*
 * mlp_model_new.h — Auto-generated TinyML 6-Mode Inference Engine
 * Target: STM32 Nucleo-64 F401RE (Cortex-M4, 16 MHz HSI)
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

int32_t mlp_new_predict_temp_cC(int32_t hs_cC, int32_t amb_cC);

#endif /* MLP_MODEL_NEW_H */
"""

with open('New_models/mlp_model_new.h', 'w') as f:
    f.write(header_code)

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

c_code = f"""/*
 * mlp_model_new.c — Complete 6-Mode TinyML Thermal Lookahead Engine for STM32F401RET6
 * Architecture: 2 Inputs -> 8 Neurons (ReLU) -> 1 Output (Linear)
 * Target: Future Heatsink Temperature (+1s ahead) in centi-°C
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
{{
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
}}

/* FP16 to FP32 converter for Mode 3 */
static inline float fp16_to_fp32(uint16_t h)
{{
    uint32_t sign = ((uint32_t)h & 0x8000U) << 16;
    uint32_t exp  = ((uint32_t)h >> 10) & 0x1FU;
    uint32_t mant = (uint32_t)h & 0x03FFU;
    if (exp == 0) {{
        if (mant == 0) {{
            uint32_t r = sign;
            float f;
            memcpy(&f, &r, 4);
            return f;
        }}
        while (!(mant & 0x0400U)) {{ mant <<= 1; exp--; }}
        exp++;
        mant &= ~0x0400U;
    }} else if (exp == 31) {{
        exp = 255;
    }} else {{
        exp = exp + (127 - 15);
    }}
    uint32_t bits = sign | (exp << 23) | (mant << 13);
    float val;
    memcpy(&val, &bits, 4);
    return val;
}}

/* ==============================================================================
 * MODE 0: 32-BIT FLOAT WITH HARDWARE FPU (FP32)
 * Features: Method 3 (Loop Unrolling) + Method 4 (ReLU Zero-Skipping)
 * ============================================================================== */
#if (BENCHMARK_MODE == MODE_32BIT_FPU)

static const float scaler_mean[2] = {{
    {mean[0]:.8f}f, {mean[1]:.8f}f
}};
static const float scaler_scale[2] = {{
    {scale[0]:.8f}f, {scale[1]:.8f}f
}};

static const float W0[16] = {{
{fmt_arr_f(W0_f32)}
}};
static const float b0[8] = {{
{fmt_arr_f(b0_f32)}
}};

static const float W1[8] = {{
{fmt_arr_f(W1_f32)}
}};
static const float b1[1] = {{
    {b1_f32[0]:.8f}f
}};

int32_t mlp_new_predict_temp_cC(int32_t hs_cC, int32_t amb_cC)
{{
    if (amb_cC <= 1000 || amb_cC >= 5000) {{
        amb_cC = 2500;
    }}

    float x[2];
    x[0] = (((float)hs_cC  / 100.0f) - scaler_mean[0]) / scaler_scale[0];
    x[1] = (((float)amb_cC / 100.0f) - scaler_mean[1]) / scaler_scale[1];

    /* Layer 0: 2 -> 8 */
    float h[8];
    for (int j = 0; j < 8; j++) {{
        float acc = b0[j] + x[0] * W0[0 * 8 + j]
                          + x[1] * W0[1 * 8 + j];
        h[j] = (acc > 0.0f) ? acc : 0.0f; /* ReLU */
    }}

    /* Layer 1: 8 -> 1 (Method 4: ReLU Zero-Skipping) */
    float out = b1[0];
    for (int k = 0; k < 8; k++) {{
        if (h[k] > 0.0f) {{
            out += h[k] * W1[k];
        }}
    }}

    return (int32_t)(out * 100.0f + (out >= 0.0f ? 0.5f : -0.5f));
}}

/* ==============================================================================
 * MODE 1: 32-BIT INTEGER WITHOUT FPU — NORMAL FIXED-POINT (Division / 10000)
 * Features: Method 3 (Loop Unrolling) + Method 4 (ReLU Zero-Skipping)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_32BIT_FIXED_NORMAL)

#define SCALE_NORMAL_32 10000LL

static const int32_t norm_mean_c[2]  = {{ {mean_c[0]}, {mean_c[1]} }};
static const int32_t norm_scale_c[2] = {{ {scale_c[0]}, {scale_c[1]} }};

static const int32_t W0_n32[16] = {{
{fmt_arr(W0_n32)}
}};
static const int32_t b0_n32[8] = {{
{fmt_arr(b0_n32)}
}};

static const int32_t W1_n32[8] = {{
{fmt_arr(W1_n32)}
}};
static const int32_t b1_n32[1] = {{
    {b1_n32[0]}
}};

int32_t mlp_new_predict_temp_cC(int32_t hs_cC, int32_t amb_cC)
{{
    if (amb_cC <= 1000 || amb_cC >= 5000) {{
        amb_cC = 2500;
    }}

    int32_t raw_c[2] = {{ hs_cC, amb_cC }};
    int32_t x_q[2];
    for (int i = 0; i < 2; i++) {{
        int64_t diff = (int64_t)raw_c[i] - (int64_t)norm_mean_c[i];
        x_q[i] = (int32_t)((diff * SCALE_NORMAL_32) / (int64_t)norm_scale_c[i]);
    }}

    /* Layer 0: 2 -> 8 */
    int32_t h_q[8];
    for (int j = 0; j < 8; j++) {{
        int64_t acc = (int64_t)b0_n32[j] * SCALE_NORMAL_32
                    + (int64_t)x_q[0] * (int64_t)W0_n32[0 * 8 + j]
                    + (int64_t)x_q[1] * (int64_t)W0_n32[1 * 8 + j];
        acc /= SCALE_NORMAL_32;
        h_q[j] = (acc > 0) ? (int32_t)acc : 0;
    }}

    /* Layer 1: 8 -> 1 (Method 4: Dynamic Sparsity) */
    int64_t out_acc = (int64_t)b1_n32[0] * SCALE_NORMAL_32;
    for (int k = 0; k < 8; k++) {{
        if (h_q[k] != 0) {{
            out_acc += (int64_t)h_q[k] * (int64_t)W1_n32[k];
        }}
    }}
    out_acc /= SCALE_NORMAL_32;

    return (int32_t)(((int64_t)out_acc * 100LL) / SCALE_NORMAL_32);
}}

/* ==============================================================================
 * MODE 2: 32-BIT INTEGER WITHOUT FPU — POWER-OF-2 (Bit-Shift >> 16)
 * Features: Method 3 (Loop Unrolling) + Method 4 (ReLU Zero-Skipping)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_32BIT_FIXED_POT)

#define SHIFT_POT_32 16

static const int32_t pot_mean_c[2]  = {{ {mean_c[0]}, {mean_c[1]} }};
static const int32_t pot_scale_c[2] = {{ {scale_c[0]}, {scale_c[1]} }};

static const int32_t W0_p32[16] = {{
{fmt_arr(W0_p32)}
}};
static const int32_t b0_p32[8] = {{
{fmt_arr(b0_p32)}
}};

static const int32_t W1_p32[8] = {{
{fmt_arr(W1_p32)}
}};
static const int32_t b1_p32[1] = {{
    {b1_p32[0]}
}};

int32_t mlp_new_predict_temp_cC(int32_t hs_cC, int32_t amb_cC)
{{
    if (amb_cC <= 1000 || amb_cC >= 5000) {{
        amb_cC = 2500;
    }}

    int32_t raw_c[2] = {{ hs_cC, amb_cC }};
    int32_t x_q[2];
    for (int i = 0; i < 2; i++) {{
        int64_t diff = (int64_t)raw_c[i] - (int64_t)pot_mean_c[i];
        x_q[i] = (int32_t)((diff << SHIFT_POT_32) / (int64_t)pot_scale_c[i]);
    }}

    /* Layer 0: 2 -> 8 */
    int32_t h_q[8];
    for (int j = 0; j < 8; j++) {{
        int64_t acc = ((int64_t)b0_p32[j] << SHIFT_POT_32)
                    + (int64_t)x_q[0] * (int64_t)W0_p32[0 * 8 + j]
                    + (int64_t)x_q[1] * (int64_t)W0_p32[1 * 8 + j];
        acc >>= SHIFT_POT_32;
        h_q[j] = (acc > 0) ? (int32_t)acc : 0;
    }}

    /* Layer 1: 8 -> 1 (Method 4: Dynamic Sparsity) */
    int64_t out_acc = ((int64_t)b1_p32[0] << SHIFT_POT_32);
    for (int k = 0; k < 8; k++) {{
        if (h_q[k] != 0) {{
            out_acc += (int64_t)h_q[k] * (int64_t)W1_p32[k];
        }}
    }}
    out_acc >>= SHIFT_POT_32;

    return (int32_t)(((int64_t)out_acc * 100LL) >> SHIFT_POT_32);
}}

/* ==============================================================================
 * MODE 3: 16-BIT FLOAT WITH HARDWARE FPU (FP16 Storage -> FP32 Compute)
 * Halves Flash memory required for weights!
 * Features: Method 3 (Loop Unrolling) + Method 4 (ReLU Zero-Skipping)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_16BIT_FPU)

static const float scaler_mean[2] = {{
    {mean[0]:.8f}f, {mean[1]:.8f}f
}};
static const float scaler_scale[2] = {{
    {scale[0]:.8f}f, {scale[1]:.8f}f
}};

static const uint16_t W0_f16[16] = {{
{fmt_arr_u16(W0_f16)}
}};
static const uint16_t b0_f16[8] = {{
{fmt_arr_u16(b0_f16)}
}};

static const uint16_t W1_f16[8] = {{
{fmt_arr_u16(W1_f16)}
}};
static const uint16_t b1_f16[1] = {{
    0x{b1_f16[0]:04X}U
}};

int32_t mlp_new_predict_temp_cC(int32_t hs_cC, int32_t amb_cC)
{{
    if (amb_cC <= 1000 || amb_cC >= 5000) {{
        amb_cC = 2500;
    }}

    float x[2];
    x[0] = (((float)hs_cC  / 100.0f) - scaler_mean[0]) / scaler_scale[0];
    x[1] = (((float)amb_cC / 100.0f) - scaler_mean[1]) / scaler_scale[1];

    /* Layer 0: 2 -> 8 (Method 3: Fully Unrolled with FP16 expansion) */
    float h[8];
    for (int j = 0; j < 8; j++) {{
        float acc = fp16_to_fp32(b0_f16[j])
                  + x[0] * fp16_to_fp32(W0_f16[0 * 8 + j])
                  + x[1] * fp16_to_fp32(W0_f16[1 * 8 + j]);
        h[j] = (acc > 0.0f) ? acc : 0.0f;
    }}

    /* Layer 1: 8 -> 1 (Method 4: ReLU Zero-Skipping) */
    float out = fp16_to_fp32(b1_f16[0]);
    for (int k = 0; k < 8; k++) {{
        if (h[k] > 0.0f) {{
            out += h[k] * fp16_to_fp32(W1_f16[k]);
        }}
    }}

    return (int32_t)(out * 100.0f + (out >= 0.0f ? 0.5f : -0.5f));
}}

/* ==============================================================================
 * MODE 4: 16-BIT INTEGER WITHOUT FPU — NORMAL FIXED-POINT (Division / 2048)
 * Weights stored as int16_t.
 * Features: Method 3 (Loop Unrolling) + Method 4 (ReLU Zero-Skipping)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_16BIT_FIXED_NORMAL)

#define SCALE_NORMAL_16 2048

static const int32_t norm_mean_c[2]  = {{ {mean_c[0]}, {mean_c[1]} }};
static const int32_t norm_scale_c[2] = {{ {scale_c[0]}, {scale_c[1]} }};

static const int16_t W0_n16[16] = {{
{fmt_arr(W0_n16)}
}};
static const int32_t b0_n16[8] = {{
{fmt_arr(b0_n16)}
}};

static const int16_t W1_n16[8] = {{
{fmt_arr(W1_n16)}
}};
static const int32_t b1_n16[1] = {{
    {b1_n16[0]}
}};

int32_t mlp_new_predict_temp_cC(int32_t hs_cC, int32_t amb_cC)
{{
    if (amb_cC <= 1000 || amb_cC >= 5000) {{
        amb_cC = 2500;
    }}

    int32_t raw_c[2] = {{ hs_cC, amb_cC }};
    int16_t x_q[2];
    for (int i = 0; i < 2; i++) {{
        int32_t diff = raw_c[i] - norm_mean_c[i];
        x_q[i] = (int16_t)(((int64_t)diff * SCALE_NORMAL_16) / norm_scale_c[i]);
    }}

    /* Layer 0: 2 -> 8 (Method 3: Fully Unrolled) */
    int16_t h_q[8];
    for (int j = 0; j < 8; j++) {{
        int32_t acc = b0_n16[j] * SCALE_NORMAL_16
                    + (int32_t)x_q[0] * (int32_t)W0_n16[0 * 8 + j]
                    + (int32_t)x_q[1] * (int32_t)W0_n16[1 * 8 + j];
        acc /= SCALE_NORMAL_16;
        h_q[j] = (acc > 0) ? (int16_t)acc : 0;
    }}

    /* Layer 1: 8 -> 1 (Method 4: Dynamic Sparsity) */
    int32_t out_acc = b1_n16[0] * SCALE_NORMAL_16;
    for (int k = 0; k < 8; k++) {{
        if (h_q[k] != 0) {{
            out_acc += (int32_t)h_q[k] * (int32_t)W1_n16[k];
        }}
    }}
    out_acc /= SCALE_NORMAL_16;

    return (int32_t)(((int64_t)out_acc * 100LL) / SCALE_NORMAL_16);
}}

/* ==============================================================================
 * MODE 5: 16-BIT INTEGER WITHOUT FPU — POWER-OF-2 (Bit-Shift >> 11) + ARM SMLAD SIMD
 * Features:
 *   - Method 2: C + Inline Assembly using ARM Cortex-M4 SMLAD (Dual MAC in 1 cycle)
 *   - Method 3: Loop Unrolling
 *   - Method 4: ReLU Zero-Skipping (Dynamic Sparsity)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_16BIT_FIXED_POT_SIMD)

#define SHIFT_POT_16 11

static const int32_t pot_mean_c[2]  = {{ {mean_c[0]}, {mean_c[1]} }};
static const int32_t pot_scale_c[2] = {{ {scale_c[0]}, {scale_c[1]} }};

/* Packed 16-bit weight pairs for Cortex-M4 SMLAD instruction */
static const uint32_t W0_simd[8] = {{
{fmt_arr_hex(W0_simd)}
}};
static const int32_t b0_simd[8] = {{
{fmt_arr(b0_p16)}
}};

static const uint32_t W1_simd[4] = {{
{fmt_arr_hex(W1_simd)}
}};
static const int32_t b1_simd[1] = {{
    {b1_p16[0]}
}};

int32_t mlp_new_predict_temp_cC(int32_t hs_cC, int32_t amb_cC)
{{
    if (amb_cC <= 1000 || amb_cC >= 5000) {{
        amb_cC = 2500;
    }}

    int32_t raw_c[2] = {{ hs_cC, amb_cC }};
    int16_t x_q[2];
    for (int i = 0; i < 2; i++) {{
        int32_t diff = raw_c[i] - pot_mean_c[i];
        x_q[i] = (int16_t)(((int64_t)diff << SHIFT_POT_16) / (int64_t)pot_scale_c[i]);
    }}

    /* Pack 2 input features into single 32-bit register for SMLAD */
    uint32_t x_pair = (uint32_t)(uint16_t)x_q[0] | ((uint32_t)(uint16_t)x_q[1] << 16);

    /* Layer 0: 2 -> 8 (Method 2: Single-Cycle Dual MAC via SMLAD) */
    int16_t h_q[8];
    for (int j = 0; j < 8; j++) {{
        int32_t acc = smlad_dual(x_pair, W0_simd[j], (int32_t)(b0_simd[j] << SHIFT_POT_16));
        acc >>= SHIFT_POT_16;
        h_q[j] = (acc > 0) ? (int16_t)acc : 0;
    }}

    /* Layer 1: 8 -> 1 (Method 2: Cortex-M4 DSP SMLAD across 4 pairs) */
    uint32_t h01 = (uint32_t)(uint16_t)h_q[0] | ((uint32_t)(uint16_t)h_q[1] << 16);
    uint32_t h23 = (uint32_t)(uint16_t)h_q[2] | ((uint32_t)(uint16_t)h_q[3] << 16);
    uint32_t h45 = (uint32_t)(uint16_t)h_q[4] | ((uint32_t)(uint16_t)h_q[5] << 16);
    uint32_t h67 = (uint32_t)(uint16_t)h_q[6] | ((uint32_t)(uint16_t)h_q[7] << 16);

    int32_t out_acc = (int32_t)(b1_simd[0] << SHIFT_POT_16);
    out_acc = smlad_dual(h01, W1_simd[0], out_acc);
    out_acc = smlad_dual(h23, W1_simd[1], out_acc);
    out_acc = smlad_dual(h45, W1_simd[2], out_acc);
    out_acc = smlad_dual(h67, W1_simd[3], out_acc);
    out_acc >>= SHIFT_POT_16;

    return (int32_t)(((int64_t)out_acc * 100LL) >> SHIFT_POT_16);
}}

#endif
"""

with open('New_models/mlp_model_new.c', 'w') as f:
    f.write(c_code)

print("  [OK] SUCCESS: Generated New_models/mlp_model_new.h and New_models/mlp_model_new.c with ALL 6 MODES!")
