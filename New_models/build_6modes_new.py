import json
import numpy as np

# Load scaler and weights for mlp_model_new (4-input stateless dual-horizon model)
with open('New_models/scaler_params_new.json') as f:
    scaler = json.load(f)
with open('New_models/mlp_weights_new.json') as f:
    weights = json.load(f)

n_inputs = len(scaler['mean'])          # 4
n_hidden = weights['layers'][0]['weights_shape'][1]  # 32

mean  = np.array(scaler['mean'],  dtype=np.float32)
scale = np.array(scaler['scale'], dtype=np.float32)

W0 = np.array(weights['layers'][0]['weights'], dtype=np.float32)  # (n_inputs, n_hidden)
b0 = np.array(weights['layers'][0]['biases'],  dtype=np.float32)  # (n_hidden,)
W1 = np.array(weights['layers'][1]['weights'], dtype=np.float32)  # (n_hidden, 2)
b1 = np.array(weights['layers'][1]['biases'],  dtype=np.float32)  # (2,)

W1_3s  = W1[:, 0]
W1_10s = W1[:, 1]
b1_3s  = float(b1[0])
b1_10s = float(b1[1])

S_N32 = 10000
SHIFT_P32 = 16
S_P32 = 1 << SHIFT_P32
S_N16 = 2048
SHIFT_P16 = 11
S_P16 = 1 << SHIFT_P16

mean_c  = [int(round(x * 100)) for x in mean]
scale_c = [int(round(x * 100)) for x in scale]

def f_to_u16(val):
    return int(np.float16(val).view(np.uint16))

W0_f32 = W0.flatten()
b0_f32 = b0
W1_3s_f32 = W1_3s
W1_10s_f32 = W1_10s

W0_n32 = np.round(W0 * S_N32).astype(int).flatten()
b0_n32 = np.round(b0 * S_N32).astype(int)
W1_3s_n32  = np.round(W1_3s * S_N32).astype(int)
W1_10s_n32 = np.round(W1_10s * S_N32).astype(int)
b1_3s_n32  = int(round(b1_3s * S_N32))
b1_10s_n32 = int(round(b1_10s * S_N32))

W0_p32 = np.round(W0 * S_P32).astype(int).flatten()
b0_p32 = np.round(b0 * S_P32).astype(int)
W1_3s_p32  = np.round(W1_3s * S_P32).astype(int)
W1_10s_p32 = np.round(W1_10s * S_P32).astype(int)
b1_3s_p32  = int(round(b1_3s * S_P32))
b1_10s_p32 = int(round(b1_10s * S_P32))

W0_f16 = [f_to_u16(x) for x in W0.flatten()]
b0_f16 = [f_to_u16(x) for x in b0]
W1_3s_f16  = [f_to_u16(x) for x in W1_3s]
W1_10s_f16 = [f_to_u16(x) for x in W1_10s]
b1_3s_f16  = f_to_u16(b1_3s)
b1_10s_f16 = f_to_u16(b1_10s)

W0_n16 = np.round(W0 * S_N16).astype(np.int16).flatten()
b0_n16 = np.round(b0 * S_N16).astype(int)
W1_3s_n16  = np.round(W1_3s * S_N16).astype(np.int16)
W1_10s_n16 = np.round(W1_10s * S_N16).astype(np.int16)
b1_3s_n16  = int(round(b1_3s * S_N16))
b1_10s_n16 = int(round(b1_10s * S_N16))

W0_p16 = np.round(W0 * S_P16).astype(np.int16)
b0_p16 = np.round(b0 * S_P16).astype(int)
W1_3s_p16  = np.round(W1_3s * S_P16).astype(np.int16)
W1_10s_p16 = np.round(W1_10s * S_P16).astype(np.int16)
b1_3s_p16  = int(round(b1_3s * S_P16))
b1_10s_p16 = int(round(b1_10s * S_P16))

W0_simd_01 = []
W0_simd_23 = []
for j in range(n_hidden):
    w0 = int(W0_p16[0, j]) & 0xFFFF
    w1 = int(W0_p16[1, j]) & 0xFFFF
    w2 = int(W0_p16[2, j]) & 0xFFFF
    w3 = int(W0_p16[3, j]) & 0xFFFF
    W0_simd_01.append((w1 << 16) | w0)
    W0_simd_23.append((w3 << 16) | w2)

W1_simd_3s = []
W1_simd_10s = []
for k in range(0, n_hidden, 2):
    w0_3s = int(W1_3s_p16[k])   & 0xFFFF
    w1_3s = int(W1_3s_p16[k+1]) & 0xFFFF
    W1_simd_3s.append((w1_3s << 16) | w0_3s)

    w0_10s = int(W1_10s_p16[k])   & 0xFFFF
    w1_10s = int(W1_10s_p16[k+1]) & 0xFFFF
    W1_simd_10s.append((w1_10s << 16) | w0_10s)

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

NI = n_inputs
NH = n_hidden

header_code = f"""/*
 * mlp_model_new.h -- Auto-generated TinyML 6-Mode Stateless Dual Thermal Predictor
 * Target: STM32 Nucleo-64 F401RE (Cortex-M4, 16 MHz HSI)
 *
 * Architecture: {NI} inputs -> {NH} neurons (ReLU) -> 2 outputs (Linear: 3s & 10s)
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

typedef struct {{
    int32_t pred_3s_cC;
    int32_t pred_10s_cC;
}} mlp_prediction_t;

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
"""

with open('New_models/mlp_model_new.h', 'w', encoding='utf-8') as f:
    f.write(header_code)

amb_clamp = (
    "    if (amb_cC <= 1000 || amb_cC >= 5500) {\n"
    "        amb_cC = 2500;\n"
    "    }"
)

def gen_x_f32():
    return (
        "    float x[4];\n"
        "    x[0] = (((float)hs_cC     / 100.0f) - scaler_mean[0]) / scaler_scale[0];\n"
        "    x[1] = (((float)amb_cC    / 100.0f) - scaler_mean[1]) / scaler_scale[1];\n"
        "    x[2] = (((float)ploss_cmW / 100.0f) - scaler_mean[2]) / scaler_scale[2];\n"
        "    x[3] = (((float)i_cmA     / 100.0f) - scaler_mean[3]) / scaler_scale[3];"
    )

def gen_x_int():
    return (
        f"    int32_t raw_c[{NI}];\n"
        f"    raw_c[0] = hs_cC;\n"
        f"    raw_c[1] = amb_cC;\n"
        f"    raw_c[2] = ploss_cmW;\n"
        f"    raw_c[3] = i_cmA;"
    )

def gen_bound_check():
    return (
        "    mlp_prediction_t res;\n"
        "    res.pred_3s_cC  = hs_cC + delta_3s_cC;\n"
        "    res.pred_10s_cC = hs_cC + delta_10s_cC;\n"
        "    if (ploss_cmW > 0) {\n"
        "        if (res.pred_3s_cC < hs_cC)  res.pred_3s_cC = hs_cC;\n"
        "        if (res.pred_10s_cC < hs_cC) res.pred_10s_cC = hs_cC;\n"
        "    }\n"
        "    if (res.pred_3s_cC > 12500)  res.pred_3s_cC = 12500;\n"
        "    if (res.pred_10s_cC > 12500) res.pred_10s_cC = 12500;\n"
        "    return res;"
    )

c_code = f"""/*
 * mlp_model_new.c -- 6-Mode TinyML Stateless Dual Thermal Predictor for STM32F401RET6
 * Architecture: {NI} Inputs -> {NH} Neurons (ReLU) -> 2 Outputs (Linear: 3s & 10s)
 * Features: [heatsink_temp_C, ambient_temp_C, power_loss_mW, current_mA]
 * Targets:  [Future +3s ahead, Future +10s ahead] in centi-degC
 *
 * ALL 6 MODES ARE SIMULTANEOUSLY COMPILED INTO THIS FILE.
 */

#include "mlp_model_new.h"
#include <stdint.h>
#include <string.h>

static inline int32_t smlad_dual(uint32_t x_pair, uint32_t w_pair, int32_t acc)
{{
#if defined(__arm__) || defined(__thumb__) || defined(__thumb2__)
    __asm__ volatile ("smlad %0, %1, %2, %0"
                      : "+r" (acc)
                      : "r" (x_pair), "r" (w_pair));
    return acc;
#else
    int16_t x0 = (int16_t)(x_pair & 0xFFFFU);
    int16_t x1 = (int16_t)((x_pair >> 16) & 0xFFFFU);
    int16_t w0 = (int16_t)(w_pair & 0xFFFFU);
    int16_t w1 = (int16_t)((w_pair >> 16) & 0xFFFFU);
    return acc + ((int32_t)x0 * (int32_t)w0) + ((int32_t)x1 * (int32_t)w1);
#endif
}}

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
 * ============================================================================== */
static const float scaler_mean[{NI}] = {{
{fmt_arr_f(mean)}
}};
static const float scaler_scale[{NI}] = {{
{fmt_arr_f(scale)}
}};

static const float W0[{NI * NH}] = {{
{fmt_arr_f(W0_f32)}
}};
static const float b0[{NH}] = {{
{fmt_arr_f(b0_f32)}
}};
static const float W1_3s[{NH}] = {{
{fmt_arr_f(W1_3s_f32)}
}};
static const float W1_10s[{NH}] = {{
{fmt_arr_f(W1_10s_f32)}
}};
static const float b1_f32[2] = {{
    {b1_3s:.8f}f, {b1_10s:.8f}f
}};

mlp_prediction_t mlp_new_predict_mode0_fp32(int32_t hs_cC, int32_t amb_cC,
                                           int32_t ploss_cmW, int32_t i_cmA)
{{
{amb_clamp}

{gen_x_f32()}

    float h[{NH}];
    for (int j = 0; j < {NH}; j++) {{
        float acc = b0[j];
        for (int i = 0; i < {NI}; i++) {{
            acc += x[i] * W0[i * {NH} + j];
        }}
        h[j] = (acc > 0.0f) ? acc : 0.0f;
    }}

    float out_3s = b1_f32[0];
    float out_10s = b1_f32[1];
    for (int k = 0; k < {NH}; k++) {{
        if (h[k] > 0.0f) {{
            out_3s  += h[k] * W1_3s[k];
            out_10s += h[k] * W1_10s[k];
        }}
    }}

    int32_t delta_3s_cC  = (int32_t)(out_3s * 100.0f + (out_3s >= 0.0f ? 0.5f : -0.5f));
    int32_t delta_10s_cC = (int32_t)(out_10s * 100.0f + (out_10s >= 0.0f ? 0.5f : -0.5f));

{gen_bound_check()}
}}

/* ==============================================================================
 * MODE 1: 32-BIT INTEGER -- NORMAL FIXED-POINT (Division / 10000)
 * ============================================================================== */
#define SCALE_NORMAL_32 10000LL

static const int32_t norm_mean_c[{NI}]  = {{ {', '.join(str(x) for x in mean_c)} }};
static const int32_t norm_scale_c[{NI}] = {{ {', '.join(str(x) for x in scale_c)} }};

static const int32_t W0_n32[{NI * NH}] = {{
{fmt_arr(W0_n32)}
}};
static const int32_t b0_n32[{NH}] = {{
{fmt_arr(b0_n32)}
}};
static const int32_t W1_3s_n32[{NH}] = {{
{fmt_arr(W1_3s_n32)}
}};
static const int32_t W1_10s_n32[{NH}] = {{
{fmt_arr(W1_10s_n32)}
}};
static const int32_t b1_n32[2] = {{ {b1_3s_n32}, {b1_10s_n32} }};

mlp_prediction_t mlp_new_predict_mode1_int32_norm(int32_t hs_cC, int32_t amb_cC,
                                                int32_t ploss_cmW, int32_t i_cmA)
{{
{amb_clamp}

{gen_x_int()}
    int32_t x_q[{NI}];
    for (int i = 0; i < {NI}; i++) {{
        int64_t diff = (int64_t)raw_c[i] - (int64_t)norm_mean_c[i];
        x_q[i] = (int32_t)((diff * SCALE_NORMAL_32) / (int64_t)norm_scale_c[i]);
    }}

    int32_t h_q[{NH}];
    for (int j = 0; j < {NH}; j++) {{
        int64_t acc = (int64_t)b0_n32[j] * SCALE_NORMAL_32;
        for (int i = 0; i < {NI}; i++) {{
            acc += (int64_t)x_q[i] * (int64_t)W0_n32[i * {NH} + j];
        }}
        acc /= SCALE_NORMAL_32;
        h_q[j] = (acc > 0) ? (int32_t)acc : 0;
    }}

    int64_t out_acc_3s  = (int64_t)b1_n32[0] * SCALE_NORMAL_32;
    int64_t out_acc_10s = (int64_t)b1_n32[1] * SCALE_NORMAL_32;
    for (int k = 0; k < {NH}; k++) {{
        if (h_q[k] != 0) {{
            out_acc_3s  += (int64_t)h_q[k] * (int64_t)W1_3s_n32[k];
            out_acc_10s += (int64_t)h_q[k] * (int64_t)W1_10s_n32[k];
        }}
    }}
    out_acc_3s  /= SCALE_NORMAL_32;
    out_acc_10s /= SCALE_NORMAL_32;

    int32_t delta_3s_cC  = (int32_t)(((int64_t)out_acc_3s * 100LL) / SCALE_NORMAL_32);
    int32_t delta_10s_cC = (int32_t)(((int64_t)out_acc_10s * 100LL) / SCALE_NORMAL_32);

{gen_bound_check()}
}}

/* ==============================================================================
 * MODE 2: 32-BIT INTEGER -- POWER-OF-2 (Bit-Shift >> 16)
 * ============================================================================== */
#define SHIFT_POT_32 16

static const int32_t pot_mean_c[{NI}]  = {{ {', '.join(str(x) for x in mean_c)} }};
static const int32_t pot_scale_c[{NI}] = {{ {', '.join(str(x) for x in scale_c)} }};

static const int32_t W0_p32[{NI * NH}] = {{
{fmt_arr(W0_p32)}
}};
static const int32_t b0_p32[{NH}] = {{
{fmt_arr(b0_p32)}
}};
static const int32_t W1_3s_p32[{NH}] = {{
{fmt_arr(W1_3s_p32)}
}};
static const int32_t W1_10s_p32[{NH}] = {{
{fmt_arr(W1_10s_p32)}
}};
static const int32_t b1_p32[2] = {{ {b1_3s_p32}, {b1_10s_p32} }};

mlp_prediction_t mlp_new_predict_mode2_int32_pot(int32_t hs_cC, int32_t amb_cC,
                                               int32_t ploss_cmW, int32_t i_cmA)
{{
{amb_clamp}

{gen_x_int()}
    int32_t x_q[{NI}];
    for (int i = 0; i < {NI}; i++) {{
        int64_t diff = (int64_t)raw_c[i] - (int64_t)pot_mean_c[i];
        x_q[i] = (int32_t)((diff << SHIFT_POT_32) / (int64_t)pot_scale_c[i]);
    }}

    int32_t h_q[{NH}];
    for (int j = 0; j < {NH}; j++) {{
        int64_t acc = ((int64_t)b0_p32[j] << SHIFT_POT_32);
        for (int i = 0; i < {NI}; i++) {{
            acc += (int64_t)x_q[i] * (int64_t)W0_p32[i * {NH} + j];
        }}
        acc >>= SHIFT_POT_32;
        h_q[j] = (acc > 0) ? (int32_t)acc : 0;
    }}

    int64_t out_acc_3s  = ((int64_t)b1_p32[0] << SHIFT_POT_32);
    int64_t out_acc_10s = ((int64_t)b1_p32[1] << SHIFT_POT_32);
    for (int k = 0; k < {NH}; k++) {{
        if (h_q[k] != 0) {{
            out_acc_3s  += (int64_t)h_q[k] * (int64_t)W1_3s_p32[k];
            out_acc_10s += (int64_t)h_q[k] * (int64_t)W1_10s_p32[k];
        }}
    }}
    out_acc_3s  >>= SHIFT_POT_32;
    out_acc_10s >>= SHIFT_POT_32;

    int32_t delta_3s_cC  = (int32_t)(((int64_t)out_acc_3s * 100LL) >> SHIFT_POT_32);
    int32_t delta_10s_cC = (int32_t)(((int64_t)out_acc_10s * 100LL) >> SHIFT_POT_32);

{gen_bound_check()}
}}

/* ==============================================================================
 * MODE 3: 16-BIT FLOAT WITH HARDWARE FPU (FP16 Storage -> FP32 Compute)
 * ============================================================================== */
static const uint16_t W0_f16[{NI * NH}] = {{
{fmt_arr_u16(W0_f16)}
}};
static const uint16_t b0_f16[{NH}] = {{
{fmt_arr_u16(b0_f16)}
}};
static const uint16_t W1_3s_f16[{NH}] = {{
{fmt_arr_u16(W1_3s_f16)}
}};
static const uint16_t W1_10s_f16[{NH}] = {{
{fmt_arr_u16(W1_10s_f16)}
}};
static const uint16_t b1_f16[2] = {{
    0x{b1_3s_f16:04X}U, 0x{b1_10s_f16:04X}U
}};

mlp_prediction_t mlp_new_predict_mode3_fp16(int32_t hs_cC, int32_t amb_cC,
                                          int32_t ploss_cmW, int32_t i_cmA)
{{
{amb_clamp}

{gen_x_f32()}

    float h[{NH}];
    for (int j = 0; j < {NH}; j++) {{
        float acc = fp16_to_fp32(b0_f16[j]);
        for (int i = 0; i < {NI}; i++) {{
            acc += x[i] * fp16_to_fp32(W0_f16[i * {NH} + j]);
        }}
        h[j] = (acc > 0.0f) ? acc : 0.0f;
    }}

    float out_3s  = fp16_to_fp32(b1_f16[0]);
    float out_10s = fp16_to_fp32(b1_f16[1]);
    for (int k = 0; k < {NH}; k++) {{
        if (h[k] > 0.0f) {{
            out_3s  += h[k] * fp16_to_fp32(W1_3s_f16[k]);
            out_10s += h[k] * fp16_to_fp32(W1_10s_f16[k]);
        }}
    }}

    int32_t delta_3s_cC  = (int32_t)(out_3s * 100.0f + (out_3s >= 0.0f ? 0.5f : -0.5f));
    int32_t delta_10s_cC = (int32_t)(out_10s * 100.0f + (out_10s >= 0.0f ? 0.5f : -0.5f));

{gen_bound_check()}
}}

/* ==============================================================================
 * MODE 4: 16-BIT INTEGER -- NORMAL FIXED-POINT (Division / 2048)
 * ============================================================================== */
#define SCALE_NORMAL_16 2048

static const int16_t W0_n16[{NI * NH}] = {{
{fmt_arr(W0_n16)}
}};
static const int32_t b0_n16[{NH}] = {{
{fmt_arr(b0_n16)}
}};
static const int16_t W1_3s_n16[{NH}] = {{
{fmt_arr(W1_3s_n16)}
}};
static const int16_t W1_10s_n16[{NH}] = {{
{fmt_arr(W1_10s_n16)}
}};
static const int32_t b1_n16[2] = {{ {b1_3s_n16}, {b1_10s_n16} }};

mlp_prediction_t mlp_new_predict_mode4_int16_norm(int32_t hs_cC, int32_t amb_cC,
                                                int32_t ploss_cmW, int32_t i_cmA)
{{
{amb_clamp}

{gen_x_int()}
    int16_t x_q[{NI}];
    for (int i = 0; i < {NI}; i++) {{
        int32_t diff = raw_c[i] - norm_mean_c[i];
        x_q[i] = (int16_t)((diff * SCALE_NORMAL_16) / norm_scale_c[i]);
    }}

    int16_t h_q[{NH}];
    for (int j = 0; j < {NH}; j++) {{
        int32_t acc = b0_n16[j] * SCALE_NORMAL_16;
        for (int i = 0; i < {NI}; i++) {{
            acc += (int32_t)x_q[i] * (int32_t)W0_n16[i * {NH} + j];
        }}
        acc /= SCALE_NORMAL_16;
        h_q[j] = (acc > 0) ? (int16_t)acc : 0;
    }}

    int32_t out_acc_3s  = b1_n16[0] * SCALE_NORMAL_16;
    int32_t out_acc_10s = b1_n16[1] * SCALE_NORMAL_16;
    for (int k = 0; k < {NH}; k++) {{
        if (h_q[k] != 0) {{
            out_acc_3s  += (int32_t)h_q[k] * (int32_t)W1_3s_n16[k];
            out_acc_10s += (int32_t)h_q[k] * (int32_t)W1_10s_n16[k];
        }}
    }}
    out_acc_3s  /= SCALE_NORMAL_16;
    out_acc_10s /= SCALE_NORMAL_16;

    int32_t delta_3s_cC  = (int32_t)(((int64_t)out_acc_3s * 100LL) / SCALE_NORMAL_16);
    int32_t delta_10s_cC = (int32_t)(((int64_t)out_acc_10s * 100LL) / SCALE_NORMAL_16);

{gen_bound_check()}
}}

/* ==============================================================================
 * MODE 5: 16-BIT INTEGER -- POT (>> 11) + ARM SMLAD SIMD
 * ============================================================================== */
#define SHIFT_POT_16 11

static const uint32_t W0_simd_01[{NH}] = {{
{fmt_arr_hex(W0_simd_01)}
}};
static const uint32_t W0_simd_23[{NH}] = {{
{fmt_arr_hex(W0_simd_23)}
}};
static const int32_t b0_simd[{NH}] = {{
{fmt_arr(b0_p16)}
}};

static const uint32_t W1_simd_3s[{NH // 2}] = {{
{fmt_arr_hex(W1_simd_3s)}
}};
static const uint32_t W1_simd_10s[{NH // 2}] = {{
{fmt_arr_hex(W1_simd_10s)}
}};
static const int32_t b1_simd[2] = {{ {b1_3s_p16}, {b1_10s_p16} }};

mlp_prediction_t mlp_new_predict_mode5_int16_simd(int32_t hs_cC, int32_t amb_cC,
                                                int32_t ploss_cmW, int32_t i_cmA)
{{
{amb_clamp}

{gen_x_int()}
    int16_t x_q[{NI}];
    for (int i = 0; i < {NI}; i++) {{
        int32_t diff = raw_c[i] - pot_mean_c[i];
        x_q[i] = (int16_t)((diff << SHIFT_POT_16) / pot_scale_c[i]);
    }}

    /* Pack 4 inputs into two 32-bit SIMD registers */
    uint32_t x_pair_01 = ((uint32_t)(uint16_t)x_q[1] << 16) | ((uint32_t)(uint16_t)x_q[0]);
    uint32_t x_pair_23 = ((uint32_t)(uint16_t)x_q[3] << 16) | ((uint32_t)(uint16_t)x_q[2]);

    /* Layer 0: {NI} -> {NH} (SMLAD dual MAC across both input pairs) */
    int16_t h_q[{NH}];
    for (int j = 0; j < {NH}; j++) {{
        int32_t acc = b0_simd[j] << SHIFT_POT_16;
        acc = smlad_dual(x_pair_01, W0_simd_01[j], acc);
        acc = smlad_dual(x_pair_23, W0_simd_23[j], acc);
        acc >>= SHIFT_POT_16;
        h_q[j] = (acc > 0) ? (int16_t)acc : 0;
    }}

    /* Layer 1: {NH} -> 2 (Dual SMLAD across {NH // 2} pairs) */
    int32_t out_acc_3s  = b1_simd[0] << SHIFT_POT_16;
    int32_t out_acc_10s = b1_simd[1] << SHIFT_POT_16;
    for (int p = 0; p < {NH // 2}; p++) {{
        uint32_t h_pair = ((uint32_t)(uint16_t)h_q[p*2+1] << 16) | ((uint32_t)(uint16_t)h_q[p*2]);
        out_acc_3s  = smlad_dual(h_pair, W1_simd_3s[p],  out_acc_3s);
        out_acc_10s = smlad_dual(h_pair, W1_simd_10s[p], out_acc_10s);
    }}
    out_acc_3s  >>= SHIFT_POT_16;
    out_acc_10s >>= SHIFT_POT_16;

    int32_t delta_3s_cC  = (int32_t)(((int64_t)out_acc_3s * 100LL) >> SHIFT_POT_16);
    int32_t delta_10s_cC = (int32_t)(((int64_t)out_acc_10s * 100LL) >> SHIFT_POT_16);

{gen_bound_check()}
}}

/* ==============================================================================
 * MASTER ENTRYPOINT DISPATCHERS
 * ============================================================================== */
mlp_prediction_t mlp_new_predict_dual_cC(int32_t hs_cC, int32_t amb_cC,
                                         int32_t ploss_cmW, int32_t i_cmA)
{{
#if (BENCHMARK_MODE == MODE_32BIT_FPU)
    return mlp_new_predict_mode0_fp32(hs_cC, amb_cC, ploss_cmW, i_cmA);
#elif (BENCHMARK_MODE == MODE_32BIT_FIXED_NORMAL)
    return mlp_new_predict_mode1_int32_norm(hs_cC, amb_cC, ploss_cmW, i_cmA);
#elif (BENCHMARK_MODE == MODE_32BIT_FIXED_POT)
    return mlp_new_predict_mode2_int32_pot(hs_cC, amb_cC, ploss_cmW, i_cmA);
#elif (BENCHMARK_MODE == MODE_16BIT_FPU)
    return mlp_new_predict_mode3_fp16(hs_cC, amb_cC, ploss_cmW, i_cmA);
#elif (BENCHMARK_MODE == MODE_16BIT_FIXED_NORMAL)
    return mlp_new_predict_mode4_int16_norm(hs_cC, amb_cC, ploss_cmW, i_cmA);
#elif (BENCHMARK_MODE == MODE_16BIT_FIXED_POT_SIMD)
    return mlp_new_predict_mode5_int16_simd(hs_cC, amb_cC, ploss_cmW, i_cmA);
#else
    return mlp_new_predict_mode5_int16_simd(hs_cC, amb_cC, ploss_cmW, i_cmA);
#endif
}}

int32_t mlp_new_predict_temp_cC(int32_t hs_cC, int32_t amb_cC,
                                 int32_t ploss_cmW, int32_t i_cmA)
{{
    mlp_prediction_t res = mlp_new_predict_dual_cC(hs_cC, amb_cC, ploss_cmW, i_cmA);
    return res.pred_3s_cC;
}}
"""

with open('New_models/mlp_model_new.c', 'w', encoding='utf-8') as f:
    f.write(c_code)

print("SUCCESS: Generated New_models/mlp_model_new.h and mlp_model_new.c with ALL 6 MODES (Dual 3s & 10s Horizons)!")
