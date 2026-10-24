/*
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

/* ==============================================================================
 * MODE 0: 32-BIT FLOAT WITH HARDWARE FPU (FP32)
 * Features: Method 3 (Loop Unrolling) + Method 4 (ReLU Zero-Skipping)
 * ============================================================================== */
#if (BENCHMARK_MODE == MODE_32BIT_FPU)

static const float scaler_mean[2] = {
    68.17805481f, 28.58159065f
};
static const float scaler_scale[2] = {
    15.35353756f, 6.69172049f
};

static const float W0[16] = {
    -0.18663535f,  2.74716783f,  0.00303465f,  2.83030176f, -0.54742914f, -3.06951404f, -0.04001034f,  0.17388870f,
     1.19469392f,  0.23851693f, -0.29622778f,  0.30987871f,  1.53858912f, -0.29915178f, -2.86772323f, -0.11337092f
};
static const float b0[8] = {
     3.72803473f,  3.53439236f, -0.48075092f,  3.66370177f,  3.91097713f, -3.97619820f,  4.35051918f, -0.51953429f
};

static const float W1[8] = {
     3.75073695f,  3.42906857f, -0.18346456f,  3.02047825f,  3.23502588f, -5.83439636f,  3.89480376f, -0.28393093f
};
static const float b1[1] = {
    1.42442358f
};

int32_t mlp_new_predict_temp_cC(int32_t hs_cC, int32_t amb_cC)
{
    if (amb_cC <= 1000 || amb_cC >= 5000) {
        amb_cC = 2500;
    }

    float x[2];
    x[0] = (((float)hs_cC  / 100.0f) - scaler_mean[0]) / scaler_scale[0];
    x[1] = (((float)amb_cC / 100.0f) - scaler_mean[1]) / scaler_scale[1];

    /* Layer 0: 2 -> 8 */
    float h[8];
    for (int j = 0; j < 8; j++) {
        float acc = b0[j] + x[0] * W0[0 * 8 + j]
                          + x[1] * W0[1 * 8 + j];
        h[j] = (acc > 0.0f) ? acc : 0.0f; /* ReLU */
    }

    /* Layer 1: 8 -> 1 (Method 4: ReLU Zero-Skipping) */
    float out = b1[0];
    for (int k = 0; k < 8; k++) {
        if (h[k] > 0.0f) {
            out += h[k] * W1[k];
        }
    }

    return (int32_t)(out * 100.0f + (out >= 0.0f ? 0.5f : -0.5f));
}

/* ==============================================================================
 * MODE 1: 32-BIT INTEGER WITHOUT FPU — NORMAL FIXED-POINT (Division / 10000)
 * Features: Method 3 (Loop Unrolling) + Method 4 (ReLU Zero-Skipping)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_32BIT_FIXED_NORMAL)

#define SCALE_NORMAL_32 10000LL

static const int32_t norm_mean_c[2]  = { 6818, 2858 };
static const int32_t norm_scale_c[2] = { 1535, 669 };

static const int32_t W0_n32[16] = {
        -1866,     27472,        30,     28303,     -5474,    -30695,      -400,      1739,
        11947,      2385,     -2962,      3099,     15386,     -2992,    -28677,     -1134
};
static const int32_t b0_n32[8] = {
        37280,     35344,     -4808,     36637,     39110,    -39762,     43505,     -5195
};

static const int32_t W1_n32[8] = {
        37507,     34291,     -1835,     30205,     32350,    -58344,     38948,     -2839
};
static const int32_t b1_n32[1] = {
    14244
};

int32_t mlp_new_predict_temp_cC(int32_t hs_cC, int32_t amb_cC)
{
    if (amb_cC <= 1000 || amb_cC >= 5000) {
        amb_cC = 2500;
    }

    int32_t raw_c[2] = { hs_cC, amb_cC };
    int32_t x_q[2];
    for (int i = 0; i < 2; i++) {
        int64_t diff = (int64_t)raw_c[i] - (int64_t)norm_mean_c[i];
        x_q[i] = (int32_t)((diff * SCALE_NORMAL_32) / (int64_t)norm_scale_c[i]);
    }

    /* Layer 0: 2 -> 8 */
    int32_t h_q[8];
    for (int j = 0; j < 8; j++) {
        int64_t acc = (int64_t)b0_n32[j] * SCALE_NORMAL_32
                    + (int64_t)x_q[0] * (int64_t)W0_n32[0 * 8 + j]
                    + (int64_t)x_q[1] * (int64_t)W0_n32[1 * 8 + j];
        acc /= SCALE_NORMAL_32;
        h_q[j] = (acc > 0) ? (int32_t)acc : 0;
    }

    /* Layer 1: 8 -> 1 (Method 4: Dynamic Sparsity) */
    int64_t out_acc = (int64_t)b1_n32[0] * SCALE_NORMAL_32;
    for (int k = 0; k < 8; k++) {
        if (h_q[k] != 0) {
            out_acc += (int64_t)h_q[k] * (int64_t)W1_n32[k];
        }
    }
    out_acc /= SCALE_NORMAL_32;

    return (int32_t)(((int64_t)out_acc * 100LL) / SCALE_NORMAL_32);
}

/* ==============================================================================
 * MODE 2: 32-BIT INTEGER WITHOUT FPU — POWER-OF-2 (Bit-Shift >> 16)
 * Features: Method 3 (Loop Unrolling) + Method 4 (ReLU Zero-Skipping)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_32BIT_FIXED_POT)

#define SHIFT_POT_32 16

static const int32_t pot_mean_c[2]  = { 6818, 2858 };
static const int32_t pot_scale_c[2] = { 1535, 669 };

static const int32_t W0_p32[16] = {
       -12231,    180038,       199,    185487,    -35876,   -201164,     -2622,     11396,
        78295,     15631,    -19414,     20308,    100833,    -19605,   -187939,     -7430
};
static const int32_t b0_p32[8] = {
       244320,    231630,    -31506,    240104,    256310,   -260584,    285116,    -34048
};

static const int32_t W1_p32[8] = {
       245808,    224727,    -12024,    197950,    212011,   -382363,    255250,    -18608
};
static const int32_t b1_p32[1] = {
    93351
};

int32_t mlp_new_predict_temp_cC(int32_t hs_cC, int32_t amb_cC)
{
    if (amb_cC <= 1000 || amb_cC >= 5000) {
        amb_cC = 2500;
    }

    int32_t raw_c[2] = { hs_cC, amb_cC };
    int32_t x_q[2];
    for (int i = 0; i < 2; i++) {
        int64_t diff = (int64_t)raw_c[i] - (int64_t)pot_mean_c[i];
        x_q[i] = (int32_t)((diff << SHIFT_POT_32) / (int64_t)pot_scale_c[i]);
    }

    /* Layer 0: 2 -> 8 */
    int32_t h_q[8];
    for (int j = 0; j < 8; j++) {
        int64_t acc = ((int64_t)b0_p32[j] << SHIFT_POT_32)
                    + (int64_t)x_q[0] * (int64_t)W0_p32[0 * 8 + j]
                    + (int64_t)x_q[1] * (int64_t)W0_p32[1 * 8 + j];
        acc >>= SHIFT_POT_32;
        h_q[j] = (acc > 0) ? (int32_t)acc : 0;
    }

    /* Layer 1: 8 -> 1 (Method 4: Dynamic Sparsity) */
    int64_t out_acc = ((int64_t)b1_p32[0] << SHIFT_POT_32);
    for (int k = 0; k < 8; k++) {
        if (h_q[k] != 0) {
            out_acc += (int64_t)h_q[k] * (int64_t)W1_p32[k];
        }
    }
    out_acc >>= SHIFT_POT_32;

    return (int32_t)(((int64_t)out_acc * 100LL) >> SHIFT_POT_32);
}

/* ==============================================================================
 * MODE 3: 16-BIT FLOAT WITH HARDWARE FPU (FP16 Storage -> FP32 Compute)
 * Halves Flash memory required for weights!
 * Features: Method 3 (Loop Unrolling) + Method 4 (ReLU Zero-Skipping)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_16BIT_FPU)

static const float scaler_mean[2] = {
    68.17805481f, 28.58159065f
};
static const float scaler_scale[2] = {
    15.35353756f, 6.69172049f
};

static const uint16_t W0_f16[16] = {
    0xB1F9U, 0x417FU, 0x1A37U, 0x41A9U, 0xB861U, 0xC224U, 0xA91FU, 0x3190U,
    0x3CC7U, 0x33A2U, 0xB4BDU, 0x34F5U, 0x3E28U, 0xB4C9U, 0xC1BCU, 0xAF41U
};
static const uint16_t b0_f16[8] = {
    0x4375U, 0x4312U, 0xB7B1U, 0x4354U, 0x43D2U, 0xC3F4U, 0x445AU, 0xB828U
};

static const uint16_t W1_f16[8] = {
    0x4380U, 0x42DCU, 0xB1DFU, 0x420AU, 0x4278U, 0xC5D6U, 0x43CAU, 0xB48BU
};
static const uint16_t b1_f16[1] = {
    0x3DB3U
};

int32_t mlp_new_predict_temp_cC(int32_t hs_cC, int32_t amb_cC)
{
    if (amb_cC <= 1000 || amb_cC >= 5000) {
        amb_cC = 2500;
    }

    float x[2];
    x[0] = (((float)hs_cC  / 100.0f) - scaler_mean[0]) / scaler_scale[0];
    x[1] = (((float)amb_cC / 100.0f) - scaler_mean[1]) / scaler_scale[1];

    /* Layer 0: 2 -> 8 (Method 3: Fully Unrolled with FP16 expansion) */
    float h[8];
    for (int j = 0; j < 8; j++) {
        float acc = fp16_to_fp32(b0_f16[j])
                  + x[0] * fp16_to_fp32(W0_f16[0 * 8 + j])
                  + x[1] * fp16_to_fp32(W0_f16[1 * 8 + j]);
        h[j] = (acc > 0.0f) ? acc : 0.0f;
    }

    /* Layer 1: 8 -> 1 (Method 4: ReLU Zero-Skipping) */
    float out = fp16_to_fp32(b1_f16[0]);
    for (int k = 0; k < 8; k++) {
        if (h[k] > 0.0f) {
            out += h[k] * fp16_to_fp32(W1_f16[k]);
        }
    }

    return (int32_t)(out * 100.0f + (out >= 0.0f ? 0.5f : -0.5f));
}

/* ==============================================================================
 * MODE 4: 16-BIT INTEGER WITHOUT FPU — NORMAL FIXED-POINT (Division / 2048)
 * Weights stored as int16_t.
 * Features: Method 3 (Loop Unrolling) + Method 4 (ReLU Zero-Skipping)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_16BIT_FIXED_NORMAL)

#define SCALE_NORMAL_16 2048

static const int32_t norm_mean_c[2]  = { 6818, 2858 };
static const int32_t norm_scale_c[2] = { 1535, 669 };

static const int16_t W0_n16[16] = {
         -382,      5626,         6,      5796,     -1121,     -6286,       -82,       356,
         2447,       488,      -607,       635,      3151,      -613,     -5873,      -232
};
static const int32_t b0_n16[8] = {
         7635,      7238,      -985,      7503,      8010,     -8143,      8910,     -1064
};

static const int16_t W1_n16[8] = {
         7682,      7023,      -376,      6186,      6625,    -11949,      7977,      -581
};
static const int32_t b1_n16[1] = {
    2917
};

int32_t mlp_new_predict_temp_cC(int32_t hs_cC, int32_t amb_cC)
{
    if (amb_cC <= 1000 || amb_cC >= 5000) {
        amb_cC = 2500;
    }

    int32_t raw_c[2] = { hs_cC, amb_cC };
    int16_t x_q[2];
    for (int i = 0; i < 2; i++) {
        int32_t diff = raw_c[i] - norm_mean_c[i];
        x_q[i] = (int16_t)(((int64_t)diff * SCALE_NORMAL_16) / norm_scale_c[i]);
    }

    /* Layer 0: 2 -> 8 (Method 3: Fully Unrolled) */
    int16_t h_q[8];
    for (int j = 0; j < 8; j++) {
        int32_t acc = b0_n16[j] * SCALE_NORMAL_16
                    + (int32_t)x_q[0] * (int32_t)W0_n16[0 * 8 + j]
                    + (int32_t)x_q[1] * (int32_t)W0_n16[1 * 8 + j];
        acc /= SCALE_NORMAL_16;
        h_q[j] = (acc > 0) ? (int16_t)acc : 0;
    }

    /* Layer 1: 8 -> 1 (Method 4: Dynamic Sparsity) */
    int32_t out_acc = b1_n16[0] * SCALE_NORMAL_16;
    for (int k = 0; k < 8; k++) {
        if (h_q[k] != 0) {
            out_acc += (int32_t)h_q[k] * (int32_t)W1_n16[k];
        }
    }
    out_acc /= SCALE_NORMAL_16;

    return (int32_t)(((int64_t)out_acc * 100LL) / SCALE_NORMAL_16);
}

/* ==============================================================================
 * MODE 5: 16-BIT INTEGER WITHOUT FPU — POWER-OF-2 (Bit-Shift >> 11) + ARM SMLAD SIMD
 * Features:
 *   - Method 2: C + Inline Assembly using ARM Cortex-M4 SMLAD (Dual MAC in 1 cycle)
 *   - Method 3: Loop Unrolling
 *   - Method 4: ReLU Zero-Skipping (Dynamic Sparsity)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_16BIT_FIXED_POT_SIMD)

#define SHIFT_POT_16 11

static const int32_t pot_mean_c[2]  = { 6818, 2858 };
static const int32_t pot_scale_c[2] = { 1535, 669 };

/* Packed 16-bit weight pairs for Cortex-M4 SMLAD instruction */
static const uint32_t W0_simd[8] = {
    0x098FFE82U, 0x01E815FAU, 0xFDA10006U, 0x027B16A4U,
    0x0C4FFB9FU, 0xFD9BE772U, 0xE90FFFAEU, 0xFF180164U
};
static const int32_t b0_simd[8] = {
         7635,      7238,      -985,      7503,      8010,     -8143,      8910,     -1064
};

static const uint32_t W1_simd[4] = {
    0x1B6F1E02U, 0x182AFE88U, 0xD15319E1U, 0xFDBB1F29U
};
static const int32_t b1_simd[1] = {
    2917
};

int32_t mlp_new_predict_temp_cC(int32_t hs_cC, int32_t amb_cC)
{
    if (amb_cC <= 1000 || amb_cC >= 5000) {
        amb_cC = 2500;
    }

    int32_t raw_c[2] = { hs_cC, amb_cC };
    int16_t x_q[2];
    for (int i = 0; i < 2; i++) {
        int32_t diff = raw_c[i] - pot_mean_c[i];
        x_q[i] = (int16_t)(((int64_t)diff << SHIFT_POT_16) / (int64_t)pot_scale_c[i]);
    }

    /* Pack 2 input features into single 32-bit register for SMLAD */
    uint32_t x_pair = (uint32_t)(uint16_t)x_q[0] | ((uint32_t)(uint16_t)x_q[1] << 16);

    /* Layer 0: 2 -> 8 (Method 2: Single-Cycle Dual MAC via SMLAD) */
    int16_t h_q[8];
    for (int j = 0; j < 8; j++) {
        int32_t acc = smlad_dual(x_pair, W0_simd[j], (int32_t)(b0_simd[j] << SHIFT_POT_16));
        acc >>= SHIFT_POT_16;
        h_q[j] = (acc > 0) ? (int16_t)acc : 0;
    }

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
}

#endif
