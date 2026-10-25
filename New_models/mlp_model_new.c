/*
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

/* ==============================================================================
 * MODE 0: 32-BIT FLOAT WITH HARDWARE FPU (FP32)
 * Features: Method 3 (Loop Unrolling) + Method 4 (ReLU Zero-Skipping)
 * ============================================================================== */
#if (BENCHMARK_MODE == MODE_32BIT_FPU)

static const float scaler_mean[4] = {
     2355.15893555f,  541.39794922f,  31.19205475f,  66.04267120f
};
static const float scaler_scale[4] = {
     929.11730957f,  221.20219421f,  6.86826229f,  16.83120346f
};

static const float W0[64] = {
    -0.40325031f,  0.86675137f, -0.00982565f,  0.05170405f, -0.68571812f, -0.22616072f, -0.32968035f,  0.57480723f,
     0.24518226f,  0.31007272f, -0.58190829f,  0.59291583f,  0.38995776f,  0.01700541f, -0.41819766f, -0.62102687f,
    -0.13987482f, -0.15374528f, -0.05924047f, -0.17434645f,  0.03044346f, -0.57025009f, -0.31004241f, -0.30657333f,
     0.09031302f, -0.09269860f, -0.38622412f, -0.11898052f, -0.19703451f, -0.73076630f, -0.03170312f, -0.03839556f,
    -0.32966092f,  0.34772500f,  0.52668548f,  0.22824498f, -0.37622961f, -0.31610617f,  0.28839684f, -0.06766623f,
    -0.09498078f,  0.34241828f, -0.41037783f,  0.46940354f,  0.01958005f,  0.10383786f, -0.17490765f,  0.28747320f,
    -0.19525856f, -0.72301888f,  0.34963882f,  0.40426731f,  0.91911215f,  0.62295794f, -0.11608328f,  0.50920016f,
    -0.67628068f, -0.59085763f, -0.28374732f, -0.55042171f, -0.38772643f, -0.45217308f,  0.51754528f, -0.74843991f
};
static const float b0[16] = {
    -0.11848060f,  0.08538616f, -0.45526862f,  0.31282303f, -0.35623512f,  0.66258496f,  0.46399432f, -0.26348725f,
    -0.36210671f,  0.56275469f,  0.09731917f,  0.36200941f,  0.37845290f, -0.30746010f, -0.43903401f, -1.09279561f
};

static const float W1[16] = {
     0.31935340f,  0.44103932f, -0.08308986f, -0.45356864f, -0.92996979f, -0.30487373f,  0.33021024f,  0.30209345f,
     0.51644868f,  0.21927749f, -0.32487220f,  0.40549740f,  0.41191700f,  0.19783261f,  0.53173596f, -0.66979170f
};
static const float b1[1] = {
     0.10305865f
};

int32_t mlp_new_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC)
{
    if (amb_cC <= 1000 || amb_cC >= 5000) {
        amb_cC = 2500;
    }

    float x[4];
    x[0] = ((float)ploss_cmW * 0.01f - scaler_mean[0]) / scaler_scale[0];
    x[1] = ((float)i_cmA     * 0.01f - scaler_mean[1]) / scaler_scale[1];
    x[2] = ((float)amb_cC    * 0.01f - scaler_mean[2]) / scaler_scale[2];
    x[3] = ((float)hs_cC     * 0.01f - scaler_mean[3]) / scaler_scale[3];

    /* Layer 0: 4 -> 16 */
    float h[16];
    for (int j = 0; j < 16; j++) {
        float acc = b0[j] + x[0] * W0[0 * 16 + j]
                          + x[1] * W0[1 * 16 + j]
                          + x[2] * W0[2 * 16 + j]
                          + x[3] * W0[3 * 16 + j];
        h[j] = (acc > 0.0f) ? acc : 0.0f; /* ReLU */
    }

    /* Layer 1: 16 -> 1 (Method 4: ReLU Zero-Skipping) */
    float delta_T = b1[0];
    for (int k = 0; k < 16; k++) {
        if (h[k] > 0.0f) {
            delta_T += h[k] * W1[k];
        }
    }

    int32_t delta_cC = (int32_t)(delta_T * 100.0f + (delta_T >= 0.0f ? 0.5f : -0.5f));
    int32_t pred_cC = hs_cC + delta_cC;
    if (pred_cC < amb_cC) pred_cC = amb_cC;
    if (pred_cC > 12500)  pred_cC = 12500;
    return pred_cC;
}

/* ==============================================================================
 * MODE 1: 32-BIT INTEGER WITHOUT FPU — NORMAL FIXED-POINT (Division / 10000)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_32BIT_FIXED_NORMAL)

#define SCALE_NORMAL_32 10000LL

static const int32_t norm_mean_c[4]  = { 235516, 54140, 3119, 6604 };
static const int32_t norm_scale_c[4] = { 92912, 22120, 687, 1683 };

static const int32_t W0_n32[64] = {
        -4033,      8668,       -98,       517,     -6857,     -2262,     -3297,      5748,
         2452,      3101,     -5819,      5929,      3900,       170,     -4182,     -6210,
        -1399,     -1537,      -592,     -1743,       304,     -5703,     -3100,     -3066,
          903,      -927,     -3862,     -1190,     -1970,     -7308,      -317,      -384,
        -3297,      3477,      5267,      2282,     -3762,     -3161,      2884,      -677,
         -950,      3424,     -4104,      4694,       196,      1038,     -1749,      2875,
        -1953,     -7230,      3496,      4043,      9191,      6230,     -1161,      5092,
        -6763,     -5909,     -2837,     -5504,     -3877,     -4522,      5175,     -7484
};
static const int32_t b0_n32[16] = {
        -1185,       854,     -4553,      3128,     -3562,      6626,      4640,     -2635,
        -3621,      5628,       973,      3620,      3785,     -3075,     -4390,    -10928
};

static const int32_t W1_n32[16] = {
         3194,      4410,      -831,     -4536,     -9300,     -3049,      3302,      3021,
         5164,      2193,     -3249,      4055,      4119,      1978,      5317,     -6698
};
static const int32_t b1_n32[1] = {
    1031
};

int32_t mlp_new_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC)
{
    if (amb_cC <= 1000 || amb_cC >= 5000) {
        amb_cC = 2500;
    }

    int32_t raw_c[4] = { ploss_cmW, i_cmA, amb_cC, hs_cC };
    int32_t x_q[4];
    for (int i = 0; i < 4; i++) {
        int64_t diff = (int64_t)raw_c[i] - (int64_t)norm_mean_c[i];
        x_q[i] = (int32_t)((diff * SCALE_NORMAL_32) / (int64_t)norm_scale_c[i]);
    }

    /* Layer 0: 4 -> 16 */
    int32_t h_q[16];
    for (int j = 0; j < 16; j++) {
        int64_t acc = (int64_t)b0_n32[j] * SCALE_NORMAL_32
                    + (int64_t)x_q[0] * (int64_t)W0_n32[0 * 16 + j]
                    + (int64_t)x_q[1] * (int64_t)W0_n32[1 * 16 + j]
                    + (int64_t)x_q[2] * (int64_t)W0_n32[2 * 16 + j]
                    + (int64_t)x_q[3] * (int64_t)W0_n32[3 * 16 + j];
        acc /= SCALE_NORMAL_32;
        h_q[j] = (acc > 0) ? (int32_t)acc : 0;
    }

    /* Layer 1: 16 -> 1 */
    int64_t delta_acc = (int64_t)b1_n32[0] * SCALE_NORMAL_32;
    for (int k = 0; k < 16; k++) {
        if (h_q[k] != 0) {
            delta_acc += (int64_t)h_q[k] * (int64_t)W1_n32[k];
        }
    }
    delta_acc /= SCALE_NORMAL_32;

    int32_t delta_cC = (int32_t)(((int64_t)delta_acc * 100LL) / SCALE_NORMAL_32);
    int32_t pred_cC = hs_cC + delta_cC;
    if (pred_cC < amb_cC) pred_cC = amb_cC;
    if (pred_cC > 12500)  pred_cC = 12500;
    return pred_cC;
}

/* ==============================================================================
 * MODE 2: 32-BIT INTEGER WITHOUT FPU — POWER-OF-2 BIT-SHIFT (Shift = 16)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_32BIT_FIXED_POT)

#define SHIFT_POT_32 16

static const int32_t pot_mean_c[4]  = { 235516, 54140, 3119, 6604 };
static const int32_t pot_scale_c[4] = { 92912, 22120, 687, 1683 };

static const int32_t W0_p32[64] = {
       -26427,     56803,      -644,      3388,    -44939,    -14822,    -21606,     37671,
        16068,     20321,    -38136,     38857,     25556,      1114,    -27407,    -40700,
        -9167,    -10076,     -3882,    -11426,      1995,    -37372,    -20319,    -20092,
         5919,     -6075,    -25312,     -7798,    -12913,    -47892,     -2078,     -2516,
       -21605,     22789,     34517,     14958,    -24657,    -20716,     18900,     -4435,
        -6225,     22441,    -26895,     30763,      1283,      6805,    -11463,     18840,
       -12796,    -47384,     22914,     26494,     60235,     40826,     -7608,     33371,
       -44321,    -38722,    -18596,    -36072,    -25410,    -29634,     33918,    -49050
};
static const int32_t b0_p32[16] = {
        -7765,      5596,    -29836,     20501,    -23346,     43423,     30408,    -17268,
       -23731,     36881,      6378,     23725,     24802,    -20150,    -28773,    -71617
};

static const int32_t W1_p32[16] = {
        20929,     28904,     -5445,    -29725,    -60946,    -19980,     21641,     19798,
        33846,     14371,    -21291,     26575,     26995,     12965,     34848,    -43895
};
static const int32_t b1_p32[1] = {
    6754
};

int32_t mlp_new_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC)
{
    if (amb_cC <= 1000 || amb_cC >= 5000) {
        amb_cC = 2500;
    }

    int32_t raw_c[4] = { ploss_cmW, i_cmA, amb_cC, hs_cC };
    int32_t x_q[4];
    for (int i = 0; i < 4; i++) {
        int64_t diff = (int64_t)raw_c[i] - (int64_t)pot_mean_c[i];
        x_q[i] = (int32_t)((diff << SHIFT_POT_32) / (int64_t)pot_scale_c[i]);
    }

    /* Layer 0: 4 -> 16 */
    int32_t h_q[16];
    for (int j = 0; j < 16; j++) {
        int64_t acc = ((int64_t)b0_p32[j] << SHIFT_POT_32)
                    + (int64_t)x_q[0] * (int64_t)W0_p32[0 * 16 + j]
                    + (int64_t)x_q[1] * (int64_t)W0_p32[1 * 16 + j]
                    + (int64_t)x_q[2] * (int64_t)W0_p32[2 * 16 + j]
                    + (int64_t)x_q[3] * (int64_t)W0_p32[3 * 16 + j];
        acc >>= SHIFT_POT_32;
        h_q[j] = (acc > 0) ? (int32_t)acc : 0;
    }

    /* Layer 1: 16 -> 1 */
    int64_t delta_acc = ((int64_t)b1_p32[0] << SHIFT_POT_32);
    for (int k = 0; k < 16; k++) {
        if (h_q[k] != 0) {
            delta_acc += (int64_t)h_q[k] * (int64_t)W1_p32[k];
        }
    }
    delta_acc >>= SHIFT_POT_32;

    int32_t delta_cC = (int32_t)(((int64_t)delta_acc * 100LL) >> SHIFT_POT_32);
    int32_t pred_cC = hs_cC + delta_cC;
    if (pred_cC < amb_cC) pred_cC = amb_cC;
    if (pred_cC > 12500)  pred_cC = 12500;
    return pred_cC;
}

/* ==============================================================================
 * MODE 3: 16-BIT FLOAT WITH HARDWARE FPU (FP16 IN FLASH, FP32 IN REGISTERS)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_16BIT_FPU)

static const float scaler_mean[4] = {
     2355.15893555f,  541.39794922f,  31.19205475f,  66.04267120f
};
static const float scaler_scale[4] = {
     929.11730957f,  221.20219421f,  6.86826229f,  16.83120346f
};

static const uint16_t W0_f16[64] = {
    0xB674U, 0x3AEFU, 0xA108U, 0x2A9EU, 0xB97CU, 0xB33DU, 0xB546U, 0x3899U,
    0x33D9U, 0x34F6U, 0xB8A8U, 0x38BEU, 0x363DU, 0x245AU, 0xB6B1U, 0xB8F8U,
    0xB07AU, 0xB0EBU, 0xAB95U, 0xB194U, 0x27CBU, 0xB890U, 0xB4F6U, 0xB4E8U,
    0x2DC8U, 0xADEFU, 0xB62EU, 0xAF9DU, 0xB24EU, 0xB9D9U, 0xA80FU, 0xA8EAU,
    0xB546U, 0x3590U, 0x3837U, 0x334EU, 0xB605U, 0xB50FU, 0x349DU, 0xAC55U,
    0xAE14U, 0x357BU, 0xB691U, 0x3783U, 0x2503U, 0x2EA5U, 0xB199U, 0x3499U,
    0xB240U, 0xB9C9U, 0x3598U, 0x3678U, 0x3B5AU, 0x38FCU, 0xAF6EU, 0x3813U,
    0xB969U, 0xB8BAU, 0xB48AU, 0xB867U, 0xB634U, 0xB73CU, 0x3824U, 0xB9FDU
};
static const uint16_t b0_f16[16] = {
    0xAF95U, 0x2D77U, 0xB749U, 0x3501U, 0xB5B3U, 0x394DU, 0x376DU, 0xB437U,
    0xB5CBU, 0x3881U, 0x2E3AU, 0x35CBU, 0x360EU, 0xB4EBU, 0xB706U, 0xBC5FU
};

static const uint16_t W1_f16[16] = {
    0x351CU, 0x370EU, 0xAD51U, 0xB742U, 0xBB71U, 0xB4E1U, 0x3549U, 0x34D5U,
    0x3822U, 0x3304U, 0xB533U, 0x367DU, 0x3697U, 0x3255U, 0x3841U, 0xB95CU
};
static const uint16_t b1_f16[1] = {
    0x2E99U
};

int32_t mlp_new_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC)
{
    if (amb_cC <= 1000 || amb_cC >= 5000) {
        amb_cC = 2500;
    }

    float x[4];
    x[0] = ((float)ploss_cmW * 0.01f - scaler_mean[0]) / scaler_scale[0];
    x[1] = ((float)i_cmA     * 0.01f - scaler_mean[1]) / scaler_scale[1];
    x[2] = ((float)amb_cC    * 0.01f - scaler_mean[2]) / scaler_scale[2];
    x[3] = ((float)hs_cC     * 0.01f - scaler_mean[3]) / scaler_scale[3];

    /* Layer 0: 4 -> 16 */
    float h[16];
    for (int j = 0; j < 16; j++) {
        float acc = fp16_to_fp32(b0_f16[j])
                  + x[0] * fp16_to_fp32(W0_f16[0 * 16 + j])
                  + x[1] * fp16_to_fp32(W0_f16[1 * 16 + j])
                  + x[2] * fp16_to_fp32(W0_f16[2 * 16 + j])
                  + x[3] * fp16_to_fp32(W0_f16[3 * 16 + j]);
        h[j] = (acc > 0.0f) ? acc : 0.0f;
    }

    /* Layer 1: 16 -> 1 */
    float delta_T = fp16_to_fp32(b1_f16[0]);
    for (int k = 0; k < 16; k++) {
        if (h[k] > 0.0f) {
            delta_T += h[k] * fp16_to_fp32(W1_f16[k]);
        }
    }

    int32_t delta_cC = (int32_t)(delta_T * 100.0f + (delta_T >= 0.0f ? 0.5f : -0.5f));
    int32_t pred_cC = hs_cC + delta_cC;
    if (pred_cC < amb_cC) pred_cC = amb_cC;
    if (pred_cC > 12500)  pred_cC = 12500;
    return pred_cC;
}

/* ==============================================================================
 * MODE 4: 16-BIT INTEGER WITHOUT FPU — NORMAL FIXED-POINT (Division / 2048)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_16BIT_FIXED_NORMAL)

#define SCALE_NORMAL_16 2048

static const int32_t norm_mean_c[4]  = { 235516, 54140, 3119, 6604 };
static const int32_t norm_scale_c[4] = { 92912, 22120, 687, 1683 };

static const int16_t W0_n16[64] = {
         -826,      1775,       -20,       106,     -1404,      -463,      -675,      1177,
          502,       635,     -1192,      1214,       799,        35,      -856,     -1272,
         -286,      -315,      -121,      -357,        62,     -1168,      -635,      -628,
          185,      -190,      -791,      -244,      -404,     -1497,       -65,       -79,
         -675,       712,      1079,       467,      -771,      -647,       591,      -139,
         -195,       701,      -840,       961,        40,       213,      -358,       589,
         -400,     -1481,       716,       828,      1882,      1276,      -238,      1043,
        -1385,     -1210,      -581,     -1127,      -794,      -926,      1060,     -1533
};
static const int32_t b0_n16[16] = {
         -243,       175,      -932,       641,      -730,      1357,       950,      -540,
         -742,      1153,       199,       741,       775,      -630,      -899,     -2238
};

static const int16_t W1_n16[16] = {
          654,       903,      -170,      -929,     -1905,      -624,       676,       619,
         1058,       449,      -665,       830,       844,       405,      1089,     -1372
};
static const int32_t b1_n16[1] = {
    211
};

int32_t mlp_new_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC)
{
    if (amb_cC <= 1000 || amb_cC >= 5000) {
        amb_cC = 2500;
    }

    int32_t raw_c[4] = { ploss_cmW, i_cmA, amb_cC, hs_cC };
    int32_t x_q[4];
    for (int i = 0; i < 4; i++) {
        int32_t diff = raw_c[i] - norm_mean_c[i];
        x_q[i] = (int32_t)(((int64_t)diff * SCALE_NORMAL_16) / norm_scale_c[i]);
    }

    /* Layer 0: 4 -> 16 */
    int16_t h_q[16];
    for (int j = 0; j < 16; j++) {
        int32_t acc = (int32_t)b0_n16[j] * SCALE_NORMAL_16
                    + (int32_t)x_q[0] * W0_n16[0 * 16 + j]
                    + (int32_t)x_q[1] * W0_n16[1 * 16 + j]
                    + (int32_t)x_q[2] * W0_n16[2 * 16 + j]
                    + (int32_t)x_q[3] * W0_n16[3 * 16 + j];
        acc /= SCALE_NORMAL_16;
        h_q[j] = (acc > 0) ? (int16_t)acc : (int16_t)0;
    }

    /* Layer 1: 16 -> 1 */
    int32_t delta_acc = (int32_t)b1_n16[0] * SCALE_NORMAL_16;
    for (int k = 0; k < 16; k++) {
        if (h_q[k] != 0) {
            delta_acc += (int32_t)h_q[k] * W1_n16[k];
        }
    }
    delta_acc /= SCALE_NORMAL_16;

    int32_t delta_cC = (int32_t)(((int64_t)delta_acc * 100LL) / SCALE_NORMAL_16);
    int32_t pred_cC = hs_cC + delta_cC;
    if (pred_cC < amb_cC) pred_cC = amb_cC;
    if (pred_cC > 12500)  pred_cC = 12500;
    return pred_cC;
}

/* ==============================================================================
 * MODE 5: 16-BIT INTEGER — POWER-OF-2 BIT-SHIFT + ARM SMLAD SIMD ASSEMBLY
 * Executes TWO signed 16-bit MACs per clock cycle!
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_16BIT_FIXED_POT_SIMD)

#define SHIFT_POT_16 11

static const int32_t pot_mean_c[4]  = { 235516, 54140, 3119, 6604 };
static const int32_t pot_scale_c[4] = { 92912, 22120, 687, 1683 };

/* Packed 32-bit SIMD weights: Pair 0 (w0, w1) and Pair 1 (w2, w3) */
static const uint32_t W0_simd_01[16] = {
    0xFEE2FCC6U, 0xFEC506EFU, 0xFF87FFECU, 0xFE9B006AU,
    0x003EFA84U, 0xFB70FE31U, 0xFD85FD5DU, 0xFD8C0499U,
    0x00B901F6U, 0xFF42027BU, 0xFCE9FB58U, 0xFF0C04BEU,
    0xFE6C031FU, 0xFA270023U, 0xFFBFFCA8U, 0xFFB1FB08U
};

static const uint32_t W0_simd_23[16] = {
    0xFE70FD5DU, 0xFA3702C8U, 0x02CC0437U, 0x033C01D3U,
    0x075AFCFDU, 0x04FCFD79U, 0xFF12024FU, 0x0413FF75U,
    0xFA97FF3DU, 0xFB4602BDU, 0xFDBBFCB8U, 0xFB9903C1U,
    0xFCE60028U, 0xFC6200D5U, 0x0424FE9AU, 0xFA03024DU
};

static const int32_t b0_simd[16] = {
         -243,       175,      -932,       641,      -730,      1357,       950,      -540,
         -742,      1153,       199,       741,       775,      -630,      -899,     -2238
};

/* Packed 32-bit SIMD weights for Layer 1: 8 pairs */
static const uint32_t W1_simd[8] = {
    0x0387028EU, 0xFC5FFF56U, 0xFD90F88FU, 0x026B02A4U,
    0x01C10422U, 0x033EFD67U, 0x0195034CU, 0xFAA40441U
};

static const int32_t b1_simd[1] = {
    211
};

int32_t mlp_new_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC)
{
    if (amb_cC <= 1000 || amb_cC >= 5000) {
        amb_cC = 2500;
    }

    int32_t raw_c[4] = { ploss_cmW, i_cmA, amb_cC, hs_cC };
    int16_t x_q[4];
    for (int i = 0; i < 4; i++) {
        int32_t diff = raw_c[i] - pot_mean_c[i];
        x_q[i] = (int16_t)(((int64_t)diff << SHIFT_POT_16) / pot_scale_c[i]);
    }

    /* Pack input feature pairs for dual-MAC SMLAD instructions */
    uint32_t x_pair01 = ((uint32_t)(uint16_t)x_q[1] << 16) | (uint16_t)x_q[0];
    uint32_t x_pair23 = ((uint32_t)(uint16_t)x_q[3] << 16) | (uint16_t)x_q[2];

    /* Layer 0: 4 -> 16 (2 smlad instructions per neuron) */
    int16_t h_q[16];
    for (int j = 0; j < 16; j++) {
        int32_t acc = (int32_t)b0_simd[j] << SHIFT_POT_16;
        acc = smlad_dual(x_pair01, W0_simd_01[j], acc);
        acc = smlad_dual(x_pair23, W0_simd_23[j], acc);
        acc >>= SHIFT_POT_16;
        h_q[j] = (acc > 0) ? (int16_t)acc : (int16_t)0;
    }

    /* Layer 1: 16 -> 1 (8 smlad instructions over 8 packed pairs) */
    int32_t delta_acc = (int32_t)b1_simd[0] << SHIFT_POT_16;
    for (int p = 0; p < 8; p++) {
        uint32_t h_pair = ((uint32_t)(uint16_t)h_q[2 * p + 1] << 16) | (uint16_t)h_q[2 * p];
        if (h_pair != 0U) {
            delta_acc = smlad_dual(h_pair, W1_simd[p], delta_acc);
        }
    }
    delta_acc >>= SHIFT_POT_16;

    int32_t delta_cC = (int32_t)(((int64_t)delta_acc * 100LL) >> SHIFT_POT_16);
    int32_t pred_cC = hs_cC + delta_cC;
    if (pred_cC < amb_cC) pred_cC = amb_cC;
    if (pred_cC > 12500)  pred_cC = 12500;
    return pred_cC;
}

#endif
