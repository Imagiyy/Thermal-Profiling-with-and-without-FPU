/*
 * mlp_model.c — TinyML 6-Mode Inference Engine for STM32F401RET6
 * Architecture: 4 Inputs -> 32 Neurons (ReLU) -> 1 Output (Linear)
 *
 * ZERO-LEAKAGE: Evaluated strictly on unseen future time horizon.
 */

#include "mlp_model.h"
#include <stdint.h>

static int32_t g_sample_count = 0;

void mlp_reset_timebase(void)
{
    g_sample_count = 0;
}

/* ==============================================================================
 * MODE 0: 32-BIT FLOAT WITH HARDWARE FPU (FP32)
 * ============================================================================== */
#if (BENCHMARK_MODE == MODE_32BIT_FPU)

static const float f32_mean[4] = {
    2327.2650470f,  536.5514767f,   30.6669503f,    6.5170872f
};

static const float f32_scale[4] = {
     928.3359692f,  225.6783391f,    8.5495057f,    4.2793284f
};

static const float W0_f32[4 * 32] = {
      -0.7359609f,    1.6919244f,    1.6952842f,    1.5626949f,   -1.4711716f,    1.3821629f,
      -1.5599550f,    0.0252599f,   -0.2425567f,    1.2149592f,    0.2258803f,   -0.1754080f,
       1.4410204f,    0.3957714f,    0.6034296f,    0.4303072f,   -1.5024773f,   -0.7145879f,
      -1.1503555f,   -1.0331305f,    1.7659189f,    0.6600626f,    1.0058233f,   -1.3312721f,
       0.1424810f,    1.6765944f,    1.0424100f,    0.7896649f,    1.2249757f,   -1.6366392f,
       0.3645507f,   -1.1724636f,   -1.7317847f,    1.0233475f,    0.9784929f,    0.6652204f,
      -0.3426733f,   -0.2735306f,    0.8480366f,    0.0017699f,   -0.9884611f,   -0.1589099f,
      -2.1280241f,   -0.6177338f,   -0.3121069f,    0.1727304f,   -1.2223426f,    0.3881702f,
       0.9134658f,   -1.6761686f,    1.0409333f,   -1.5509634f,    1.6377424f,    0.6233866f,
       0.3685074f,    1.0310053f,    0.7858433f,   -0.3340155f,    0.0581109f,   -1.9142287f,
      -0.2656337f,    0.8751672f,    0.3424234f,    0.6498332f,    0.0314248f,    0.0484843f,
      -0.5267152f,    0.5093486f,   -0.0045841f,    0.0029236f,   -0.0023497f,   -0.0011069f,
      -0.0016105f,    0.0009529f,    0.0017087f,   -0.0012880f,    0.0010193f,   -0.0058987f,
       0.0023869f,    0.0016814f,    0.6873761f,   -0.0262131f,   -0.5333654f,   -0.0043306f,
       0.0013454f,    0.0022054f,    0.0005816f,    0.0827938f,   -0.0023745f,    0.0030678f,
       0.0000158f,    0.0004137f,    0.0053375f,   -0.0024160f,   -0.0037366f,   -0.1644280f,
       1.3627100f,    0.0527180f,   -0.3500127f,   -0.2606664f,   -2.0296943f,    1.3935664f,
      -0.2667474f,   -0.0000000f,   -0.6461157f,    0.7385443f,    0.6526845f,   -0.4307012f,
       0.6964719f,   -2.9720873f,    0.0691939f,   -3.0620355f,    0.7462252f,    1.4518799f,
       0.4735692f,    1.6020705f,    1.8228888f,   -3.0767100f,    1.3214937f,    0.4179677f,
       2.4589654f,    1.6959619f,    1.1502129f,    0.6351030f,    0.6351351f,   -0.4165790f,
       0.7739527f,    0.6616275f
};

static const float b0_f32[32] = {
      -2.3379719f,    1.1154752f,    1.1641906f,    1.0167967f,   -2.0718057f,    0.8424808f,
       1.6724683f,   -0.2835816f,   -0.1971131f,    1.8624276f,    0.8564604f,   -0.0744650f,
       2.1856394f,   -3.9837020f,    1.4036258f,   -3.2017184f,    1.2610753f,   -2.2980743f,
       1.3325019f,   -1.1948185f,    0.4398625f,   -1.8403846f,    1.7849833f,    1.0913320f,
      -0.9083549f,    1.0265897f,    1.4959510f,    0.7356946f,    1.9063932f,    1.5332595f,
      -1.6487459f,    1.4040111f
};

static const float W1_f32[32] = {
      -2.0380685f,    1.1692684f,    1.3118070f,    1.2547536f,   -2.6973341f,    1.4773856f,
       1.0303016f,   -0.0179380f,   -0.7196947f,    1.2617435f,    1.5079926f,   -0.2715157f,
       1.1361701f,   -6.4229322f,    1.2556242f,   -5.1125185f,    1.1659163f,   -2.0827281f,
       1.3210101f,   -2.6343391f,   -2.0230176f,   -3.6990908f,    1.6743385f,    1.3754502f,
      -2.1143859f,    1.3707257f,    1.3532849f,    1.2229379f,    1.2140666f,    1.2951165f,
      -1.7483131f,    1.3303846f
};

static const float b1_f32[1] = {
    1.0996176f
};

int32_t cnn_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC)
{
    (void)hs_cC;
    g_sample_count++;

    float x[4];
    x[0] = ((float)ploss_cmW * 0.01f - f32_mean[0]) / f32_scale[0];
    x[1] = ((float)i_cmA * 0.01f - f32_mean[1]) / f32_scale[1];
    x[2] = ((float)amb_cC * 0.01f - f32_mean[2]) / f32_scale[2];
    x[3] = (((float)g_sample_count * (0.5f / 60.0f)) - f32_mean[3]) / f32_scale[3];

    float h[32];
    for (int j = 0; j < 32; j++) {
        float acc = b0_f32[j];
        for (int i = 0; i < 4; i++) {
            acc += x[i] * W0_f32[i * 32 + j];
        }
        h[j] = (acc > 0.0f) ? acc : 0.0f;
    }

    float out = b1_f32[0];
    for (int k = 0; k < 32; k++) {
        out += h[k] * W1_f32[k];
    }

    int32_t delta_cC = (int32_t)(out * 100.0f);
    int32_t final_cC = amb_cC + delta_cC;
    if (final_cC < amb_cC) {
        final_cC = amb_cC;
    }
    if (final_cC > 12500) {
        final_cC = 12500;
    }
    return final_cC;
}

/* ==============================================================================
 * MODE 1: 32-BIT FIXED-POINT NORMAL DIVISION (Scale = 10000)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_32BIT_FIXED_NORMAL)

#define SCALE_NORMAL_32 10000

static const int32_t n32_mean_c[4] = {
       232727,     53655,      3067,       652
};

static const int32_t n32_scale_c[4] = {
        92834,     22568,       855,       428
};

static const int32_t W0_n32[4 * 32] = {
        -7360,     16919,     16953,     15627,    -14712,     13822,    -15600,       253,
        -2426,     12150,      2259,     -1754,     14410,      3958,      6034,      4303,
       -15025,     -7146,    -11504,    -10331,     17659,      6601,     10058,    -13313,
         1425,     16766,     10424,      7897,     12250,    -16366,      3646,    -11725,
       -17318,     10233,      9785,      6652,     -3427,     -2735,      8480,        18,
        -9885,     -1589,    -21280,     -6177,     -3121,      1727,    -12223,      3882,
         9135,    -16762,     10409,    -15510,     16377,      6234,      3685,     10310,
         7858,     -3340,       581,    -19142,     -2656,      8752,      3424,      6498,
          314,       485,     -5267,      5093,       -46,        29,       -23,       -11,
          -16,        10,        17,       -13,        10,       -59,        24,        17,
         6874,      -262,     -5334,       -43,        13,        22,         6,       828,
          -24,        31,         0,         4,        53,       -24,       -37,     -1644,
        13627,       527,     -3500,     -2607,    -20297,     13936,     -2667,         0,
        -6461,      7385,      6527,     -4307,      6965,    -29721,       692,    -30620,
         7462,     14519,      4736,     16021,     18229,    -30767,     13215,      4180,
        24590,     16960,     11502,      6351,      6351,     -4166,      7740,      6616
};

static const int32_t b0_n32[32] = {
       -23380,     11155,     11642,     10168,    -20718,      8425,     16725,     -2836,
        -1971,     18624,      8565,      -745,     21856,    -39837,     14036,    -32017,
        12611,    -22981,     13325,    -11948,      4399,    -18404,     17850,     10913,
        -9084,     10266,     14960,      7357,     19064,     15333,    -16487,     14040
};

static const int32_t W1_n32[32] = {
       -20381,     11693,     13118,     12548,    -26973,     14774,     10303,      -179,
        -7197,     12617,     15080,     -2715,     11362,    -64229,     12556,    -51125,
        11659,    -20827,     13210,    -26343,    -20230,    -36991,     16743,     13755,
       -21144,     13707,     13533,     12229,     12141,     12951,    -17483,     13304
};

static const int32_t b1_n32[1] = {
    10996
};

int32_t cnn_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC)
{
    (void)hs_cC;
    g_sample_count++;

    int32_t raw_c[4];
    raw_c[0] = ploss_cmW;
    raw_c[1] = i_cmA;
    raw_c[2] = amb_cC;
    raw_c[3] = (int32_t)(((int64_t)g_sample_count * 100LL) / 120LL);

    int32_t x_q[4];
    for (int i = 0; i < 4; i++) {
        int32_t diff = raw_c[i] - n32_mean_c[i];
        x_q[i] = (int32_t)(((int64_t)diff * SCALE_NORMAL_32) / n32_scale_c[i]);
    }

    int32_t h_q[32];
    for (int j = 0; j < 32; j++) {
        int64_t acc = (int64_t)b0_n32[j] * SCALE_NORMAL_32;
        for (int i = 0; i < 4; i++) {
            acc += (int64_t)x_q[i] * (int64_t)W0_n32[i * 32 + j];
        }
        acc /= SCALE_NORMAL_32;
        h_q[j] = (acc > 0) ? (int32_t)acc : 0;
    }

    int64_t out_acc = (int64_t)b1_n32[0] * SCALE_NORMAL_32;
    for (int k = 0; k < 32; k++) {
        out_acc += (int64_t)h_q[k] * (int64_t)W1_n32[k];
    }
    out_acc /= SCALE_NORMAL_32;

    int32_t delta_cC = (int32_t)((out_acc * 100LL) / SCALE_NORMAL_32);
    int32_t final_cC = amb_cC + delta_cC;
    if (final_cC < amb_cC) {
        final_cC = amb_cC;
    }
    if (final_cC > 12500) {
        final_cC = 12500;
    }
    return final_cC;
}

/* ==============================================================================
 * MODE 2: 32-BIT FIXED-POINT POWER-OF-2 BIT-SHIFT (Q16.16)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_32BIT_FIXED_POT)

#define SHIFT_POT_32 16

static const int32_t pot32_mean_c[4] = {
       232727,     53655,      3067,       652
};

static const int32_t pot32_scale_c[4] = {
        92834,     22568,       855,       428
};

static const int32_t W0_p32[4 * 32] = {
       -48232,    110882,    111102,    102413,    -96415,     90581,   -102233,      1655,
       -15896,     79624,     14803,    -11496,     94439,     25937,     39546,     28201,
       -98466,    -46831,    -75390,    -67707,    115731,     43258,     65918,    -87246,
         9338,    109877,     68315,     51751,     80280,   -107259,     23891,    -76839,
      -113494,     67066,     64127,     43596,    -22457,    -17926,     55577,       116,
       -64780,    -10414,   -139462,    -40484,    -20454,     11320,    -80107,     25439,
        59865,   -109849,     68219,   -101644,    107331,     40854,     24151,     67568,
        51501,    -21890,      3808,   -125451,    -17409,     57355,     22441,     42587,
         2059,      3177,    -34519,     33381,      -300,       192,      -154,       -73,
         -106,        62,       112,       -84,        67,      -387,       156,       110,
        45048,     -1718,    -34955,      -284,        88,       145,        38,      5426,
         -156,       201,         1,        27,       350,      -158,      -245,    -10776,
        89307,      3455,    -22938,    -17083,   -133018,     91329,    -17482,         0,
       -42344,     48401,     42774,    -28226,     45644,   -194779,      4535,   -200674,
        48905,     95150,     31036,    104993,    119465,   -201635,     86605,     27392,
       161151,    111147,     75380,     41622,     41624,    -27301,     50722,     43360
};

static const int32_t b0_p32[32] = {
      -153221,     73104,     76296,     66637,   -135778,     55213,    109607,    -18585,
       -12918,    122056,     56129,     -4880,    143238,   -261076,     91988,   -209828,
        82646,   -150607,     87327,    -78304,     28827,   -120611,    116981,     71522,
       -59530,     67279,     98039,     48214,    124937,    100484,   -108052,     92013
};

static const int32_t W1_p32[32] = {
      -133567,     76629,     85971,     82232,   -176772,     96822,     67522,     -1176,
       -47166,     82690,     98828,    -17794,     74460,   -420933,     82289,   -335054,
        76409,   -136494,     86574,   -172644,   -132580,   -242424,    109729,     90142,
      -138568,     89832,     88689,     80146,     79565,     84877,   -114577,     87188
};

static const int32_t b1_p32[1] = {
    72065
};

int32_t cnn_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC)
{
    (void)hs_cC;
    g_sample_count++;

    int32_t raw_c[4];
    raw_c[0] = ploss_cmW;
    raw_c[1] = i_cmA;
    raw_c[2] = amb_cC;
    raw_c[3] = (int32_t)(((int64_t)g_sample_count * 100LL) / 120LL);

    int32_t x_q[4];
    for (int i = 0; i < 4; i++) {
        int32_t diff = raw_c[i] - pot32_mean_c[i];
        x_q[i] = (int32_t)(((int64_t)diff << SHIFT_POT_32) / pot32_scale_c[i]);
    }

    int32_t h_q[32];
    for (int j = 0; j < 32; j++) {
        int64_t acc = (int64_t)b0_p32[j] << SHIFT_POT_32;
        for (int i = 0; i < 4; i++) {
            acc += (int64_t)x_q[i] * (int64_t)W0_p32[i * 32 + j];
        }
        acc >>= SHIFT_POT_32;
        h_q[j] = (acc > 0) ? (int32_t)acc : 0;
    }

    int64_t out_acc = (int64_t)b1_p32[0] << SHIFT_POT_32;
    for (int k = 0; k < 32; k++) {
        out_acc += (int64_t)h_q[k] * (int64_t)W1_p32[k];
    }
    out_acc >>= SHIFT_POT_32;

    int32_t delta_cC = (int32_t)(((int64_t)out_acc * 100LL) >> SHIFT_POT_32);
    int32_t final_cC = amb_cC + delta_cC;
    if (final_cC < amb_cC) {
        final_cC = amb_cC;
    }
    if (final_cC > 12500) {
        final_cC = 12500;
    }
    return final_cC;
}

/* ==============================================================================
 * MODE 3: 16-BIT FLOAT (FP16 STORAGE / FP32 COMPUTE WITH FPU)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_16BIT_FPU)

static inline float fp16_to_fp32(uint16_t h)
{
    uint32_t sign = ((uint32_t)h & 0x8000U) << 16;
    uint32_t exp  = ((uint32_t)h & 0x7C00U) >> 10;
    uint32_t mant = ((uint32_t)h & 0x03FFU);
    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            while ((mant & 0x0400U) == 0) {
                mant <<= 1;
                exp--;
            }
            exp++;
            mant &= ~0x0400U;
            f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000U | (mant << 13);
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }

    union {
        uint32_t u;
        float f;
    } conv;
    conv.u = f;
    return conv.f;
}

static const float f32_mean[4] = {
    2327.2650470f,  536.5514767f,   30.6669503f,    6.5170872f
};

static const float f32_scale[4] = {
     928.3359692f,  225.6783391f,    8.5495057f,    4.2793284f
};

static const uint16_t W0_f16[4 * 32] = {
    0x0000B9E3U, 0x00003EC5U, 0x00003EC8U, 0x00003E40U, 0x0000BDE2U, 0x00003D87U, 0x0000BE3DU, 0x00002677U,
    0x0000B3C3U, 0x00003CDCU, 0x0000333AU, 0x0000B19DU, 0x00003DC4U, 0x00003655U, 0x000038D4U, 0x000036E3U,
    0x0000BE03U, 0x0000B9B7U, 0x0000BC9AU, 0x0000BC22U, 0x00003F10U, 0x00003948U, 0x00003C06U, 0x0000BD53U,
    0x0000308FU, 0x00003EB5U, 0x00003C2BU, 0x00003A51U, 0x00003CE6U, 0x0000BE8CU, 0x000035D5U, 0x0000BCB1U,
    0x0000BEEDU, 0x00003C18U, 0x00003BD4U, 0x00003952U, 0x0000B57CU, 0x0000B460U, 0x00003AC9U, 0x00001740U,
    0x0000BBE8U, 0x0000B116U, 0x0000C042U, 0x0000B8F1U, 0x0000B4FEU, 0x00003187U, 0x0000BCE4U, 0x00003636U,
    0x00003B4FU, 0x0000BEB4U, 0x00003C2AU, 0x0000BE34U, 0x00003E8DU, 0x000038FDU, 0x000035E5U, 0x00003C20U,
    0x00003A49U, 0x0000B558U, 0x00002B70U, 0x0000BFA8U, 0x0000B440U, 0x00003B00U, 0x0000357BU, 0x00003933U,
    0x00002806U, 0x00002A35U, 0x0000B837U, 0x00003813U, 0x00009CB2U, 0x000019FDU, 0x000098D0U, 0x00009489U,
    0x00009699U, 0x000013CEU, 0x00001700U, 0x00009547U, 0x0000142DU, 0x00009E0AU, 0x000018E3U, 0x000016E3U,
    0x00003980U, 0x0000A6B6U, 0x0000B844U, 0x00009C6FU, 0x00001583U, 0x00001884U, 0x000010C4U, 0x00002D4CU,
    0x000098DDU, 0x00001A48U, 0x0000010AU, 0x00000EC7U, 0x00001D77U, 0x000098F3U, 0x00009BA7U, 0x0000B143U,
    0x00003D73U, 0x00002ABFU, 0x0000B59AU, 0x0000B42CU, 0x0000C00FU, 0x00003D93U, 0x0000B445U, 0x00008000U,
    0x0000B92BU, 0x000039E9U, 0x00003939U, 0x0000B6E4U, 0x00003992U, 0x0000C1F2U, 0x00002C6EU, 0x0000C220U,
    0x000039F8U, 0x00003DCFU, 0x00003794U, 0x00003E69U, 0x00003F4BU, 0x0000C227U, 0x00003D49U, 0x000036B0U,
    0x000040EBU, 0x00003EC9U, 0x00003C9AU, 0x00003915U, 0x00003915U, 0x0000B6AAU, 0x00003A31U, 0x0000394BU
};

static const uint16_t b0_f16[32] = {
    0x0000C0ADU, 0x00003C76U, 0x00003CA8U, 0x00003C11U, 0x0000C025U, 0x00003ABDU, 0x00003EB1U, 0x0000B48AU,
    0x0000B24FU, 0x00003F73U, 0x00003ADAU, 0x0000ACC4U, 0x0000405FU, 0x0000C3F8U, 0x00003D9DU, 0x0000C267U,
    0x00003D0BU, 0x0000C099U, 0x00003D54U, 0x0000BCC7U, 0x0000370AU, 0x0000BF5DU, 0x00003F24U, 0x00003C5EU,
    0x0000BB44U, 0x00003C1BU, 0x00003DFCU, 0x000039E3U, 0x00003FA0U, 0x00003E22U, 0x0000BE98U, 0x00003D9EU
};

static const uint16_t W1_f16[32] = {
    0x0000C013U, 0x00003CADU, 0x00003D3FU, 0x00003D05U, 0x0000C165U, 0x00003DE9U, 0x00003C1FU, 0x0000A498U,
    0x0000B9C2U, 0x00003D0CU, 0x00003E08U, 0x0000B458U, 0x00003C8BU, 0x0000C66CU, 0x00003D06U, 0x0000C51DU,
    0x00003CAAU, 0x0000C02AU, 0x00003D49U, 0x0000C145U, 0x0000C00CU, 0x0000C366U, 0x00003EB3U, 0x00003D80U,
    0x0000C03BU, 0x00003D7CU, 0x00003D6AU, 0x00003CE4U, 0x00003CDBU, 0x00003D2EU, 0x0000BEFEU, 0x00003D52U
};

static const uint16_t b1_f16[1] = {
    0x3C66U
};

int32_t cnn_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC)
{
    (void)hs_cC;
    g_sample_count++;

    float x[4];
    x[0] = ((float)ploss_cmW * 0.01f - f32_mean[0]) / f32_scale[0];
    x[1] = ((float)i_cmA * 0.01f - f32_mean[1]) / f32_scale[1];
    x[2] = ((float)amb_cC * 0.01f - f32_mean[2]) / f32_scale[2];
    x[3] = (((float)g_sample_count * (0.5f / 60.0f)) - f32_mean[3]) / f32_scale[3];

    float h[32];
    for (int j = 0; j < 32; j++) {
        float acc = fp16_to_fp32(b0_f16[j]);
        for (int i = 0; i < 4; i++) {
            acc += x[i] * fp16_to_fp32(W0_f16[i * 32 + j]);
        }
        h[j] = (acc > 0.0f) ? acc : 0.0f;
    }

    float out = fp16_to_fp32(b1_f16[0]);
    for (int k = 0; k < 32; k++) {
        out += h[k] * fp16_to_fp32(W1_f16[k]);
    }

    int32_t delta_cC = (int32_t)(out * 100.0f);
    int32_t final_cC = amb_cC + delta_cC;
    if (final_cC < amb_cC) {
        final_cC = amb_cC;
    }
    if (final_cC > 12500) {
        final_cC = 12500;
    }
    return final_cC;
}

/* ==============================================================================
 * MODE 4: 16-BIT FIXED-POINT NORMAL DIVISION (Scale = 2048)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_16BIT_FIXED_NORMAL)

#define SCALE_NORMAL_16 2048

static const int32_t n16_mean_c[4] = {
       232727,     53655,      3067,       652
};

static const int32_t n16_scale_c[4] = {
        92834,     22568,       855,       428
};

static const int16_t W0_n16[4 * 32] = {
        -1507,      3465,      3472,      3200,     -3013,      2831,     -3195,        52,
         -497,      2488,       463,      -359,      2951,       811,      1236,       881,
        -3077,     -1463,     -2356,     -2116,      3617,      1352,      2060,     -2726,
          292,      3434,      2135,      1617,      2509,     -3352,       747,     -2401,
        -3547,      2096,      2004,      1362,      -702,      -560,      1737,         4,
        -2024,      -325,     -4358,     -1265,      -639,       354,     -2503,       795,
         1871,     -3433,      2132,     -3176,      3354,      1277,       755,      2111,
         1609,      -684,       119,     -3920,      -544,      1792,       701,      1331,
           64,        99,     -1079,      1043,        -9,         6,        -5,        -2,
           -3,         2,         3,        -3,         2,       -12,         5,         3,
         1408,       -54,     -1092,        -9,         3,         5,         1,       170,
           -5,         6,         0,         1,        11,        -5,        -8,      -337,
         2791,       108,      -717,      -534,     -4157,      2854,      -546,         0,
        -1323,      1513,      1337,      -882,      1426,     -6087,       142,     -6271,
         1528,      2973,       970,      3281,      3733,     -6301,      2706,       856,
         5036,      3473,      2356,      1301,      1301,      -853,      1585,      1355
};

static const int32_t b0_n16[32] = {
        -4788,      2284,      2384,      2082,     -4243,      1725,      3425,      -581,
         -404,      3814,      1754,      -153,      4476,     -8159,      2875,     -6557,
         2583,     -4706,      2729,     -2447,       901,     -3769,      3656,      2235,
        -1860,      2102,      3064,      1507,      3904,      3140,     -3377,      2875
};

static const int16_t W1_n16[32] = {
        -4174,      2395,      2687,      2570,     -5524,      3026,      2110,       -37,
        -1474,      2584,      3088,      -556,      2327,    -13154,      2572,    -10470,
         2388,     -4265,      2705,     -5395,     -4143,     -7576,      3429,      2817,
        -4330,      2807,      2772,      2505,      2486,      2652,     -3581,      2725
};

static const int32_t b1_n16[1] = {
    2252
};

int32_t cnn_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC)
{
    (void)hs_cC;
    g_sample_count++;

    int32_t raw_c[4];
    raw_c[0] = ploss_cmW;
    raw_c[1] = i_cmA;
    raw_c[2] = amb_cC;
    raw_c[3] = (int32_t)(((int64_t)g_sample_count * 100LL) / 120LL);

    int16_t x_q[4];
    for (int i = 0; i < 4; i++) {
        int32_t diff = raw_c[i] - n16_mean_c[i];
        x_q[i] = (int16_t)(((int32_t)diff * SCALE_NORMAL_16) / n16_scale_c[i]);
    }

    int16_t h_q[32];
    for (int j = 0; j < 32; j++) {
        int32_t acc = b0_n16[j] * SCALE_NORMAL_16;
        for (int i = 0; i < 4; i++) {
            acc += (int32_t)x_q[i] * (int32_t)W0_n16[i * 32 + j];
        }
        acc /= SCALE_NORMAL_16;
        h_q[j] = (acc > 0) ? (int16_t)acc : 0;
    }

    int32_t out_acc = b1_n16[0] * SCALE_NORMAL_16;
    for (int k = 0; k < 32; k++) {
        out_acc += (int32_t)h_q[k] * (int32_t)W1_n16[k];
    }
    out_acc /= SCALE_NORMAL_16;

    int32_t delta_cC = (int32_t)(((int32_t)out_acc * 100) / SCALE_NORMAL_16);
    int32_t final_cC = amb_cC + delta_cC;
    if (final_cC < amb_cC) {
        final_cC = amb_cC;
    }
    if (final_cC > 12500) {
        final_cC = 12500;
    }
    return final_cC;
}

/* ==============================================================================
 * MODE 5: 16-BIT FIXED-POINT POWER-OF-2 BIT-SHIFT (Q5.11) + ARM SMLAD SIMD ASSEMBLY
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_16BIT_FIXED_POT_SIMD)

#define SHIFT_POT_16 11

static inline int32_t smlad_dual(uint32_t val1, uint32_t val2, int32_t acc)
{
#if defined(__arm__) || defined(__thumb__) || defined(__thumb2__)
    __asm__ volatile ("smlad %0, %1, %2, %0" : "+r" (acc) : "r" (val1), "r" (val2));
    return acc;
#else
    int16_t v1_lo = (int16_t)(val1 & 0xFFFFU);
    int16_t v1_hi = (int16_t)(val1 >> 16);
    int16_t v2_lo = (int16_t)(val2 & 0xFFFFU);
    int16_t v2_hi = (int16_t)(val2 >> 16);
    return acc + ((int32_t)v1_lo * (int32_t)v2_lo) + ((int32_t)v1_hi * (int32_t)v2_hi);
#endif
}

static const int32_t pot_mean_c[4] = {
       232727,     53655,      3067,       652
};

static const int32_t pot_scale_c[4] = {
        92834,     22568,       855,       428
};

static const uint32_t W0_simd_01[32] = {
    0xF225FA1DU, 0x08300D89U, 0x07D40D90U, 0x05520C80U, 0xFD42F43BU, 0xFDD00B0FU, 0x06C9F385U, 0x00040034U,
    0xF818FE0FU, 0xFEBB09B8U, 0xEEFA01CFU, 0xFB0FFE99U, 0xFD810B87U, 0x0162032BU, 0xF63904D4U, 0x031B0371U,
    0x074FF3FBU, 0xF297FA49U, 0x0854F6CCU, 0xF398F7BCU, 0x0D1A0E21U, 0x04FD0548U, 0x02F3080CU, 0x083FF55AU,
    0x06490124U, 0xFD540D6AU, 0x00770857U, 0xF0B00651U, 0xFDE009CDU, 0x0700F2E8U, 0x02BD02EBU, 0x0533F69FU
};

static const uint32_t W0_simd_23[32] = {
    0x0AE70040U, 0x006C0063U, 0xFD33FBC9U, 0xFDEA0413U, 0xEFC3FFF7U, 0x0B260006U, 0xFDDEFFFBU, 0x0000FFFEU,
    0xFAD5FFFDU, 0x05E90002U, 0x05390003U, 0xFC8EFFFDU, 0x05920002U, 0xE839FFF4U, 0x008E0005U, 0xE7810003U,
    0x05F80580U, 0x0B9DFFCAU, 0x03CAFBBCU, 0x0CD1FFF7U, 0x0E950003U, 0xE7630005U, 0x0A920001U, 0x035800AAU,
    0x13ACFFFBU, 0x0D910006U, 0x09340000U, 0x05150001U, 0x0515000BU, 0xFCABFFFBU, 0x0631FFF8U, 0x054BFEAFU
};

static const int32_t b0_simd[32] = {
        -4788,      2284,      2384,      2082,     -4243,      1725,      3425,      -581,
         -404,      3814,      1754,      -153,      4476,     -8159,      2875,     -6557,
         2583,     -4706,      2729,     -2447,       901,     -3769,      3656,      2235,
        -1860,      2102,      3064,      1507,      3904,      3140,     -3377,      2875
};

static const int16_t W1_simd[32] = {
        -4174,      2395,      2687,      2570,     -5524,      3026,      2110,       -37,
        -1474,      2584,      3088,      -556,      2327,    -13154,      2572,    -10470,
         2388,     -4265,      2705,     -5395,     -4143,     -7576,      3429,      2817,
        -4330,      2807,      2772,      2505,      2486,      2652,     -3581,      2725
};

static const int32_t b1_simd[1] = {
    2252
};

int32_t cnn_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC)
{
    (void)hs_cC;
    g_sample_count++;

    int32_t raw_c[4];
    raw_c[0] = ploss_cmW;
    raw_c[1] = i_cmA;
    raw_c[2] = amb_cC;
    raw_c[3] = (int32_t)(((int64_t)g_sample_count * 100LL) / 120LL);

    int16_t x_q[4];
    for (int i = 0; i < 4; i++) {
        int32_t diff = raw_c[i] - pot_mean_c[i];
        x_q[i] = (int16_t)((diff << SHIFT_POT_16) / pot_scale_c[i]);
    }

    /* Pack 4 inputs into two 32-bit registers for ARM SIMD instructions */
    uint32_t x_pair_01 = ((uint32_t)(uint16_t)x_q[1] << 16) | ((uint32_t)(uint16_t)x_q[0]);
    uint32_t x_pair_23 = ((uint32_t)(uint16_t)x_q[3] << 16) | ((uint32_t)(uint16_t)x_q[2]);

    /* Layer 0: 4 -> 32 (Method 2: SMLAD Assembly + Method 3: Register Cached) */
    int16_t h_q[32];
    for (int j = 0; j < 32; j++) {
        int32_t acc = b0_simd[j] << SHIFT_POT_16;
        acc = smlad_dual(x_pair_01, W0_simd_01[j], acc);
        acc = smlad_dual(x_pair_23, W0_simd_23[j], acc);
        acc >>= SHIFT_POT_16;
        h_q[j] = (acc > 0) ? (int16_t)acc : 0;
    }

    /* Layer 1: 32 -> 1 (Method 4: Dynamic ReLU Zero-Skipping) */
    int32_t out_acc = b1_simd[0] << SHIFT_POT_16;
    for (int k = 0; k < 32; k++) {
        if (h_q[k] != 0) {
            out_acc += (int32_t)h_q[k] * (int32_t)W1_simd[k];
        }
    }
    out_acc >>= SHIFT_POT_16;

    int32_t delta_cC = (int32_t)(((int32_t)out_acc * 100) >> SHIFT_POT_16);
    int32_t final_cC = amb_cC + delta_cC;
    if (final_cC < amb_cC) {
        final_cC = amb_cC;
    }
    if (final_cC > 12500) {
        final_cC = 12500;
    }
    return final_cC;
}

#endif
