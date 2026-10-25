/*
 * mlp_model_new.c -- 6-Mode TinyML Stateless Dual Thermal Predictor for STM32F401RET6
 * Architecture: 4 Inputs -> 32 Neurons (ReLU) -> 2 Outputs (Linear: 3s & 10s)
 * Features: [heatsink_temp_C, ambient_temp_C, power_loss_mW, current_mA]
 * Targets:  [Future +3s ahead, Future +10s ahead] in centi-degC
 *
 * ALL 6 MODES ARE SIMULTANEOUSLY COMPILED INTO THIS FILE.
 */

#include "mlp_model_new.h"
#include <stdint.h>
#include <string.h>

static inline int32_t smlad_dual(uint32_t x_pair, uint32_t w_pair, int32_t acc)
{
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
}

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
 * ============================================================================== */
static const float scaler_mean[4] = {
     63.57460403f,  31.25141525f,  2319.23217773f,  535.38635254f
};
static const float scaler_scale[4] = {
     17.47022820f,  8.12819958f,  933.69464111f,  227.39906311f
};

static const float W0[128] = {
     0.00000000f, -0.63587940f,  0.00000000f,  0.37646383f, -0.73375034f, -0.28659528f, -0.00067777f,  0.26138976f,
     0.80754381f,  0.57578546f, -0.48843199f, -0.18087709f,  0.18931192f, -0.55942255f, -0.54916275f,  0.00004717f,
    -0.65975082f,  0.27801436f, -0.42080414f, -0.31485948f,  0.48370370f, -0.49813747f,  0.03047229f, -0.38547888f,
     0.00000000f, -0.11012955f,  0.01391958f, -0.00000000f, -0.21912520f, -0.34542209f, -0.76773524f, -0.38753310f,
     0.00000000f,  0.30422240f,  0.00000000f, -0.16616403f,  0.33730435f,  0.13237275f,  0.00152045f, -0.12045329f,
    -0.37781626f, -0.26537532f,  0.23941781f,  0.21106645f, -0.09483091f,  0.25732565f,  0.25776502f, -0.00000545f,
     0.30960920f, -0.13164802f,  0.22419521f,  0.14634268f,  0.08183595f,  0.23097998f, -0.01938921f,  0.17922391f,
     0.00000000f,  0.05903166f, -0.00348753f, -0.00000000f,  0.09690070f, -0.04848793f,  0.35987139f,  0.17893969f,
     0.00000000f,  0.18600745f, -0.00000000f, -0.14027636f, -0.50585192f,  0.88800079f,  0.00050195f, -0.72981298f,
    -0.81230229f,  0.25514710f,  0.25871447f,  0.13207412f,  0.45111641f,  0.17277876f, -0.41175967f, -0.00006840f,
     0.65428400f,  0.30215666f,  0.26100436f, -1.57586718f, -0.31029058f, -0.00183023f,  0.38386089f,  0.74429399f,
    -0.00000000f, -0.67494690f, -0.02011253f,  0.00000000f,  0.64968681f,  0.25882098f, -0.24016753f,  0.08362906f,
     0.00000000f, -0.11200746f, -0.00000000f, -0.30404690f,  0.04079409f,  0.70437056f,  0.00189988f, -0.56141585f,
     0.22254600f, -0.28816000f, -0.13316283f, -0.08512739f, -0.36219740f, -0.48638526f, -0.10808866f, -0.00000825f,
     0.24898714f,  0.39513019f, -0.01504597f,  0.25791132f,  0.12606187f, -0.07544747f,  0.48346493f,  0.08682822f,
     0.00000000f,  0.52461505f, -0.00315422f,  0.00000000f, -0.52044737f, -0.07982420f,  0.23845874f,  0.25091642f
};
static const float b0[32] = {
    -0.42675874f, -1.12006426f, -0.15007168f, -0.36815122f, -0.86413246f, -0.52467936f, -0.07111299f,  0.40061796f,
    -0.69734293f,  0.37661305f, -0.61723936f,  0.36437076f,  0.34216139f, -0.53683341f, -0.79596770f, -0.04811355f,
     0.29577360f, -0.51504624f,  0.38335839f, -0.18072610f, -0.39594433f,  0.38624316f, -0.33298787f, -0.35760853f,
    -0.41931215f, -0.56619591f, -0.22640371f, -0.38849157f,  0.12929620f,  0.24025540f, -0.59995472f, -0.36535877f
};
static const float W1_3s[32] = {
     0.00000000f,  1.72105229f, -0.00003908f, -0.05495977f, -0.43115175f, -0.55046254f, -0.00036561f, -0.39329764f,
    -1.85992301f, -0.09523735f, -0.26900291f,  0.01994545f,  0.03982083f,  0.19314162f, -0.32998738f, -0.00040347f,
     0.28706613f,  0.32297266f,  0.08009779f,  0.49944359f, -0.01545958f,  0.04292216f, -0.04546348f,  0.39277917f,
     0.00000000f,  0.35920820f, -0.00983424f,  0.00000000f,  0.14416675f,  0.01431540f,  0.24587615f, -0.33009136f
};
static const float W1_10s[32] = {
     0.00000000f,  3.26869106f, -0.00009958f, -0.17290962f, -1.10876524f, -1.47307158f,  0.00527640f, -0.91682631f,
    -5.34620762f, -0.55094671f, -0.85098284f,  0.07815012f,  0.30918580f,  0.47150698f, -0.84999031f,  0.00054681f,
     0.71907991f,  0.68426645f,  0.27918139f,  1.03426433f, -0.05664663f,  0.29626334f, -0.24310383f,  0.85718483f,
     0.00000000f,  0.65237910f, -0.01512499f,  0.00000000f,  0.48788837f,  0.08034202f,  0.88252705f, -0.32231125f
};
static const float b1_f32[2] = {
    -0.08973459f, -0.10638153f
};

mlp_prediction_t mlp_new_predict_mode0_fp32(int32_t hs_cC, int32_t amb_cC,
                                           int32_t ploss_cmW, int32_t i_cmA)
{
    if (amb_cC <= 1000 || amb_cC >= 5500) {
        amb_cC = 2500;
    }

    float x[4];
    x[0] = (((float)hs_cC     / 100.0f) - scaler_mean[0]) / scaler_scale[0];
    x[1] = (((float)amb_cC    / 100.0f) - scaler_mean[1]) / scaler_scale[1];
    x[2] = (((float)ploss_cmW / 100.0f) - scaler_mean[2]) / scaler_scale[2];
    x[3] = (((float)i_cmA     / 100.0f) - scaler_mean[3]) / scaler_scale[3];

    float h[32];
    for (int j = 0; j < 32; j++) {
        float acc = b0[j];
        for (int i = 0; i < 4; i++) {
            acc += x[i] * W0[i * 32 + j];
        }
        h[j] = (acc > 0.0f) ? acc : 0.0f;
    }

    float out_3s = b1_f32[0];
    float out_10s = b1_f32[1];
    for (int k = 0; k < 32; k++) {
        if (h[k] > 0.0f) {
            out_3s  += h[k] * W1_3s[k];
            out_10s += h[k] * W1_10s[k];
        }
    }

    int32_t delta_3s_cC  = (int32_t)(out_3s * 100.0f + (out_3s >= 0.0f ? 0.5f : -0.5f));
    int32_t delta_10s_cC = (int32_t)(out_10s * 100.0f + (out_10s >= 0.0f ? 0.5f : -0.5f));

    mlp_prediction_t res;
    res.pred_3s_cC  = hs_cC + delta_3s_cC;
    res.pred_10s_cC = hs_cC + delta_10s_cC;
    if (ploss_cmW > 0) {
        if (res.pred_3s_cC < hs_cC)  res.pred_3s_cC = hs_cC;
        if (res.pred_10s_cC < hs_cC) res.pred_10s_cC = hs_cC;
    }
    if (res.pred_3s_cC > 12500)  res.pred_3s_cC = 12500;
    if (res.pred_10s_cC > 12500) res.pred_10s_cC = 12500;
    return res;
}

/* ==============================================================================
 * MODE 1: 32-BIT INTEGER -- NORMAL FIXED-POINT (Division / 10000)
 * ============================================================================== */
#define SCALE_NORMAL_32 10000LL

static const int32_t norm_mean_c[4]  = { 6357, 3125, 231923, 53539 };
static const int32_t norm_scale_c[4] = { 1747, 813, 93369, 22740 };

static const int32_t W0_n32[128] = {
            0,     -6359,         0,      3765,     -7338,     -2866,        -7,      2614,
         8075,      5758,     -4884,     -1809,      1893,     -5594,     -5492,         0,
        -6598,      2780,     -4208,     -3149,      4837,     -4981,       305,     -3855,
            0,     -1101,       139,         0,     -2191,     -3454,     -7677,     -3875,
            0,      3042,         0,     -1662,      3373,      1324,        15,     -1205,
        -3778,     -2654,      2394,      2111,      -948,      2573,      2578,         0,
         3096,     -1316,      2242,      1463,       818,      2310,      -194,      1792,
            0,       590,       -35,         0,       969,      -485,      3599,      1789,
            0,      1860,         0,     -1403,     -5059,      8880,         5,     -7298,
        -8123,      2551,      2587,      1321,      4511,      1728,     -4118,        -1,
         6543,      3022,      2610,    -15759,     -3103,       -18,      3839,      7443,
            0,     -6749,      -201,         0,      6497,      2588,     -2402,       836,
            0,     -1120,         0,     -3040,       408,      7044,        19,     -5614,
         2225,     -2882,     -1332,      -851,     -3622,     -4864,     -1081,         0,
         2490,      3951,      -150,      2579,      1261,      -754,      4835,       868,
            0,      5246,       -32,         0,     -5204,      -798,      2385,      2509
};
static const int32_t b0_n32[32] = {
        -4268,    -11201,     -1501,     -3682,     -8641,     -5247,      -711,      4006,
        -6973,      3766,     -6172,      3644,      3422,     -5368,     -7960,      -481,
         2958,     -5150,      3834,     -1807,     -3959,      3862,     -3330,     -3576,
        -4193,     -5662,     -2264,     -3885,      1293,      2403,     -6000,     -3654
};
static const int32_t W1_3s_n32[32] = {
            0,     17211,         0,      -550,     -4312,     -5505,        -4,     -3933,
       -18599,      -952,     -2690,       199,       398,      1931,     -3300,        -4,
         2871,      3230,       801,      4994,      -155,       429,      -455,      3928,
            0,      3592,       -98,         0,      1442,       143,      2459,     -3301
};
static const int32_t W1_10s_n32[32] = {
            0,     32687,        -1,     -1729,    -11088,    -14731,        53,     -9168,
       -53462,     -5509,     -8510,       782,      3092,      4715,     -8500,         5,
         7191,      6843,      2792,     10343,      -566,      2963,     -2431,      8572,
            0,      6524,      -151,         0,      4879,       803,      8825,     -3223
};
static const int32_t b1_n32[2] = { -897, -1064 };

mlp_prediction_t mlp_new_predict_mode1_int32_norm(int32_t hs_cC, int32_t amb_cC,
                                                int32_t ploss_cmW, int32_t i_cmA)
{
    if (amb_cC <= 1000 || amb_cC >= 5500) {
        amb_cC = 2500;
    }

    int32_t raw_c[4];
    raw_c[0] = hs_cC;
    raw_c[1] = amb_cC;
    raw_c[2] = ploss_cmW;
    raw_c[3] = i_cmA;
    int32_t x_q[4];
    for (int i = 0; i < 4; i++) {
        int64_t diff = (int64_t)raw_c[i] - (int64_t)norm_mean_c[i];
        x_q[i] = (int32_t)((diff * SCALE_NORMAL_32) / (int64_t)norm_scale_c[i]);
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

    int64_t out_acc_3s  = (int64_t)b1_n32[0] * SCALE_NORMAL_32;
    int64_t out_acc_10s = (int64_t)b1_n32[1] * SCALE_NORMAL_32;
    for (int k = 0; k < 32; k++) {
        if (h_q[k] != 0) {
            out_acc_3s  += (int64_t)h_q[k] * (int64_t)W1_3s_n32[k];
            out_acc_10s += (int64_t)h_q[k] * (int64_t)W1_10s_n32[k];
        }
    }
    out_acc_3s  /= SCALE_NORMAL_32;
    out_acc_10s /= SCALE_NORMAL_32;

    int32_t delta_3s_cC  = (int32_t)(((int64_t)out_acc_3s * 100LL) / SCALE_NORMAL_32);
    int32_t delta_10s_cC = (int32_t)(((int64_t)out_acc_10s * 100LL) / SCALE_NORMAL_32);

    mlp_prediction_t res;
    res.pred_3s_cC  = hs_cC + delta_3s_cC;
    res.pred_10s_cC = hs_cC + delta_10s_cC;
    if (ploss_cmW > 0) {
        if (res.pred_3s_cC < hs_cC)  res.pred_3s_cC = hs_cC;
        if (res.pred_10s_cC < hs_cC) res.pred_10s_cC = hs_cC;
    }
    if (res.pred_3s_cC > 12500)  res.pred_3s_cC = 12500;
    if (res.pred_10s_cC > 12500) res.pred_10s_cC = 12500;
    return res;
}

/* ==============================================================================
 * MODE 2: 32-BIT INTEGER -- POWER-OF-2 (Bit-Shift >> 16)
 * ============================================================================== */
#define SHIFT_POT_32 16

static const int32_t pot_mean_c[4]  = { 6357, 3125, 231923, 53539 };
static const int32_t pot_scale_c[4] = { 1747, 813, 93369, 22740 };

static const int32_t W0_p32[128] = {
            0,    -41673,         0,     24672,    -48087,    -18782,       -44,     17130,
        52923,     37735,    -32010,    -11854,     12407,    -36662,    -35990,         3,
       -43237,     18220,    -27578,    -20635,     31700,    -32646,      1997,    -25263,
            0,     -7217,       912,         0,    -14361,    -22638,    -50314,    -25397,
            0,     19938,         0,    -10890,     22106,      8675,       100,     -7894,
       -24761,    -17392,     15690,     13832,     -6215,     16864,     16893,         0,
        20291,     -8628,     14693,      9591,      5363,     15138,     -1271,     11746,
            0,      3869,      -229,         0,      6350,     -3178,     23585,     11727,
            0,     12190,         0,     -9193,    -33152,     58196,        33,    -47829,
       -53235,     16721,     16955,      8656,     29564,     11323,    -26985,        -4,
        42879,     19802,     17105,   -103276,    -20335,      -120,     25157,     48778,
            0,    -44233,     -1318,         0,     42578,     16962,    -15740,      5481,
            0,     -7341,         0,    -19926,      2673,     46162,       125,    -36793,
        14585,    -18885,     -8727,     -5579,    -23737,    -31876,     -7084,        -1,
        16318,     25895,      -986,     16902,      8262,     -4945,     31684,      5690,
            0,     34381,      -207,         0,    -34108,     -5231,     15628,     16444
};
static const int32_t b0_p32[32] = {
       -27968,    -73405,     -9835,    -24127,    -56632,    -34385,     -4660,     26255,
       -45701,     24682,    -40451,     23879,     22424,    -35182,    -52165,     -3153,
        19384,    -33754,     25124,    -11844,    -25949,     25313,    -21823,    -23436,
       -27480,    -37106,    -14838,    -25460,      8474,     15745,    -39319,    -23944
};
static const int32_t W1_3s_p32[32] = {
            0,    112791,        -3,     -3602,    -28256,    -36075,       -24,    -25775,
      -121892,     -6241,    -17629,      1307,      2610,     12658,    -21626,       -26,
        18813,     21166,      5249,     32732,     -1013,      2813,     -2979,     25741,
            0,     23541,      -644,         0,      9448,       938,     16114,    -21633
};
static const int32_t W1_10s_p32[32] = {
            0,    214217,        -7,    -11332,    -72664,    -96539,       346,    -60085,
      -350369,    -36107,    -55770,      5122,     20263,     30901,    -55705,        36,
        47126,     44844,     18296,     67782,     -3712,     19416,    -15932,     56176,
            0,     42754,      -991,         0,     31974,      5265,     57837,    -21123
};
static const int32_t b1_p32[2] = { -5881, -6972 };

mlp_prediction_t mlp_new_predict_mode2_int32_pot(int32_t hs_cC, int32_t amb_cC,
                                               int32_t ploss_cmW, int32_t i_cmA)
{
    if (amb_cC <= 1000 || amb_cC >= 5500) {
        amb_cC = 2500;
    }

    int32_t raw_c[4];
    raw_c[0] = hs_cC;
    raw_c[1] = amb_cC;
    raw_c[2] = ploss_cmW;
    raw_c[3] = i_cmA;
    int32_t x_q[4];
    for (int i = 0; i < 4; i++) {
        int64_t diff = (int64_t)raw_c[i] - (int64_t)pot_mean_c[i];
        x_q[i] = (int32_t)((diff << SHIFT_POT_32) / (int64_t)pot_scale_c[i]);
    }

    int32_t h_q[32];
    for (int j = 0; j < 32; j++) {
        int64_t acc = ((int64_t)b0_p32[j] << SHIFT_POT_32);
        for (int i = 0; i < 4; i++) {
            acc += (int64_t)x_q[i] * (int64_t)W0_p32[i * 32 + j];
        }
        acc >>= SHIFT_POT_32;
        h_q[j] = (acc > 0) ? (int32_t)acc : 0;
    }

    int64_t out_acc_3s  = ((int64_t)b1_p32[0] << SHIFT_POT_32);
    int64_t out_acc_10s = ((int64_t)b1_p32[1] << SHIFT_POT_32);
    for (int k = 0; k < 32; k++) {
        if (h_q[k] != 0) {
            out_acc_3s  += (int64_t)h_q[k] * (int64_t)W1_3s_p32[k];
            out_acc_10s += (int64_t)h_q[k] * (int64_t)W1_10s_p32[k];
        }
    }
    out_acc_3s  >>= SHIFT_POT_32;
    out_acc_10s >>= SHIFT_POT_32;

    int32_t delta_3s_cC  = (int32_t)(((int64_t)out_acc_3s * 100LL) >> SHIFT_POT_32);
    int32_t delta_10s_cC = (int32_t)(((int64_t)out_acc_10s * 100LL) >> SHIFT_POT_32);

    mlp_prediction_t res;
    res.pred_3s_cC  = hs_cC + delta_3s_cC;
    res.pred_10s_cC = hs_cC + delta_10s_cC;
    if (ploss_cmW > 0) {
        if (res.pred_3s_cC < hs_cC)  res.pred_3s_cC = hs_cC;
        if (res.pred_10s_cC < hs_cC) res.pred_10s_cC = hs_cC;
    }
    if (res.pred_3s_cC > 12500)  res.pred_3s_cC = 12500;
    if (res.pred_10s_cC > 12500) res.pred_10s_cC = 12500;
    return res;
}

/* ==============================================================================
 * MODE 3: 16-BIT FLOAT WITH HARDWARE FPU (FP16 Storage -> FP32 Compute)
 * ============================================================================== */
static const uint16_t W0_f16[128] = {
    0x0000U, 0xB916U, 0x0000U, 0x3606U, 0xB9DFU, 0xB496U, 0x918DU, 0x342FU,
    0x3A76U, 0x389BU, 0xB7D1U, 0xB1CAU, 0x320FU, 0xB87AU, 0xB865U, 0x0317U,
    0xB947U, 0x3473U, 0xB6BCU, 0xB50AU, 0x37BDU, 0xB7F8U, 0x27CDU, 0xB62BU,
    0x0000U, 0xAF0CU, 0x2320U, 0x8000U, 0xB303U, 0xB587U, 0xBA24U, 0xB633U,
    0x0000U, 0x34DEU, 0x0000U, 0xB151U, 0x3566U, 0x303CU, 0x163AU, 0xAFB6U,
    0xB60CU, 0xB43FU, 0x33A9U, 0x32C1U, 0xAE12U, 0x341EU, 0x3420U, 0x805BU,
    0x34F4U, 0xB036U, 0x332DU, 0x30AFU, 0x2D3DU, 0x3364U, 0xA4F7U, 0x31BCU,
    0x0000U, 0x2B8EU, 0x9B24U, 0x8000U, 0x2E34U, 0xAA35U, 0x35C2U, 0x31BAU,
    0x0000U, 0x31F4U, 0x8000U, 0xB07DU, 0xB80CU, 0x3B1BU, 0x101DU, 0xB9D7U,
    0xBA80U, 0x3415U, 0x3424U, 0x303AU, 0x3738U, 0x3187U, 0xB697U, 0x847CU,
    0x393CU, 0x34D6U, 0x342DU, 0xBE4EU, 0xB4F7U, 0x977FU, 0x3624U, 0x39F4U,
    0x8000U, 0xB966U, 0xA526U, 0x0000U, 0x3933U, 0x3424U, 0xB3AFU, 0x2D5AU,
    0x0000U, 0xAF2BU, 0x8000U, 0xB4DDU, 0x2939U, 0x39A3U, 0x17C8U, 0xB87EU,
    0x331FU, 0xB49CU, 0xB043U, 0xAD73U, 0xB5CCU, 0xB7C8U, 0xAEEBU, 0x808AU,
    0x33F8U, 0x3652U, 0xA3B4U, 0x3420U, 0x3009U, 0xACD4U, 0x37BCU, 0x2D8FU,
    0x0000U, 0x3832U, 0x9A76U, 0x0000U, 0xB82AU, 0xAD1CU, 0x33A1U, 0x3404U
};
static const uint16_t b0_f16[32] = {
    0xB6D4U, 0xBC7BU, 0xB0CDU, 0xB5E4U, 0xBAEAU, 0xB833U, 0xAC8DU, 0x3669U,
    0xB994U, 0x3607U, 0xB8F0U, 0x35D4U, 0x3579U, 0xB84BU, 0xBA5EU, 0xAA29U,
    0x34BBU, 0xB81FU, 0x3622U, 0xB1C9U, 0xB656U, 0x362EU, 0xB554U, 0xB5B9U,
    0xB6B6U, 0xB888U, 0xB33FU, 0xB637U, 0x3023U, 0x33B0U, 0xB8CDU, 0xB5D9U
};
static const uint16_t W1_3s_f16[32] = {
    0x0000U, 0x3EE2U, 0x8290U, 0xAB09U, 0xB6E6U, 0xB867U, 0x8DFDU, 0xB64BU,
    0xBF71U, 0xAE18U, 0xB44EU, 0x251BU, 0x2919U, 0x322EU, 0xB548U, 0x8E9CU,
    0x3498U, 0x352BU, 0x2D20U, 0x37FEU, 0xA3EAU, 0x297EU, 0xA9D2U, 0x3649U,
    0x0000U, 0x35BFU, 0xA109U, 0x0000U, 0x309DU, 0x2354U, 0x33DEU, 0xB548U
};
static const uint16_t W1_10s_f16[32] = {
    0x0000U, 0x428AU, 0x8687U, 0xB188U, 0xBC6FU, 0xBDE4U, 0x1D67U, 0xBB56U,
    0xC559U, 0xB868U, 0xBACFU, 0x2D00U, 0x34F2U, 0x378BU, 0xBACDU, 0x107BU,
    0x39C1U, 0x3979U, 0x3478U, 0x3C23U, 0xAB40U, 0x34BDU, 0xB3C8U, 0x3ADCU,
    0x0000U, 0x3938U, 0xA3BEU, 0x0000U, 0x37CEU, 0x2D24U, 0x3B0FU, 0xB528U
};
static const uint16_t b1_f16[2] = {
    0xADBEU, 0xAECFU
};

mlp_prediction_t mlp_new_predict_mode3_fp16(int32_t hs_cC, int32_t amb_cC,
                                          int32_t ploss_cmW, int32_t i_cmA)
{
    if (amb_cC <= 1000 || amb_cC >= 5500) {
        amb_cC = 2500;
    }

    float x[4];
    x[0] = (((float)hs_cC     / 100.0f) - scaler_mean[0]) / scaler_scale[0];
    x[1] = (((float)amb_cC    / 100.0f) - scaler_mean[1]) / scaler_scale[1];
    x[2] = (((float)ploss_cmW / 100.0f) - scaler_mean[2]) / scaler_scale[2];
    x[3] = (((float)i_cmA     / 100.0f) - scaler_mean[3]) / scaler_scale[3];

    float h[32];
    for (int j = 0; j < 32; j++) {
        float acc = fp16_to_fp32(b0_f16[j]);
        for (int i = 0; i < 4; i++) {
            acc += x[i] * fp16_to_fp32(W0_f16[i * 32 + j]);
        }
        h[j] = (acc > 0.0f) ? acc : 0.0f;
    }

    float out_3s  = fp16_to_fp32(b1_f16[0]);
    float out_10s = fp16_to_fp32(b1_f16[1]);
    for (int k = 0; k < 32; k++) {
        if (h[k] > 0.0f) {
            out_3s  += h[k] * fp16_to_fp32(W1_3s_f16[k]);
            out_10s += h[k] * fp16_to_fp32(W1_10s_f16[k]);
        }
    }

    int32_t delta_3s_cC  = (int32_t)(out_3s * 100.0f + (out_3s >= 0.0f ? 0.5f : -0.5f));
    int32_t delta_10s_cC = (int32_t)(out_10s * 100.0f + (out_10s >= 0.0f ? 0.5f : -0.5f));

    mlp_prediction_t res;
    res.pred_3s_cC  = hs_cC + delta_3s_cC;
    res.pred_10s_cC = hs_cC + delta_10s_cC;
    if (ploss_cmW > 0) {
        if (res.pred_3s_cC < hs_cC)  res.pred_3s_cC = hs_cC;
        if (res.pred_10s_cC < hs_cC) res.pred_10s_cC = hs_cC;
    }
    if (res.pred_3s_cC > 12500)  res.pred_3s_cC = 12500;
    if (res.pred_10s_cC > 12500) res.pred_10s_cC = 12500;
    return res;
}

/* ==============================================================================
 * MODE 4: 16-BIT INTEGER -- NORMAL FIXED-POINT (Division / 2048)
 * ============================================================================== */
#define SCALE_NORMAL_16 2048

static const int16_t W0_n16[128] = {
            0,     -1302,         0,       771,     -1503,      -587,        -1,       535,
         1654,      1179,     -1000,      -370,       388,     -1146,     -1125,         0,
        -1351,       569,      -862,      -645,       991,     -1020,        62,      -789,
            0,      -226,        29,         0,      -449,      -707,     -1572,      -794,
            0,       623,         0,      -340,       691,       271,         3,      -247,
         -774,      -543,       490,       432,      -194,       527,       528,         0,
          634,      -270,       459,       300,       168,       473,       -40,       367,
            0,       121,        -7,         0,       198,       -99,       737,       366,
            0,       381,         0,      -287,     -1036,      1819,         1,     -1495,
        -1664,       523,       530,       270,       924,       354,      -843,         0,
         1340,       619,       535,     -3227,      -635,        -4,       786,      1524,
            0,     -1382,       -41,         0,      1331,       530,      -492,       171,
            0,      -229,         0,      -623,        84,      1443,         4,     -1150,
          456,      -590,      -273,      -174,      -742,      -996,      -221,         0,
          510,       809,       -31,       528,       258,      -155,       990,       178,
            0,      1074,        -6,         0,     -1066,      -163,       488,       514
};
static const int32_t b0_n16[32] = {
         -874,     -2294,      -307,      -754,     -1770,     -1075,      -146,       820,
        -1428,       771,     -1264,       746,       701,     -1099,     -1630,       -99,
          606,     -1055,       785,      -370,      -811,       791,      -682,      -732,
         -859,     -1160,      -464,      -796,       265,       492,     -1229,      -748
};
static const int16_t W1_3s_n16[32] = {
            0,      3525,         0,      -113,      -883,     -1127,        -1,      -805,
        -3809,      -195,      -551,        41,        82,       396,      -676,        -1,
          588,       661,       164,      1023,       -32,        88,       -93,       804,
            0,       736,       -20,         0,       295,        29,       504,      -676
};
static const int16_t W1_10s_n16[32] = {
            0,      6694,         0,      -354,     -2271,     -3017,        11,     -1878,
       -10949,     -1128,     -1743,       160,       633,       966,     -1741,         1,
         1473,      1401,       572,      2118,      -116,       607,      -498,      1756,
            0,      1336,       -31,         0,       999,       165,      1807,      -660
};
static const int32_t b1_n16[2] = { -184, -218 };

mlp_prediction_t mlp_new_predict_mode4_int16_norm(int32_t hs_cC, int32_t amb_cC,
                                                int32_t ploss_cmW, int32_t i_cmA)
{
    if (amb_cC <= 1000 || amb_cC >= 5500) {
        amb_cC = 2500;
    }

    int32_t raw_c[4];
    raw_c[0] = hs_cC;
    raw_c[1] = amb_cC;
    raw_c[2] = ploss_cmW;
    raw_c[3] = i_cmA;
    int16_t x_q[4];
    for (int i = 0; i < 4; i++) {
        int32_t diff = raw_c[i] - norm_mean_c[i];
        x_q[i] = (int16_t)((diff * SCALE_NORMAL_16) / norm_scale_c[i]);
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

    int32_t out_acc_3s  = b1_n16[0] * SCALE_NORMAL_16;
    int32_t out_acc_10s = b1_n16[1] * SCALE_NORMAL_16;
    for (int k = 0; k < 32; k++) {
        if (h_q[k] != 0) {
            out_acc_3s  += (int32_t)h_q[k] * (int32_t)W1_3s_n16[k];
            out_acc_10s += (int32_t)h_q[k] * (int32_t)W1_10s_n16[k];
        }
    }
    out_acc_3s  /= SCALE_NORMAL_16;
    out_acc_10s /= SCALE_NORMAL_16;

    int32_t delta_3s_cC  = (int32_t)(((int64_t)out_acc_3s * 100LL) / SCALE_NORMAL_16);
    int32_t delta_10s_cC = (int32_t)(((int64_t)out_acc_10s * 100LL) / SCALE_NORMAL_16);

    mlp_prediction_t res;
    res.pred_3s_cC  = hs_cC + delta_3s_cC;
    res.pred_10s_cC = hs_cC + delta_10s_cC;
    if (ploss_cmW > 0) {
        if (res.pred_3s_cC < hs_cC)  res.pred_3s_cC = hs_cC;
        if (res.pred_10s_cC < hs_cC) res.pred_10s_cC = hs_cC;
    }
    if (res.pred_3s_cC > 12500)  res.pred_3s_cC = 12500;
    if (res.pred_10s_cC > 12500) res.pred_10s_cC = 12500;
    return res;
}

/* ==============================================================================
 * MODE 5: 16-BIT INTEGER -- POT (>> 11) + ARM SMLAD SIMD
 * ============================================================================== */
#define SHIFT_POT_16 11

static const uint32_t W0_simd_01[32] = {
    0x00000000U, 0x026FFAEAU, 0x00000000U, 0xFEAC0303U,
    0x02B3FA21U, 0x010FFDB5U, 0x0003FFFFU, 0xFF090217U,
    0xFCFA0676U, 0xFDE1049BU, 0x01EAFC18U, 0x01B0FE8EU,
    0xFF3E0184U, 0x020FFB86U, 0x0210FB9BU, 0x00000000U,
    0x027AFAB9U, 0xFEF20239U, 0x01CBFCA2U, 0x012CFD7BU,
    0x00A803DFU, 0x01D9FC04U, 0xFFD8003EU, 0x016FFCEBU,
    0x00000000U, 0x0079FF1EU, 0xFFF9001DU, 0x00000000U,
    0x00C6FE3FU, 0xFF9DFD3DU, 0x02E1F9DCU, 0x016EFCE6U
};
static const uint32_t W0_simd_23[32] = {
    0x00000000U, 0xFF1B017DU, 0x00000000U, 0xFD91FEE1U,
    0x0054FBF4U, 0x05A3071BU, 0x00040001U, 0xFB82FA29U,
    0x01C8F980U, 0xFDB2020BU, 0xFEEF0212U, 0xFF52010EU,
    0xFD1A039CU, 0xFC1C0162U, 0xFF23FCB5U, 0x00000000U,
    0x01FE053CU, 0x0329026BU, 0xFFE10217U, 0x0210F365U,
    0x0102FD85U, 0xFF65FFFCU, 0x03DE0312U, 0x00B205F4U,
    0x00000000U, 0x0432FA9AU, 0xFFFAFFD7U, 0x00000000U,
    0xFBD60533U, 0xFF5D0212U, 0x01E8FE14U, 0x020200ABU
};
static const int32_t b0_simd[32] = {
         -874,     -2294,      -307,      -754,     -1770,     -1075,      -146,       820,
        -1428,       771,     -1264,       746,       701,     -1099,     -1630,       -99,
          606,     -1055,       785,      -370,      -811,       791,      -682,      -732,
         -859,     -1160,      -464,      -796,       265,       492,     -1229,      -748
};

static const uint32_t W1_simd_3s[16] = {
    0x0DC50000U, 0xFF8F0000U, 0xFB99FC8DU, 0xFCDBFFFFU,
    0xFF3DF11FU, 0x0029FDD9U, 0x018C0052U, 0xFFFFFD5CU,
    0x0295024CU, 0x03FF00A4U, 0x0058FFE0U, 0x0324FFA3U,
    0x02E00000U, 0x0000FFECU, 0x001D0127U, 0xFD5C01F8U
};
static const uint32_t W1_simd_10s[16] = {
    0x1A260000U, 0xFE9E0000U, 0xF437F721U, 0xF8AA000BU,
    0xFB98D53BU, 0x00A0F931U, 0x03C60279U, 0x0001F933U,
    0x057905C1U, 0x0846023CU, 0x025FFF8CU, 0x06DCFE0EU,
    0x05380000U, 0x0000FFE1U, 0x00A503E7U, 0xFD6C070FU
};
static const int32_t b1_simd[2] = { -184, -218 };

mlp_prediction_t mlp_new_predict_mode5_int16_simd(int32_t hs_cC, int32_t amb_cC,
                                                int32_t ploss_cmW, int32_t i_cmA)
{
    if (amb_cC <= 1000 || amb_cC >= 5500) {
        amb_cC = 2500;
    }

    int32_t raw_c[4];
    raw_c[0] = hs_cC;
    raw_c[1] = amb_cC;
    raw_c[2] = ploss_cmW;
    raw_c[3] = i_cmA;
    int16_t x_q[4];
    for (int i = 0; i < 4; i++) {
        int32_t diff = raw_c[i] - pot_mean_c[i];
        x_q[i] = (int16_t)((diff << SHIFT_POT_16) / pot_scale_c[i]);
    }

    /* Pack 4 inputs into two 32-bit SIMD registers */
    uint32_t x_pair_01 = ((uint32_t)(uint16_t)x_q[1] << 16) | ((uint32_t)(uint16_t)x_q[0]);
    uint32_t x_pair_23 = ((uint32_t)(uint16_t)x_q[3] << 16) | ((uint32_t)(uint16_t)x_q[2]);

    /* Layer 0: 4 -> 32 (SMLAD dual MAC across both input pairs) */
    int16_t h_q[32];
    for (int j = 0; j < 32; j++) {
        int32_t acc = b0_simd[j] << SHIFT_POT_16;
        acc = smlad_dual(x_pair_01, W0_simd_01[j], acc);
        acc = smlad_dual(x_pair_23, W0_simd_23[j], acc);
        acc >>= SHIFT_POT_16;
        h_q[j] = (acc > 0) ? (int16_t)acc : 0;
    }

    /* Layer 1: 32 -> 2 (Dual SMLAD across 16 pairs) */
    int32_t out_acc_3s  = b1_simd[0] << SHIFT_POT_16;
    int32_t out_acc_10s = b1_simd[1] << SHIFT_POT_16;
    for (int p = 0; p < 16; p++) {
        uint32_t h_pair = ((uint32_t)(uint16_t)h_q[p*2+1] << 16) | ((uint32_t)(uint16_t)h_q[p*2]);
        out_acc_3s  = smlad_dual(h_pair, W1_simd_3s[p],  out_acc_3s);
        out_acc_10s = smlad_dual(h_pair, W1_simd_10s[p], out_acc_10s);
    }
    out_acc_3s  >>= SHIFT_POT_16;
    out_acc_10s >>= SHIFT_POT_16;

    int32_t delta_3s_cC  = (int32_t)(((int64_t)out_acc_3s * 100LL) >> SHIFT_POT_16);
    int32_t delta_10s_cC = (int32_t)(((int64_t)out_acc_10s * 100LL) >> SHIFT_POT_16);

    mlp_prediction_t res;
    res.pred_3s_cC  = hs_cC + delta_3s_cC;
    res.pred_10s_cC = hs_cC + delta_10s_cC;
    if (ploss_cmW > 0) {
        if (res.pred_3s_cC < hs_cC)  res.pred_3s_cC = hs_cC;
        if (res.pred_10s_cC < hs_cC) res.pred_10s_cC = hs_cC;
    }
    if (res.pred_3s_cC > 12500)  res.pred_3s_cC = 12500;
    if (res.pred_10s_cC > 12500) res.pred_10s_cC = 12500;
    return res;
}

/* ==============================================================================
 * MASTER ENTRYPOINT DISPATCHERS
 * ============================================================================== */
mlp_prediction_t mlp_new_predict_dual_cC(int32_t hs_cC, int32_t amb_cC,
                                         int32_t ploss_cmW, int32_t i_cmA)
{
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
}

int32_t mlp_new_predict_temp_cC(int32_t hs_cC, int32_t amb_cC,
                                 int32_t ploss_cmW, int32_t i_cmA)
{
    mlp_prediction_t res = mlp_new_predict_dual_cC(hs_cC, amb_cC, ploss_cmW, i_cmA);
    return res.pred_3s_cC;
}
