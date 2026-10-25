/*
 * mlp_model_new.c -- 6-Mode TinyML Stateless Thermal Predictor for STM32F401RET6
 * Architecture: 4 Inputs -> 32 Neurons (ReLU) -> 1 Output (Linear)
 * Features: [heatsink_temp_C, ambient_temp_C, power_loss_mW, current_mA]
 * Target: Future Heatsink Temperature +1s ahead (centi-degC)
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
     63.71195984f,  31.25789261f,  2330.16552734f,  537.22021484f
};
static const float scaler_scale[4] = {
     17.56591797f,  8.12421036f,  925.26922607f,  225.02209473f
};

static const float W0[128] = {
    -1.03183162f,  1.03696680f,  0.59214526f,  1.50096583f, -0.50820553f, -0.37810901f,  0.60702527f, -0.00000000f,
    -0.00000000f,  0.69982725f,  0.09300920f,  0.12366483f,  1.76909542f, -0.62284994f, -0.15199254f, -0.61936486f,
     0.72384888f, -0.00000000f,  0.48341811f, -1.34733355f, -0.50794291f, -0.31955242f,  1.27653944f,  0.68886840f,
    -1.49904132f,  1.91299331f, -0.15465443f,  0.45667449f,  0.39637622f,  0.63197082f,  0.56873429f, -0.32006168f,
     0.48546290f,  0.84730870f,  0.59762102f,  1.03411007f,  0.16118665f, -1.43137753f,  0.60475266f,  0.00000000f,
    -0.00000000f,  0.70965016f, -1.16535866f,  0.81330514f, -0.81539011f,  0.35855991f, -0.91729534f,  0.31377214f,
     0.44122133f,  0.00000000f,  0.67241579f,  0.09996632f,  1.80650854f,  0.11306284f, -0.11674577f,  0.43267509f,
    -1.63623381f, -0.88485748f, -1.22928786f, -1.21878481f, -1.57169008f,  0.22511309f,  0.49107587f, -0.63133174f,
    -1.52961767f,  0.73605633f, -0.43472981f,  0.83858776f, -0.59486085f,  0.86140841f, -0.75393265f,  0.00000000f,
     0.00000000f, -0.04086806f,  0.19588447f,  0.11646786f,  0.94797343f, -0.62217081f, -1.76673281f, -0.16579255f,
    -1.94586360f, -0.00000000f,  0.64898276f, -0.37633976f,  0.41154501f, -0.11972667f,  1.16020966f,  1.54895902f,
    -1.37063682f,  0.27903292f, -2.01869893f,  0.23638456f,  1.09706151f, -1.81637502f,  0.34158733f, -1.48338616f,
    -2.69787312f,  0.27286986f, -0.77292240f, -3.16006017f, -1.38502169f,  2.09286618f, -0.84972656f, -0.00000000f,
    -0.00000000f, -0.62632972f, -2.62471414f,  0.70603919f,  0.73684442f, -1.62615693f, -3.39742684f, -1.35976529f,
     1.15827584f, -0.00000000f,  1.46384978f,  0.90267545f,  1.11564636f, -1.16307676f,  1.39978874f,  0.97878182f,
    -0.63354295f,  1.65752220f,  1.32137930f, -2.34531641f,  0.87427306f,  0.79775578f, -0.51164740f,  1.31500673f
};
static const float b0[32] = {
    -3.78425169f,  2.24434876f,  2.49310398f,  0.39419153f, -1.62107134f,  1.69142842f,  2.57801890f, -0.19743522f,
    -0.20244841f,  2.67722034f,  1.82280076f, -0.04541713f,  2.66029453f, -1.28892291f,  1.89466000f, -1.37940443f,
     1.66986525f, -0.45269486f,  0.85791326f,  0.37448412f,  0.22493976f, -1.16593945f,  1.74145198f,  1.58425462f,
    -3.78289557f,  1.83126521f,  1.68330073f,  1.76149058f,  2.73883653f,  1.80323088f, -0.75157595f,  1.58026516f
};
static const float W1[32] = {
    -3.34575796f,  1.40338707f,  1.58770001f,  1.45492160f, -1.74031401f,  1.91586816f,  1.75675666f,  0.00000000f,
     0.00000000f,  1.56573486f,  1.55957639f, -0.67172033f,  1.43659520f, -1.48477781f,  2.09057379f, -1.50518894f,
     1.66756463f, -0.00000000f,  1.04799926f, -1.09505963f, -1.61653829f, -1.39277363f,  1.65534937f,  1.72900236f,
    -3.01520085f,  1.64076316f,  1.53003478f,  1.45889854f,  1.83782279f,  1.36229110f, -1.31017232f,  1.59875977f
};
static const float b1[1] = {
    1.23477602f
};

int32_t mlp_new_predict_mode0_fp32(int32_t hs_cC, int32_t amb_cC,
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

    float out = b1[0];
    for (int k = 0; k < 32; k++) {
        if (h[k] > 0.0f) {
            out += h[k] * W1[k];
        }
    }

    return (int32_t)(out * 100.0f + (out >= 0.0f ? 0.5f : -0.5f));
}

/* ==============================================================================
 * MODE 1: 32-BIT INTEGER -- NORMAL FIXED-POINT (Division / 10000)
 * ============================================================================== */
#define SCALE_NORMAL_32 10000LL

static const int32_t norm_mean_c[4]  = { 6371, 3126, 233017, 53722 };
static const int32_t norm_scale_c[4] = { 1757, 812, 92527, 22502 };

static const int32_t W0_n32[128] = {
       -10318,     10370,      5921,     15010,     -5082,     -3781,      6070,         0,
            0,      6998,       930,      1237,     17691,     -6228,     -1520,     -6194,
         7238,         0,      4834,    -13473,     -5079,     -3196,     12765,      6889,
       -14990,     19130,     -1547,      4567,      3964,      6320,      5687,     -3201,
         4855,      8473,      5976,     10341,      1612,    -14314,      6048,         0,
            0,      7097,    -11654,      8133,     -8154,      3586,     -9173,      3138,
         4412,         0,      6724,      1000,     18065,      1131,     -1167,      4327,
       -16362,     -8849,    -12293,    -12188,    -15717,      2251,      4911,     -6313,
       -15296,      7361,     -4347,      8386,     -5949,      8614,     -7539,         0,
            0,      -409,      1959,      1165,      9480,     -6222,    -17667,     -1658,
       -19459,         0,      6490,     -3763,      4115,     -1197,     11602,     15490,
       -13706,      2790,    -20187,      2364,     10971,    -18164,      3416,    -14834,
       -26979,      2729,     -7729,    -31601,    -13850,     20929,     -8497,         0,
            0,     -6263,    -26247,      7060,      7368,    -16262,    -33974,    -13598,
        11583,         0,     14638,      9027,     11156,    -11631,     13998,      9788,
        -6335,     16575,     13214,    -23453,      8743,      7978,     -5116,     13150
};
static const int32_t b0_n32[32] = {
       -37843,     22443,     24931,      3942,    -16211,     16914,     25780,     -1974,
        -2024,     26772,     18228,      -454,     26603,    -12889,     18947,    -13794,
        16699,     -4527,      8579,      3745,      2249,    -11659,     17415,     15843,
       -37829,     18313,     16833,     17615,     27388,     18032,     -7516,     15803
};
static const int32_t W1_n32[32] = {
       -33458,     14034,     15877,     14549,    -17403,     19159,     17568,         0,
            0,     15657,     15596,     -6717,     14366,    -14848,     20906,    -15052,
        16676,         0,     10480,    -10951,    -16165,    -13928,     16553,     17290,
       -30152,     16408,     15300,     14589,     18378,     13623,    -13102,     15988
};
static const int32_t b1_n32[1] = { 12348 };

int32_t mlp_new_predict_mode1_int32_norm(int32_t hs_cC, int32_t amb_cC,
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

    int64_t out_acc = (int64_t)b1_n32[0] * SCALE_NORMAL_32;
    for (int k = 0; k < 32; k++) {
        if (h_q[k] != 0) {
            out_acc += (int64_t)h_q[k] * (int64_t)W1_n32[k];
        }
    }
    out_acc /= SCALE_NORMAL_32;

    return (int32_t)(((int64_t)out_acc * 100LL) / SCALE_NORMAL_32);
}

/* ==============================================================================
 * MODE 2: 32-BIT INTEGER -- POWER-OF-2 (Bit-Shift >> 16)
 * ============================================================================== */
#define SHIFT_POT_32 16

static const int32_t pot_mean_c[4]  = { 6371, 3126, 233017, 53722 };
static const int32_t pot_scale_c[4] = { 1757, 812, 92527, 22502 };

static const int32_t W0_p32[128] = {
       -67622,     67959,     38807,     98367,    -33306,    -24780,     39782,         0,
            0,     45864,      6095,      8104,    115939,    -40819,     -9961,    -40591,
        47438,         0,     31681,    -88299,    -33289,    -20942,     83659,     45146,
       -98241,    125370,    -10135,     29929,     25977,     41417,     37273,    -20976,
        31815,     55529,     39166,     67771,     10564,    -93807,     39633,         0,
            0,     46508,    -76373,     53301,    -53437,     23499,    -60116,     20563,
        28916,         0,     44067,      6551,    118391,      7410,     -7651,     28356,
      -107232,    -57990,    -80563,    -79874,   -103002,     14753,     32183,    -41375,
      -100245,     48238,    -28490,     54958,    -38985,     56453,    -49410,         0,
            0,     -2678,     12837,      7633,     62126,    -40775,   -115785,    -10865,
      -127524,         0,     42532,    -24664,     26971,     -7846,     76036,    101513,
       -89826,     18287,   -132297,     15492,     71897,   -119038,     22386,    -97215,
      -176808,     17883,    -50654,   -207098,    -90769,    137158,    -55688,         0,
            0,    -41047,   -172013,     46271,     48290,   -106572,   -222654,    -89114,
        75909,         0,     95935,     59158,     73115,    -76223,     91737,     64145,
       -41520,    108627,     86598,   -153703,     57296,     52282,    -33531,     86180
};
static const int32_t b0_p32[32] = {
      -248005,    147086,    163388,     25834,   -106239,    110849,    168953,    -12939,
       -13268,    175454,    119459,     -2976,    174345,    -84471,    124168,    -90401,
       109436,    -29668,     56224,     24542,     14742,    -76411,    114128,    103826,
      -247916,    120014,    110317,    115441,    179492,    118177,    -49255,    103564
};
static const int32_t W1_p32[32] = {
      -219268,     91972,    104052,     95350,   -114053,    125558,    115131,         0,
            0,    102612,    102208,    -44022,     94149,    -97306,    137008,    -98644,
       109286,         0,     68682,    -71766,   -105941,    -91277,    108485,    113312,
      -197604,    107529,    100272,     95610,    120444,     89279,    -85863,    104776
};
static const int32_t b1_p32[1] = { 80922 };

int32_t mlp_new_predict_mode2_int32_pot(int32_t hs_cC, int32_t amb_cC,
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

    int64_t out_acc = ((int64_t)b1_p32[0] << SHIFT_POT_32);
    for (int k = 0; k < 32; k++) {
        if (h_q[k] != 0) {
            out_acc += (int64_t)h_q[k] * (int64_t)W1_p32[k];
        }
    }
    out_acc >>= SHIFT_POT_32;

    return (int32_t)(((int64_t)out_acc * 100LL) >> SHIFT_POT_32);
}

/* ==============================================================================
 * MODE 3: 16-BIT FLOAT WITH HARDWARE FPU (FP16 Storage -> FP32 Compute)
 * ============================================================================== */
static const uint16_t W0_f16[128] = {
    0xBC21U, 0x3C26U, 0x38BDU, 0x3E01U, 0xB811U, 0xB60DU, 0x38DBU, 0x8000U,
    0x8000U, 0x3999U, 0x2DF4U, 0x2FEAU, 0x3F14U, 0xB8FCU, 0xB0DDU, 0xB8F4U,
    0x39CAU, 0x8000U, 0x37BCU, 0xBD64U, 0xB810U, 0xB51DU, 0x3D1BU, 0x3983U,
    0xBDFFU, 0x3FA7U, 0xB0F3U, 0x374FU, 0x3658U, 0x390EU, 0x388DU, 0xB51FU,
    0x37C4U, 0x3AC7U, 0x38C8U, 0x3C23U, 0x3128U, 0xBDBAU, 0x38D7U, 0x0000U,
    0x8000U, 0x39ADU, 0xBCA9U, 0x3A82U, 0xBA86U, 0x35BDU, 0xBB57U, 0x3505U,
    0x370FU, 0x0000U, 0x3961U, 0x2E66U, 0x3F3AU, 0x2F3CU, 0xAF79U, 0x36ECU,
    0xBE8CU, 0xBB14U, 0xBCEBU, 0xBCE0U, 0xBE49U, 0x3334U, 0x37DBU, 0xB90DU,
    0xBE1EU, 0x39E3U, 0xB6F5U, 0x3AB5U, 0xB8C2U, 0x3AE4U, 0xBA08U, 0x0000U,
    0x0000U, 0xA93BU, 0x3245U, 0x2F74U, 0x3B95U, 0xB8FAU, 0xBF11U, 0xB14EU,
    0xBFC9U, 0x8000U, 0x3931U, 0xB605U, 0x3696U, 0xAFAAU, 0x3CA4U, 0x3E32U,
    0xBD7CU, 0x3477U, 0xC00AU, 0x3390U, 0x3C63U, 0xBF44U, 0x3577U, 0xBDEFU,
    0xC165U, 0x345EU, 0xBA2FU, 0xC252U, 0xBD8AU, 0x4030U, 0xBACCU, 0x8000U,
    0x8000U, 0xB903U, 0xC140U, 0x39A6U, 0x39E5U, 0xBE81U, 0xC2CBU, 0xBD70U,
    0x3CA2U, 0x8000U, 0x3DDBU, 0x3B39U, 0x3C76U, 0xBCA7U, 0x3D99U, 0x3BD5U,
    0xB911U, 0x3EA1U, 0x3D49U, 0xC0B1U, 0x3AFFU, 0x3A62U, 0xB818U, 0x3D43U
};
static const uint16_t b0_f16[32] = {
    0xC392U, 0x407DU, 0x40FCU, 0x364FU, 0xBE7CU, 0x3EC4U, 0x4128U, 0xB251U,
    0xB27AU, 0x415BU, 0x3F4BU, 0xA9D0U, 0x4152U, 0xBD28U, 0x3F94U, 0xBD85U,
    0x3EAEU, 0xB73EU, 0x3ADDU, 0x35FEU, 0x3333U, 0xBCAAU, 0x3EF7U, 0x3E56U,
    0xC391U, 0x3F53U, 0x3EBCU, 0x3F0CU, 0x417AU, 0x3F37U, 0xBA03U, 0x3E52U
};
static const uint16_t W1_f16[32] = {
    0xC2B1U, 0x3D9DU, 0x3E5AU, 0x3DD2U, 0xBEF6U, 0x3FAAU, 0x3F07U, 0x0000U,
    0x0000U, 0x3E43U, 0x3E3DU, 0xB960U, 0x3DBFU, 0xBDF0U, 0x402EU, 0xBE05U,
    0x3EACU, 0x8000U, 0x3C31U, 0xBC61U, 0xBE77U, 0xBD92U, 0x3E9FU, 0x3EEAU,
    0xC208U, 0x3E90U, 0x3E1FU, 0x3DD6U, 0x3F5AU, 0x3D73U, 0xBD3EU, 0x3E65U
};
static const uint16_t b1_f16[1] = {
    0x3CF0U
};

int32_t mlp_new_predict_mode3_fp16(int32_t hs_cC, int32_t amb_cC,
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

    float out = fp16_to_fp32(b1_f16[0]);
    for (int k = 0; k < 32; k++) {
        if (h[k] > 0.0f) {
            out += h[k] * fp16_to_fp32(W1_f16[k]);
        }
    }

    return (int32_t)(out * 100.0f + (out >= 0.0f ? 0.5f : -0.5f));
}

/* ==============================================================================
 * MODE 4: 16-BIT INTEGER -- NORMAL FIXED-POINT (Division / 2048)
 * ============================================================================== */
#define SCALE_NORMAL_16 2048

static const int16_t W0_n16[128] = {
        -2113,      2124,      1213,      3074,     -1041,      -774,      1243,         0,
            0,      1433,       190,       253,      3623,     -1276,      -311,     -1268,
         1482,         0,       990,     -2759,     -1040,      -654,      2614,      1411,
        -3070,      3918,      -317,       935,       812,      1294,      1165,      -655,
          994,      1735,      1224,      2118,       330,     -2931,      1239,         0,
            0,      1453,     -2387,      1666,     -1670,       734,     -1879,       643,
          904,         0,      1377,       205,      3700,       232,      -239,       886,
        -3351,     -1812,     -2518,     -2496,     -3219,       461,      1006,     -1293,
        -3133,      1507,      -890,      1717,     -1218,      1764,     -1544,         0,
            0,       -84,       401,       239,      1941,     -1274,     -3618,      -340,
        -3985,         0,      1329,      -771,       843,      -245,      2376,      3172,
        -2807,       571,     -4134,       484,      2247,     -3720,       700,     -3038,
        -5525,       559,     -1583,     -6472,     -2837,      4286,     -1740,         0,
            0,     -1283,     -5375,      1446,      1509,     -3330,     -6958,     -2785,
         2372,         0,      2998,      1849,      2285,     -2382,      2867,      2005,
        -1297,      3395,      2706,     -4803,      1791,      1634,     -1048,      2693
};
static const int32_t b0_n16[32] = {
        -7750,      4596,      5106,       807,     -3320,      3464,      5280,      -404,
         -415,      5483,      3733,       -93,      5448,     -2640,      3880,     -2825,
         3420,      -927,      1757,       767,       461,     -2388,      3566,      3245,
        -7747,      3750,      3447,      3608,      5609,      3693,     -1539,      3236
};
static const int16_t W1_n16[32] = {
        -6852,      2874,      3252,      2980,     -3564,      3924,      3598,         0,
            0,      3207,      3194,     -1376,      2942,     -3041,      4281,     -3083,
         3415,         0,      2146,     -2243,     -3311,     -2852,      3390,      3541,
        -6175,      3360,      3134,      2988,      3764,      2790,     -2683,      3274
};
static const int32_t b1_n16[1] = { 2529 };

int32_t mlp_new_predict_mode4_int16_norm(int32_t hs_cC, int32_t amb_cC,
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

    int32_t out_acc = b1_n16[0] * SCALE_NORMAL_16;
    for (int k = 0; k < 32; k++) {
        if (h_q[k] != 0) {
            out_acc += (int32_t)h_q[k] * (int32_t)W1_n16[k];
        }
    }
    out_acc /= SCALE_NORMAL_16;

    return (int32_t)(((int64_t)out_acc * 100LL) / SCALE_NORMAL_16);
}

/* ==============================================================================
 * MODE 5: 16-BIT INTEGER -- POT (>> 11) + ARM SMLAD SIMD
 * ============================================================================== */
#define SHIFT_POT_16 11

/* Packed 16-bit weight pairs for SMLAD: input pair [0,1] */
static const uint32_t W0_simd_01[32] = {
    0x03E2F7BFU, 0x06C7084CU, 0x04C804BDU, 0x08460C02U,
    0x014AFBEFU, 0xF48DFCFAU, 0x04D704DBU, 0x00000000U,
    0x00000000U, 0x05AD0599U, 0xF6AD00BEU, 0x068200FDU,
    0xF97A0E27U, 0x02DEFB04U, 0xF8A9FEC9U, 0x0283FB0CU,
    0x038805CAU, 0x00000000U, 0x056103DEU, 0x00CDF539U,
    0x0E74FBF0U, 0x00E8FD72U, 0xFF110A36U, 0x03760583U,
    0xF2E9F402U, 0xF8EC0F4EU, 0xF62AFEC3U, 0xF64003A7U,
    0xF36D032CU, 0x01CD050EU, 0x03EE048DU, 0xFAF3FD71U
};
/* Packed 16-bit weight pairs for SMLAD: input pair [2,3] */
static const uint32_t W0_simd_23[32] = {
    0xEA6BF3C3U, 0x022F05E3U, 0xF9D1FC86U, 0xE6B806B5U,
    0xF4EBFB3EU, 0x10BE06E4U, 0xF934F9F8U, 0x00000000U,
    0x00000000U, 0xFAFDFFACU, 0xEB010191U, 0x05A600EFU,
    0x05E50795U, 0xF2FEFB06U, 0xE4D2F1DEU, 0xF51FFEACU,
    0x0944F06FU, 0x00000000U, 0x0BB60531U, 0x0739FCFDU,
    0x08ED034BU, 0xF6B2FF0BU, 0x0B330948U, 0x07D50C64U,
    0xFAEFF509U, 0x0D43023BU, 0x0A92EFDAU, 0xED3D01E4U,
    0x06FF08C7U, 0x0662F178U, 0xFBE802BCU, 0x0A85F422U
};
static const int32_t b0_simd[32] = {
        -7750,      4596,      5106,       807,     -3320,      3464,      5280,      -404,
         -415,      5483,      3733,       -93,      5448,     -2640,      3880,     -2825,
         3420,      -927,      1757,       767,       461,     -2388,      3566,      3245,
        -7747,      3750,      3447,      3608,      5609,      3693,     -1539,      3236
};

static const uint32_t W1_simd[16] = {
    0x0B3AE53CU, 0x0BA40CB4U, 0x0F54F214U, 0x00000E0EU,
    0x0C870000U, 0xFAA00C7AU, 0xF41F0B7EU, 0xF3F510B9U,
    0x00000D57U, 0xF73D0862U, 0xF4DCF311U, 0x0DD50D3EU,
    0x0D20E7E1U, 0x0BAC0C3EU, 0x0AE60EB4U, 0x0CCAF585U
};
static const int32_t b1_simd[1] = { 2529 };

int32_t mlp_new_predict_mode5_int16_simd(int32_t hs_cC, int32_t amb_cC,
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

    /* Layer 1: 32 -> 1 (SMLAD across 16 pairs, ReLU zero-skipped) */
    int32_t out_acc = b1_simd[0] << SHIFT_POT_16;
    for (int p = 0; p < 16; p++) {
        uint32_t h_pair = ((uint32_t)(uint16_t)h_q[p*2+1] << 16) | ((uint32_t)(uint16_t)h_q[p*2]);
        out_acc = smlad_dual(h_pair, W1_simd[p], out_acc);
    }
    out_acc >>= SHIFT_POT_16;

    return (int32_t)(((int64_t)out_acc * 100LL) >> SHIFT_POT_16);
}

/* ==============================================================================
 * MASTER ENTRYPOINT DISPATCHER
 * ============================================================================== */
int32_t mlp_new_predict_temp_cC(int32_t hs_cC, int32_t amb_cC,
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
