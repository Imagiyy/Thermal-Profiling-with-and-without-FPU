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
    2369.2202554f,  546.1087207f,   33.1790060f,    6.4172036f
};

static const float f32_scale[4] = {
     882.7450924f,  215.8827859f,    0.2144650f,    4.1182603f
};

static const float W0_f32[4 * 32] = {
      -0.0733915f,    2.9565426f,   -1.9386904f,    1.6366461f,    0.3426283f,    0.8419034f,
      -2.7954732f,    0.1853358f,    0.2432335f,    1.5342416f,    0.6832459f,    0.0142513f,
       1.7522328f,   -2.0859981f,    0.7513775f,   -1.0399110f,   -2.2321324f,    0.0230498f,
      -2.2588320f,   -0.1195203f,    0.2110407f,   -0.1850602f,    2.6917530f,   -2.3613099f,
       0.0660551f,    2.0974821f,    1.0061386f,    1.4418242f,    1.5354559f,   -2.8036339f,
       1.3445049f,   -1.9949985f,   -0.3206339f,    1.3488569f,    1.9451860f,   -1.2252476f,
       0.8040690f,   -1.4165395f,    2.0298350f,    0.0287708f,   -0.2153136f,   -1.2787785f,
      -1.8711726f,   -0.3282810f,   -1.7539281f,    1.0118577f,   -1.7417025f,    0.5303960f,
       1.6491174f,   -0.2047627f,    1.5772545f,    0.1034084f,    0.2435141f,    0.1672323f,
       1.2914928f,    1.5922193f,   -0.2195314f,   -1.4174107f,   -1.7976064f,   -1.6191393f,
      -1.6221714f,    1.5942058f,   -0.2597247f,    1.2637403f,   -0.1408425f,   -0.8822776f,
      -1.6109180f,    0.8243766f,    0.8136203f,    0.8110408f,    0.0912136f,   -0.1284066f,
      -3.4772929f,    0.6362181f,    0.6052467f,    1.8954603f,    0.6716865f,   -0.3289075f,
       0.3726757f,    0.0484824f,    0.0351868f,    0.0414976f,   -0.7357017f,   -0.4517706f,
      -0.0455737f,    0.0001607f,   -0.2429546f,   -0.5380109f,    0.1246658f,    0.3433723f,
       0.0871544f,    0.6628321f,    0.6676096f,   -0.3689201f,    1.3213451f,   -0.5795400f,
       0.0419573f,    0.7702160f,   -0.5360750f,    0.0535340f,   -3.4148967f,    0.4858972f,
      -1.3655989f,   -0.0210553f,    1.0126511f,    0.2114419f,    0.2631910f,   -0.4984433f,
       0.0649585f,   -2.3475754f,    0.1799830f,   -2.5910080f,    0.5931956f,    0.1934766f,
       0.1700613f,    0.3192969f,    0.1010156f,   -0.0593084f,    1.0842883f,    0.1592964f,
       1.1234359f,    2.1405367f,    0.1973505f,    0.2315026f,    0.0930629f,   -1.1798070f,
       0.9803541f,    0.4008875f
};

static const float b0_f32[32] = {
      -0.4306348f,    2.0478930f,    1.8719468f,    1.2730739f,   -2.3138650f,    1.4079627f,
       2.3517018f,   -0.2612222f,   -0.7637485f,    1.7401064f,    1.3727329f,   -0.1200377f,
       1.9071666f,    0.3059364f,    1.4827598f,   -0.9726681f,    1.8363290f,   -0.4309244f,
       1.8920735f,   -0.1823623f,   -0.4814676f,   -0.3161727f,    1.9494278f,    1.6495566f,
      -0.8081746f,    1.1297601f,    1.6026029f,    1.3058661f,    1.6561532f,    2.4159683f,
      -0.7336383f,    1.8727978f
};

static const float W1_f32[32] = {
      -0.0819430f,    1.9963790f,    1.7808913f,    1.2976440f,   -4.4055466f,    1.4591306f,
       2.0049015f,   -0.1826167f,   -2.7488707f,    1.2243793f,    1.3885017f,   -0.5988494f,
       1.1109760f,   -2.3316720f,    1.4558803f,   -2.7101443f,    1.9086996f,   -0.0490557f,
       1.9093877f,   -0.3622511f,   -0.0284199f,   -0.2211880f,    2.1947069f,    2.0776770f,
      -0.7806193f,    1.5782132f,    1.4632065f,    1.2807018f,    1.1857290f,    1.9774451f,
      -2.3782849f,    1.8866177f
};

static const float b1_f32[1] = {
    1.3126396f
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

    return (int32_t)(out * 100.0f);
}

/* ==============================================================================
 * MODE 1: 32-BIT FIXED-POINT NORMAL DIVISION (Scale = 10000)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_32BIT_FIXED_NORMAL)

#define SCALE_NORMAL_32 10000

static const int32_t n32_mean_c[4] = {
       236922,     54611,      3318,       642
};

static const int32_t n32_scale_c[4] = {
        88275,     21588,        21,       412
};

static const int32_t W0_n32[4 * 32] = {
         -734,     29565,    -19387,     16366,      3426,      8419,    -27955,      1853,
         2432,     15342,      6832,       143,     17522,    -20860,      7514,    -10399,
       -22321,       230,    -22588,     -1195,      2110,     -1851,     26918,    -23613,
          661,     20975,     10061,     14418,     15355,    -28036,     13445,    -19950,
        -3206,     13489,     19452,    -12252,      8041,    -14165,     20298,       288,
        -2153,    -12788,    -18712,     -3283,    -17539,     10119,    -17417,      5304,
        16491,     -2048,     15773,      1034,      2435,      1672,     12915,     15922,
        -2195,    -14174,    -17976,    -16191,    -16222,     15942,     -2597,     12637,
        -1408,     -8823,    -16109,      8244,      8136,      8110,       912,     -1284,
       -34773,      6362,      6052,     18955,      6717,     -3289,      3727,       485,
          352,       415,     -7357,     -4518,      -456,         2,     -2430,     -5380,
         1247,      3434,       872,      6628,      6676,     -3689,     13213,     -5795,
          420,      7702,     -5361,       535,    -34149,      4859,    -13656,      -211,
        10127,      2114,      2632,     -4984,       650,    -23476,      1800,    -25910,
         5932,      1935,      1701,      3193,      1010,      -593,     10843,      1593,
        11234,     21405,      1974,      2315,       931,    -11798,      9804,      4009
};

static const int32_t b0_n32[32] = {
        -4306,     20479,     18719,     12731,    -23139,     14080,     23517,     -2612,
        -7637,     17401,     13727,     -1200,     19072,      3059,     14828,     -9727,
        18363,     -4309,     18921,     -1824,     -4815,     -3162,     19494,     16496,
        -8082,     11298,     16026,     13059,     16562,     24160,     -7336,     18728
};

static const int32_t W1_n32[32] = {
         -819,     19964,     17809,     12976,    -44055,     14591,     20049,     -1826,
       -27489,     12244,     13885,     -5988,     11110,    -23317,     14559,    -27101,
        19087,      -491,     19094,     -3623,      -284,     -2212,     21947,     20777,
        -7806,     15782,     14632,     12807,     11857,     19774,    -23783,     18866
};

static const int32_t b1_n32[1] = {
    13126
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

    return (int32_t)((out_acc * 100LL) / SCALE_NORMAL_32);
}

/* ==============================================================================
 * MODE 2: 32-BIT FIXED-POINT POWER-OF-2 BIT-SHIFT (Q16.16)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_32BIT_FIXED_POT)

#define SHIFT_POT_32 16

static const int32_t pot32_mean_c[4] = {
       236922,     54611,      3318,       642
};

static const int32_t pot32_scale_c[4] = {
        88275,     21588,        21,       412
};

static const int32_t W0_p32[4 * 32] = {
        -4810,    193760,   -127054,    107259,     22454,     55175,   -183204,     12146,
        15941,    100548,     44777,       934,    114834,   -136708,     49242,    -68152,
      -146285,      1511,   -148035,     -7833,     13831,    -12128,    176407,   -154751,
         4329,    137461,     65938,     94491,    100628,   -183739,     88113,   -130744,
       -21013,     88399,    127480,    -80298,     52695,    -92834,    133027,      1886,
       -14111,    -83806,   -122629,    -21514,   -114945,     66313,   -114144,     34760,
       108077,    -13419,    103367,      6777,     15959,     10960,     84639,    104348,
       -14387,    -92891,   -117808,   -106112,   -106311,    104478,    -17021,     82820,
        -9230,    -57821,   -105573,     54026,     53321,     53152,      5978,     -8415,
      -227888,     41695,     39665,    124221,     44020,    -21555,     24424,      3177,
         2306,      2720,    -48215,    -29607,     -2987,        11,    -15922,    -35259,
         8170,     22503,      5712,     43439,     43752,    -24178,     86596,    -37981,
         2750,     50477,    -35132,      3508,   -223799,     31844,    -89496,     -1380,
        66365,     13857,     17248,    -32666,      4257,   -153851,     11795,   -169804,
        38876,     12680,     11145,     20925,      6620,     -3887,     71060,     10440,
        73625,    140282,     12934,     15172,      6099,    -77320,     64248,     26273
};

static const int32_t b0_p32[32] = {
       -28222,    134211,    122680,     83432,   -151641,     92272,    154121,    -17119,
       -50053,    114040,     89963,     -7867,    124988,     20050,     97174,    -63745,
       120346,    -28241,    123999,    -11951,    -31553,    -20721,    127758,    108105,
       -52965,     74040,    105028,     85581,    108538,    158333,    -48080,    122736
};

static const int32_t W1_p32[32] = {
        -5370,    130835,    116712,     85042,   -288722,     95626,    131393,    -11968,
      -180150,     80241,     90997,    -39246,     72809,   -152808,     95413,   -177612,
       125089,     -3215,    125134,    -23740,     -1863,    -14496,    143832,    136163,
       -51159,    103430,     95893,     83932,     77708,    129594,   -155863,    123641
};

static const int32_t b1_p32[1] = {
    86025
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

    return (int32_t)(((int64_t)out_acc * 100LL) >> SHIFT_POT_32);
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

    float result;
    *((uint32_t *)(void *)&result) = f;
    return result;
}

static const float f32_mean[4] = {
    2369.2202554f,  546.1087207f,   33.1790060f,    6.4172036f
};

static const float f32_scale[4] = {
     882.7450924f,  215.8827859f,    0.2144650f,    4.1182603f
};

static const uint16_t W0_f16[4 * 32] = {
    0x0000ACB2U, 0x000041EAU, 0x0000BFC1U, 0x00003E8CU, 0x0000357BU, 0x00003ABCU, 0x0000C197U, 0x000031EEU,
    0x000033C9U, 0x00003E23U, 0x00003977U, 0x0000234CU, 0x00003F02U, 0x0000C02CU, 0x00003A03U, 0x0000BC29U,
    0x0000C077U, 0x000025E7U, 0x0000C085U, 0x0000AFA6U, 0x000032C1U, 0x0000B1ECU, 0x00004162U, 0x0000C0B9U,
    0x00002C3AU, 0x00004032U, 0x00003C06U, 0x00003DC4U, 0x00003E24U, 0x0000C19BU, 0x00003D61U, 0x0000BFFBU,
    0x0000B521U, 0x00003D65U, 0x00003FC8U, 0x0000BCE7U, 0x00003A6FU, 0x0000BDABU, 0x0000400FU, 0x0000275EU,
    0x0000B2E4U, 0x0000BD1DU, 0x0000BF7CU, 0x0000B541U, 0x0000BF04U, 0x00003C0CU, 0x0000BEF8U, 0x0000383EU,
    0x00003E99U, 0x0000B28DU, 0x00003E4FU, 0x00002E9EU, 0x000033CBU, 0x0000315AU, 0x00003D2AU, 0x00003E5EU,
    0x0000B306U, 0x0000BDABU, 0x0000BF31U, 0x0000BE7AU, 0x0000BE7DU, 0x00003E60U, 0x0000B428U, 0x00003D0EU,
    0x0000B082U, 0x0000BB0FU, 0x0000BE72U, 0x00003A98U, 0x00003A82U, 0x00003A7DU, 0x00002DD6U, 0x0000B01CU,
    0x0000C2F4U, 0x00003917U, 0x000038D8U, 0x00003F95U, 0x00003960U, 0x0000B543U, 0x000035F6U, 0x00002A35U,
    0x00002881U, 0x00002950U, 0x0000B9E3U, 0x0000B73AU, 0x0000A9D5U, 0x00000944U, 0x0000B3C6U, 0x0000B84EU,
    0x00002FFBU, 0x0000357EU, 0x00002D94U, 0x0000394DU, 0x00003957U, 0x0000B5E7U, 0x00003D49U, 0x0000B8A3U,
    0x0000295FU, 0x00003A29U, 0x0000B84AU, 0x00002ADAU, 0x0000C2D4U, 0x000037C6U, 0x0000BD76U, 0x0000A564U,
    0x00003C0DU, 0x000032C4U, 0x00003436U, 0x0000B7FAU, 0x00002C28U, 0x0000C0B2U, 0x000031C2U, 0x0000C12FU,
    0x000038BFU, 0x00003231U, 0x00003171U, 0x0000351CU, 0x00002E77U, 0x0000AB97U, 0x00003C56U, 0x00003119U,
    0x00003C7EU, 0x00004048U, 0x00003251U, 0x00003368U, 0x00002DF5U, 0x0000BCB8U, 0x00003BD8U, 0x0000366AU
};

static const uint16_t b0_f16[32] = {
    0x0000B6E4U, 0x00004019U, 0x00003F7DU, 0x00003D18U, 0x0000C0A1U, 0x00003DA2U, 0x000040B4U, 0x0000B42EU,
    0x0000BA1CU, 0x00003EF6U, 0x00003D7EU, 0x0000AFAFU, 0x00003FA1U, 0x000034E5U, 0x00003DEEU, 0x0000BBC8U,
    0x00003F58U, 0x0000B6E5U, 0x00003F91U, 0x0000B1D6U, 0x0000B7B4U, 0x0000B50FU, 0x00003FCCU, 0x00003E99U,
    0x0000BA77U, 0x00003C85U, 0x00003E69U, 0x00003D39U, 0x00003EA0U, 0x000040D5U, 0x0000B9DEU, 0x00003F7EU
};

static const uint16_t W1_f16[32] = {
    0x0000AD3FU, 0x00003FFCU, 0x00003F20U, 0x00003D31U, 0x0000C468U, 0x00003DD6U, 0x00004003U, 0x0000B1D8U,
    0x0000C17FU, 0x00003CE6U, 0x00003D8EU, 0x0000B8CAU, 0x00003C72U, 0x0000C0AAU, 0x00003DD3U, 0x0000C16CU,
    0x00003FA3U, 0x0000AA47U, 0x00003FA3U, 0x0000B5CCU, 0x0000A747U, 0x0000B314U, 0x00004064U, 0x00004028U,
    0x0000BA3FU, 0x00003E50U, 0x00003DDAU, 0x00003D1FU, 0x00003CBEU, 0x00003FE9U, 0x0000C0C2U, 0x00003F8CU
};

static const uint16_t b1_f16[1] = {
    0x3D40U
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

    return (int32_t)(out * 100.0f);
}

/* ==============================================================================
 * MODE 4: 16-BIT FIXED-POINT NORMAL DIVISION (Scale = 2048)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_16BIT_FIXED_NORMAL)

#define SCALE_NORMAL_16 2048

static const int32_t n16_mean_c[4] = {
       236922,     54611,      3318,       642
};

static const int32_t n16_scale_c[4] = {
        88275,     21588,        21,       412
};

static const int16_t W0_n16[4 * 32] = {
         -150,      6055,     -3970,      3352,       702,      1724,     -5725,       380,
          498,      3142,      1399,        29,      3589,     -4272,      1539,     -2130,
        -4571,        47,     -4626,      -245,       432,      -379,      5513,     -4836,
          135,      4296,      2061,      2953,      3145,     -5742,      2754,     -4086,
         -657,      2762,      3984,     -2509,      1647,     -2901,      4157,        59,
         -441,     -2619,     -3832,      -672,     -3592,      2072,     -3567,      1086,
         3377,      -419,      3230,       212,       499,       342,      2645,      3261,
         -450,     -2903,     -3681,     -3316,     -3322,      3265,      -532,      2588,
         -288,     -1807,     -3299,      1688,      1666,      1661,       187,      -263,
        -7121,      1303,      1240,      3882,      1376,      -674,       763,        99,
           72,        85,     -1507,      -925,       -93,         0,      -498,     -1102,
          255,       703,       178,      1357,      1367,      -756,      2706,     -1187,
           86,      1577,     -1098,       110,     -6994,       995,     -2797,       -43,
         2074,       433,       539,     -1021,       133,     -4808,       369,     -5306,
         1215,       396,       348,       654,       207,      -121,      2221,       326,
         2301,      4384,       404,       474,       191,     -2416,      2008,       821
};

static const int32_t b0_n16[32] = {
         -882,      4194,      3834,      2607,     -4739,      2884,      4816,      -535,
        -1564,      3564,      2811,      -246,      3906,       627,      3037,     -1992,
         3761,      -883,      3875,      -373,      -986,      -648,      3992,      3378,
        -1655,      2314,      3282,      2674,      3392,      4948,     -1502,      3835
};

static const int16_t W1_n16[32] = {
         -168,      4089,      3647,      2658,     -9023,      2988,      4106,      -374,
        -5630,      2508,      2844,     -1226,      2275,     -4775,      2982,     -5550,
         3909,      -100,      3910,      -742,       -58,      -453,      4495,      4255,
        -1599,      3232,      2997,      2623,      2428,      4050,     -4871,      3864
};

static const int32_t b1_n16[1] = {
    2688
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

    return (int32_t)(((int32_t)out_acc * 100) / SCALE_NORMAL_16);
}

/* ==============================================================================
 * MODE 5: 16-BIT FIXED-POINT POWER-OF-2 BIT-SHIFT (Q5.11) + ARM SMLAD SIMD ASSEMBLY
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_16BIT_FIXED_POT_SIMD)

#define SHIFT_POT_16 11

static inline int32_t smlad_dual(uint32_t val1, uint32_t val2, int32_t acc)
{
#if defined(__arm__) || defined(__thumb__)
    __asm__ ("smlad %0, %1, %2, %3" : "=r" (acc) : "r" (val1), "r" (val2), "r" (acc));
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
       236922,     54611,      3318,       642
};

static const int32_t pot_scale_c[4] = {
        88275,     21588,        21,       412
};

static const uint32_t W0_simd_01[32] = {
    0xFD6FFF6AU, 0x0ACA17A7U, 0x0F90F07EU, 0xF6330D18U, 0x066F02BEU, 0xF4AB06BCU, 0x103DE9A3U, 0x003B017CU,
    0xFE4701F2U, 0xF5C50C46U, 0xF1080577U, 0xFD60001DU, 0xF1F80E05U, 0x0818EF50U, 0xF2110603U, 0x043EF7AEU,
    0x0D31EE25U, 0xFE5D002FU, 0x0C9EEDEEU, 0x00D4FF0BU, 0x01F301B0U, 0x0156FE85U, 0x0A551589U, 0x0CBDED1CU,
    0xFE3E0087U, 0xF4A910C8U, 0xF19F080DU, 0xF30C0B89U, 0xF3060C49U, 0x0CC1E992U, 0xFDEC0AC2U, 0x0A1CF00AU
};

static const uint32_t W0_simd_23[32] = {
    0x0056FEE0U, 0x0629F8F1U, 0xFBB6F31DU, 0x006E0698U, 0xE4AE0682U, 0x03E3067DU, 0xF51300BBU, 0xFFD5FEF9U,
    0x081AE42FU, 0x01B10517U, 0x021B04D8U, 0xFC030F2AU, 0x00850560U, 0xED38FD5EU, 0x017102FBU, 0xEB460063U,
    0x04BF0048U, 0x018C0055U, 0x015CFA1DU, 0x028EFC63U, 0x00CFFFA3U, 0xFF870000U, 0x08ADFE0EU, 0x0146FBB2U,
    0x08FD00FFU, 0x112002BFU, 0x019400B2U, 0x01DA054DU, 0x00BF0557U, 0xF690FD0CU, 0x07D80A92U, 0x0335FB5DU
};

static const int32_t b0_simd[32] = {
         -882,      4194,      3834,      2607,     -4739,      2884,      4816,      -535,
        -1564,      3564,      2811,      -246,      3906,       627,      3037,     -1992,
         3761,      -883,      3875,      -373,      -986,      -648,      3992,      3378,
        -1655,      2314,      3282,      2674,      3392,      4948,     -1502,      3835
};

static const int16_t W1_simd[32] = {
         -168,      4089,      3647,      2658,     -9023,      2988,      4106,      -374,
        -5630,      2508,      2844,     -1226,      2275,     -4775,      2982,     -5550,
         3909,      -100,      3910,      -742,       -58,      -453,      4495,      4255,
        -1599,      3232,      2997,      2623,      2428,      4050,     -4871,      3864
};

static const int32_t b1_simd[1] = {
    2688
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

    return (int32_t)(((int32_t)out_acc * 100) >> SHIFT_POT_16);
}

#endif
