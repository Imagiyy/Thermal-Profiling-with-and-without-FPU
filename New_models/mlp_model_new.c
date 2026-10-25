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
     63.68672943f,  31.25336266f,  2320.69970703f,  535.03784180f
};
static const float scaler_scale[4] = {
     17.49801064f,  8.12818432f,  935.20721436f,  227.14947510f
};

static const float W0[128] = {
     0.00000000f, -0.00477088f,  0.00000000f, -0.19875368f,  0.20293917f, -0.35543165f, -0.07373660f,  0.58276427f,
     0.23291650f,  0.00000000f, -0.27035403f,  0.26336381f,  0.31120399f,  0.02671019f, -0.23812672f, -0.10821409f,
     0.00000000f, -0.00000000f, -0.09789277f, -0.23772001f, -0.45788109f, -0.00000000f, -0.36831555f, -0.38183019f,
     0.02822347f,  0.19474758f, -0.06030774f, -0.43977305f,  0.00000000f,  0.01982356f, -0.00002489f, -0.12639095f,
    -0.00000000f,  0.02265307f,  0.00000000f,  0.10826995f, -0.08746849f,  0.21359512f,  0.03779886f, -0.27762038f,
    -0.10967639f, -0.00000000f,  0.05111378f, -0.05964182f, -0.20411745f, -0.00266911f, -0.00370324f,  0.06372789f,
     0.00000000f, -0.00000000f,  0.15380415f,  0.12672962f,  0.22455265f,  0.00000002f,  0.16274609f,  0.17827635f,
    -0.09882439f, -0.09090358f,  0.02022445f,  0.21212430f,  0.00000000f, -0.01341835f,  0.00070464f,  0.09730500f,
     0.00000000f,  0.00304871f, -0.00000000f,  0.18809380f, -0.24168631f,  0.25511241f,  0.10254359f, -0.40013143f,
    -0.20648664f,  0.00000000f,  0.24868140f, -0.11060765f, -0.15379494f, -0.06026885f,  0.10841760f, -0.10956023f,
     0.00000000f,  0.00000000f,  0.01015268f, -0.28166372f,  0.05606656f,  0.00000000f,  0.35293064f,  0.28718677f,
     0.25051987f, -0.26887766f,  0.03722078f,  0.14353383f,  0.00000000f, -0.02722102f,  0.00000004f,  0.03440034f,
    -0.00000000f,  0.00114318f, -0.00000000f, -0.04575149f, -0.29797968f,  0.17239887f,  0.00386318f,  0.00022429f,
     0.17235436f, -0.00000000f, -0.12774055f,  0.11348799f, -0.12874947f, -0.00857379f, -0.12394851f,  0.04209317f,
     0.00000000f,  0.00000000f,  0.01824661f,  0.27933753f,  0.04919350f, -0.00000000f,  0.09288272f, -0.03953998f,
     0.23282051f,  0.13381270f, -0.00188626f, -0.07719218f,  0.00000000f, -0.00983019f,  0.00051031f,  0.08313408f
};
static const float b0[32] = {
    -0.45233878f, -0.17581944f, -0.40040892f, -0.51008940f, -0.22334078f, -0.34944141f, -0.14891796f, -0.27581033f,
     0.02412324f, -0.09633094f, -0.46568650f,  0.30834469f,  0.29252696f, -0.37591285f, -0.41001385f, -0.26203978f,
    -0.28039184f, -0.48660162f, -0.27337986f,  0.06305258f, -0.82791513f, -0.36138162f,  0.13748102f, -0.08141204f,
    -0.41404948f, -0.31984299f, -0.11468600f, -0.77796817f, -0.04748674f, -0.01212883f, -0.35377863f, -0.18186592f
};
static const float W1[32] = {
     0.00000000f,  0.00046891f,  0.00008824f,  0.03094990f, -0.06566146f,  0.11158147f,  0.00174342f, -0.37878731f,
    -0.05351267f, -0.00000000f,  0.05796075f, -0.10793212f, -0.07478331f,  0.03586829f,  0.03371601f, -0.00980320f,
    -0.00000000f, -0.00000000f, -0.00086310f, -0.08896444f, -0.84977853f,  0.00052257f,  0.12488046f,  0.11046546f,
     0.00075991f,  0.05032289f,  0.00135667f,  0.84129977f, -0.00000000f,  0.00277020f,  0.00003638f,  0.00644050f
};
static const float b1[1] = {
    0.12022017f
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

    int32_t delta_cC = (int32_t)(out * 100.0f + (out >= 0.0f ? 0.5f : -0.5f));
    return hs_cC + delta_cC;
}

/* ==============================================================================
 * MODE 1: 32-BIT INTEGER -- NORMAL FIXED-POINT (Division / 10000)
 * ============================================================================== */
#define SCALE_NORMAL_32 10000LL

static const int32_t norm_mean_c[4]  = { 6369, 3125, 232070, 53504 };
static const int32_t norm_scale_c[4] = { 1750, 813, 93521, 22715 };

static const int32_t W0_n32[128] = {
            0,       -48,         0,     -1988,      2029,     -3554,      -737,      5828,
         2329,         0,     -2704,      2634,      3112,       267,     -2381,     -1082,
            0,         0,      -979,     -2377,     -4579,         0,     -3683,     -3818,
          282,      1947,      -603,     -4398,         0,       198,         0,     -1264,
            0,       227,         0,      1083,      -875,      2136,       378,     -2776,
        -1097,         0,       511,      -596,     -2041,       -27,       -37,       637,
            0,         0,      1538,      1267,      2246,         0,      1627,      1783,
         -988,      -909,       202,      2121,         0,      -134,         7,       973,
            0,        30,         0,      1881,     -2417,      2551,      1025,     -4001,
        -2065,         0,      2487,     -1106,     -1538,      -603,      1084,     -1096,
            0,         0,       102,     -2817,       561,         0,      3529,      2872,
         2505,     -2689,       372,      1435,         0,      -272,         0,       344,
            0,        11,         0,      -458,     -2980,      1724,        39,         2,
         1724,         0,     -1277,      1135,     -1287,       -86,     -1239,       421,
            0,         0,       182,      2793,       492,         0,       929,      -395,
         2328,      1338,       -19,      -772,         0,       -98,         5,       831
};
static const int32_t b0_n32[32] = {
        -4523,     -1758,     -4004,     -5101,     -2233,     -3494,     -1489,     -2758,
          241,      -963,     -4657,      3083,      2925,     -3759,     -4100,     -2620,
        -2804,     -4866,     -2734,       631,     -8279,     -3614,      1375,      -814,
        -4140,     -3198,     -1147,     -7780,      -475,      -121,     -3538,     -1819
};
static const int32_t W1_n32[32] = {
            0,         5,         1,       309,      -657,      1116,        17,     -3788,
         -535,         0,       580,     -1079,      -748,       359,       337,       -98,
            0,         0,        -9,      -890,     -8498,         5,      1249,      1105,
            8,       503,        14,      8413,         0,        28,         0,        64
};
static const int32_t b1_n32[1] = { 1202 };

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

    int32_t delta_cC = (int32_t)(((int64_t)out_acc * 100LL) / SCALE_NORMAL_32);
    return hs_cC + delta_cC;
}

/* ==============================================================================
 * MODE 2: 32-BIT INTEGER -- POWER-OF-2 (Bit-Shift >> 16)
 * ============================================================================== */
#define SHIFT_POT_32 16

static const int32_t pot_mean_c[4]  = { 6369, 3125, 232070, 53504 };
static const int32_t pot_scale_c[4] = { 1750, 813, 93521, 22715 };

static const int32_t W0_p32[128] = {
            0,      -313,         0,    -13026,     13300,    -23294,     -4832,     38192,
        15264,         0,    -17718,     17260,     20395,      1750,    -15606,     -7092,
            0,         0,     -6416,    -15579,    -30008,         0,    -24138,    -25024,
         1850,     12763,     -3952,    -28821,         0,      1299,        -2,     -8283,
            0,      1485,         0,      7096,     -5732,     13998,      2477,    -18194,
        -7188,         0,      3350,     -3909,    -13377,      -175,      -243,      4176,
            0,         0,     10080,      8305,     14716,         0,     10666,     11684,
        -6477,     -5957,      1325,     13902,         0,      -879,        46,      6377,
            0,       200,         0,     12327,    -15839,     16719,      6720,    -26223,
       -13532,         0,     16298,     -7249,    -10079,     -3950,      7105,     -7180,
            0,         0,       665,    -18459,      3674,         0,     23130,     18821,
        16418,    -17621,      2439,      9407,         0,     -1784,         0,      2254,
            0,        75,         0,     -2998,    -19528,     11298,       253,        15,
        11295,         0,     -8372,      7438,     -8438,      -562,     -8123,      2759,
            0,         0,      1196,     18307,      3224,         0,      6087,     -2591,
        15258,      8770,      -124,     -5059,         0,      -644,        33,      5448
};
static const int32_t b0_p32[32] = {
       -29644,    -11523,    -26241,    -33429,    -14637,    -22901,     -9759,    -18076,
         1581,     -6313,    -30519,     20208,     19171,    -24636,    -26871,    -17173,
       -18376,    -31890,    -17916,      4132,    -54258,    -23684,      9010,     -5335,
       -27135,    -20961,     -7516,    -50985,     -3112,      -795,    -23185,    -11919
};
static const int32_t W1_p32[32] = {
            0,        31,         6,      2028,     -4303,      7313,       114,    -24824,
        -3507,         0,      3799,     -7073,     -4901,      2351,      2210,      -642,
            0,         0,       -57,     -5830,    -55691,        34,      8184,      7239,
           50,      3298,        89,     55135,         0,       182,         2,       422
};
static const int32_t b1_p32[1] = { 7879 };

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

    int32_t delta_cC = (int32_t)(((int64_t)out_acc * 100LL) >> SHIFT_POT_32);
    return hs_cC + delta_cC;
}

/* ==============================================================================
 * MODE 3: 16-BIT FLOAT WITH HARDWARE FPU (FP16 Storage -> FP32 Compute)
 * ============================================================================== */
static const uint16_t W0_f16[128] = {
    0x0000U, 0x9CE3U, 0x0000U, 0xB25CU, 0x327EU, 0xB5B0U, 0xACB8U, 0x38AAU,
    0x3374U, 0x0000U, 0xB453U, 0x3437U, 0x34FBU, 0x26D6U, 0xB39FU, 0xAEEDU,
    0x0000U, 0x8000U, 0xAE44U, 0xB39BU, 0xB753U, 0x8000U, 0xB5E5U, 0xB61CU,
    0x273AU, 0x323BU, 0xABB8U, 0xB709U, 0x0000U, 0x2513U, 0x81A2U, 0xB00BU,
    0x8000U, 0x25CDU, 0x0000U, 0x2EEEU, 0xAD99U, 0x32D6U, 0x28D7U, 0xB471U,
    0xAF05U, 0x8000U, 0x2A8BU, 0xABA2U, 0xB288U, 0x9977U, 0x9B96U, 0x2C14U,
    0x0000U, 0x8000U, 0x30ECU, 0x300EU, 0x3330U, 0x0000U, 0x3135U, 0x31B4U,
    0xAE53U, 0xADD1U, 0x252DU, 0x32CAU, 0x0000U, 0xA2DFU, 0x11C6U, 0x2E3AU,
    0x0000U, 0x1A3EU, 0x8000U, 0x3205U, 0xB3BCU, 0x3415U, 0x2E90U, 0xB667U,
    0xB29CU, 0x0000U, 0x33F5U, 0xAF14U, 0xB0ECU, 0xABB7U, 0x2EF0U, 0xAF03U,
    0x0000U, 0x0000U, 0x2133U, 0xB482U, 0x2B2DU, 0x0000U, 0x35A6U, 0x3498U,
    0x3402U, 0xB44DU, 0x28C4U, 0x3098U, 0x0000U, 0xA6F8U, 0x0001U, 0x2867U,
    0x8000U, 0x14AFU, 0x8000U, 0xA9DBU, 0xB4C5U, 0x3184U, 0x1BE9U, 0x0B59U,
    0x3184U, 0x8000U, 0xB016U, 0x2F43U, 0xB01FU, 0xA064U, 0xAFEFU, 0x2963U,
    0x0000U, 0x0000U, 0x24ACU, 0x3478U, 0x2A4CU, 0x8000U, 0x2DF2U, 0xA910U,
    0x3373U, 0x3048U, 0x97BAU, 0xACF1U, 0x0000U, 0xA108U, 0x102EU, 0x2D52U
};
static const uint16_t b0_f16[32] = {
    0xB73DU, 0xB1A0U, 0xB668U, 0xB815U, 0xB326U, 0xB597U, 0xB0C4U, 0xB46AU,
    0x262DU, 0xAE2AU, 0xB773U, 0x34EFU, 0x34AEU, 0xB604U, 0xB68FU, 0xB431U,
    0xB47CU, 0xB7C9U, 0xB460U, 0x2C09U, 0xBAA0U, 0xB5C8U, 0x3066U, 0xAD36U,
    0xB6A0U, 0xB51EU, 0xAF57U, 0xBA39U, 0xAA14U, 0xA236U, 0xB5A9U, 0xB1D2U
};
static const uint16_t W1_f16[32] = {
    0x0000U, 0x0FAFU, 0x05C8U, 0x27ECU, 0xAC34U, 0x2F24U, 0x1724U, 0xB610U,
    0xAADAU, 0x8000U, 0x2B6BU, 0xAEE8U, 0xACC9U, 0x2897U, 0x2851U, 0xA105U,
    0x8000U, 0x8000U, 0x9312U, 0xADB2U, 0xBACCU, 0x1048U, 0x2FFEU, 0x2F12U,
    0x123AU, 0x2A71U, 0x158FU, 0x3ABBU, 0x8000U, 0x19ACU, 0x0262U, 0x1E98U
};
static const uint16_t b1_f16[1] = {
    0x2FB2U
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

    int32_t delta_cC = (int32_t)(out * 100.0f + (out >= 0.0f ? 0.5f : -0.5f));
    return hs_cC + delta_cC;
}

/* ==============================================================================
 * MODE 4: 16-BIT INTEGER -- NORMAL FIXED-POINT (Division / 2048)
 * ============================================================================== */
#define SCALE_NORMAL_16 2048

static const int16_t W0_n16[128] = {
            0,       -10,         0,      -407,       416,      -728,      -151,      1194,
          477,         0,      -554,       539,       637,        55,      -488,      -222,
            0,         0,      -200,      -487,      -938,         0,      -754,      -782,
           58,       399,      -124,      -901,         0,        41,         0,      -259,
            0,        46,         0,       222,      -179,       437,        77,      -569,
         -225,         0,       105,      -122,      -418,        -5,        -8,       131,
            0,         0,       315,       260,       460,         0,       333,       365,
         -202,      -186,        41,       434,         0,       -27,         1,       199,
            0,         6,         0,       385,      -495,       522,       210,      -819,
         -423,         0,       509,      -227,      -315,      -123,       222,      -224,
            0,         0,        21,      -577,       115,         0,       723,       588,
          513,      -551,        76,       294,         0,       -56,         0,        70,
            0,         2,         0,       -94,      -610,       353,         8,         0,
          353,         0,      -262,       232,      -264,       -18,      -254,        86,
            0,         0,        37,       572,       101,         0,       190,       -81,
          477,       274,        -4,      -158,         0,       -20,         1,       170
};
static const int32_t b0_n16[32] = {
         -926,      -360,      -820,     -1045,      -457,      -716,      -305,      -565,
           49,      -197,      -954,       631,       599,      -770,      -840,      -537,
         -574,      -997,      -560,       129,     -1696,      -740,       282,      -167,
         -848,      -655,      -235,     -1593,       -97,       -25,      -725,      -372
};
static const int16_t W1_n16[32] = {
            0,         1,         0,        63,      -134,       229,         4,      -776,
         -110,         0,       119,      -221,      -153,        73,        69,       -20,
            0,         0,        -2,      -182,     -1740,         1,       256,       226,
            2,       103,         3,      1723,         0,         6,         0,        13
};
static const int32_t b1_n16[1] = { 246 };

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

    int32_t delta_cC = (int32_t)(((int64_t)out_acc * 100LL) / SCALE_NORMAL_16);
    return hs_cC + delta_cC;
}

/* ==============================================================================
 * MODE 5: 16-BIT INTEGER -- POT (>> 11) + ARM SMLAD SIMD
 * ============================================================================== */
#define SHIFT_POT_16 11

/* Packed 16-bit weight pairs for SMLAD: input pair [0,1] */
static const uint32_t W0_simd_01[32] = {
    0x00000000U, 0x002EFFF6U, 0x00000000U, 0x00DEFE69U,
    0xFF4D01A0U, 0x01B5FD28U, 0x004DFF69U, 0xFDC704AAU,
    0xFF1F01DDU, 0x00000000U, 0x0069FDD6U, 0xFF86021BU,
    0xFE5E027DU, 0xFFFB0037U, 0xFFF8FE18U, 0x0083FF22U,
    0x00000000U, 0x00000000U, 0x013BFF38U, 0x0104FE19U,
    0x01CCFC56U, 0x00000000U, 0x014DFD0EU, 0x016DFCF2U,
    0xFF36003AU, 0xFF46018FU, 0x0029FF84U, 0x01B2FC7BU,
    0x00000000U, 0xFFE50029U, 0x00010000U, 0x00C7FEFDU
};
/* Packed 16-bit weight pairs for SMLAD: input pair [2,3] */
static const uint32_t W0_simd_23[32] = {
    0x00000000U, 0x00020006U, 0x00000000U, 0xFFA20181U,
    0xFD9EFE11U, 0x0161020AU, 0x000800D2U, 0x0000FCCDU,
    0x0161FE59U, 0x00000000U, 0xFEFA01FDU, 0x00E8FF1DU,
    0xFEF8FEC5U, 0xFFEEFF85U, 0xFF0200DEU, 0x0056FF20U,
    0x00000000U, 0x00000000U, 0x00250015U, 0x023CFDBFU,
    0x00650073U, 0x00000000U, 0x00BE02D3U, 0xFFAF024CU,
    0x01DD0201U, 0x0112FDD9U, 0xFFFC004CU, 0xFF620126U,
    0x00000000U, 0xFFECFFC8U, 0x00010000U, 0x00AA0046U
};
static const int32_t b0_simd[32] = {
         -926,      -360,      -820,     -1045,      -457,      -716,      -305,      -565,
           49,      -197,      -954,       631,       599,      -770,      -840,      -537,
         -574,      -997,      -560,       129,     -1696,      -740,       282,      -167,
         -848,      -655,      -235,     -1593,       -97,       -25,      -725,      -372
};

static const uint32_t W1_simd[16] = {
    0x00010000U, 0x003F0000U, 0x00E5FF7AU, 0xFCF80004U,
    0x0000FF92U, 0xFF230077U, 0x0049FF67U, 0xFFEC0045U,
    0x00000000U, 0xFF4AFFFEU, 0x0001F934U, 0x00E20100U,
    0x00670002U, 0x06BB0003U, 0x00060000U, 0x000D0000U
};
static const int32_t b1_simd[1] = { 246 };

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

    int32_t delta_cC = (int32_t)(((int64_t)out_acc * 100LL) >> SHIFT_POT_16);
    return hs_cC + delta_cC;
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
