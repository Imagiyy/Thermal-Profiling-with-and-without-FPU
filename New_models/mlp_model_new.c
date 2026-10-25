/*
 * mlp_model_new.c -- 6-Mode TinyML Stateless Thermal Predictor for STM32F401RET6
 * Architecture: 4 Inputs -> 32 Neurons (ReLU) -> 1 Output (Linear)
 * Features: [heatsink_temp_C, ambient_temp_C, power_loss_mW, current_mA]
 * Target: Future Heatsink Temperature +3s ahead (centi-degC)
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
     63.66351700f,  31.25230217f,  2320.55468750f,  535.13977051f
};
static const float scaler_scale[4] = {
     17.49335670f,  8.12835026f,  934.88092041f,  227.21328735f
};

static const float W0[128] = {
    -0.00000000f, -0.00091267f,  0.41233796f, -0.27732638f,  0.00994459f, -0.00000000f, -0.54077333f,  0.65935022f,
     0.27145451f, -0.13746682f, -0.12610744f,  0.26030007f,  0.21460530f, -0.41567069f, -0.28061840f, -0.44751593f,
    -0.35471541f, -0.00000000f, -0.00039718f, -0.03408562f,  0.00000000f, -0.45514157f, -0.50320613f, -0.26543036f,
     0.27813557f,  0.02081435f,  0.00137489f, -0.44208351f, -0.54178995f,  0.00020224f, -0.18607546f, -0.66128045f,
     0.00000000f,  0.00117926f, -0.19345148f,  0.13173066f, -0.00282148f,  0.00000000f,  0.25868490f, -0.30754495f,
    -0.20984924f,  0.08661235f, -0.06994619f,  0.00749366f, -0.11454774f,  0.18775342f, -0.13133353f,  0.20262012f,
     0.16764945f,  0.00000000f,  0.00331317f,  0.03226573f, -0.00000000f,  0.21898857f,  0.23793340f,  0.26888591f,
    -0.13376994f, -0.00332824f, -0.00250775f,  0.20554076f,  0.25277567f, -0.00017064f,  0.09296998f,  0.31120458f,
     0.00000000f,  0.00021581f, -0.46810430f,  0.52376717f, -0.01256669f,  0.00000000f,  0.37394640f, -0.51581013f,
    -0.14475746f,  0.25306579f,  0.06958726f, -0.10822783f, -0.19715849f, -0.60916221f,  0.09617186f, -0.62757438f,
     0.85578167f, -0.00000000f, -0.00001995f, -0.06928343f, -0.00000000f,  0.09902919f,  0.49249959f,  0.19179949f,
     0.24283253f, -0.34368116f, -0.00033497f,  0.37762302f,  0.16846083f, -0.00012288f,  0.00253332f, -0.74297649f,
     0.00000000f,  0.00012623f, -0.55928284f, -0.03246143f, -0.00240832f,  0.00000000f, -0.08541531f, -0.02806580f,
     0.15723254f,  0.03093438f, -0.00839556f,  0.08293712f, -0.08693123f, -0.16707343f, -0.10301190f, -0.17026635f,
     0.63644063f, -0.00000000f,  0.00103376f,  0.04930491f, -0.00000000f, -0.05374757f,  0.12121733f, -0.14575213f,
     0.49211797f,  0.25408617f, -0.00202711f, -0.51432431f, -0.09663565f, -0.00019344f,  0.02943282f,  0.27576220f
};
static const float b0[32] = {
    -0.47512162f, -0.18677603f, -0.82349622f, -0.46663022f, -0.21267962f, -0.48990908f, -0.11331431f, -0.24648111f,
     0.17480776f, -0.36973542f, -0.47684851f,  0.20543407f,  0.06867225f, -0.50749415f, -0.39578101f, -0.55333829f,
    -0.43314201f, -0.48847890f, -0.26561835f, -0.13133709f, -0.50088835f, -0.68908387f,  0.19097766f, -0.27097902f,
    -0.50675046f, -0.38455078f, -0.04690455f, -0.25545302f, -0.96344656f, -0.00376105f, -0.52452731f, -0.03324101f
};
static const float W1[32] = {
    -0.00000000f, -0.00373912f,  0.53274715f,  0.11775768f,  0.00292581f, -0.00000748f,  0.24983631f, -0.65504295f,
    -0.08395581f,  0.01094219f,  0.00141068f, -0.05658756f, -0.00781974f, -0.41381392f,  0.02290420f, -0.57886624f,
    -0.83515185f,  0.00000000f, -0.00535187f, -0.00338036f,  0.00000000f, -0.70392025f,  0.27291957f,  0.02614647f,
    -0.25619435f,  0.00330923f,  0.00074390f,  0.23900008f,  2.09960055f,  0.00007649f, -0.07915515f,  0.56916809f
};
static const float b1[1] = {
    0.08450621f
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

static const int32_t norm_mean_c[4]  = { 6366, 3125, 232055, 53514 };
static const int32_t norm_scale_c[4] = { 1749, 813, 93488, 22721 };

static const int32_t W0_n32[128] = {
            0,        -9,      4123,     -2773,        99,         0,     -5408,      6594,
         2715,     -1375,     -1261,      2603,      2146,     -4157,     -2806,     -4475,
        -3547,         0,        -4,      -341,         0,     -4551,     -5032,     -2654,
         2781,       208,        14,     -4421,     -5418,         2,     -1861,     -6613,
            0,        12,     -1935,      1317,       -28,         0,      2587,     -3075,
        -2098,       866,      -699,        75,     -1145,      1878,     -1313,      2026,
         1676,         0,        33,       323,         0,      2190,      2379,      2689,
        -1338,       -33,       -25,      2055,      2528,        -2,       930,      3112,
            0,         2,     -4681,      5238,      -126,         0,      3739,     -5158,
        -1448,      2531,       696,     -1082,     -1972,     -6092,       962,     -6276,
         8558,         0,         0,      -693,         0,       990,      4925,      1918,
         2428,     -3437,        -3,      3776,      1685,        -1,        25,     -7430,
            0,         1,     -5593,      -325,       -24,         0,      -854,      -281,
         1572,       309,       -84,       829,      -869,     -1671,     -1030,     -1703,
         6364,         0,        10,       493,         0,      -537,      1212,     -1458,
         4921,      2541,       -20,     -5143,      -966,        -2,       294,      2758
};
static const int32_t b0_n32[32] = {
        -4751,     -1868,     -8235,     -4666,     -2127,     -4899,     -1133,     -2465,
         1748,     -3697,     -4768,      2054,       687,     -5075,     -3958,     -5533,
        -4331,     -4885,     -2656,     -1313,     -5009,     -6891,      1910,     -2710,
        -5068,     -3846,      -469,     -2555,     -9634,       -38,     -5245,      -332
};
static const int32_t W1_n32[32] = {
            0,       -37,      5327,      1178,        29,         0,      2498,     -6550,
         -840,       109,        14,      -566,       -78,     -4138,       229,     -5789,
        -8352,         0,       -54,       -34,         0,     -7039,      2729,       261,
        -2562,        33,         7,      2390,     20996,         1,      -792,      5692
};
static const int32_t b1_n32[1] = { 845 };

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

static const int32_t pot_mean_c[4]  = { 6366, 3125, 232055, 53514 };
static const int32_t pot_scale_c[4] = { 1749, 813, 93488, 22721 };

static const int32_t W0_p32[128] = {
            0,       -60,     27023,    -18175,       652,         0,    -35440,     43211,
        17790,     -9009,     -8265,     17059,     14064,    -27241,    -18391,    -29328,
       -23247,         0,       -26,     -2234,         0,    -29828,    -32978,    -17395,
        18228,      1364,        90,    -28972,    -35507,        13,    -12195,    -43338,
            0,        77,    -12678,      8633,      -185,         0,     16953,    -20155,
       -13753,      5676,     -4584,       491,     -7507,     12305,     -8607,     13279,
        10987,         0,       217,      2115,         0,     14352,     15593,     17622,
        -8767,      -218,      -164,     13470,     16566,       -11,      6093,     20395,
            0,        14,    -30678,     34326,      -824,         0,     24507,    -33804,
        -9487,     16585,      4560,     -7093,    -12921,    -39922,      6303,    -41129,
        56085,         0,        -1,     -4541,         0,      6490,     32276,     12570,
        15914,    -22523,       -22,     24748,     11040,        -8,       166,    -48692,
            0,         8,    -36653,     -2127,      -158,         0,     -5598,     -1839,
        10304,      2027,      -550,      5435,     -5697,    -10949,     -6751,    -11159,
        41710,         0,        68,      3231,         0,     -3522,      7944,     -9552,
        32251,     16652,      -133,    -33707,     -6333,       -13,      1929,     18072
};
static const int32_t b0_p32[32] = {
       -31138,    -12241,    -53969,    -30581,    -13938,    -32107,     -7426,    -16153,
        11456,    -24231,    -31251,     13463,      4501,    -33259,    -25938,    -36264,
       -28386,    -32013,    -17408,     -8607,    -32826,    -45160,     12516,    -17759,
       -33210,    -25202,     -3074,    -16741,    -63140,      -246,    -34375,     -2178
};
static const int32_t W1_p32[32] = {
            0,      -245,     34914,      7717,       192,         0,     16373,    -42929,
        -5502,       717,        92,     -3709,      -512,    -27120,      1501,    -37937,
       -54733,         0,      -351,      -222,         0,    -46132,     17886,      1714,
       -16790,       217,        49,     15663,    137599,         5,     -5188,     37301
};
static const int32_t b1_p32[1] = { 5538 };

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
    0x8000U, 0x937AU, 0x3699U, 0xB470U, 0x2117U, 0x8000U, 0xB854U, 0x3946U,
    0x3458U, 0xB066U, 0xB009U, 0x342AU, 0x32DEU, 0xB6A7U, 0xB47DU, 0xB729U,
    0xB5ADU, 0x8000U, 0x8E82U, 0xA85DU, 0x0000U, 0xB748U, 0xB807U, 0xB43FU,
    0x3473U, 0x2554U, 0x15A2U, 0xB713U, 0xB856U, 0x0AA1U, 0xB1F4U, 0xB94AU,
    0x0000U, 0x14D5U, 0xB231U, 0x3037U, 0x99C7U, 0x0000U, 0x3424U, 0xB4ECU,
    0xB2B7U, 0x2D8BU, 0xAC7AU, 0x1FACU, 0xAF55U, 0x3202U, 0xB034U, 0x327CU,
    0x315DU, 0x0000U, 0x1AC9U, 0x2821U, 0x8000U, 0x3302U, 0x339DU, 0x344DU,
    0xB048U, 0x9AD1U, 0x9923U, 0x3294U, 0x340BU, 0x8997U, 0x2DF3U, 0x34FBU,
    0x0000U, 0x0B12U, 0xB77DU, 0x3831U, 0xA26FU, 0x0000U, 0x35FCU, 0xB820U,
    0xB0A2U, 0x340DU, 0x2C74U, 0xAEEDU, 0xB24FU, 0xB8E0U, 0x2E28U, 0xB905U,
    0x3AD9U, 0x8000U, 0x814FU, 0xAC6FU, 0x8000U, 0x2E56U, 0x37E1U, 0x3223U,
    0x33C5U, 0xB580U, 0x8D7DU, 0x360BU, 0x3164U, 0x8807U, 0x1930U, 0xB9F2U,
    0x0000U, 0x0823U, 0xB879U, 0xA828U, 0x98EFU, 0x0000U, 0xAD77U, 0xA72FU,
    0x3108U, 0x27EBU, 0xA04CU, 0x2D4FU, 0xAD90U, 0xB159U, 0xAE98U, 0xB173U,
    0x3917U, 0x8000U, 0x143CU, 0x2A50U, 0x8000U, 0xAAE1U, 0x2FC2U, 0xB0AAU,
    0x37E0U, 0x3411U, 0x9827U, 0xB81DU, 0xAE2FU, 0x8A57U, 0x2789U, 0x346AU
};
static const uint16_t b0_f16[32] = {
    0xB79AU, 0xB1FAU, 0xBA97U, 0xB777U, 0xB2CEU, 0xB7D7U, 0xAF41U, 0xB3E3U,
    0x3198U, 0xB5EAU, 0xB7A1U, 0x3293U, 0x2C65U, 0xB80FU, 0xB655U, 0xB86DU,
    0xB6EEU, 0xB7D1U, 0xB440U, 0xB034U, 0xB802U, 0xB983U, 0x321CU, 0xB456U,
    0xB80EU, 0xB627U, 0xAA01U, 0xB416U, 0xBBB5U, 0x9BB4U, 0xB832U, 0xA841U
};
static const uint16_t W1_f16[32] = {
    0x8000U, 0x9BA8U, 0x3843U, 0x2F89U, 0x19FEU, 0x807EU, 0x33FFU, 0xB93EU,
    0xAD60U, 0x219AU, 0x15C7U, 0xAB3EU, 0xA001U, 0xB69FU, 0x25DDU, 0xB8A2U,
    0xBAAEU, 0x0000U, 0x9D7BU, 0x9AECU, 0x0000U, 0xB9A2U, 0x345EU, 0x26B2U,
    0xB419U, 0x1AC7U, 0x1218U, 0x33A6U, 0x4033U, 0x0503U, 0xAD11U, 0x388EU
};
static const uint16_t b1_f16[1] = {
    0x2D69U
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
            0,        -2,       844,      -568,        20,         0,     -1108,      1350,
          556,      -282,      -258,       533,       440,      -851,      -575,      -917,
         -726,         0,        -1,       -70,         0,      -932,     -1031,      -544,
          570,        43,         3,      -905,     -1110,         0,      -381,     -1354,
            0,         2,      -396,       270,        -6,         0,       530,      -630,
         -430,       177,      -143,        15,      -235,       385,      -269,       415,
          343,         0,         7,        66,         0,       448,       487,       551,
         -274,        -7,        -5,       421,       518,         0,       190,       637,
            0,         0,      -959,      1073,       -26,         0,       766,     -1056,
         -296,       518,       143,      -222,      -404,     -1248,       197,     -1285,
         1753,         0,         0,      -142,         0,       203,      1009,       393,
          497,      -704,        -1,       773,       345,         0,         5,     -1522,
            0,         0,     -1145,       -66,        -5,         0,      -175,       -57,
          322,        63,       -17,       170,      -178,      -342,      -211,      -349,
         1303,         0,         2,       101,         0,      -110,       248,      -299,
         1008,       520,        -4,     -1053,      -198,         0,        60,       565
};
static const int32_t b0_n16[32] = {
         -973,      -383,     -1687,      -956,      -436,     -1003,      -232,      -505,
          358,      -757,      -977,       421,       141,     -1039,      -811,     -1133,
         -887,     -1000,      -544,      -269,     -1026,     -1411,       391,      -555,
        -1038,      -788,       -96,      -523,     -1973,        -8,     -1074,       -68
};
static const int16_t W1_n16[32] = {
            0,        -8,      1091,       241,         6,         0,       512,     -1342,
         -172,        22,         3,      -116,       -16,      -847,        47,     -1186,
        -1710,         0,       -11,        -7,         0,     -1442,       559,        54,
         -525,         7,         2,       489,      4300,         0,      -162,      1166
};
static const int32_t b1_n16[1] = { 173 };

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
    0x00000000U, 0x0002FFFEU, 0xFE74034CU, 0x010EFDC8U,
    0xFFFA0014U, 0x00000000U, 0x0212FBACU, 0xFD8A0546U,
    0xFE52022CU, 0x00B1FEE6U, 0xFF71FEFEU, 0x000F0215U,
    0xFF1501B8U, 0x0181FCADU, 0xFEF3FDC1U, 0x019FFC6BU,
    0x0157FD2AU, 0x00000000U, 0x0007FFFFU, 0x0042FFBAU,
    0x00000000U, 0x01C0FC5CU, 0x01E7FBF9U, 0x0227FDE0U,
    0xFEEE023AU, 0xFFF9002BU, 0xFFFB0003U, 0x01A5FC77U,
    0x0206FBAAU, 0x00000000U, 0x00BEFE83U, 0x027DFAB6U
};
/* Packed 16-bit weight pairs for SMLAD: input pair [2,3] */
static const uint32_t W0_simd_23[32] = {
    0x00000000U, 0x00000000U, 0xFB87FC41U, 0xFFBE0431U,
    0xFFFBFFE6U, 0x00000000U, 0xFF5102FEU, 0xFFC7FBE0U,
    0x0142FED8U, 0x003F0206U, 0xFFEF008FU, 0x00AAFF22U,
    0xFF4EFE6CU, 0xFEAAFB20U, 0xFF2D00C5U, 0xFEA3FAFBU,
    0x051706D9U, 0x00000000U, 0x00020000U, 0x0065FF72U,
    0x00000000U, 0xFF9200CBU, 0x00F803F1U, 0xFED50189U,
    0x03F001F1U, 0x0208FD40U, 0xFFFCFFFFU, 0xFBE30305U,
    0xFF3A0159U, 0x00000000U, 0x003C0005U, 0x0235FA0EU
};
static const int32_t b0_simd[32] = {
         -973,      -383,     -1687,      -956,      -436,     -1003,      -232,      -505,
          358,      -757,      -977,       421,       141,     -1039,      -811,     -1133,
         -887,     -1000,      -544,      -269,     -1026,     -1411,       391,      -555,
        -1038,      -788,       -96,      -523,     -1973,        -8,     -1074,       -68
};

static const uint32_t W1_simd[16] = {
    0xFFF80000U, 0x00F10443U, 0x00000006U, 0xFAC20200U,
    0x0016FF54U, 0xFF8C0003U, 0xFCB1FFF0U, 0xFB5E002FU,
    0x0000F952U, 0xFFF9FFF5U, 0xFA5E0000U, 0x0036022FU,
    0x0007FDF3U, 0x01E90002U, 0x000010CCU, 0x048EFF5EU
};
static const int32_t b1_simd[1] = { 173 };

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
