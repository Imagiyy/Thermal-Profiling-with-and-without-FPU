#include "stm32f401xe.h"
#include <stdint.h>

// Select your target fixed-point weights header:
#include "case2_weights.h"
// #include "case3_weights.h"
// #include "case5_weights_2.h"
// #include "case6_weights_2.h"

/* Input bounds expressed in exact centi-units to avoid all floats */
static const int32_t IN_MIN[4]  = { 0,      0,     3279, 3175 };
static const int32_t IN_SPAN[4] = { 393974, 78850, 96,   6675 };

int32_t cnn_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC)
{
    const int32_t raw_in[4] = { ploss_cmW, i_cmA, amb_cC, hs_cC };
    int32_t in_q[4];

    /* 1. Pure Integer Min-Max Normalization -> Q(MODEL_SHIFT_BITS) */
    for (int i = 0; i < 4; i++) {
        int32_t diff = raw_in[i] - IN_MIN[i];
        if (diff <= 0) {
            in_q[i] = 0;
        } else if (diff >= IN_SPAN[i]) {
            in_q[i] = (int32_t)(1UL << MODEL_SHIFT_BITS);
        } else {
            in_q[i] = (int32_t)(((int64_t)diff << MODEL_SHIFT_BITS) / IN_SPAN[i]);
        }
    }

    /* 2. Layer 0: Dense (4 -> 16, ReLU) */
    int32_t h_q[16];
    for (int j = 0; j < 16; j++) {
        int64_t acc = ((int64_t)L0_B[j]) << MODEL_SHIFT_BITS;
        for (int i = 0; i < 4; i++) {
            acc += (int64_t)in_q[i] * (int64_t)L0_W[i * 16 + j];
        }
        acc >>= MODEL_SHIFT_BITS;
        h_q[j] = (acc > 0) ? (int32_t)acc : 0;
    }

    /* 3. Layer 1: Dense (16 -> 2, Linear) -> Target 0 (t+1s) */
    int64_t out_acc = ((int64_t)L1_B[0]) << MODEL_SHIFT_BITS;
    for (int j = 0; j < 16; j++) {
        out_acc += (int64_t)h_q[j] * (int64_t)L1_W[j * 2 + 0];
    }
    int32_t out_q = (int32_t)(out_acc >> MODEL_SHIFT_BITS);

    /* 4. Integer Dequantization to centi-degrees Celsius */
    return (int32_t)(((int64_t)out_q * 100) >> MODEL_SHIFT_BITS);
}