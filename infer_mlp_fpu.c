#include "stm32f401xe.h"
#include <stdint.h>

// Select one header:
#include "case1_weights.h"
// #include "case4_weights.h"

int32_t cnn_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC)
{
    float in[4] = {
        (float)ploss_cmW / 100.0f,
        (float)i_cmA / 100.0f,
        (float)amb_cC / 100.0f,
        (float)hs_cC / 100.0f
    };

    for (int i = 0; i < 4; i++) {
        in[i] = (in[i] - INPUT_MIN[i]) / (INPUT_MAX[i] - INPUT_MIN[i]);
        if (in[i] < 0.0f) in[i] = 0.0f;
        else if (in[i] > 1.0f) in[i] = 1.0f;
    }

    float h[16];
    for (int j = 0; j < 16; j++) {
        float sum = L0_B[j];
        for (int i = 0; i < 4; i++) {
            sum += in[i] * L0_W[i * 16 + j];
        }
        h[j] = (sum > 0.0f) ? sum : 0.0f; // ReLU
    }

    float pred_t1 = L1_B[0];
    for (int j = 0; j < 16; j++) {
        pred_t1 += h[j] * L1_W[j * 2 + 0];
    }

    return (int32_t)(pred_t1 * 100.0f);
}
