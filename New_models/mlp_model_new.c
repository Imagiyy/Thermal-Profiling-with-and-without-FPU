/*
 * mlp_model_new.c — Pure C Hardware-FPU TinyML Thermal Predictor for STM32F401RET6
 *
 * Predicts future heatsink temperature (+1s ahead) from present thermal readings.
 * Features: [heatsink_temp_C, ambient_temp_C]
 * Architecture: 2 -> 8 -> 1
 * Parameters:   33 floats (132 bytes flash)
 */

#include "mlp_model_new.h"

/* Feature Normalisation (StandardScaler) */
static const float scaler_mean[2] = {
    68.17805822f, 33.20679808f
};
static const float scaler_scale[2] = {
    15.35353778f, 0.20280992f
};

/* Layer 0: Weights [2 x 8] */
static const float W0[16] = {
    -0.12035448f,  1.92869996f,  0.19087808f,  1.76751756f,  1.40198652f, -0.28563257f, -0.86826510f,  0.24875921f,
     0.09334612f, -0.44885203f, -0.27328899f,  1.25753733f,  1.26653665f, -0.24495334f, -1.94424009f, -0.24065915f
};
static const float b0[8] = {
    -0.36480568f,  4.38642455f, -0.45042314f,  4.60872958f,  4.47527867f, -0.76644414f,  4.66252550f, -0.47535494f
};

/* Layer 1: Weights [8 x 1] */
static const float W1[8] = {
    -0.01704816f,  3.80263226f, -0.21193366f,  3.34556220f,  3.79997410f, -0.59523408f,  3.75771813f, -0.34896849f
};
static const float b1[1] = {
     1.59955612f
};

int32_t mlp_new_predict_temp_cC(int32_t hs_cC, int32_t amb_cC)
{
    /*
     * Feature extraction:
     *   x[0] = heatsink_temp_C = hs_cC  / 100.0f  (centi-°C -> °C)
     *   x[1] = ambient_temp_C  = amb_cC / 100.0f  (centi-°C -> °C)
     */
    float x[2];
    x[0] = (((float)hs_cC  / 100.0f) - scaler_mean[0]) / scaler_scale[0];
    x[1] = (((float)amb_cC / 100.0f) - scaler_mean[1]) / scaler_scale[1];

    /* Forward Pass (Single Precision HW FPU) */
    float buf_a[8];
    float buf_b[8];
    const float *in_vec = x;
    float *out_vec;

    /* Layer 0: 2 -> 8 */
    out_vec = buf_a;
    for (int j = 0; j < 8; j++) {
        float acc = b0[j];
        for (int k = 0; k < 2; k++) {
            acc += in_vec[k] * W0[k * 8 + j];
        }
        out_vec[j] = (acc > 0.0f) ? acc : 0.0f; /* ReLU */
    }
    in_vec = out_vec;

    /* Layer 1: 8 -> 1 */
    out_vec = buf_b;
    for (int j = 0; j < 1; j++) {
        float acc = b1[j];
        for (int k = 0; k < 8; k++) {
            acc += in_vec[k] * W1[k * 1 + j];
        }
        out_vec[j] = acc; /* Linear Output */
    }
    in_vec = out_vec;

    /* Convert predicted °C float back to centi-°C integer */
    float pred_C = out_vec[0];
    return (int32_t)(pred_C * 100.0f + (pred_C >= 0.0f ? 0.5f : -0.5f));
}
