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
    68.17805822f, 28.58158989f
};
static const float scaler_scale[2] = {
    15.35353778f, 6.69172072f
};

/* Layer 0: Weights [2 x 8] */
static const float W0[16] = {
    -0.18663535f,  2.74716792f,  0.00303465f,  2.83030169f, -0.54742916f, -3.06951409f, -0.04001035f,  0.17388869f,
     1.19469390f,  0.23851692f, -0.29622779f,  0.30987871f,  1.53858908f, -0.29915177f, -2.86772334f, -0.11337092f
};
static const float b0[8] = {
     3.72803479f,  3.53439226f, -0.48075091f,  3.66370184f,  3.91097712f, -3.97619813f,  4.35051909f, -0.51953428f
};

/* Layer 1: Weights [8 x 1] */
static const float W1[8] = {
     3.75073703f,  3.42906860f, -0.18346456f,  3.02047823f,  3.23502585f, -5.83439638f,  3.89480379f, -0.28393094f
};
static const float b1[1] = {
     1.42442354f
};

int32_t mlp_new_predict_temp_cC(int32_t hs_cC, int32_t amb_cC)
{
    /*
     * Sanity clamp ambient reading to realistic operating room conditions [10 °C, 45 °C].
     * If BME280 sensor is disconnected or reading <= 0, default to 25 °C (2500 cC).
     */
    if (amb_cC <= 1000 || amb_cC >= 5000) {
        amb_cC = 2500;
    }

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
