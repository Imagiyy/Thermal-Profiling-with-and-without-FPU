#include <stdio.h>
#include <stdint.h>
#include "ml_model/mlp_model.h"

extern void mlp_reset_timebase(void);
extern int32_t cnn_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC);

int main() {
    mlp_reset_timebase();
    // Suppose 10 ohm load: ploss ~ 2000 mW (200000 cmW), I ~ 800 mA (80000 cmA), amb ~ 25 C (2500 cC)
    int32_t ploss = 200000;
    int32_t i_ma = 80000;
    int32_t amb = 2500;
    for (int t = 0; t <= 30; t++) {
        int32_t pred = cnn_predict_temp_cC(ploss, i_ma, amb, amb);
        if (t % 5 == 0) {
            printf("t=%d min -> pred=%.2f C\n", t, (double)pred/100.0);
        }
        for (int step = 0; step < 120; step++) {
            cnn_predict_temp_cC(ploss, i_ma, amb, amb);
        }
    }
    return 0;
}
