#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "ml_model/mlp_model.h"

extern void mlp_reset_timebase(void);
extern int32_t cnn_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC);

int main() {
    FILE *f = fopen("scratch/test_inputs.csv", "r");
    if (!f) return 1;
    char line[256];
    if (!fgets(line, sizeof(line), f)) return 1; // header
    
    FILE *out = fopen("scratch/test_preds.csv", "w");
    fprintf(out, "pred_temp_C\n");

    mlp_reset_timebase();
    double ploss, i_ma, amb, elapsed, target;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "%lf,%lf,%lf,%lf,%lf", &ploss, &i_ma, &amb, &elapsed, &target) == 5) {
            int32_t ploss_cmW = (int32_t)(ploss * 100.0);
            int32_t i_cmA = (int32_t)(i_ma * 100.0);
            int32_t amb_cC = (int32_t)(amb * 100.0);
            int32_t pred_cC = cnn_predict_temp_cC(ploss_cmW, i_cmA, amb_cC, amb_cC);
            fprintf(out, "%.2f\n", (double)pred_cC / 100.0);
        }
    }
    fclose(f);
    fclose(out);
    return 0;
}
