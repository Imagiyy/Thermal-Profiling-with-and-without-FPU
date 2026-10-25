#include <stdio.h>
#include <stdint.h>
#include "mlp_model_new.h"

int main() {
    int32_t amb = 2500;
    for (int hs = 2500; hs <= 9000; hs += 1000) {
        int32_t pred = mlp_new_predict_temp_cC(hs, amb);
        printf("%d->%d ", hs, pred);
    }
    printf("\n");
    return 0;
}
