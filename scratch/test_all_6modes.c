#include <stdio.h>
#include <stdint.h>
#include "mlp_model_new.h"

int main() {
    // Test conditions:
    // 1. Cold start / Standby: Ploss=0, I=0, Tamb=25.0C, Ths=25.0C
    // 2. Active 10-ohm heating (mid-heating): Ploss=2200 mW, I=780 mA, Tamb=25.0C, Ths=45.0C
    // 3. Active 20-ohm heating: Ploss=1300 mW, I=390 mA, Tamb=25.0C, Ths=45.0C
    // 4. Equilibrium at 73C: Ploss=2200 mW, I=780 mA, Tamb=25.0C, Ths=73.0C
    // 5. Unpowered cooling at 55C: Ploss=0, I=0, Tamb=25.0C, Ths=55.0C

    int32_t tests[5][4] = {
        {0, 0, 2500, 2500},
        {220000, 78000, 2500, 4500},
        {130000, 39000, 2500, 4500},
        {220000, 78000, 2500, 7300},
        {0, 0, 2500, 5500}
    };

    printf("Mode %d:\n", BENCHMARK_MODE);
    for (int i = 0; i < 5; i++) {
        int32_t p = mlp_new_predict_temp_cC(tests[i][0], tests[i][1], tests[i][2], tests[i][3]);
        int32_t hs = tests[i][3];
        printf("  Test %d: Ths=%.2f -> Pred=%.2f C (dT=%+.2f C)\n", 
               i+1, (double)hs/100.0, (double)p/100.0, (double)(p - hs)/100.0);
    }
    return 0;
}
