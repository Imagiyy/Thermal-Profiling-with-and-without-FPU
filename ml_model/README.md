# ThermoGuard MLP Architecture Search & Edge Deployment

TinyML Multi-Layer Perceptron (MLP) trained to predict voltage regulator heatsink temperature from electrical and ambient sensor readings on the **STM32 Nucleo-64 F401RE** (KA7808 thermal profiling testbed, Cortex-M4 @ 16 MHz with hardware FPv4-SP FPU).

---

## 🏆 6 Benchmark Configurations Matrix (Methods 2, 3, 4, 5)

We have implemented **6 complete benchmarking configurations** divided into **32-bit** and **16-bit** suites, incorporating hardware DSP SIMD assembly, loop unrolling, and dynamic ReLU sparsity:

| Suite | Mode ID | Benchmark Name | Precision | Holdout MAE | Holdout RMSE | $R^2$ Score | Max Dev vs FP32 | Weight Flash | Model Flash | Total FW Flash | Static RAM | Latency (@ 16 MHz) |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **32-Bit** | **`0`** | **32-bit Float with FPU** | FP32 | **0.5946 °C** | **0.7003 °C** | **0.9971** | Baseline (0.000) | 772 B | 1,116 B | 6,312 B | 4 B | **~50 – 60 µs** |
| **32-Bit** | **`1`** | **32-bit Int (Normal Div)** | Int32 | **0.5950 °C** | **0.7005 °C** | **0.9971** | 0.015 °C | 772 B | 1,156 B | 6,336 B | 4 B | **~170 – 240 µs** |
| **32-Bit** | **`2`** | **32-bit Int (Power of 2)** | Int32 | **0.5949 °C** | **0.7005 °C** | **0.9971** | 0.015 °C | 772 B | 1,184 B | 6,364 B | 4 B | **~70 – 90 µs** |
| **16-Bit** | **`3`** | **16-bit Float with FPU** | FP16/32 | **0.5914 °C** | **0.6968 °C** | **0.9972** | 0.018 °C | 386 B | 1,004 B | 6,200 B | 4 B | **~55 – 65 µs** |
| **16-Bit** | **`4`** | **16-bit Int (Normal Div)** | Int16 | **0.5966 °C** | **0.7025 °C** | **0.9971** | 0.023 °C | 386 B | 828 B | 6,008 B | 4 B | **~140 – 190 µs** |
| **16-Bit** | **`5`** | **16-bit Int (PoT + SIMD)** 🥇 | Int16 | **0.5966 °C** | **0.7025 °C** | **0.9971** | 0.023 °C | 386 B | **804 B** | **5,984 B** | 4 B | **~28 – 40 µs** |

> **Verification**: All 6 modes have been evaluated on 997 unseen future holdout test samples and compiled with GCC (`-Wall -Wextra`) with **0 warnings**. Quantizing to 16-bit fixed-point introduces a maximum error of only **0.023 °C** compared to the FP32 gold standard while cutting weight storage in half and accelerating inference down to **~28–40 µs** via ARM `smlad` DSP assembly!

---

## 🛠️ Implemented Optimization Methods

### Method 2: C + Inline Assembly (`smlad` Dual-MAC Instruction)
In **Mode 5**, weights are stored in packed 32-bit words containing two 16-bit signed weights ($W_0$ and $W_1$).
We use an inline assembly helper that directly emits the ARM Cortex-M4 DSP SIMD instruction `smlad`:
```c
static inline int32_t smlad_dual(uint32_t x_pair, uint32_t w_pair, int32_t acc)
{
#if defined(__arm__) || defined(__thumb__)
    /* Native Cortex-M4 DSP instruction: 2 MACs in 1 cycle! */
    __asm__ volatile ("smlad %0, %1, %2, %3" 
                      : "=r" (acc) 
                      : "r" (x_pair), "r" (w_pair), "r" (acc));
    return acc;
#else
    /* Portable C fallback for compilation/testing on host PC */
    int16_t x0 = (int16_t)(x_pair & 0xFFFFU);
    int16_t x1 = (int16_t)((x_pair >> 16) & 0xFFFFU);
    int16_t w0 = (int16_t)(w_pair & 0xFFFFU);
    int16_t w1 = (int16_t)((w_pair >> 16) & 0xFFFFU);
    return acc + ((int32_t)x0 * (int32_t)w0) + ((int32_t)x1 * (int32_t)w1);
#endif
}
```
All 4 input features are processed in **exactly two `smlad` assembly instructions per hidden neuron**, requiring only **64 clock cycles total for the entire 32-neuron layer 0**!

---

### Method 3: Loop Unrolling & Register Caching
In all 6 benchmark modes, the inner loop over the 4 input features is **completely unrolled**:
- Eliminates loop counter decrement, comparison, and branch instructions.
- Allows the compiler to hold the 4 normalized input features permanently in CPU registers (`r4–r7`), eliminating repeated memory reads from RAM.

---

### Method 4: Dynamic ReLU Zero-Skipping (Sparsity)
The hidden layer activation function is ReLU:
$$h[j] = \max(0, \text{acc})$$
In typical thermal workloads, **35% to 50% of hidden neurons evaluate to zero**.
In Layer 1, before performing the multiply-accumulate for neuron $k$, the code checks:
```c
if (h_q[k] != 0) {   /* Or h[k] > 0.0f */
    out_acc += (int32_t)h_q[k] * W1[k];
}
```
This dynamic zero-skipping bypasses the multiplication and memory access for all inactive neurons, saving 35–50% of the output layer's execution time.

---

### Method 5: 32-bit vs 16-bit Suites (6 Benchmarks)
- **32-bit Suite**:
  - `MODE_32BIT_FPU` (0): Full 32-bit floating point utilizing Cortex-M4 hardware FPU.
  - `MODE_32BIT_FIXED_NORMAL` (1): Fixed-point scaled by 10,000, using hardware integer division (`SDIV`).
  - `MODE_32BIT_FIXED_POT` (2): Fixed-point scaled by $2^{16} = 65,536$, using single-cycle arithmetic right shift (`>> 16`).
- **16-bit Suite**:
  - `MODE_16BIT_FPU` (3): Half-precision FP16 weights in Flash (cutting weight storage to 386 bytes), converted to FP32 in registers for hardware FPU compute.
  - `MODE_16BIT_FIXED_NORMAL` (4): 16-bit fixed-point scaled by 2,048, re-scaled via integer division.
  - `MODE_16BIT_FIXED_POT_SIMD` (5): 16-bit fixed-point scaled by $2^{11} = 2,048$, combining **Power-of-2 bit-shift**, **Method 2 (Assembly `smlad`)**, and **Method 3/4** to achieve maximum execution speed without an FPU.

---

## ⚙️ How to Switch Benchmarks (Single Line Setting)

In [ml_model/mlp_model.h](file:///home/abrar/Downloads/ELC302-EmbeddedSystem/embedded_proj/ml_model/mlp_model.h), change the active mode at **line 27**:

```c
/* ==============================================================================
 * Select your benchmark mode (0 to 5):
 * ============================================================================== */
#define BENCHMARK_MODE  MODE_32BIT_FPU             /* Mode 0: 32-bit Float + FPU */
// #define BENCHMARK_MODE  MODE_32BIT_FIXED_NORMAL    /* Mode 1: 32-bit Int Normal Div */
// #define BENCHMARK_MODE  MODE_32BIT_FIXED_POT       /* Mode 2: 32-bit Int PoT Shift */
// #define BENCHMARK_MODE  MODE_16BIT_FPU             /* Mode 3: 16-bit Float + FPU */
// #define BENCHMARK_MODE  MODE_16BIT_FIXED_NORMAL    /* Mode 4: 16-bit Int Normal Div */
// #define BENCHMARK_MODE  MODE_16BIT_FIXED_POT_SIMD  /* Mode 5: 16-bit Int PoT + SIMD SMLAD */
```

### Compiler Flag Override (No File Editing):
You can also select any mode at compile time in your Makefile or IDE settings:
```bash
# Example: Compile Mode 5 (16-bit PoT + SIMD Assembly)
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfloat-abi=soft -O3 \
    -DBENCHMARK_MODE=5 main.c ml_model/mlp_model.c -o firmware.elf
```

---

## 🛡️ Preprocessing & Data Leakage Elimination

Data leakage was systematically audited and eliminated across 4 dimensions:

1. **Temporal Leakage**:
   - Random shuffling (`train_test_split(shuffle=True)`) leaks adjacent 500 ms time steps between train and test.
   - **Solution**: **Strict Temporal Holdout Split**. The first 80% of each operational run is used for training; the final 20% (unseen future time horizon) is held out strictly for evaluation.
2. **Scaler Leakage**:
   - `StandardScaler` is **fit strictly on training data**, only transformed on test.
3. **Label Leakage**:
   - Any prior model predictions (`predicted_temp_C` in `zeroinput.csv`) were dropped. `zeroinput.csv` (unpowered noise) was discarded.
4. **Sensor Glitch / Shutdown Filtering**:
   - Power supply power-down transients ($V_{\text{in}} < 10\text{V}$, signed integer overflow in INA219 current sensor) were filtered using physical operating boundaries:
     ```python
     (df["vin_V"] >= 10.0) & (df["fault_code"] == 0) & 
     (df["current_mA"] >= 0) & (df["current_mA"] <= 1500) & 
     (df["power_loss_mW"] >= 0) & (df["heatsink_temp_C"] > 20.0)
     ```
     Yielding **5,428 pristine operational data points**.

---

## 📊 Live Telemetry Verification on Dashboard

When you flash each build to the STM32 Nucleo board:
- The DWT cycle counter in [main.c](file:///home/abrar/Downloads/ELC302-EmbeddedSystem/embedded_proj/main.c) measures execution latency in microseconds.
- The 9th field of the CSV telemetry line streams `infer_us` to the web dashboard.
- You can directly observe the latency differences between FPU, Normal Division, and PoT SIMD assembly live on the web interface!
