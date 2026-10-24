# ThermoGuard — TinyML Thermal Profiling Dashboard

Real-time telemetry and edge TinyML inference benchmarking dashboard for the **STM32 Nucleo-64 F401RE** (KA7808 voltage regulator thermal profiling), featuring an MLP model trained from sensor data and deployed as pure C for bare-metal inference.

---

## 🚀 Quick Start

### 1. Install Dependencies

Ensure Python 3.8+ is installed, then install the required packages:

```bash
pip install -r requirements.txt
pip install -r ml_model/requirements.txt
```

*(Core: `fastapi`, `uvicorn[standard]`, `pyserial` — ML: `numpy`, `pandas`, `scikit-learn`, `matplotlib`)*

---

### 2. Train the ML Model (First Time Setup)

```bash
cd ml_model
python train_mlp.py
```

This trains an MLP on your sensor dataset and auto-generates `mlp_model.c` / `mlp_model.h` for STM32 deployment. See the [ML Model section](#-ml-model-training--deployment) below for full details.

---

### 3. Start the Server

From the project root directory, run:

```bash
uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
```

*Alternatively, run directly with Python:*
```bash
python3 backend/server.py
```

---

### 4. Open the Dashboard

Open your web browser and navigate to:

👉 **[http://localhost:8000](http://localhost:8000)** (or `http://127.0.0.1:8000`)

---

## 🔌 Connecting Data Streams

### Option A: Physical STM32 Nucleo Board

1. Plug your **STM32 Nucleo-F401RE** into your computer via USB.
2. Open the dashboard in your browser.
3. In the top navigation bar:
   - Click the **Refresh** icon next to **Port** if your device is not listed.
   - Select your board port (e.g., `/dev/ttyACM0` on Linux, `COM3` on Windows).
   - Set Baud Rate to **`115200`**.
   - Click **Connect**.

> **Linux Port Permission Note:**  
> If you get `Permission Denied` on `/dev/ttyACM0`, grant access with:  
> ```bash
> sudo usermod -a -G dialout $USER
> # or temporarily:
> sudo chmod 666 /dev/ttyACM0
> ```
> *(Log out and log back in for dialout group changes to take effect).*

---

### Option B: Virtual Simulation (No Hardware Needed)

If your physical board is disconnected or not yet flashed:

1. Select **`VIRTUAL_STM32`** from the **Port** dropdown.
2. Click **Connect**.
3. The dashboard will immediately stream realistic 10-field telemetry at 500 ms intervals with live DWT inference benchmarking (~245 µs).

---

## 📊 Dashboard Features

* **Prediction Time**: Real-time TinyML inference latency measured directly on Cortex-M4 hardware via the **DWT cycle counter** ($\mu\text{s}$ and $\text{ms}$).
* **Avg Pred. Time**: Cumulative session mean inference latency.
* **Thermal Tracking Chart**: Live rolling 120s window plotting:
  - $T_{\text{measured}}$ (MAX31855 thermocouple on regulator heatsink)
  - $T_{\text{predicted}}$ (TinyML model output)
  - $T_{\text{ambient}}$ (BME280 room temperature)
* **Predicted Temp KPI**: Real-time TinyML model output ($T_{\text{predicted}}$ in $^\circ\text{C}$).
* **Heatsink Temp KPI & Safety Zone**: Measured temperature ($T_{\text{measured}}$) with active warning and critical trip zones.
* **Tracking Error Metrics**: Instantaneous tracking error, session **MAE**, and **RMSE**.
* **Reset & Session Management**: Click **Reset** to clear all charts, counters, and statistics for a fresh session.
* **Export CSV**: Click **Export CSV** at any time to export the full in-memory session dataset.

---

## 🧠 TinyML 6-Benchmark Suite (32-Bit & 16-Bit Quantization)

The project implements **6 distinct inference benchmark configurations** spanning full 32-bit floating-point and integer quantization down to 16-bit half-precision and fixed-point with ARM Cortex-M4 DSP SIMD assembly optimizations:

### 📊 Model Accuracy & Digital Footprint Comparison

All metrics evaluated on the **unseen future holdout test set** (997 continuous operational telemetry samples across 10Ω, 20Ω, and no-load conditions under strict zero-data-leakage constraints):

| Mode | Benchmark Name | Precision | Holdout MAE | Holdout RMSE | $R^2$ Score | Max Error | Max Dev vs FP32 | Weight Flash | Model Flash | Total FW Flash | Static RAM | Latency (@ 16 MHz) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`0`** | **32-bit Float with FPU** | FP32 | **0.5946 °C** | **0.7003 °C** | **0.9971** | 1.81 °C | Baseline (0.000) | 772 B | 1,116 B | 6,312 B | 4 B | **~50 – 60 µs** |
| **`1`** | **32-bit Int Without FPU (Normal Div)** | Int32 | **0.5950 °C** | **0.7005 °C** | **0.9971** | 1.82 °C | 0.015 °C | 772 B | 1,156 B | 6,336 B | 4 B | **~170 – 240 µs** |
| **`2`** | **32-bit Int Without FPU (Power-of-2)** | Int32 | **0.5949 °C** | **0.7005 °C** | **0.9971** | 1.81 °C | 0.015 °C | 772 B | 1,184 B | 6,364 B | 4 B | **~70 – 90 µs** |
| **`3`** | **16-bit Float with FPU (FP16/FP32)** | FP16/32 | **0.5914 °C** | **0.6968 °C** | **0.9972** | 1.79 °C | 0.018 °C | 386 B | 1,004 B | 6,200 B | 4 B | **~55 – 65 µs** |
| **`4`** | **16-bit Int Without FPU (Normal Div)** | Int16 | **0.5966 °C** | **0.7025 °C** | **0.9971** | 1.82 °C | 0.023 °C | 386 B | 828 B | 6,008 B | 4 B | **~140 – 190 µs** |
| **`5`** | **16-bit Int (PoT + ARM SMLAD SIMD)** 🥇 | Int16 | **0.5966 °C** | **0.7025 °C** | **0.9971** | 1.82 °C | 0.023 °C | 386 B | **804 B** | **5,984 B** | 4 B | **~28 – 40 µs** |

#### Key Engineering Insights:
* **Negligible Quantization Loss:** Quantizing weights from 32-bit float to 16-bit integer introduces only **$0.023^\circ\text{C}$ maximum deviation** compared to the FP32 baseline, maintaining $R^2 = 0.9971$ across all modes.
* **50% Flash Weight Reduction:** The 16-bit suite (Modes 3, 4, 5) cuts weight storage in half from **772 bytes to 386 bytes**.
* **Smallest Footprint & Lowest Latency (Mode 5):** Combining 16-bit Power-of-2 bit-shift with ARM Cortex-M4 DSP inline assembly (`SMLAD`) delivers the smallest flash footprint (**804 B** isolated ML model / **5,984 B** total firmware) while achieving the lowest cycle latency (**~28–40 µs** at 16 MHz).
* **Zero Dynamic Allocation:** All 6 modes require **0 bytes heap** (`malloc`/`free` strictly avoided) and only **4 bytes static RAM** (`g_sample_count`).

---

### 🚀 How to Run Each Benchmark Mode

#### Option 1: Automated Script (Fastest — 1 Command)
Use the included automated build and flash script [`build_firmware.sh`](build_firmware.sh):

```bash
# 32-Bit Suite:
./build_firmware.sh 0 --flash    # Mode 0: 32-bit Float + Hardware FPU
./build_firmware.sh 1 --flash    # Mode 1: 32-bit Fixed-Point (Normal Div, no FPU)
./build_firmware.sh 2 --flash    # Mode 2: 32-bit Fixed-Point (Power-of-2 Shift, no FPU)

# 16-Bit Suite:
./build_firmware.sh 3 --flash    # Mode 3: 16-bit Float + Hardware FPU (FP16 storage)
./build_firmware.sh 4 --flash    # Mode 4: 16-bit Fixed-Point (Normal Div, no FPU)
./build_firmware.sh 5 --flash    # Mode 5: 16-bit Fixed-Point (PoT + ARM SMLAD SIMD Assembly)
```
*(Omit `--flash` if you only want to compile without flashing immediately, e.g. `./build_firmware.sh 5`).*

#### Option 2: Inside STM32CubeIDE (GUI)
1. In your project, open [`Core/Inc/mlp_model.h`](ml_model/mlp_model.h).
2. Edit line 37 to select the active mode:
   ```c
   #ifndef BENCHMARK_MODE
   #define BENCHMARK_MODE    MODE_16BIT_FIXED_POT_SIMD   /* Change to 0, 1, 2, 3, 4, or 5 */
   #endif
   ```
3. Click **Build (Hammer icon)** and **Run (Play icon)** to flash to your Nucleo board.

---

### 📊 Comparing Performance on the Live Dashboard
1. Open `http://localhost:8000` and click **Connect**.
2. Observe the top **PREDICTION TIME** KPI badge:
   - Notice how **Mode 5 (ARM SMLAD SIMD)** runs in only **~30 µs**, while **Mode 1 (Normal division)** requires **~200 µs**!
   - All 6 modes output matching thermal predictions (within 0.15 °C) while providing vastly different execution latencies!

1. Add new CSV files to `dataset/` (same column format as existing files)
2. Run `python ml_model/train_mlp.py`
3. Copy the regenerated `mlp_model.c` / `.h` to your STM32 project
4. Rebuild and re-flash

For full details, see [`ml_model/README.md`](ml_model/README.md).

---

## 📁 Project Structure

```text
├── backend/
│   └── server.py          # FastAPI + Uvicorn server, WebSocket broadcaster, serial reader
├── frontend/
│   ├── index.html         # Tailwind CSS + Chart.js responsive dashboard UI
│   └── app.js             # WebSocket client, KPI calculations, dynamic charts
├── dataset/               # Thermal profiling CSV datasets (10Ω, 20Ω, no-load, zero-input)
├── ml_model/
│   ├── train_mlp.py       # MLP training pipeline (load → train → evaluate → export C)
│   ├── mlp_model.c        # Auto-generated C inference code for STM32F401RE
│   ├── mlp_model.h        # Auto-generated C header (function prototype)
│   ├── mlp_model.joblib   # Serialised sklearn model + scaler
│   ├── mlp_weights.json   # Raw weights/biases (JSON)
│   ├── scaler_params.json # StandardScaler normalisation parameters
│   ├── metrics.json       # Train/test evaluation metrics
│   ├── training_report.png# Evaluation plots
│   ├── requirements.txt   # ML Python dependencies
│   └── README.md          # Detailed ML model documentation
├── logs/                  # Timestamped telemetry CSV recordings
├── main.c                 # Bare-metal STM32F401RE C firmware (10-field CSV, DWT benchmarking)
├── logfile.py             # Standalone Python CLI logger for terminal recording
├── requirements.txt       # Python package dependencies (server)
└── README.md              # Project documentation (this file)
```

---

## 📟 Standalone CLI Logger (Optional)

To log telemetry directly to a CSV file from the terminal without the web UI:

```bash
python logfile.py /dev/ttyACM0 --baud 115200 -o my_run.csv
```
List available ports:
```bash
python logfile.py --list
```
