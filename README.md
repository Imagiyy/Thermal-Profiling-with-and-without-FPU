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

## 🧠 ML Model Training & Deployment

### Overview

The `ml_model/` folder contains a complete pipeline to:
1. **Train** an MLP regressor (scikit-learn) on your thermal dataset
2. **Evaluate** with plots and metrics
3. **Export** the trained model as pure C code for STM32F401RE

### Architecture

```
Input (3 features)  →  Dense(16, ReLU)  →  Dense(8, ReLU)  →  Dense(1)  →  Predicted Temp
   power_loss_mW                                                              heatsink_temp_C
   current_mA
   ambient_temp_C
```

Total: **209 parameters** — fits comfortably in flash with ~1 KB footprint.

### Training

```bash
cd ml_model
python train_mlp.py
```

**Custom options:**
```bash
python train_mlp.py --epochs 5000              # more iterations
python train_mlp.py --hidden 32,16,8           # bigger network
python train_mlp.py --test-size 0.3            # 70/30 split
python train_mlp.py --no-export                # skip C export
```

### Deploying to STM32

After training, two C files are auto-generated: `mlp_model.h` and `mlp_model.c`.

**Step 1:** Copy them next to `main.c`:
```bash
cp ml_model/mlp_model.h  .
cp ml_model/mlp_model.c  .
```

**Step 2:** Add to your build. For bare-metal GCC:
```bash
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 \
    -O2 -Wall \
    -T linker_script.ld \
    main.c mlp_model.c \
    -o thermoguard.elf \
    -nostartfiles -lm
```

**Step 3:** Flash via ST-Link:
```bash
st-flash write thermoguard.bin 0x08000000
```

> **Important:** The generated `cnn_predict_temp_cC()` function overrides the `__attribute__((weak))` stub in `main.c` automatically — **no firmware code changes needed!**

> **FPU flags are mandatory:** `-mfloat-abi=hard -mfpu=fpv4-sp-d16` — without these, the compiler uses slow software floating-point.

### Re-training After New Data

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
