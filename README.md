# ThermoGuard — TinyML Thermal Profiling Dashboard

Real-time telemetry and edge TinyML inference benchmarking dashboard for the **STM32 Nucleo-64 F401RE** (KA7808 voltage regulator thermal profiling).

---

## 🚀 Quick Start

### 1. Install Dependencies

Ensure Python 3.8+ is installed, then install the required packages:

```bash
pip install -r requirements.txt
```

*(Packages installed: `fastapi`, `uvicorn[standard]`, `pyserial`)*

---

### 2. Start the Server

From the project root directory, run:

```bash
uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
```

*Alternatively, run directly with Python:*
```bash
python3 backend/server.py
```

---

### 3. Open the Dashboard

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

## 📁 Project Structure

```text
├── backend/
│   └── server.py          # FastAPI + Uvicorn server, WebSocket broadcaster, serial reader
├── frontend/
│   ├── index.html         # Tailwind CSS + Chart.js responsive dashboard UI
│   └── app.js             # WebSocket client, KPI calculations, dynamic charts
├── logs/                  # Timestamped telemetry CSV recordings
├── main.c                 # Bare-metal STM32F401RE C firmware (10-field CSV, DWT benchmarking)
├── logfile.py             # Standalone Python CLI logger for terminal recording
├── requirements.txt       # Python package dependencies
└── README.md              # Project documentation
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
