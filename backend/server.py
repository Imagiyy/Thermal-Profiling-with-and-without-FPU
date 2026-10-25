#!/usr/bin/env python3
"""
server.py – Project ThermoGuard telemetry backend.

FastAPI + Uvicorn server that:
  1. Scans Linux serial ports and exposes them via REST.
  2. Opens a serial connection in a background thread, parses
     ASCII CSV frames from the STM32 Nucleo-F401RE, and broadcasts
     validated telemetry JSON over WebSockets.
  3. Optionally records incoming frames to a timestamped CSV file.

Protocol frame (10 fields, ASCII CSV, 115200 baud, ~500 ms interval):
  timestamp_ms, V_in, V_out, I_mA, P_mW, T_ambient, T_measured, T_predicted, infer_us, fault_code

Run:
    uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
from collections import deque
import csv
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import serial
import serial.tools.list_ports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("thermoguard")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIELD_NAMES = [
    "timestamp_ms",
    "V_in",
    "V_out",
    "I_mA",
    "P_mW",
    "T_ambient",
    "T_measured",
    "T_predicted",
    "infer_us",
    "fault_code",
]
NUM_FIELDS = len(FIELD_NAMES)

# Comprehensive fields written when exporting / recording CSV
EXPORT_FIELD_NAMES = [
    "timestamp_ms",
    "iso_time",
    "V_in",
    "V_out",
    "V_drop",
    "I_mA",
    "P_mW",
    "T_ambient",
    "T_measured",
    "T_predicted",
    "infer_us",
    "tracking_error",
    "R_theta_JA",
    "fault_code",
    "fault_flags",
]

# Fault bitmask decoding (MAX31855)
FAULT_FLAGS = {
    0: "FAULT",
    1: "SCG",
    2: "SCV",
    3: "OC",
}

# Default serial settings
DEFAULT_BAUD = 115_200

# CSV log directory
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

# Frontend directory (served as static files)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# ---------------------------------------------------------------------------
# Shared State
# ---------------------------------------------------------------------------

class TelemetryState:
    """Thread-safe shared state between the serial reader and async WS layer."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.serial_conn: serial.Serial | None = None
        self.reader_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.connected = False
        self.port: str = ""
        self.baud: int = DEFAULT_BAUD

        # Recording
        self.recording = False
        self.csv_file: Any = None
        self.csv_writer: Any = None
        self.csv_path: str = ""
        self.record_start: float = 0.0
        self.record_samples: int = 0

        # WebSocket subscribers
        self.ws_clients: set[WebSocket] = set()

        # Latest frame (for new clients)
        self.latest_frame: dict | None = None

        # Async event loop reference (set at startup)
        self.loop: asyncio.AbstractEventLoop | None = None

        # Stats
        self.total_frames: int = 0
        self.malformed_frames: int = 0

        # Anti-glitch freeze filter state (freezes display if temp fluctuates >= 10.0 °C for <= 2 samples)
        self.clean_t_meas: float | None = None
        self.meas_glitch_cnt: int = 0
        self.clean_t_pred: float | None = None
        self.pred_glitch_cnt: int = 0

        # 3-second lookahead prediction verification buffer (6 samples @ 500 ms)
        self.pred_buffer: deque[float] = deque(maxlen=7)

    def filter_temperatures(self, t_meas: float, t_pred: float) -> tuple[float, float]:
        """
        Anti-glitch freezing filter:
        If temperature fluctuates >= 10.0 °C for <= 2 samples (or drops to <= 0 °C on SPI error),
        freeze display to the last valid stable temperature.
        Only accept change if it persists for >= 3 consecutive samples.
        """
        with self.lock:
            # 1. Measured heatsink filter
            if self.clean_t_meas is None:
                if t_meas > 0.0:
                    self.clean_t_meas = t_meas
                out_meas = t_meas
            else:
                is_meas_glitch = (t_meas <= 0.0) or (abs(t_meas - self.clean_t_meas) >= 10.0)
                if is_meas_glitch:
                    self.meas_glitch_cnt += 1
                    if self.meas_glitch_cnt <= 2:
                        out_meas = self.clean_t_meas  # Freeze display to last stable reading
                    else:
                        self.clean_t_meas = t_meas
                        self.meas_glitch_cnt = 0
                        out_meas = t_meas
                else:
                    self.clean_t_meas = t_meas
                    self.meas_glitch_cnt = 0
                    out_meas = t_meas

            # 2. Predicted temperature filter
            if self.clean_t_pred is None:
                if t_pred > 0.0:
                    self.clean_t_pred = t_pred
                out_pred = t_pred
            else:
                # If t_pred is actively closing the gap towards measured heatsink temp, accept immediately
                if self.clean_t_meas is not None and self.clean_t_pred < self.clean_t_meas and t_pred > self.clean_t_pred and t_pred <= (self.clean_t_meas + 2.0):
                    self.clean_t_pred = t_pred
                    self.pred_glitch_cnt = 0
                    out_pred = t_pred
                else:
                    is_pred_glitch = (t_pred <= 0.0) or (abs(t_pred - self.clean_t_pred) >= 10.0)
                    if is_pred_glitch:
                        self.pred_glitch_cnt += 1
                        if self.pred_glitch_cnt <= 2:
                            out_pred = self.clean_t_pred  # Freeze display to last stable reading
                        else:
                            self.clean_t_pred = t_pred
                            self.pred_glitch_cnt = 0
                            out_pred = t_pred
                    else:
                        self.clean_t_pred = t_pred
                        self.pred_glitch_cnt = 0
                        out_pred = t_pred

            return round(out_meas, 2), round(out_pred, 2)

    def reset_filters(self) -> None:
        with self.lock:
            self.clean_t_meas = None
            self.meas_glitch_cnt = 0
            self.clean_t_pred = None
            self.pred_glitch_cnt = 0


state = TelemetryState()


# ---------------------------------------------------------------------------
# Frame parsing
# ---------------------------------------------------------------------------

def decode_faults(code: int) -> list[str]:
    """Decode MAX31855 fault bitmask into human-readable flag list."""
    active: list[str] = []
    for bit, name in FAULT_FLAGS.items():
        if code & (1 << bit):
            active.append(name)
    return active


def parse_frame(raw_line: str) -> dict | None:
    """
    Parse a single CSV telemetry line into a validated dict.
    Supports both 10-field (with DWT inference time) and legacy 9-field formats.
    Returns None on malformed input (caller can safely skip).
    """
    parts = [p.strip() for p in raw_line.split(",")]
    if len(parts) not in (9, 10):
        return None
    try:
        ts = int(parts[0])
        v_in = float(parts[1])
        v_out = float(parts[2])
        i_ma = float(parts[3])
        p_mw = float(parts[4])
        t_amb = float(parts[5])
        t_meas = float(parts[6])
        t_pred = float(parts[7])
        if len(parts) == 10:
            infer_us = float(parts[8])
            fault = int(parts[9], 0)  # supports hex 0x prefix
        else:
            infer_us = 0.0
            fault = int(parts[8], 0)
    except (ValueError, IndexError):
        return None

    # Software Rule: Vin threshold (< 8.5 V -> regulator unpowered / cooling phase)
    if v_in < 8.5:
        i_ma = 0.0
        p_mw = 0.0
        v_drop = 0.0
    else:
        v_drop = round(max(0.0, v_in - v_out), 3)

    # Anti-glitch freeze filter: suppress temperature fluctuations >= 10.0 °C lasting <= 2 samples
    t_meas, t_pred = state.filter_temperatures(t_meas, t_pred)

    # 3-second lookahead error (evaluates prediction made 6 samples = 3.0s ago against current measurement)
    with state.lock:
        state.pred_buffer.append(t_pred)
        if len(state.pred_buffer) >= 6:
            past_pred = state.pred_buffer[0]
            tracking_error = round(abs(t_meas - past_pred), 3)
        else:
            tracking_error = round(abs(t_meas - t_pred), 3)

    return {
        "timestamp_ms": ts,
        "V_in": round(v_in, 3),
        "V_out": round(v_out, 3),
        "I_mA": round(i_ma, 2),
        "P_mW": round(p_mw, 2),
        "T_ambient": round(t_amb, 2),
        "T_measured": round(t_meas, 2),
        "T_predicted": round(t_pred, 2),
        "infer_us": round(infer_us, 1),
        "fault_code": fault,
        "faults": decode_faults(fault),
        "V_drop": v_drop,
        "tracking_error": tracking_error,
        "R_theta_JA": round(
            (t_meas - t_amb) / (p_mw / 1000.0), 3
        ) if p_mw > 0 else None,
        "server_ts": time.time(),
    }


# ---------------------------------------------------------------------------
# Async broadcast helper
# ---------------------------------------------------------------------------

async def _broadcast(payload: dict) -> None:
    """Send JSON payload to every connected WebSocket client."""
    dead: list[WebSocket] = []
    for ws in list(state.ws_clients):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        state.ws_clients.discard(ws)


def broadcast(payload: dict) -> None:
    """Schedule an async broadcast from a sync context (reader thread)."""
    if state.loop is not None:
        asyncio.run_coroutine_threadsafe(_broadcast(payload), state.loop)


# ---------------------------------------------------------------------------
# Serial reader thread
# ---------------------------------------------------------------------------

def serial_reader() -> None:
    """Background thread: read lines, parse, broadcast, optionally record."""
    log.info("Serial reader started on %s @ %d", state.port, state.baud)
    ser = state.serial_conn
    assert ser is not None

    try:
        ser.reset_input_buffer()
    except Exception:
        pass

    while not state.stop_event.is_set():
        try:
            raw = ser.readline()
        except serial.SerialException as exc:
            log.error("Serial read error: %s", exc)
            broadcast({"type": "error", "message": f"Serial error: {exc}"})
            break
        except Exception:
            break

        if not raw:
            continue

        try:
            line = raw.decode("utf-8", errors="replace").strip()
        except Exception:
            continue

        if not line or line.startswith("#"):
            continue

        frame = parse_frame(line)
        if frame is None:
            state.malformed_frames += 1
            log.debug("Malformed frame skipped: %s", line[:80])
            continue

        state.total_frames += 1
        state.latest_frame = frame

        # Broadcast to WebSocket subscribers
        broadcast({"type": "telemetry", **frame})

        # CSV recording
        with state.lock:
            if state.recording and state.csv_writer is not None:
                try:
                    state.csv_writer.writerow([
                        frame["timestamp_ms"],
                        datetime.now().isoformat(),
                        frame["V_in"],
                        frame["V_out"],
                        frame["V_drop"],
                        frame["I_mA"],
                        frame["P_mW"],
                        frame["T_ambient"],
                        frame["T_measured"],
                        frame["T_predicted"],
                        frame["infer_us"],
                        frame["tracking_error"],
                        frame["R_theta_JA"] if frame["R_theta_JA"] is not None else "",
                        frame["fault_code"],
                        "|".join(frame["faults"]) if frame["faults"] else "OK",
                    ])
                    state.csv_file.flush()
                    state.record_samples += 1
                except Exception as exc:
                    log.error("CSV write error: %s", exc)

    # Cleanup on exit
    log.info("Serial reader stopped.")
    with state.lock:
        state.connected = False
        try:
            if ser.is_open:
                ser.close()
        except Exception:
            pass
    broadcast({"type": "status", "connected": False, "reason": "reader_stopped"})


def virtual_reader() -> None:
    """
    Virtual telemetry streamer generating realistic STM32 10-field CSV frames:
      timestamp_ms, vin, vout, current, power, amb_temp, hs_temp, pred_temp, infer_us, fault_code
    Runs every 500 ms (matching firmware SAMPLE_PERIOD_MS) for benchmarking and live demonstration.
    """
    import random
    log.info("Virtual STM32 reader started (10-field stream @ 500 ms).")
    sim_t0 = time.time()
    heat = 34.5

    while not state.stop_event.is_set():
        now = time.time()
        elapsed_ms = int((now - sim_t0) * 1000)
        v_in = 12.015 + random.uniform(-0.02, 0.02)
        v_out = 8.005 + random.uniform(-0.01, 0.01)
        i_ma = 245.0 + random.uniform(-3.0, 3.0)
        p_mw = (v_in - v_out) * i_ma
        t_amb = 29.5 + random.uniform(-0.1, 0.1)
        
        # Thermal equilibrium model
        target_heat = t_amb + (p_mw / 1000.0) * 12.8
        heat += (target_heat - heat) * 0.04 + random.uniform(-0.08, 0.08)
        t_meas = heat
        t_pred = heat + random.uniform(-0.35, 0.35)
        # Latency in microseconds benchmarked via Cortex-M4 DWT (~240-265 us)
        infer_us = int(random.gauss(248, 12))
        fault = 0

        line = f"{elapsed_ms},{v_in:.3f},{v_out:.3f},{i_ma:.2f},{p_mw:.2f},{t_amb:.2f},{t_meas:.2f},{t_pred:.2f},{infer_us},{fault}"
        frame = parse_frame(line)
        if frame is not None:
            state.total_frames += 1
            state.latest_frame = frame
            broadcast({"type": "telemetry", **frame})

            with state.lock:
                if state.recording and state.csv_writer is not None:
                    try:
                        state.csv_writer.writerow([
                            frame["timestamp_ms"],
                            datetime.now().isoformat(),
                            frame["V_in"],
                            frame["V_out"],
                            frame["V_drop"],
                            frame["I_mA"],
                            frame["P_mW"],
                            frame["T_ambient"],
                            frame["T_measured"],
                            frame["T_predicted"],
                            frame["infer_us"],
                            frame["tracking_error"],
                            frame["R_theta_JA"] if frame["R_theta_JA"] is not None else "",
                            frame["fault_code"],
                            "|".join(frame["faults"]) if frame["faults"] else "OK",
                        ])
                        state.csv_file.flush()
                        state.record_samples += 1
                    except Exception as exc:
                        log.error("CSV write error: %s", exc)

        time.sleep(0.5)

    log.info("Virtual STM32 reader stopped.")
    with state.lock:
        state.connected = False
    broadcast({"type": "status", "connected": False, "reason": "reader_stopped"})



# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    state.loop = asyncio.get_running_loop()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log.info("ThermoGuard backend ready.")
    yield
    # Shutdown
    _disconnect()
    log.info("ThermoGuard backend shutdown.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Project ThermoGuard – Telemetry Server",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    """Serve the frontend index.html."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return JSONResponse({"error": "Frontend not found"}, status_code=404)


@app.get("/api/ports")
async def list_ports():
    """Return available serial ports on this Linux host + Virtual Nucleo demo stream."""
    ports = serial.tools.list_ports.comports()
    devs = [
        {
            "device": p.device,
            "description": p.description,
            "hwid": p.hwid,
        }
        for p in sorted(ports, key=lambda p: p.device)
    ]
    # Provide virtual Nucleo simulator option for benchmarking & dashboard testing
    devs.insert(0, {
        "device": "VIRTUAL_STM32",
        "description": "Virtual STM32 Nucleo Stream (10 fields, 500ms)",
        "hwid": "DEMO_STREAM",
    })
    return devs


@app.get("/api/status")
async def get_status():
    """Return current connection and recording status."""
    rec_info = {}
    if state.recording:
        elapsed = time.time() - state.record_start
        rec_info = {
            "recording": True,
            "elapsed_s": round(elapsed, 1),
            "samples": state.record_samples,
            "file": state.csv_path,
        }
    else:
        rec_info = {"recording": False}

    return {
        "connected": state.connected,
        "port": state.port,
        "baud": state.baud,
        "total_frames": state.total_frames,
        "malformed_frames": state.malformed_frames,
        **rec_info,
    }


@app.post("/api/connect")
async def connect(port: str, baud: int = DEFAULT_BAUD):
    """Open a serial connection (or virtual simulator) and start the reader thread."""
    if state.connected:
        return JSONResponse(
            {"error": "Already connected. Disconnect first."},
            status_code=409,
        )

    if port == "VIRTUAL_STM32":
        state.serial_conn = None
        state.port = port
        state.baud = baud
        state.connected = True
        state.stop_event.clear()
        state.total_frames = 0
        state.malformed_frames = 0
        state.reset_filters()

        t = threading.Thread(target=virtual_reader, daemon=True, name="virtual-reader")
        state.reader_thread = t
        t.start()

        log.info("Connected to %s", port)
        broadcast({"type": "status", "connected": True, "port": port, "baud": baud})
        return {"connected": True, "port": port, "baud": baud}

    try:
        ser = serial.Serial(port, baud, timeout=1)
    except serial.SerialException as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    state.serial_conn = ser
    state.port = port
    state.baud = baud
    state.connected = True
    state.stop_event.clear()
    state.total_frames = 0
    state.malformed_frames = 0
    state.reset_filters()

    t = threading.Thread(target=serial_reader, daemon=True, name="serial-reader")
    state.reader_thread = t
    t.start()

    log.info("Connected to %s @ %d", port, baud)
    broadcast({"type": "status", "connected": True, "port": port, "baud": baud})
    return {"connected": True, "port": port, "baud": baud}


def _disconnect() -> None:
    """Internal disconnect logic (used by endpoint and shutdown)."""
    state.stop_event.set()
    _stop_recording()
    if state.serial_conn and state.serial_conn.is_open:
        try:
            state.serial_conn.close()
        except Exception:
            pass
    state.connected = False
    state.serial_conn = None
    state.reset_filters()


@app.post("/api/disconnect")
async def disconnect():
    """Close the serial connection."""
    if not state.connected:
        return JSONResponse({"error": "Not connected."}, status_code=409)
    _disconnect()
    log.info("Disconnected.")
    broadcast({"type": "status", "connected": False, "reason": "user_disconnect"})
    return {"connected": False}


@app.post("/api/reset")
async def reset_session():
    """Reset the session: disconnect serial port, clear frame cache and counters."""
    if state.connected:
        _disconnect()
    with state.lock:
        state.latest_frame = None
        state.total_frames = 0
        state.malformed_frames = 0
    state.reset_filters()
    broadcast({"type": "status", "connected": False, "reason": "reset"})
    log.info("Session reset via API.")
    return {"reset": True, "connected": False}


# ---------------------------------------------------------------------------
# CSV recording endpoints
# ---------------------------------------------------------------------------

def _stop_recording() -> None:
    with state.lock:
        state.recording = False
        if state.csv_file:
            try:
                state.csv_file.close()
            except Exception:
                pass
            state.csv_file = None
            state.csv_writer = None


@app.post("/api/record/start")
async def start_recording():
    """Start recording telemetry frames to a CSV file."""
    if state.recording:
        return JSONResponse({"error": "Already recording."}, status_code=409)
    if not state.connected:
        return JSONResponse({"error": "Not connected."}, status_code=400)

    fname = datetime.now().strftime("thermoguard_%Y%m%d_%H%M%S.csv")
    fpath = LOG_DIR / fname
    try:
        f = open(fpath, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(EXPORT_FIELD_NAMES)
        f.flush()
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

    with state.lock:
        state.csv_file = f
        state.csv_writer = writer
        state.csv_path = str(fpath)
        state.record_start = time.time()
        state.record_samples = 0
        state.recording = True

    log.info("Recording started -> %s", fpath)
    return {"recording": True, "file": str(fpath)}


@app.post("/api/record/stop")
async def stop_recording():
    """Stop CSV recording."""
    if not state.recording:
        return JSONResponse({"error": "Not recording."}, status_code=409)
    samples = state.record_samples
    path = state.csv_path
    _stop_recording()
    log.info("Recording stopped (%d samples).", samples)
    return {"recording": False, "samples": samples, "file": path}


@app.get("/api/logs/download/latest")
async def download_latest_log():
    """Download the most recently recorded CSV file."""
    logs = sorted(LOG_DIR.glob("*.csv"), key=os.path.getmtime, reverse=True)
    if not logs:
        return JSONResponse({"error": "No recorded CSV logs found."}, status_code=404)
    return FileResponse(
        str(logs[0]),
        media_type="text/csv",
        filename=logs[0].name,
    )


@app.get("/api/logs")
async def list_logs():
    """List all recorded CSV log files."""
    logs = sorted(LOG_DIR.glob("*.csv"), key=os.path.getmtime, reverse=True)
    return [
        {
            "name": p.name,
            "size": p.stat().st_size,
            "modified": p.stat().st_mtime,
        }
        for p in logs
    ]


@app.get("/api/logs/{filename}")
async def download_log(filename: str):
    """Download a specific recorded CSV file."""
    fpath = LOG_DIR / filename
    if not fpath.exists() or not fpath.is_file():
        return JSONResponse({"error": "File not found."}, status_code=404)
    return FileResponse(
        str(fpath),
        media_type="text/csv",
        filename=filename,
    )


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state.ws_clients.add(ws)
    log.info("WS client connected (%d total).", len(state.ws_clients))

    # Send current status snapshot
    await ws.send_json({
        "type": "status",
        "connected": state.connected,
        "port": state.port,
        "baud": state.baud,
    })

    # Send latest frame so new clients aren't staring at empty charts
    if state.latest_frame:
        await ws.send_json({"type": "telemetry", **state.latest_frame})

    try:
        while True:
            # Keep the connection alive; handle client-side pings
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        state.ws_clients.discard(ws)
        log.info("WS client disconnected (%d remaining).", len(state.ws_clients))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
