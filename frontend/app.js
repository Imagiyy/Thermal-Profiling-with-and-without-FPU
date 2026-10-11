/**
 * app.js – Project ThermoGuard frontend controller.
 *
 * Handles:
 *  - WebSocket connection to the FastAPI backend
 *  - REST calls for port scanning, connect/disconnect, recording
 *  - Chart.js thermal & electrical chart rendering with 120 s rolling window
 *  - Live KPI badge updates (tracking error, MAE, RMSE, RθJA)
 *  - Safety zone indicators with audible alarm at >70°C
 *  - MAX31855 fault decoding visualization
 */

"use strict";

// ═══════════════════════════════════════════════════════════════════════════
// CONFIG
// ═══════════════════════════════════════════════════════════════════════════
const WS_URL = `ws://${location.host}/ws`;
const API    = `${location.origin}/api`;

const WINDOW_SEC          = 120;            // default rolling chart window (120 s)
const BUFFER_RETENTION_MS = 600 * 1000;     // retain up to 10 minutes (600 s) in memory for zoom/pan
const MAX_POINTS          = 1500;           // maximum buffer headroom (12.5 mins at 500ms intervals)
const CHART_COLORS = {
  measured : "rgb(34,211,238)",   // cyan
  predicted: "rgb(163,230,53)",   // lime
  ambient  : "rgba(148,163,184,.6)", // gray-400
  vdrop    : "rgb(251,191,36)",   // amber
  current  : "rgb(96,165,250)",   // blue-400
  power    : "rgb(251,113,133)",  // rose-400
};

// ═══════════════════════════════════════════════════════════════════════════
// DOM REFS
// ═══════════════════════════════════════════════════════════════════════════
const $  = (id) => document.getElementById(id);

const portSelect   = $("port-select");
const baudSelect   = $("baud-select");
const btnRefresh   = $("btn-refresh-ports");
const btnConnect   = $("btn-connect");
const btnDisconnect= $("btn-disconnect");
const connDot      = $("conn-dot");
const connLabel    = $("conn-label");

const btnRecStart  = $("btn-rec-start");
const btnRecStop   = $("btn-rec-stop");
const btnExportCsv = $("btn-export-csv");
const recBadge     = $("rec-badge");
const recTime      = $("rec-time");
const recCount     = $("rec-count");

const kpiErrorVal  = $("kpi-error-val");
const kpiMaeVal    = $("kpi-mae-val");
const kpiRmseVal   = $("kpi-rmse-val");
const kpiRthetaVal = $("kpi-rtheta-val");
const kpiTempVal   = $("kpi-temp-val");
const kpiFaultVal  = $("kpi-fault-val");
const thermalDot   = $("thermal-dot");
const thermalLabel = $("thermal-label");
const thermalMarker= $("thermal-marker");
const faultPills   = $("fault-pills");
const kpiThermal   = $("kpi-thermal");
const kpiFault     = $("kpi-fault");

// Live readings
const liveVin   = $("live-vin");
const liveVout  = $("live-vout");
const liveVdrop = $("live-vdrop");
const liveIload = $("live-iload");
const livePdiss = $("live-pdiss");
const liveTamb  = $("live-tamb");
const liveTmeas = $("live-tmeas");
const liveTpred = $("live-tpred");

const statFrames    = $("stat-frames");
const statMalformed = $("stat-malformed");
const statUptime    = $("stat-uptime");

// ═══════════════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════════════
let ws = null;
let isConnected = false;
let sessionStart = null;
let recStartTime = null;
let recTimerHandle = null;

// Error accumulation for MAE / RMSE
let errorSum   = 0;
let errorSqSum = 0;
let errorCount = 0;
let frameCount = 0;
let malformedCount = 0;

// Complete session telemetry buffer for CSV export (all 9 raw values + derived metrics)
const sessionTelemetryLog = [];

// Alarm state
let alarmActive = false;
let alarmOsc    = null;

// ═══════════════════════════════════════════════════════════════════════════
// CHARTS
// ═══════════════════════════════════════════════════════════════════════════

const chartGridColor  = "rgba(255,255,255,.04)";
const chartTickColor  = "rgba(148,163,184,.5)";

let currentLatestTs = 0;

function updateZoomLabel(chart) {
  const span = chart._userSpan || (WINDOW_SEC * 1000);
  const s = Math.round(span / 1000);
  const lbl = chart === thermalChart ? $("thermal-window-lbl") : $("elec-window-lbl");
  if (lbl) {
    lbl.textContent = s >= 60 ? `${(s / 60).toFixed(1).replace(/\.0$/, "")}m` : `${s}s`;
  }
}

// -- Shared zoom/pan plugin options (keeps live stream going at zoomed resolution) --
function createZoomConfig() {
  return {
    zoom: {
      wheel : { enabled: true, speed: 0.1 },
      pinch : { enabled: true },
      mode  : "x",
      onZoom: ({ chart }) => {
        if (chart.scales.x) {
          const s = chart.scales.x.max - chart.scales.x.min;
          if (s >= 2000) {
            chart._userSpan = s;
            chart._isZoomed = true;
            chart._isPannedBack = false;
            updateZoomLabel(chart);
          }
        }
      },
      onZoomComplete: ({ chart }) => {
        if (chart.scales.x) {
          const s = chart.scales.x.max - chart.scales.x.min;
          if (s >= 2000) {
            chart._userSpan = s;
            chart._isZoomed = true;
            chart._isPannedBack = false;
            updateZoomLabel(chart);
          }
        }
      },
    },
    pan: {
      enabled: true,
      mode   : "x",
      onPan: ({ chart }) => {
        if (chart.scales.x) {
          chart._isZoomed = true;
          const latest = currentLatestTs || (chart.data.labels.length ? chart.data.labels[chart.data.labels.length - 1] : 0);
          chart._isPannedBack = (latest - chart.scales.x.max > 2500);
          if (!chart._isPannedBack) {
            chart._userSpan = chart.scales.x.max - chart.scales.x.min;
            updateZoomLabel(chart);
          }
        }
      },
      onPanComplete: ({ chart }) => {
        if (chart.scales.x) {
          chart._isZoomed = true;
          const latest = currentLatestTs || (chart.data.labels.length ? chart.data.labels[chart.data.labels.length - 1] : 0);
          chart._isPannedBack = (latest - chart.scales.x.max > 2500);
          if (!chart._isPannedBack) {
            chart._userSpan = chart.scales.x.max - chart.scales.x.min;
            updateZoomLabel(chart);
          }
        }
      },
    },
    limits: {
      x: { minRange: 2000 },
    },
  };
}

// -- Thermal Chart --
const thermalCtx = $("chart-thermal").getContext("2d");
const thermalChart = new Chart(thermalCtx, {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label: "T measured (°C)",
        data: [],
        borderColor: CHART_COLORS.measured,
        backgroundColor: "rgba(34,211,238,.08)",
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.3,
        fill: false,
      },
      {
        label: "T predicted (°C)",
        data: [],
        borderColor: CHART_COLORS.predicted,
        backgroundColor: "rgba(163,230,53,.08)",
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.3,
        fill: false,
        borderDash: [6, 3],
      },
      {
        label: "T ambient (°C)",
        data: [],
        borderColor: CHART_COLORS.ambient,
        backgroundColor: "rgba(148,163,184,.04)",
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
        fill: true,
      },
    ],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 0 },
    interaction: { intersect: false, mode: "index" },
    scales: {
      x: {
        type: "linear",
        title: { display: true, text: "Time (s)", color: chartTickColor, font: { size: 10 } },
        ticks: { color: chartTickColor, font: { size: 9 }, maxTicksLimit: 12,
          callback: (v) => (v / 1000).toFixed(0) + "s",
        },
        grid: { color: chartGridColor },
      },
      y: {
        title: { display: true, text: "Temperature (°C)", color: chartTickColor, font: { size: 10 } },
        ticks: { color: chartTickColor, font: { size: 9 } },
        grid: { color: chartGridColor },
        grace: "8%",
      },
    },
    plugins: {
      legend: { labels: { color: chartTickColor, font: { size: 10 }, boxWidth: 14, padding: 12 } },
      zoom: createZoomConfig(),
      tooltip: {
        backgroundColor: "rgba(13,15,18,.9)",
        titleColor: "#e2e8f0",
        bodyColor: "#cbd5e1",
        borderColor: "rgba(255,255,255,.08)",
        borderWidth: 1,
        cornerRadius: 8,
        padding: 10,
      },
    },
  },
});

// -- Electrical Chart --
const elecCtx = $("chart-electrical").getContext("2d");
const elecChart = new Chart(elecCtx, {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label: "V drop (V)",
        data: [],
        borderColor: CHART_COLORS.vdrop,
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.3,
        yAxisID: "yV",
      },
      {
        label: "I load (mA)",
        data: [],
        borderColor: CHART_COLORS.current,
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.3,
        yAxisID: "yI",
      },
      {
        label: "P diss (mW)",
        data: [],
        borderColor: CHART_COLORS.power,
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.3,
        yAxisID: "yP",
        borderDash: [4, 2],
      },
    ],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 0 },
    interaction: { intersect: false, mode: "index" },
    scales: {
      x: {
        type: "linear",
        title: { display: true, text: "Time (s)", color: chartTickColor, font: { size: 10 } },
        ticks: { color: chartTickColor, font: { size: 9 }, maxTicksLimit: 12,
          callback: (v) => (v / 1000).toFixed(0) + "s",
        },
        grid: { color: chartGridColor },
      },
      yV: {
        position: "left",
        title: { display: true, text: "V drop (V)", color: CHART_COLORS.vdrop, font: { size: 10 } },
        ticks: { color: CHART_COLORS.vdrop, font: { size: 9 } },
        grid: { color: chartGridColor },
      },
      yI: {
        position: "right",
        title: { display: true, text: "I load (mA)", color: CHART_COLORS.current, font: { size: 10 } },
        ticks: { color: CHART_COLORS.current, font: { size: 9 } },
        grid: { drawOnChartArea: false },
      },
      yP: {
        position: "right",
        title: { display: true, text: "P diss (mW)", color: CHART_COLORS.power, font: { size: 10 } },
        ticks: { color: CHART_COLORS.power, font: { size: 9 } },
        grid: { drawOnChartArea: false },
        offset: true,
      },
    },
    plugins: {
      legend: { labels: { color: chartTickColor, font: { size: 10 }, boxWidth: 14, padding: 12 } },
      zoom: createZoomConfig(),
      tooltip: {
        backgroundColor: "rgba(13,15,18,.9)",
        titleColor: "#e2e8f0",
        bodyColor: "#cbd5e1",
        borderColor: "rgba(255,255,255,.08)",
        borderWidth: 1,
        cornerRadius: 8,
        padding: 10,
      },
    },
  },
});

function zoomChartByFactor(chart, factor) {
  const currentSpan = chart._userSpan || (WINDOW_SEC * 1000);
  const newSpan = Math.max(3000, Math.min(BUFFER_RETENTION_MS, currentSpan * factor));
  chart._userSpan = newSpan;
  chart._isZoomed = true;
  chart._isPannedBack = false;

  const latest = currentLatestTs || (chart.data.labels.length ? chart.data.labels[chart.data.labels.length - 1] : 0);
  if (latest > 0) {
    chart.options.scales.x.max = latest;
    chart.options.scales.x.min = latest - newSpan;
  }
  updateZoomLabel(chart);
  chart.update("none");
}

function resetChartZoom(chart) {
  chart.resetZoom();
  chart._isZoomed = false;
  chart._isPannedBack = false;
  chart._userSpan = WINDOW_SEC * 1000;
  if (currentLatestTs > 0) {
    chart.options.scales.x.min = currentLatestTs - (WINDOW_SEC * 1000);
    chart.options.scales.x.max = currentLatestTs;
  } else {
    delete chart.options.scales.x.min;
    delete chart.options.scales.x.max;
  }
  updateZoomLabel(chart);
  chart.update("none");
}

$("btn-thermal-zoom-in").addEventListener("click", () => zoomChartByFactor(thermalChart, 0.7));
$("btn-thermal-zoom-out").addEventListener("click", () => zoomChartByFactor(thermalChart, 1.4));
$("btn-reset-thermal-zoom").addEventListener("click", () => resetChartZoom(thermalChart));

$("btn-elec-zoom-in").addEventListener("click", () => zoomChartByFactor(elecChart, 0.7));
$("btn-elec-zoom-out").addEventListener("click", () => zoomChartByFactor(elecChart, 1.4));
$("btn-reset-elec-zoom").addEventListener("click", () => resetChartZoom(elecChart));

// ═══════════════════════════════════════════════════════════════════════════
// AUDIBLE ALARM (Web Audio API — no external file needed)
// ═══════════════════════════════════════════════════════════════════════════
let audioCtx = null;

function startAlarm() {
  if (alarmActive) return;
  alarmActive = true;
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    alarmOsc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    alarmOsc.type = "square";
    alarmOsc.frequency.setValueAtTime(880, audioCtx.currentTime);
    gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
    alarmOsc.connect(gain).connect(audioCtx.destination);
    alarmOsc.start();
    // Pulse the tone
    const lfo = audioCtx.createOscillator();
    lfo.frequency.setValueAtTime(4, audioCtx.currentTime);
    const lfoGain = audioCtx.createGain();
    lfoGain.gain.setValueAtTime(0.06, audioCtx.currentTime);
    lfo.connect(lfoGain).connect(gain.gain);
    lfo.start();
    alarmOsc._lfo = lfo;
    alarmOsc._gain = gain;
    alarmOsc._lfoGain = lfoGain;
  } catch (e) {
    console.warn("Audio alarm failed:", e);
  }
}

function stopAlarm() {
  alarmActive = false;
  if (alarmOsc) {
    try {
      alarmOsc.stop();
      alarmOsc._lfo?.stop();
    } catch (_) {}
    alarmOsc = null;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// TELEMETRY HANDLER
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Main handler for each telemetry frame received over WebSocket.
 */
function handleTelemetry(d) {
  frameCount++;
  statFrames.textContent = frameCount;

  const ts = d.timestamp_ms;

  // ── Push to thermal chart ──
  thermalChart.data.labels.push(ts);
  thermalChart.data.datasets[0].data.push({ x: ts, y: d.T_measured });
  thermalChart.data.datasets[1].data.push({ x: ts, y: d.T_predicted });
  thermalChart.data.datasets[2].data.push({ x: ts, y: d.T_ambient });

  // ── Push to electrical chart ──
  elecChart.data.labels.push(ts);
  elecChart.data.datasets[0].data.push({ x: ts, y: d.V_drop });
  elecChart.data.datasets[1].data.push({ x: ts, y: d.I_mA });
  elecChart.data.datasets[2].data.push({ x: ts, y: d.P_mW });

  // ── Record frame into session export log (all 9 raw values + error + metrics) ──
  sessionTelemetryLog.push({
    timestamp_ms: d.timestamp_ms,
    iso_time: new Date().toISOString(),
    V_in: d.V_in,
    V_out: d.V_out,
    V_drop: d.V_drop,
    I_mA: d.I_mA,
    P_mW: d.P_mW,
    T_ambient: d.T_ambient,
    T_measured: d.T_measured,
    T_predicted: d.T_predicted,
    tracking_error: d.tracking_error,
    R_theta_JA: d.R_theta_JA !== null ? d.R_theta_JA : "",
    fault_code: d.fault_code,
    fault_flags: (d.faults && d.faults.length > 0) ? d.faults.join("|") : "OK",
  });

  currentLatestTs = ts;

  // ── Advance live rolling window (maintains custom zoom level in real-time) ──
  advanceLiveWindow(thermalChart, ts);
  advanceLiveWindow(elecChart, ts);

  // ── Trim historical buffer (retains up to 10 minutes) ──
  trimChart(thermalChart, ts);
  trimChart(elecChart, ts);

  // ── Update charts (batched) ──
  requestAnimationFrame(() => {
    thermalChart.update("none");
    elecChart.update("none");
  });

  // ── KPI: Tracking error ──
  const tErr = d.tracking_error;
  kpiErrorVal.textContent = tErr.toFixed(2);

  // ── KPI: MAE & RMSE ──
  errorSum   += tErr;
  errorSqSum += tErr * tErr;
  errorCount++;
  kpiMaeVal.textContent  = (errorSum / errorCount).toFixed(3);
  kpiRmseVal.textContent = Math.sqrt(errorSqSum / errorCount).toFixed(3);

  // ── KPI: RθJA ──
  kpiRthetaVal.textContent = d.R_theta_JA != null ? d.R_theta_JA.toFixed(2) : "—";

  // ── KPI: Heatsink temperature + safety zone ──
  const t = d.T_measured;
  kpiTempVal.textContent = t.toFixed(1) + " °C";
  updateThermalZone(t);

  // ── KPI: MAX31855 fault decode ──
  updateFaults(d.fault_code, d.faults);

  // ── Live readings ──
  liveVin.textContent   = d.V_in.toFixed(3)   + " V";
  liveVout.textContent  = d.V_out.toFixed(3)   + " V";
  liveVdrop.textContent = d.V_drop.toFixed(3)  + " V";
  liveIload.textContent = d.I_mA.toFixed(1)    + " mA";
  livePdiss.textContent = d.P_mW.toFixed(1)    + " mW";
  liveTamb.textContent  = d.T_ambient.toFixed(2)  + " °C";
  liveTmeas.textContent = d.T_measured.toFixed(2) + " °C";
  liveTpred.textContent = d.T_predicted.toFixed(2)+ " °C";
}

function advanceLiveWindow(chart, latestTs) {
  const span = (chart._isZoomed && chart._userSpan)
    ? chart._userSpan
    : (WINDOW_SEC * 1000);

  // If user hasn't actively panned into history, keep advancing live
  if (!chart._isPannedBack) {
    chart.options.scales.x.max = latestTs;
    chart.options.scales.x.min = latestTs - span;
  }
}

function trimChart(chart, latestTs) {
  const cutoff = latestTs - BUFFER_RETENTION_MS;
  while (chart.data.labels.length > 0 && chart.data.labels[0] < cutoff) {
    chart.data.labels.shift();
    chart.data.datasets.forEach((ds) => ds.data.shift());
  }
  // Also cap absolute length
  while (chart.data.labels.length > MAX_POINTS) {
    chart.data.labels.shift();
    chart.data.datasets.forEach((ds) => ds.data.shift());
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// THERMAL SAFETY ZONE (KA7808 Profile: 125°C Max Op, 150°C Shutdown)
// ═══════════════════════════════════════════════════════════════════════════

const T_MIN = 25;
const T_MAX_OPERATING = 125;
const T_SHUTDOWN = 150;

function updateThermalZone(tempC) {
  // Marker position (25°C → 0%, 150°C → 100%)
  const rawPct = ((tempC - T_MIN) / (T_SHUTDOWN - T_MIN)) * 100;
  const pct = Math.max(1, Math.min(99, rawPct));
  thermalMarker.style.left = pct.toFixed(1) + "%";

  if (tempC >= T_SHUTDOWN) {
    // CRITICAL THERMAL SHUTDOWN (≥150°C) — Flashing red + audible alarm
    kpiThermal.classList.add("glow-red");
    kpiThermal.classList.remove("glow-amber");
    thermalDot.className = "w-2.5 h-2.5 rounded-full bg-red-600 dot-blink";
    thermalLabel.textContent = "SHUTDOWN (≥150°C)";
    thermalLabel.className = "text-[10px] font-bold text-red-400 tracking-tight";
    kpiTempVal.className = "text-xl font-bold text-red-500 tabular-nums";
    startAlarm();
  } else if (tempC >= T_MAX_OPERATING) {
    // OPERATING MAX EXCEEDED (125°C - 150°C) — Warning alarm + critical red
    kpiThermal.classList.add("glow-red");
    kpiThermal.classList.remove("glow-amber");
    thermalDot.className = "w-2.5 h-2.5 rounded-full bg-rose-500 dot-blink";
    thermalLabel.textContent = "OVER-MAX (≥125°C)";
    thermalLabel.className = "text-[10px] font-semibold text-rose-400";
    kpiTempVal.className = "text-xl font-bold text-rose-400 tabular-nums";
    startAlarm();
  } else if (tempC >= 90) {
    // ELEVATED / WARNING (90°C - 125°C) — Amber glow
    kpiThermal.classList.remove("glow-red");
    kpiThermal.classList.add("glow-amber");
    thermalDot.className = "w-2.5 h-2.5 rounded-full bg-amber-400";
    thermalLabel.textContent = "WARNING (≥90°C)";
    thermalLabel.className = "text-[10px] font-semibold text-amber-400";
    kpiTempVal.className = "text-xl font-bold text-amber-400 tabular-nums";
    stopAlarm();
  } else {
    // NOMINAL SAFE (< 90°C) — Green
    kpiThermal.classList.remove("glow-red", "glow-amber");
    thermalDot.className = "w-2.5 h-2.5 rounded-full bg-emerald-500";
    thermalLabel.textContent = "SAFE (<90°C)";
    thermalLabel.className = "text-[10px] font-semibold text-emerald-400";
    kpiTempVal.className = "text-xl font-bold text-emerald-400 tabular-nums";
    stopAlarm();
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// FAULT DECODE
// ═══════════════════════════════════════════════════════════════════════════

const FAULT_NAMES = ["FAULT", "SCG", "SCV", "OC"];
const DIAG_DOTS   = [$("diag-f0"), $("diag-f1"), $("diag-f2"), $("diag-f3")];

function updateFaults(code, faultList) {
  if (code === 0) {
    kpiFaultVal.textContent = "OK";
    kpiFaultVal.className = "text-xl font-bold text-emerald-400 tabular-nums";
    kpiFault.classList.remove("glow-red");
    faultPills.innerHTML = "";
  } else {
    kpiFaultVal.textContent = "FAULT";
    kpiFaultVal.className = "text-xl font-bold text-rose-400 tabular-nums";
    kpiFault.classList.add("glow-red");
    faultPills.innerHTML = faultList
      .map((f) => `<span class="text-[9px] px-1.5 py-0.5 rounded-full bg-rose-500/20 text-rose-300 font-semibold">${f}</span>`)
      .join("");
  }

  // Diagnostic indicator dots
  for (let i = 0; i < 4; i++) {
    if (code & (1 << i)) {
      DIAG_DOTS[i].className = "w-2 h-2 rounded-full bg-rose-500 dot-blink";
    } else {
      DIAG_DOTS[i].className = "w-2 h-2 rounded-full bg-gray-600";
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// WEBSOCKET
// ═══════════════════════════════════════════════════════════════════════════

function wsConnect() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    console.log("[WS] Connected");
  };

  ws.onmessage = (evt) => {
    let msg;
    try {
      msg = JSON.parse(evt.data);
    } catch (_) {
      return;
    }

    if (msg.type === "telemetry") {
      handleTelemetry(msg);
    } else if (msg.type === "status") {
      setConnectionUI(msg.connected);
      if (!msg.connected && msg.reason) {
        console.warn("[WS] Disconnected:", msg.reason);
      }
    } else if (msg.type === "error") {
      console.error("[WS] Error:", msg.message);
      setConnectionUI(false);
    }
  };

  ws.onclose = () => {
    console.log("[WS] Closed — reconnecting in 2s");
    setTimeout(wsConnect, 2000);
  };

  ws.onerror = (e) => {
    console.error("[WS] Error:", e);
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// CONNECTION UI
// ═══════════════════════════════════════════════════════════════════════════

function setConnectionUI(connected) {
  isConnected = connected;
  if (connected) {
    connDot.className      = "w-2 h-2 rounded-full bg-emerald-500 dot-blink";
    connLabel.textContent  = "Online";
    connLabel.className    = "text-xs text-emerald-400";
    btnConnect.classList.add("hidden");
    btnDisconnect.classList.remove("hidden");
    btnRecStart.disabled   = false;
    sessionStart = sessionStart || Date.now();
  } else {
    connDot.className      = "w-2 h-2 rounded-full bg-gray-600";
    connLabel.textContent  = "Offline";
    connLabel.className    = "text-xs text-gray-500";
    btnConnect.classList.remove("hidden");
    btnDisconnect.classList.add("hidden");
    btnRecStart.disabled   = true;
    stopAlarm();
    sessionStart = null;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// REST HELPERS
// ═══════════════════════════════════════════════════════════════════════════

async function fetchPorts() {
  try {
    const res = await fetch(`${API}/ports`);
    const ports = await res.json();
    portSelect.innerHTML = '<option value="">— select port —</option>';
    ports.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p.device;
      opt.textContent = `${p.device} — ${p.description}`;
      portSelect.appendChild(opt);
    });
  } catch (e) {
    console.error("Failed to fetch ports:", e);
  }
}

async function doConnect() {
  const port = portSelect.value;
  const baud = parseInt(baudSelect.value);
  if (!port) {
    portSelect.focus();
    return;
  }
  try {
    const res = await fetch(`${API}/connect?port=${encodeURIComponent(port)}&baud=${baud}`, { method: "POST" });
    const data = await res.json();
    if (res.ok) {
      setConnectionUI(true);
      // Reset session metrics
      errorSum = errorSqSum = errorCount = frameCount = malformedCount = 0;
    } else {
      alert(data.error || "Connection failed");
    }
  } catch (e) {
    alert("Network error: " + e.message);
  }
}

async function doDisconnect() {
  try {
    await fetch(`${API}/disconnect`, { method: "POST" });
    setConnectionUI(false);
    stopRecordingUI();
  } catch (e) {
    console.error("Disconnect error:", e);
  }
}

async function doStartRecording() {
  try {
    const res = await fetch(`${API}/record/start`, { method: "POST" });
    const data = await res.json();
    if (res.ok) {
      btnRecStart.classList.add("hidden");
      btnRecStop.classList.remove("hidden");
      recBadge.classList.remove("hidden");
      recStartTime = Date.now();
      recTimerHandle = setInterval(updateRecTimer, 1000);
    } else {
      alert(data.error || "Recording failed");
    }
  } catch (e) {
    alert("Network error: " + e.message);
  }
}

async function doStopRecording() {
  try {
    const res = await fetch(`${API}/record/stop`, { method: "POST" });
    const data = await res.json();
    stopRecordingUI();
    if (res.ok && data.samples > 0) {
      const ask = confirm(
        `Recording stopped (${data.samples} samples saved on server).\n\nDo you want to download this CSV file now?`
      );
      if (ask) {
        window.location.href = `${API}/logs/download/latest`;
      }
    }
  } catch (e) {
    console.error("Stop rec error:", e);
  }
}

function exportSessionCSV() {
  if (sessionTelemetryLog.length === 0) {
    alert("No telemetry data received yet to export. Connect to a port first!");
    return;
  }
  const headers = [
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
    "tracking_error",
    "R_theta_JA",
    "fault_code",
    "fault_flags",
  ];
  const lines = [headers.join(",")];
  for (const row of sessionTelemetryLog) {
    lines.push(headers.map((h) => row[h] !== undefined ? row[h] : "").join(","));
  }
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const tsStr = new Date().toISOString().replace(/[:.]/g, "-");
  a.href = url;
  a.download = `thermoguard_export_${tsStr}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function stopRecordingUI() {
  btnRecStop.classList.add("hidden");
  btnRecStart.classList.remove("hidden");
  recBadge.classList.add("hidden");
  clearInterval(recTimerHandle);
  recStartTime = null;
}

function updateRecTimer() {
  if (!recStartTime) return;
  const elapsed = Math.floor((Date.now() - recStartTime) / 1000);
  const min = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const sec = String(elapsed % 60).padStart(2, "0");
  recTime.textContent = `${min}:${sec}`;
  recCount.textContent = frameCount;
}

// ── Uptime ticker ──
setInterval(() => {
  if (sessionStart) {
    const s = Math.floor((Date.now() - sessionStart) / 1000);
    const h = String(Math.floor(s / 3600)).padStart(2, "0");
    const m = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
    const sec = String(s % 60).padStart(2, "0");
    statUptime.textContent = `${h}:${m}:${sec}`;
  }
}, 1000);

// ═══════════════════════════════════════════════════════════════════════════
// EVENT WIRING
// ═══════════════════════════════════════════════════════════════════════════

btnRefresh.addEventListener("click", fetchPorts);
btnConnect.addEventListener("click", doConnect);
btnDisconnect.addEventListener("click", doDisconnect);
btnRecStart.addEventListener("click", doStartRecording);
btnRecStop.addEventListener("click", doStopRecording);
btnExportCsv.addEventListener("click", exportSessionCSV);

// ═══════════════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════════════

(function init() {
  fetchPorts();
  wsConnect();
})();
