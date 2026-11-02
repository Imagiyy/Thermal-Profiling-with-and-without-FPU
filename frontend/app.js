/**
 * app.js – Project ThermoGuard frontend controller.
 *
 * Handles:
 *  - WebSocket connection to the FastAPI backend
 *  - REST calls for port scanning, connect/disconnect, recording
 *  - Chart.js thermal & electrical chart rendering with 120 s rolling window
 *  - Live KPI badge updates (tracking error, MAE, RMSE, RθJA)
 *  - Safety zone indicators with audible alarm at cutoff (>=80°C, scale max 100°C)
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

// ═══════════════════════════════════════════════════════════════════════════
// THEME & PALETTES (Default: White Background)
// ═══════════════════════════════════════════════════════════════════════════
const THEME_KEY = "thermoguard_theme";
let isDarkMode = false; // Default: White Background!

function getThemeColors(dark) {
  if (dark) {
    return {
      grid: "rgba(255,255,255,.05)",
      tick: "rgba(148,163,184,.6)",
      tooltipBg: "rgba(13,15,18,.92)",
      tooltipTitle: "#e2e8f0",
      tooltipBody: "#cbd5e1",
      tooltipBorder: "rgba(255,255,255,.1)",
      measured: "rgb(34,211,238)",     // cyan-400
      pred3s:   "rgb(163,230,53)",     // lime-400
      pred10s:  "rgb(251,146,60)",     // orange-400
      ambient:  "rgba(148,163,184,.6)",// slate-400
      vdrop:    "rgb(251,191,36)",     // amber-400
      current:  "rgb(96,165,250)",     // blue-400
      power:    "rgb(251,113,133)",    // rose-400
      fillMeas: "rgba(34,211,238,.08)",
      fill3s:   "rgba(163,230,53,.08)",
      fill10s:  "rgba(251,146,60,.08)",
      fillAmb:  "rgba(148,163,184,.04)",
      cutoff:   "rgba(244,63,94,0.85)",
    };
  } else {
    // Light mode (default white background) — rich high-contrast palette
    return {
      grid: "rgba(0,0,0,.08)",
      tick: "rgba(71,85,105,.9)",      // slate-600
      tooltipBg: "rgba(255,255,255,.98)",
      tooltipTitle: "#0f172a",          // slate-900
      tooltipBody: "#334155",           // slate-700
      tooltipBorder: "rgba(0,0,0,.12)",
      measured: "rgb(8,145,178)",       // cyan-600
      pred3s:   "rgb(101,163,13)",      // lime-600
      pred10s:  "rgb(234,88,12)",       // orange-600
      ambient:  "rgba(100,116,139,.8)", // slate-500
      vdrop:    "rgb(217,119,6)",       // amber-600
      current:  "rgb(37,99,235)",       // blue-600
      power:    "rgb(225,29,72)",       // rose-600
      fillMeas: "rgba(8,145,178,.08)",
      fill3s:   "rgba(101,163,13,.08)",
      fill10s:  "rgba(234,88,12,.08)",
      fillAmb:  "rgba(100,116,139,.06)",
      cutoff:   "rgba(225,29,72,0.9)",
    };
  }
}

let activeColors = getThemeColors(false);

const CHART_COLORS = new Proxy({}, {
  get: (_, prop) => activeColors[prop] || "#888",
});

// ═══════════════════════════════════════════════════════════════════════════
// DOM REFS
// ═══════════════════════════════════════════════════════════════════════════
const $  = (id) => document.getElementById(id);

const portSelect          = $("port-select");
const baudSelect          = $("baud-select");
const btnRefresh          = $("btn-refresh-ports");
const btnConnect          = $("btn-connect");
const btnDisconnect       = $("btn-disconnect");
const connDot             = $("conn-dot");
const connLabel           = $("conn-label");

const btnThemeToggle      = $("btn-theme-toggle");
const btnExportThermalPng = $("btn-export-thermal-png");
const btnExportElecPng    = $("btn-export-elec-png");
const btnResetPage        = $("btn-reset-page");
const btnExportCsv        = $("btn-export-csv");

const kpiErrorVal    = $("kpi-error-val");
const kpiErrorAvgVal = $("kpi-error-avg-val") || $("kpi-error-max-val");
const kpiAmbTempVal  = $("kpi-amb-temp-val");
const kpiRmseVal     = $("kpi-rmse-val");
const kpiRmseAvgVal  = $("kpi-rmse-avg-val") || $("kpi-rmse-max-val");
const kpiPred3sVal  = $("kpi-pred-3s-val");
const kpiPred10sVal = $("kpi-pred-10s-val");
const kpiRthetaVal  = $("kpi-rtheta-val");
const kpiTempVal    = $("kpi-temp-val");
const kpiPredTimeVal= $("kpi-pred-time-val");
const kpiPredTimeSub= $("kpi-pred-time-sub");
const thermalDot    = $("thermal-dot");
const thermalLabel  = $("thermal-label");
const thermalMarker = $("thermal-marker");
const kpiThermal    = $("kpi-thermal");

// Live readings
const liveVin       = $("live-vin");
const liveVout      = $("live-vout");
const liveVdrop     = $("live-vdrop");
const liveIload     = $("live-iload");
const livePdiss     = $("live-pdiss");
const liveTamb      = $("live-tamb");
const liveTmeas     = $("live-tmeas");
const liveTpred3s   = $("live-tpred-3s");
const liveTpred10s  = $("live-tpred-10s");
const liveLatency   = $("live-latency");
const modelLatency  = $("model-latency");

const statFrames    = $("stat-frames");
const statMalformed = $("stat-malformed");
const statUptime    = $("stat-uptime");

// ═══════════════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════════════
let ws = null;
let isConnected = false;
let sessionStart = null;

// Error accumulation for MAE / RMSE
let errorSum3s    = 0;
let errorSum10s   = 0;
let errorSqSum3s  = 0;
let errorSqSum10s = 0;
let errorCount    = 0;
let frameCount    = 0;
let malformedCount = 0;
let rmseSum3s     = 0;
let rmseSum10s    = 0;

// Inference latency accumulation (DWT benchmarking)
let latencySum   = 0;
let latencyCount = 0;

// 3-second (6 samples @ 500 ms) and 10-second (20 samples @ 500 ms) lookahead verification buffers
const LOOKAHEAD_3S_SAMPLES  = 6;
const LOOKAHEAD_10S_SAMPLES = 20;
const pred3sLookaheadBuffer  = [];
const pred10sLookaheadBuffer = [];

// Complete session telemetry buffer for CSV export (all 9 raw values + derived metrics)
const sessionTelemetryLog = [];

// Alarm state
let alarmActive = false;
let alarmOsc    = null;

// ═══════════════════════════════════════════════════════════════════════════
// CHARTS
// ═══════════════════════════════════════════════════════════════════════════

let chartGridColor  = activeColors.grid;
let chartTickColor  = activeColors.tick;

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
            if (chart === thermalChart && typeof updateThermalYAutoZoom === "function") updateThermalYAutoZoom(chart);
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
            if (chart === thermalChart && typeof updateThermalYAutoZoom === "function") updateThermalYAutoZoom(chart);
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
            if (chart === thermalChart && typeof updateThermalYAutoZoom === "function") updateThermalYAutoZoom(chart);
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
            if (chart === thermalChart && typeof updateThermalYAutoZoom === "function") updateThermalYAutoZoom(chart);
          }
        }
      },
    },
    limits: {
      x: { minRange: 2000 },
    },
  };
}

// Plugin to draw horizontal cutoff indicator line on thermal chart at 80°C
const thermalCutoffLinePlugin = {
  id: "thermalCutoffLine",
  afterDraw(chart) {
    if (chart.canvas.id !== "chart-thermal") return;
    const yAxis = chart.scales.y;
    const xAxis = chart.scales.x;
    if (!yAxis || !xAxis) return;
    const yPos = yAxis.getPixelForValue(80);
    if (yPos < yAxis.top || yPos > yAxis.bottom) return;

    const ctx = chart.ctx;
    ctx.save();
    ctx.strokeStyle = activeColors.cutoff;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(xAxis.left, yPos);
    ctx.lineTo(xAxis.right, yPos);
    ctx.stroke();

    ctx.fillStyle = activeColors.cutoff;
    ctx.font = "bold 9px monospace, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText("CUTOFF 80°C", xAxis.right - 6, yPos - 4);
    ctx.restore();
  },
};

// -- Thermal Chart --
const thermalCtx = $("chart-thermal").getContext("2d");
const thermalChart = new Chart(thermalCtx, {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label: "Heatsink Temp (°C)",
        data: [],
        borderColor: CHART_COLORS.measured,
        backgroundColor: "rgba(34,211,238,.08)",
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.3,
        fill: false,
      },
      {
        label: "T predicted (+3 s)",
        data: [],
        borderColor: CHART_COLORS.pred3s,
        backgroundColor: "rgba(163,230,53,.08)",
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.3,
        fill: false,
        borderDash: [5, 3],
      },
      {
        label: "T predicted (+10 s)",
        data: [],
        borderColor: CHART_COLORS.pred10s,
        backgroundColor: "rgba(251,146,60,.08)",
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.3,
        fill: false,
        borderDash: [6, 4],
      },
      {
        label: "Ambient Temp (°C)",
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
      },
    },
    plugins: {
      legend: { labels: { color: chartTickColor, font: { size: 10 }, boxWidth: 14, padding: 12 } },
      zoom: createZoomConfig(),
      tooltip: { enabled: false },
    },
  },
  plugins: [thermalCutoffLinePlugin],
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
      tooltip: { enabled: false },
    },
  },
});

// ═══════════════════════════════════════════════════════════════════════════
// STATIONARY CURSOR TOOLTIP
// Values remain completely STILL at cursor while graph scrolls;
// changes ONLY when user moves cursor to another point.
// ═══════════════════════════════════════════════════════════════════════════

function findClosestPointIndex(arr, targetX) {
  if (!arr || arr.length === 0) return -1;
  let low = 0;
  let high = arr.length - 1;
  while (low <= high) {
    const mid = (low + high) >> 1;
    const midX = arr[mid]?.x;
    if (midX === undefined) return -1;
    if (midX === targetX) return mid;
    if (midX < targetX) low = mid + 1;
    else high = mid - 1;
  }
  if (low >= arr.length) return arr.length - 1;
  if (high < 0) return 0;
  const dLow = Math.abs(arr[low].x - targetX);
  const dHigh = Math.abs(arr[high].x - targetX);
  return dLow < dHigh ? low : high;
}

function attachStationaryCursorTooltip(chart, canvasId, tooltipId, crosshairId, type) {
  const canvas = $(canvasId);
  const tooltip = $(tooltipId);
  const crosshair = $(crosshairId);
  if (!canvas || !tooltip || !crosshair) return;

  let isHovering = false;
  let lastClientX = 0;
  let lastClientY = 0;

  function updateTooltip(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const mouseX = clientX - rect.left;
    const mouseY = clientY - rect.top;

    const xAxis = chart.scales.x;
    const yAxis = chart.scales.y || chart.scales.yV;
    if (!xAxis || !yAxis) {
      hide();
      return;
    }

    // Must be inside the active chart area
    if (mouseX < xAxis.left || mouseX > xAxis.right || mouseY < yAxis.top || mouseY > yAxis.bottom) {
      hide();
      return;
    }

    const hoveredTime = xAxis.getValueForPixel(mouseX);
    const d0 = chart.data.datasets[0]?.data;
    if (!d0 || d0.length === 0) {
      hide();
      return;
    }

    const idx = findClosestPointIndex(d0, hoveredTime);
    if (idx === -1 || !d0[idx]) {
      hide();
      return;
    }

    const ptTime = d0[idx].x;
    const timeSec = (ptTime / 1000).toFixed(1);

    let html = `
      <div class="flex items-center justify-between gap-3 border-b border-slate-200 dark:border-white/10 pb-1 mb-1.5">
        <span class="font-semibold text-slate-500 dark:text-gray-400 font-mono text-[10px]">TIME</span>
        <span class="font-mono font-bold text-slate-900 dark:text-white text-xs">${timeSec}s</span>
      </div>
      <div class="space-y-1">
    `;

    if (type === "thermal") {
      const tMeas = chart.data.datasets[0]?.data[idx]?.y;
      const t3s   = chart.data.datasets[1]?.data[idx]?.y;
      const t10s  = chart.data.datasets[2]?.data[idx]?.y;
      const tAmb  = chart.data.datasets[3]?.data[idx]?.y;

      html += `
        <div class="flex items-center justify-between gap-4">
          <div class="flex items-center gap-1.5">
            <span class="w-2 h-2 rounded-full shadow-sm" style="background:${activeColors.measured}"></span>
            <span class="text-slate-600 dark:text-gray-300">Heatsink</span>
          </div>
          <span class="font-mono font-bold text-slate-800 dark:text-gray-100">${tMeas !== undefined ? tMeas.toFixed(1) + "°C" : "—"}</span>
        </div>
        <div class="flex items-center justify-between gap-4">
          <div class="flex items-center gap-1.5">
            <span class="w-2 h-2 rounded-full shadow-sm" style="background:${activeColors.pred3s}"></span>
            <span class="text-slate-600 dark:text-gray-300">Pred (+3s)</span>
          </div>
          <span class="font-mono font-bold text-slate-800 dark:text-gray-100">${t3s !== undefined ? t3s.toFixed(1) + "°C" : "—"}</span>
        </div>
        <div class="flex items-center justify-between gap-4">
          <div class="flex items-center gap-1.5">
            <span class="w-2 h-2 rounded-full shadow-sm" style="background:${activeColors.pred10s}"></span>
            <span class="text-slate-600 dark:text-gray-300">Pred (+10s)</span>
          </div>
          <span class="font-mono font-bold text-slate-800 dark:text-gray-100">${t10s !== undefined ? t10s.toFixed(1) + "°C" : "—"}</span>
        </div>
        <div class="flex items-center justify-between gap-4">
          <div class="flex items-center gap-1.5">
            <span class="w-2 h-2 rounded-full shadow-sm" style="background:${activeColors.ambient}"></span>
            <span class="text-slate-500 dark:text-gray-400">Ambient</span>
          </div>
          <span class="font-mono font-medium text-slate-600 dark:text-gray-300">${tAmb !== undefined ? tAmb.toFixed(1) + "°C" : "—"}</span>
        </div>
      `;
    } else {
      const vDrop = chart.data.datasets[0]?.data[idx]?.y;
      const iLoad = chart.data.datasets[1]?.data[idx]?.y;
      const pDiss = chart.data.datasets[2]?.data[idx]?.y;

      html += `
        <div class="flex items-center justify-between gap-4">
          <div class="flex items-center gap-1.5">
            <span class="w-2 h-2 rounded-full shadow-sm" style="background:${activeColors.vdrop}"></span>
            <span class="text-slate-600 dark:text-gray-300">V drop</span>
          </div>
          <span class="font-mono font-bold text-slate-800 dark:text-gray-100">${vDrop !== undefined ? vDrop.toFixed(2) + " V" : "—"}</span>
        </div>
        <div class="flex items-center justify-between gap-4">
          <div class="flex items-center gap-1.5">
            <span class="w-2 h-2 rounded-full shadow-sm" style="background:${activeColors.current}"></span>
            <span class="text-slate-600 dark:text-gray-300">I load</span>
          </div>
          <span class="font-mono font-bold text-slate-800 dark:text-gray-100">${iLoad !== undefined ? iLoad.toFixed(1) + " mA" : "—"}</span>
        </div>
        <div class="flex items-center justify-between gap-4">
          <div class="flex items-center gap-1.5">
            <span class="w-2 h-2 rounded-full shadow-sm" style="background:${activeColors.power}"></span>
            <span class="text-slate-600 dark:text-gray-300">P diss</span>
          </div>
          <span class="font-mono font-bold text-slate-800 dark:text-gray-100">${pDiss !== undefined ? pDiss.toFixed(1) + " mW" : "—"}</span>
        </div>
      `;
    }

    html += `</div>`;
    tooltip.innerHTML = html;

    // Show and position crosshair
    crosshair.style.left = mouseX + "px";
    crosshair.style.top = yAxis.top + "px";
    crosshair.style.height = (yAxis.bottom - yAxis.top) + "px";
    crosshair.style.display = "block";

    // Show and position tooltip
    tooltip.style.display = "block";
    const tipWidth = tooltip.offsetWidth || 180;
    const tipHeight = tooltip.offsetHeight || 130;

    let left = mouseX + 16;
    let top = mouseY - 20;

    if (left + tipWidth > rect.width - 12) {
      left = mouseX - tipWidth - 16;
    }
    if (left < 10) left = 10;

    if (top + tipHeight > rect.height - 10) {
      top = rect.height - tipHeight - 10;
    }
    if (top < yAxis.top) top = yAxis.top;

    tooltip.style.left = left + "px";
    tooltip.style.top = top + "px";
  }

  function hide() {
    isHovering = false;
    tooltip.style.display = "none";
    crosshair.style.display = "none";
  }

  // Mousemove: updates values ONLY when cursor moves to a new point
  canvas.addEventListener("mousemove", (e) => {
    isHovering = true;
    lastClientX = e.clientX;
    lastClientY = e.clientY;
    updateTooltip(lastClientX, lastClientY);
  });

  canvas.addEventListener("mouseleave", hide);

  canvas.addEventListener("wheel", () => {
    if (isHovering) {
      requestAnimationFrame(() => updateTooltip(lastClientX, lastClientY));
    }
  }, { passive: true });

  canvas.addEventListener("touchstart", (e) => {
    if (e.touches && e.touches[0]) {
      isHovering = true;
      lastClientX = e.touches[0].clientX;
      lastClientY = e.touches[0].clientY;
      updateTooltip(lastClientX, lastClientY);
    }
  }, { passive: true });

  canvas.addEventListener("touchmove", (e) => {
    if (e.touches && e.touches[0]) {
      isHovering = true;
      lastClientX = e.touches[0].clientX;
      lastClientY = e.touches[0].clientY;
      updateTooltip(lastClientX, lastClientY);
    }
  }, { passive: true });

  canvas.addEventListener("touchend", hide);
}

// Initialize stationary tooltips on both charts
attachStationaryCursorTooltip(thermalChart, "chart-thermal", "thermal-tooltip", "thermal-crosshair", "thermal");
attachStationaryCursorTooltip(elecChart, "chart-electrical", "elec-tooltip", "elec-crosshair", "elec");

// ═══════════════════════════════════════════════════════════════════════════
// DYNAMIC Y-AXIS ZOOM FOR THERMAL CHART
// Auto-scales Y-axis tightly around active temperatures so errors and dynamics
// are clearly visible instead of squashed on a static 0-100°C scale.
// ═══════════════════════════════════════════════════════════════════════════

let isThermalAutoY = true;
let currentThermalYMin = null;
let currentThermalYMax = null;

const btnToggleThermalAutoY = $("btn-toggle-thermal-autoy");
const autoyDot = $("autoy-dot");
const autoyLabel = $("autoy-label");
const thermalYBadge = $("thermal-y-badge");

function updateThermalYAutoZoom(chart) {
  if (!isThermalAutoY) return;
  if (!chart || !chart.data || !chart.data.datasets) return;

  const xAxis = chart.scales.x;
  const xMin = (xAxis && !isNaN(xAxis.min)) ? xAxis.min : (chart.options.scales.x.min || 0);
  const xMax = (xAxis && !isNaN(xAxis.max)) ? xAxis.max : (chart.options.scales.x.max || Infinity);

  let minVal = Infinity;
  let maxVal = -Infinity;
  let count = 0;

  // Inspect visible datasets (Heatsink, Pred +3s, Pred +10s, Ambient if visible)
  chart.data.datasets.forEach((ds, dsIdx) => {
    if (chart.isDatasetVisible && !chart.isDatasetVisible(dsIdx)) return;
    if (!ds.data || ds.data.length === 0) return;

    for (let i = ds.data.length - 1; i >= 0; i--) {
      const pt = ds.data[i];
      if (!pt || pt.y === undefined || pt.y === null || isNaN(pt.y)) continue;

      if (pt.x !== undefined) {
        if (pt.x < xMin) {
          // Include boundary point for continuity, then stop looking backwards
          if (pt.y < minVal) minVal = pt.y;
          if (pt.y > maxVal) maxVal = pt.y;
          count++;
          break;
        }
        if (pt.x > xMax) continue;
      }

      if (pt.y < minVal) minVal = pt.y;
      if (pt.y > maxVal) maxVal = pt.y;
      count++;
    }
  });

  if (count === 0 || !isFinite(minVal) || !isFinite(maxVal)) return;

  // Ensure minimum range of 3.0°C so micro noise doesn't over-magnify
  const rawSpan = maxVal - minVal;
  const effectiveSpan = Math.max(3.0, rawSpan);
  const center = (minVal + maxVal) / 2;

  // Add 18% headroom above and below
  const padding = effectiveSpan * 0.18;
  const targetMin = Math.floor(center - (effectiveSpan / 2) - padding);
  const targetMax = Math.ceil(center + (effectiveSpan / 2) + padding);

  // Smooth filter
  if (currentThermalYMin === null) {
    currentThermalYMin = targetMin;
    currentThermalYMax = targetMax;
  } else {
    currentThermalYMin += (targetMin - currentThermalYMin) * 0.35;
    currentThermalYMax += (targetMax - currentThermalYMax) * 0.35;
  }

  const finalMin = Math.floor(currentThermalYMin);
  const finalMax = Math.ceil(currentThermalYMax);

  chart.options.scales.y.min = finalMin;
  chart.options.scales.y.max = finalMax;

  if (thermalYBadge && isThermalAutoY) {
    thermalYBadge.textContent = `${finalMin}–${finalMax}°C (Auto Y)`;
  }
}

function setThermalAutoY(enable) {
  isThermalAutoY = enable;
  if (isThermalAutoY) {
    if (autoyDot) autoyDot.className = "w-1.5 h-1.5 rounded-full bg-emerald-500";
    if (autoyLabel) autoyLabel.textContent = "Auto Y";
    if (thermalYBadge) {
      thermalYBadge.className = "ml-1 px-1.5 py-0.5 rounded text-[9px] font-mono bg-emerald-100 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800/60 font-semibold";
    }
    currentThermalYMin = null;
    currentThermalYMax = null;
    updateThermalYAutoZoom(thermalChart);
  } else {
    if (autoyDot) autoyDot.className = "w-1.5 h-1.5 rounded-full bg-slate-400";
    if (autoyLabel) autoyLabel.textContent = "0–100°C";
    if (thermalYBadge) {
      thermalYBadge.textContent = "0–100°C";
      thermalYBadge.className = "ml-1 px-1.5 py-0.5 rounded text-[9px] font-mono bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-slate-700";
    }
    thermalChart.options.scales.y.min = 0;
    thermalChart.options.scales.y.max = 100;
  }
  thermalChart.update("none");
}

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
  if (chart === thermalChart) updateThermalYAutoZoom(chart);
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
  if (chart === thermalChart) updateThermalYAutoZoom(chart);
  chart.update("none");
}

$("btn-thermal-zoom-in").addEventListener("click", () => zoomChartByFactor(thermalChart, 0.7));
$("btn-thermal-zoom-out").addEventListener("click", () => zoomChartByFactor(thermalChart, 1.4));
$("btn-reset-thermal-zoom").addEventListener("click", () => resetChartZoom(thermalChart));

if (btnToggleThermalAutoY) {
  btnToggleThermalAutoY.addEventListener("click", () => {
    setThermalAutoY(!isThermalAutoY);
  });
}

$("btn-elec-zoom-in").addEventListener("click", () => zoomChartByFactor(elecChart, 0.7));
$("btn-elec-zoom-out").addEventListener("click", () => zoomChartByFactor(elecChart, 1.4));
$("btn-reset-elec-zoom").addEventListener("click", () => resetChartZoom(elecChart));

// ═══════════════════════════════════════════════════════════════════════════
// THEME SWITCHING (White Background Default & Dark Mode Toggle)
// ═══════════════════════════════════════════════════════════════════════════

function applyChartTheme(chart, dark) {
  if (!chart || !chart.options) return;
  const c = getThemeColors(dark);

  if (chart.options.scales.x) {
    chart.options.scales.x.ticks.color = c.tick;
    chart.options.scales.x.title.color = c.tick;
    chart.options.scales.x.grid.color  = c.grid;
  }

  if (chart === thermalChart) {
    if (chart.options.scales.y) {
      chart.options.scales.y.ticks.color = c.tick;
      chart.options.scales.y.title.color = c.tick;
      chart.options.scales.y.grid.color  = c.grid;
    }
    if (chart.data.datasets[0]) {
      chart.data.datasets[0].borderColor = c.measured;
      chart.data.datasets[0].backgroundColor = c.fillMeas;
    }
    if (chart.data.datasets[1]) {
      chart.data.datasets[1].borderColor = c.pred3s;
      chart.data.datasets[1].backgroundColor = c.fill3s;
    }
    if (chart.data.datasets[2]) {
      chart.data.datasets[2].borderColor = c.pred10s;
      chart.data.datasets[2].backgroundColor = c.fill10s;
    }
    if (chart.data.datasets[3]) {
      chart.data.datasets[3].borderColor = c.ambient;
      chart.data.datasets[3].backgroundColor = c.fillAmb;
    }
  } else if (chart === elecChart) {
    if (chart.options.scales.yV) {
      chart.options.scales.yV.ticks.color = c.vdrop;
      chart.options.scales.yV.title.color = c.vdrop;
      chart.options.scales.yV.grid.color  = c.grid;
    }
    if (chart.options.scales.yI) {
      chart.options.scales.yI.ticks.color = c.current;
      chart.options.scales.yI.title.color = c.current;
    }
    if (chart.options.scales.yP) {
      chart.options.scales.yP.ticks.color = c.power;
      chart.options.scales.yP.title.color = c.power;
    }
    if (chart.data.datasets[0]) chart.data.datasets[0].borderColor = c.vdrop;
    if (chart.data.datasets[1]) chart.data.datasets[1].borderColor = c.current;
    if (chart.data.datasets[2]) chart.data.datasets[2].borderColor = c.power;
  }

  if (chart.options.plugins?.legend?.labels) {
    chart.options.plugins.legend.labels.color = c.tick;
  }
  if (chart.options.plugins?.tooltip) {
    chart.options.plugins.tooltip.backgroundColor = c.tooltipBg;
    chart.options.plugins.tooltip.titleColor       = c.tooltipTitle;
    chart.options.plugins.tooltip.bodyColor        = c.tooltipBody;
    chart.options.plugins.tooltip.borderColor      = c.tooltipBorder;
  }
}

function setTheme(dark) {
  isDarkMode = dark;
  activeColors = getThemeColors(dark);
  chartGridColor = activeColors.grid;
  chartTickColor = activeColors.tick;
  document.documentElement.classList.toggle("dark", dark);

  const sunIcon = $("theme-icon-sun");
  const moonIcon = $("theme-icon-moon");
  const themeLabel = $("theme-label");

  if (sunIcon && moonIcon && themeLabel) {
    if (dark) {
      sunIcon.classList.remove("hidden");
      moonIcon.classList.add("hidden");
      themeLabel.textContent = "Light Mode";
    } else {
      sunIcon.classList.add("hidden");
      moonIcon.classList.remove("hidden");
      themeLabel.textContent = "Dark Mode";
    }
  }

  applyChartTheme(thermalChart, dark);
  applyChartTheme(elecChart, dark);
  thermalChart.update("none");
  elecChart.update("none");

  localStorage.setItem(THEME_KEY, dark ? "dark" : "light");
}

function initTheme() {
  const saved = localStorage.getItem(THEME_KEY);
  // Default is WHITE BACKGROUND (isDarkMode = false)
  const shouldBeDark = saved === "dark";
  setTheme(shouldBeDark);
}

// ═══════════════════════════════════════════════════════════════════════════
// EXPORT GRAPH AS PNG (Proper labels, legends, grid, and solid background)
// ═══════════════════════════════════════════════════════════════════════════

function exportChartAsPNG(chart, chartTitle, defaultFilename) {
  if (!chart || !chart.canvas) return;

  const wasDark = isDarkMode;
  if (wasDark) {
    // Temporarily switch chart colors to light theme for crisp white-background export
    applyChartTheme(chart, false);
    chart.render();
  }

  try {
    const srcCanvas = chart.canvas;
    const pixelRatio = window.devicePixelRatio || 1;
    const scale = Math.max(2, pixelRatio); // at least 2x scale for crisp print quality

    const clientWidth = srcCanvas.clientWidth || (srcCanvas.width / pixelRatio);
    const clientHeight = srcCanvas.clientHeight || (srcCanvas.height / pixelRatio);

    const padX = 24 * scale;
    const padTop = 64 * scale;
    const padBottom = 24 * scale;

    const outCanvas = document.createElement("canvas");
    outCanvas.width = clientWidth * scale + padX * 2;
    outCanvas.height = clientHeight * scale + padTop + padBottom;

    const ctx = outCanvas.getContext("2d");
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";

    // 1. Solid pure white background (never transparent)
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, outCanvas.width, outCanvas.height);

    // 2. Crisp container border
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1 * scale;
    ctx.strokeRect(scale, scale, outCanvas.width - 2 * scale, outCanvas.height - 2 * scale);

    // 3. Header title
    ctx.fillStyle = "#0f172a"; // slate-900
    ctx.font = `bold ${13 * scale}px Inter, system-ui, -apple-system, sans-serif`;
    ctx.textAlign = "left";
    ctx.fillText(chartTitle, padX, 26 * scale);

    // 4. Header subtitle with timestamp & window
    const now = new Date();
    const dateStr = now.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' }) +
      " " + now.toLocaleTimeString();
    const spanSec = chart._userSpan ? Math.round(chart._userSpan / 1000) : WINDOW_SEC;
    ctx.fillStyle = "#64748b"; // slate-500
    ctx.font = `${9 * scale}px Inter, system-ui, -apple-system, sans-serif`;
    const subText = chart === thermalChart
      ? `Project ThermoGuard • Window: ${spanSec}s • Scale: 0–100°C (Cutoff: 80°C) • ${dateStr}`
      : `Project ThermoGuard • Window: ${spanSec}s • Telemetry: V_drop, I_load, P_diss • ${dateStr}`;
    ctx.fillText(subText, padX, 42 * scale);

    // 5. Divider line
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1 * scale;
    ctx.beginPath();
    ctx.moveTo(padX, 52 * scale);
    ctx.lineTo(outCanvas.width - padX, 52 * scale);
    ctx.stroke();

    // 6. Draw chart canvas (which includes legends, labels, axis lines, grid, and data)
    ctx.drawImage(
      srcCanvas,
      0, 0, srcCanvas.width, srcCanvas.height,
      padX, padTop, clientWidth * scale, clientHeight * scale
    );

    // 7. Footer metadata
    ctx.fillStyle = "#94a3b8"; // slate-400
    ctx.font = `${8 * scale}px Inter, system-ui, -apple-system, sans-serif`;
    ctx.textAlign = "right";
    ctx.fillText("KA7808 Thermal Profiling • STM32 Nucleo-F401RE • TinyML Dual-Horizon", outCanvas.width - padX, outCanvas.height - 8 * scale);

    // 8. Trigger PNG download
    const tsStr = now.toISOString().replace(/[:.]/g, "-").slice(0, 19);
    const link = document.createElement("a");
    link.download = `${defaultFilename}_${tsStr}.png`;
    link.href = outCanvas.toDataURL("image/png");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  } finally {
    if (wasDark) {
      // Restore dark theme if it was previously dark
      applyChartTheme(chart, true);
      chart.render();
    }
  }
}

if ($("btn-export-thermal-png")) {
  $("btn-export-thermal-png").addEventListener("click", () => {
    exportChartAsPNG(thermalChart, "Project ThermoGuard — Thermal Tracking Profile", "thermoguard_thermal_graph");
  });
}

if ($("btn-export-elec-png")) {
  $("btn-export-elec-png").addEventListener("click", () => {
    exportChartAsPNG(elecChart, "Project ThermoGuard — Electrical Metrics Profile", "thermoguard_electrical_graph");
  });
}

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
// TRANSIENT GLITCH FILTER (Hold-Last-Valid on 10 °C fluctuation for <= 2 points)
// ═══════════════════════════════════════════════════════════════════════════

let cleanTmeas = null;
let measGlitchCount = 0;
let cleanTpred3s = null;
let pred3sGlitchCount = 0;
let cleanTpred10s = null;
let pred10sGlitchCount = 0;

function filterTempGlitch(type, val, threshold = 10.0, maxPoints = 2) {
  if (type === "meas") {
    if (cleanTmeas === null) {
      if (val > 0) cleanTmeas = val;
      return val;
    }
    const isGlitch = (val <= 0) || Math.abs(val - cleanTmeas) >= threshold;
    if (isGlitch) {
      measGlitchCount++;
      if (measGlitchCount <= maxPoints) {
        return cleanTmeas;
      } else {
        cleanTmeas = val;
        measGlitchCount = 0;
        return val;
      }
    } else {
      cleanTmeas = val;
      measGlitchCount = 0;
      return val;
    }
  } else if (type === "pred3s") {
    if (cleanTpred3s === null) {
      if (val > 0) cleanTpred3s = val;
      return val;
    }
    const isGlitch = (val <= 0) || Math.abs(val - cleanTpred3s) >= threshold;
    if (isGlitch) {
      pred3sGlitchCount++;
      if (pred3sGlitchCount <= maxPoints) {
        return cleanTpred3s;
      } else {
        cleanTpred3s = val;
        pred3sGlitchCount = 0;
        return val;
      }
    } else {
      cleanTpred3s = val;
      pred3sGlitchCount = 0;
      return val;
    }
  } else {
    if (cleanTpred10s === null) {
      if (val > 0) cleanTpred10s = val;
      return val;
    }
    const isGlitch = (val <= 0) || Math.abs(val - cleanTpred10s) >= threshold;
    if (isGlitch) {
      pred10sGlitchCount++;
      if (pred10sGlitchCount <= maxPoints) {
        return cleanTpred10s;
      } else {
        cleanTpred10s = val;
        pred10sGlitchCount = 0;
        return val;
      }
    } else {
      cleanTpred10s = val;
      pred10sGlitchCount = 0;
      return val;
    }
  }
}

function resetTempGlitchFilters() {
  cleanTmeas = null;
  measGlitchCount = 0;
  cleanTpred3s = null;
  pred3sGlitchCount = 0;
  cleanTpred10s = null;
  pred10sGlitchCount = 0;
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

  // ── Transient Glitch Filter ──
  const pred3sRaw = d.T_predicted_3s !== undefined ? d.T_predicted_3s : (d.T_predicted !== undefined ? d.T_predicted : d.T_measured);
  const pred10sRaw = d.T_predicted_10s !== undefined ? d.T_predicted_10s : pred3sRaw;

  d.T_measured      = filterTempGlitch("meas", d.T_measured);
  d.T_predicted_3s  = filterTempGlitch("pred3s", pred3sRaw);
  d.T_predicted_10s = filterTempGlitch("pred10s", pred10sRaw);
  d.T_predicted     = d.T_predicted_3s;

  // ── Lookahead Errors: Compare current ground truth with predictions made 3s and 10s ago ──
  pred3sLookaheadBuffer.push(d.T_predicted_3s);
  if (pred3sLookaheadBuffer.length > LOOKAHEAD_3S_SAMPLES) {
    const pastPred3s = pred3sLookaheadBuffer.shift();
    d.tracking_error_3s = Math.abs(d.T_measured - pastPred3s);
  } else {
    d.tracking_error_3s = Math.abs(d.T_measured - d.T_predicted_3s);
  }

  pred10sLookaheadBuffer.push(d.T_predicted_10s);
  if (pred10sLookaheadBuffer.length > LOOKAHEAD_10S_SAMPLES) {
    const pastPred10s = pred10sLookaheadBuffer.shift();
    d.tracking_error_10s = Math.abs(d.T_measured - pastPred10s);
  } else {
    d.tracking_error_10s = Math.abs(d.T_measured - d.T_predicted_10s);
  }

  d.tracking_error = d.tracking_error_3s;

  const ts = d.timestamp_ms;

  // ── Push 4 lines to thermal chart: Heatsink, Pred(+3s), Pred(+10s), Ambient ──
  thermalChart.data.labels.push(ts);
  thermalChart.data.datasets[0].data.push({ x: ts, y: d.T_measured });
  thermalChart.data.datasets[1].data.push({ x: ts, y: d.T_predicted_3s });
  thermalChart.data.datasets[2].data.push({ x: ts, y: d.T_predicted_10s });
  thermalChart.data.datasets[3].data.push({ x: ts, y: d.T_ambient });

  // ── Push to electrical chart ──
  elecChart.data.labels.push(ts);
  elecChart.data.datasets[0].data.push({ x: ts, y: d.V_drop });
  elecChart.data.datasets[1].data.push({ x: ts, y: d.I_mA });
  elecChart.data.datasets[2].data.push({ x: ts, y: d.P_mW });

  // ── Record frame into session export log ──
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
    T_predicted_3s: d.T_predicted_3s,
    T_predicted_10s: d.T_predicted_10s,
    infer_us: d.infer_us !== undefined ? d.infer_us : 0,
    tracking_error_3s: d.tracking_error_3s,
    tracking_error_10s: d.tracking_error_10s,
    R_theta_JA: d.R_theta_JA !== null ? d.R_theta_JA : "",
    fault_code: d.fault_code,
    fault_flags: (d.faults && d.faults.length > 0) ? d.faults.join("|") : "OK",
  });

  currentLatestTs = ts;

  // ── Advance live rolling window ──
  advanceLiveWindow(thermalChart, ts);
  advanceLiveWindow(elecChart, ts);

  // ── Dynamically zoom Y-axis on thermal chart ──
  updateThermalYAutoZoom(thermalChart);

  // ── Trim historical buffer ──
  trimChart(thermalChart, ts);
  trimChart(elecChart, ts);

  // ── Update charts (batched) ──
  requestAnimationFrame(() => {
    thermalChart.update("none");
    elecChart.update("none");
  });

  // ── Error Accumulation (3s and 10s lookahead) ──
  const tErr3s  = d.tracking_error_3s;
  const tErr10s = d.tracking_error_10s;
  errorSum3s    += tErr3s;
  errorSum10s   += tErr10s;
  errorSqSum3s  += tErr3s * tErr3s;
  errorSqSum10s += tErr10s * tErr10s;
  errorCount++;

  const avgErr3s   = errorSum3s / errorCount;
  const avgErr10s  = errorSum10s / errorCount;
  const curRmse3s  = Math.sqrt(errorSqSum3s / errorCount);
  const curRmse10s = Math.sqrt(errorSqSum10s / errorCount);
  rmseSum3s  += curRmse3s;
  rmseSum10s += curRmse10s;
  const avgRmse3s  = rmseSum3s / errorCount;
  const avgRmse10s = rmseSum10s / errorCount;

  // ── KPI: Tracking error (displays 3s and 10s lookahead error) ──
  if (kpiErrorVal) {
    kpiErrorVal.textContent = `${tErr3s.toFixed(2)} / ${tErr10s.toFixed(2)}`;
  }
  if (kpiErrorAvgVal) {
    kpiErrorAvgVal.textContent = `${avgErr3s.toFixed(2)} / ${avgErr10s.toFixed(2)}`;
  }

  // ── KPI: Ambient Temp (Replaced MAE) ──
  if (kpiAmbTempVal) {
    kpiAmbTempVal.textContent = d.T_ambient.toFixed(2) + " °C";
  }

  // ── KPI: RMSE (3s and 10s lookahead) ──
  if (kpiRmseVal) {
    kpiRmseVal.textContent = `${curRmse3s.toFixed(2)} / ${curRmse10s.toFixed(2)}`;
  }
  if (kpiRmseAvgVal) {
    kpiRmseAvgVal.textContent = `${avgRmse3s.toFixed(2)} / ${avgRmse10s.toFixed(2)}`;
  }

  // ── KPI: Predicted Temperatures (+3s and +10s) ──
  if (kpiPred3sVal) {
    kpiPred3sVal.textContent = d.T_predicted_3s.toFixed(2) + " °C";
  }
  if (kpiPred10sVal) {
    kpiPred10sVal.textContent = d.T_predicted_10s.toFixed(2) + " °C";
  }
  if (kpiRthetaVal) {
    kpiRthetaVal.textContent = d.R_theta_JA != null ? d.R_theta_JA.toFixed(2) : "—";
  }

  // ── KPI: Heatsink temperature + safety zone ──
  const t = d.T_measured;
  kpiTempVal.textContent = t.toFixed(1) + " °C";
  updateThermalZone(t);

  // ── KPI: Prediction Time (DWT Latency) ──
  const inferUs = (d.infer_us !== undefined && d.infer_us !== null) ? Number(d.infer_us) : 0;
  if (inferUs > 0) {
    latencySum += inferUs;
    latencyCount++;

    const fmtLatency = (val) => (val >= 1000 ? `${(val / 1000).toFixed(2)} ms` : `${Math.round(val)} µs`);

    if (kpiPredTimeVal) kpiPredTimeVal.textContent = fmtLatency(inferUs);
    if (kpiPredTimeSub) {
      kpiPredTimeSub.textContent = inferUs >= 1000
        ? `${Math.round(inferUs).toLocaleString()} µs · DWT`
        : `${(inferUs / 1000).toFixed(3)} ms · DWT`;
    }

    if (liveLatency) liveLatency.textContent = fmtLatency(inferUs);
    if (modelLatency) modelLatency.textContent = fmtLatency(inferUs);
  }

  // ── MAX31855 fault decode (sidebar indicators) ──
  updateFaults(d.fault_code, d.faults);

  // ── Live readings ──
  if (liveVin)   liveVin.textContent      = d.V_in.toFixed(3)   + " V";
  if (liveVout)  liveVout.textContent     = d.V_out.toFixed(3)   + " V";
  if (liveVdrop) liveVdrop.textContent    = d.V_drop.toFixed(3)  + " V";
  if (liveIload) liveIload.textContent    = d.I_mA.toFixed(1)    + " mA";
  if (livePdiss) livePdiss.textContent    = d.P_mW.toFixed(1)    + " mW";
  if (liveTamb)  liveTamb.textContent     = d.T_ambient.toFixed(2)  + " °C";
  if (liveTmeas) liveTmeas.textContent    = d.T_measured.toFixed(2) + " °C";
  if (liveTpred3s)  liveTpred3s.textContent  = d.T_predicted_3s.toFixed(2)  + " °C";
  if (liveTpred10s) liveTpred10s.textContent = d.T_predicted_10s.toFixed(2) + " °C";
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
// THERMAL SAFETY ZONE (Cutoff: 80°C, Max: 100°C)
// ═══════════════════════════════════════════════════════════════════════════

const T_MIN = 0;
const T_WARN = 80;
const T_CUTOFF = 80;
const T_MAX = 100;

function updateThermalZone(tempC) {
  // Marker position (0°C → 0%, 100°C → 100%)
  const rawPct = ((tempC - T_MIN) / (T_MAX - T_MIN)) * 100;
  const pct = Math.max(1, Math.min(99, rawPct));
  thermalMarker.style.left = pct.toFixed(1) + "%";

  if (tempC >= T_MAX) {
    // CRITICAL THERMAL OVER-MAX (≥100°C) — Flashing red + audible alarm
    kpiThermal.classList.add("glow-red");
    kpiThermal.classList.remove("glow-amber");
    thermalDot.className = "w-2.5 h-2.5 rounded-full bg-red-600 dot-blink";
    thermalLabel.textContent = "OVER-MAX (≥100°C)";
    thermalLabel.className = "text-[10px] font-bold text-red-600 dark:text-red-400 tracking-tight";
    kpiTempVal.className = "text-xl font-bold text-red-600 dark:text-red-500 tabular-nums";
    startAlarm();
  } else if (tempC >= T_CUTOFF) {
    // THERMAL CUTOFF (≥80°C) — Flashing rose/red + audible alarm
    kpiThermal.classList.add("glow-red");
    kpiThermal.classList.remove("glow-amber");
    thermalDot.className = "w-2.5 h-2.5 rounded-full bg-rose-600 dot-blink";
    thermalLabel.textContent = "CUTOFF (≥80°C)";
    thermalLabel.className = "text-[10px] font-bold text-rose-600 dark:text-rose-400 tracking-tight";
    kpiTempVal.className = "text-xl font-bold text-rose-600 dark:text-rose-400 tabular-nums";
    startAlarm();
  } else if (tempC >= T_WARN) {
    // ELEVATED / WARNING (≥80°C) — Amber glow
    kpiThermal.classList.remove("glow-red");
    kpiThermal.classList.add("glow-amber");
    thermalDot.className = "w-2.5 h-2.5 rounded-full bg-amber-500";
    thermalLabel.textContent = "WARNING (≥80°C)";
    thermalLabel.className = "text-[10px] font-semibold text-amber-600 dark:text-amber-400";
    kpiTempVal.className = "text-xl font-bold text-amber-600 dark:text-amber-400 tabular-nums";
    stopAlarm();
  } else {
    // NOMINAL SAFE (< 80°C) — Green
    kpiThermal.classList.remove("glow-red", "glow-amber");
    thermalDot.className = "w-2.5 h-2.5 rounded-full bg-emerald-500";
    thermalLabel.textContent = "SAFE (<80°C)";
    thermalLabel.className = "text-[10px] font-semibold text-emerald-600 dark:text-emerald-400";
    kpiTempVal.className = "text-xl font-bold text-emerald-600 dark:text-emerald-400 tabular-nums";
    stopAlarm();
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// FAULT DECODE
// ═══════════════════════════════════════════════════════════════════════════

const FAULT_NAMES = ["FAULT", "SCG", "SCV", "OC"];
const DIAG_DOTS   = [$("diag-f0"), $("diag-f1"), $("diag-f2"), $("diag-f3")];

function updateFaults(code, faultList) {
  // Diagnostic indicator dots in sidebar
  for (let i = 0; i < 4; i++) {
    if (DIAG_DOTS[i]) {
      if (code & (1 << i)) {
        DIAG_DOTS[i].className = "w-2 h-2 rounded-full bg-rose-500 dot-blink";
      } else {
        DIAG_DOTS[i].className = "w-2 h-2 rounded-full bg-slate-300 dark:bg-gray-600";
      }
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
    connLabel.className    = "text-xs text-emerald-600 dark:text-emerald-400 font-medium";
    btnConnect.classList.add("hidden");
    btnDisconnect.classList.remove("hidden");
    sessionStart = sessionStart || Date.now();
  } else {
    connDot.className      = "w-2 h-2 rounded-full bg-slate-400 dark:bg-gray-600";
    connLabel.textContent  = "Offline";
    connLabel.className    = "text-xs text-slate-500 dark:text-gray-500 font-medium";
    btnConnect.classList.remove("hidden");
    btnDisconnect.classList.add("hidden");
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
      pred3sLookaheadBuffer.length = 0;
      pred10sLookaheadBuffer.length = 0;
      errorSum3s = errorSum10s = errorSqSum3s = errorSqSum10s = errorCount = frameCount = malformedCount = 0;
      rmseSum3s = rmseSum10s = 0;
      latencySum = latencyCount = 0;
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
  } catch (e) {
    console.error("Disconnect error:", e);
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
    "T_predicted_3s",
    "T_predicted_10s",
    "infer_us",
    "tracking_error_3s",
    "tracking_error_10s",
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

async function resetWholePage() {
  // If currently connected or receiving, disconnect and reset backend state
  try {
    await fetch(`${API}/reset`, { method: "POST" });
  } catch (e) {
    console.warn("Backend reset request error:", e);
  }
  setConnectionUI(false);

  // Clear charts
  thermalChart.data.labels = [];
  thermalChart.data.datasets.forEach((ds) => (ds.data = []));
  thermalChart.resetZoom();
  delete thermalChart.options.scales.x.min;
  delete thermalChart.options.scales.x.max;
  thermalChart._isZoomed = false;
  thermalChart._isPannedBack = false;
  thermalChart._userSpan = WINDOW_SEC * 1000;
  updateZoomLabel(thermalChart);
  thermalChart.update("none");

  elecChart.data.labels = [];
  elecChart.data.datasets.forEach((ds) => (ds.data = []));
  elecChart.resetZoom();
  delete elecChart.options.scales.x.min;
  delete elecChart.options.scales.x.max;
  elecChart._isZoomed = false;
  elecChart._isPannedBack = false;
  elecChart._userSpan = WINDOW_SEC * 1000;
  updateZoomLabel(elecChart);
  elecChart.update("none");

  // Reset counters & accumulators
  resetTempGlitchFilters();
  errorSum3s = 0;
  errorSum10s = 0;
  errorSqSum3s = 0;
  errorSqSum10s = 0;
  errorCount = 0;
  rmseSum3s = 0;
  rmseSum10s = 0;
  frameCount = 0;
  malformedCount = 0;
  latencySum = 0;
  latencyCount = 0;
  currentLatestTs = 0;
  sessionTelemetryLog.length = 0;
  pred3sLookaheadBuffer.length = 0;
  pred10sLookaheadBuffer.length = 0;
  sessionStart = null;

  // Reset all top row KPI cards
  kpiErrorVal.textContent = "—";
  if (kpiErrorAvgVal) kpiErrorAvgVal.textContent = "—";
  if (kpiAmbTempVal) kpiAmbTempVal.textContent = "—";
  kpiRmseVal.textContent = "—";
  if (kpiRmseAvgVal) kpiRmseAvgVal.textContent = "—";
  if (kpiPred3sVal) kpiPred3sVal.textContent = "—";
  if (kpiPred10sVal) kpiPred10sVal.textContent = "—";
  kpiTempVal.textContent = "—";
  kpiTempVal.className = "text-xl font-bold text-emerald-600 dark:text-emerald-400 tabular-nums";
  if (kpiPredTimeVal) kpiPredTimeVal.textContent = "—";
  if (kpiPredTimeSub) kpiPredTimeSub.textContent = "DWT Cycle Latency";

  // Reset thermal zone & safety indicators
  thermalMarker.style.left = "1%";
  kpiThermal.classList.remove("glow-red", "glow-amber");
  thermalDot.className = "w-2.5 h-2.5 rounded-full bg-emerald-500";
  thermalLabel.textContent = "SAFE";
  thermalLabel.className = "text-[10px] font-semibold text-emerald-600 dark:text-emerald-400";
  stopAlarm();

  // Reset live readings
  liveVin.textContent = "—";
  liveVout.textContent = "—";
  liveVdrop.textContent = "—";
  liveIload.textContent = "—";
  livePdiss.textContent = "—";
  liveTamb.textContent = "—";
  liveTmeas.textContent = "—";
  if (liveTpred3s) liveTpred3s.textContent = "—";
  if (liveTpred10s) liveTpred10s.textContent = "—";
  if (liveLatency) liveLatency.textContent = "—";
  if (modelLatency) modelLatency.textContent = "—";

  // Reset session stats
  statFrames.textContent = "0";
  statMalformed.textContent = "0";
  statUptime.textContent = "—";

  // Reset diagnostic dots
  for (let i = 0; i < 4; i++) {
    if (DIAG_DOTS[i]) DIAG_DOTS[i].className = "w-2 h-2 rounded-full bg-slate-300 dark:bg-gray-600";
  }

  // Visual button feedback
  if (btnResetPage) {
    const origHtml = btnResetPage.innerHTML;
    btnResetPage.innerHTML = `<svg class="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg><span class="text-emerald-600 dark:text-emerald-400">Reset!</span>`;
    setTimeout(() => {
      btnResetPage.innerHTML = origHtml;
    }, 1000);
  }
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
if (btnThemeToggle) btnThemeToggle.addEventListener("click", () => setTheme(!isDarkMode));
if (btnResetPage) btnResetPage.addEventListener("click", resetWholePage);
btnExportCsv.addEventListener("click", exportSessionCSV);

// ═══════════════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════════════

(function init() {
  initTheme();
  fetchPorts();
  wsConnect();
})();
