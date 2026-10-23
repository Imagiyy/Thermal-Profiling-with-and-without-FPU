#!/usr/bin/env python3
"""
train_mlp.py — Rigorous Zero-Leakage TinyML Pipeline for STM32F401RE.

This script trains an MLPRegressor on KA7808 linear voltage regulator thermal telemetry
under strict zero-data-leakage constraints, validates performance on an unseen future
time horizon, generates evaluation plots, and exports the 6-benchmark C inference engine.

ZERO-LEAKAGE AUDIT GUARANTEES:
  1. Temporal Holdout Split: First 80% of time for training, last 20% for testing.
     Never uses random shuffling (train_test_split shuffle=True), which leaks thermal
     inertia across adjacent 500ms time samples.
  2. Purge & Embargo Buffer: A 15-second (30-sample) dead-band gap at the split boundary
     is discarded to prevent autoregressive boundary leakage.
  3. Scaler Fit Policy: StandardScaler.fit() is called strictly on df_train. The test set
     is strictly transformed using training statistics. C normalization constants are
     derived strictly from df_train.
  4. Label & Target Leakage Prevention: Prior prediction columns ('predicted_temp_C')
     are stripped; 'zeroinput.csv' non-operational artifacts are excluded.
  5. Physical Operating Filter: Discards power-off transient noise (Vin < 10.0 V,
     signed integer underflows I < 0, sensor fault codes).
  6. Zero Memory Leakage in C: 100% static/stack allocation (0 bytes heap malloc/free).
  7. Zero Timebase Leakage in Firmware: Reset hook mlp_reset_timebase() clears accumulator
     whenever Vin < 1.0 V.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ───────────────────────────── Paths ──────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"
OUTPUT_DIR  = SCRIPT_DIR

# ────────────────────── 1. Zero-Leakage Data Load ──────────────────
def load_clean_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load operational CSV datasets, filter electrical noise, and perform
    strict temporal holdout splitting with an embargo buffer.
    """
    files = ['10ohm.csv', '20ohm.csv', 'noload.csv']
    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []

    print("=" * 70)
    print("  [1/5] ZERO-LEAKAGE DATA INGESTION & TEMPORAL SPLIT")
    print("=" * 70)

    for fname in files:
        fpath = DATASET_DIR / fname
        if not fpath.exists():
            print(f"ERROR: Missing dataset {fpath}")
            sys.exit(1)

        df = pd.read_csv(fpath)
        df.columns = [c.strip() for c in df.columns]
        src = fname.replace('.csv', '')

        # Prevent label leakage from previous model outputs
        if 'predicted_temp_C' in df.columns:
            df = df.drop(columns=['predicted_temp_C'])

        # Filter unpowered shutdown transients and sensor fault codes
        mask = (
            (df['vin_V'] >= 10.0) &
            (df['fault_code'] == 0) &
            (df['current_mA'] >= 0) &
            (df['current_mA'] <= 1500) &
            (df['power_loss_mW'] >= 0) &
            (df['heatsink_temp_C'] > 20.0)
        )
        df_clean = df[mask].copy()

        # Compute causal elapsed operational timebase (minutes from continuous start)
        t0 = df_clean['timestamp_ms'].min()
        df_clean['elapsed_min'] = (df_clean['timestamp_ms'] - t0) / 60000.0
        df_clean['source'] = src

        # Strict Temporal Split: First 80% train, last 20% test
        n_total = len(df_clean)
        cutoff = int(n_total * 0.80)
        embargo_samples = 30  # 15 seconds at 500 ms sampling rate

        df_train_sub = df_clean.iloc[:cutoff].copy()
        df_test_sub  = df_clean.iloc[cutoff + embargo_samples:].copy()

        train_frames.append(df_train_sub)
        test_frames.append(df_test_sub)

        print(f"  {fname:12s}: Total {n_total} rows | Train: {len(df_train_sub)} | "
              f"Embargo Gap: {embargo_samples} rows (15s) | Test: {len(df_test_sub)}")

    df_train = pd.concat(train_frames, ignore_index=True)
    df_test  = pd.concat(test_frames, ignore_index=True)

    print(f"\n  Final Train Partition: {len(df_train)} samples (Strictly first 80% time)")
    print(f"  Final Test Partition:  {len(df_test)} samples (Strictly unseen future time)")
    print("  Zero temporal overlap. Zero adjacent thermal leakage.")

    return df_train, df_test

# ──────────────────────── 2. Train Model ──────────────────────────
def train_zero_leakage_mlp(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    hidden_dim: int = 32,
    max_iter: int = 1000,
    seed: int = 42,
) -> MLPRegressor:
    """Train MLP on scaled training data strictly."""
    print("\n" + "=" * 70)
    print(f"  [2/5] TRAINING MODEL (Architecture: 4 -> {hidden_dim} -> 1, MaxIter={max_iter})")
    print("=" * 70)

    mlp = MLPRegressor(
        hidden_layer_sizes=(hidden_dim,),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        learning_rate_init=2e-3,
        max_iter=max_iter,
        tol=1e-6,
        random_state=seed,
        verbose=False,
    )
    mlp.fit(X_train_scaled, y_train)

    print(f"  Converged in {mlp.n_iter_} iterations | Final Training Loss: {mlp.loss_:.6f}")
    return mlp

# ─────────────────── 3. Evaluate & Plot ───────────────────────────
def evaluate_and_plot(
    mlp: MLPRegressor,
    scaler: StandardScaler,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> dict:
    """Evaluate performance strictly on unseen future test set and generate plot."""
    print("\n" + "=" * 70)
    print("  [3/5] MODEL EVALUATION ON UNSEEN HOLDOUT TEST SET")
    print("=" * 70)

    X_train_s = scaler.transform(df_train[feature_cols].values)
    X_test_s  = scaler.transform(df_test[feature_cols].values)
    y_train   = df_train[target_col].values
    y_test    = df_test[target_col].values

    y_pred_train = mlp.predict(X_train_s)
    y_pred_test  = mlp.predict(X_test_s)

    metrics = {
        "zero_leakage_guarantee": {
            "temporal_split": "First 80% train, unseen last 20% test per continuous load dataset",
            "embargo_gap_seconds": 15,
            "scaler_policy": "Fitted strictly on df_train; only transform() called on df_test",
            "dropped_datasets": ["zeroinput.csv"],
            "dropped_columns": ["predicted_temp_C"]
        },
        "train": {
            "MAE":  float(mean_absolute_error(y_train, y_pred_train)),
            "RMSE": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
            "R2":   float(r2_score(y_train, y_pred_train)),
            "samples": len(y_train)
        },
        "holdout_test": {
            "MAE":  float(mean_absolute_error(y_test, y_pred_test)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
            "R2":   float(r2_score(y_test, y_pred_test)),
            "samples": len(y_test)
        }
    }

    print(f"  Holdout Test MAE:  {metrics['holdout_test']['MAE']:.4f} °C")
    print(f"  Holdout Test RMSE: {metrics['holdout_test']['RMSE']:.4f} °C")
    print(f"  Holdout Test R²:   {metrics['holdout_test']['R2']:.4f}")

    # Generate Report Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Zero-Leakage TinyML Thermal Prediction Report", fontsize=15, fontweight="bold")

    # (a) Actual vs Predicted (Holdout Test)
    ax = axes[0, 0]
    ax.scatter(y_test, y_pred_test, alpha=0.5, s=12, c="#1E88E5", edgecolors="none")
    mn, mx = min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.8, label="Ideal 1:1")
    ax.set_xlabel("Actual Heatsink Temp (°C)")
    ax.set_ylabel("Predicted Heatsink Temp (°C)")
    ax.set_title(f"Unseen Future Holdout Test Set\nMAE={metrics['holdout_test']['MAE']:.3f}°C  R²={metrics['holdout_test']['R2']:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Residual Histogram
    ax = axes[0, 1]
    residuals = y_test - y_pred_test
    ax.hist(residuals, bins=40, color="#43A047", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", lw=1.5)
    ax.set_xlabel("Residual Error (°C)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Distribution (Holdout Test)\nMean={residuals.mean():.3f}°C  Std={residuals.std():.3f}°C")
    ax.grid(True, alpha=0.3)

    # (c) Training Loss Curve
    ax = axes[1, 0]
    ax.plot(mlp.loss_curve_, color="#E53935", lw=1.5)
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(f"Training Convergence ({mlp.n_iter_} iterations)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # (d) Sequential Tracking over Unseen Future Time
    ax = axes[1, 1]
    n_plot = min(400, len(y_test))
    idx = np.arange(n_plot)
    ax.plot(idx, y_test[:n_plot], label="Actual (°C)", color="#1E88E5", lw=1.5)
    ax.plot(idx, y_pred_test[:n_plot], label="Predicted (°C)", color="#FB8C00", lw=1.5, alpha=0.9)
    ax.fill_between(idx, y_test[:n_plot], y_pred_test[:n_plot], alpha=0.2, color="#E53935")
    ax.set_xlabel("Sample Index in Unseen Future")
    ax.set_ylabel("Heatsink Temp (°C)")
    ax.set_title("Tracking Unseen Future Horizon (First 400 Samples)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "training_report.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Saved evaluation plot → {plot_path}")

    return metrics

# ─────────────────── 4. Save Artifacts ────────────────────────────
def save_artifacts(
    mlp: MLPRegressor,
    scaler: StandardScaler,
    feature_cols: list[str],
    metrics: dict,
):
    """Save model bundle, scaler parameters, raw weights, and metrics."""
    print("\n" + "=" * 70)
    print("  [4/5] SAVING MODEL ARTIFACTS")
    print("=" * 70)

    # 1. Joblib bundle
    bundle_path = OUTPUT_DIR / "mlp_model.joblib"
    joblib.dump({
        "model": mlp,
        "scaler": scaler,
        "features": feature_cols,
        "hidden": list(mlp.hidden_layer_sizes),
        "metrics": metrics
    }, bundle_path)
    print(f"  Saved sklearn model bundle → {bundle_path}")

    # 2. Scaler params JSON
    scaler_path = OUTPUT_DIR / "scaler_params.json"
    with open(scaler_path, "w") as f:
        json.dump({
            "features": feature_cols,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "trained_on": "df_train_strictly"
        }, f, indent=2)
    print(f"  Saved scaler params → {scaler_path}")

    # 3. Metrics JSON
    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved evaluation metrics → {metrics_path}")

    # 4. Raw weights JSON
    weights_path = OUTPUT_DIR / "mlp_weights.json"
    raw_weights = {
        "architecture": f"MLP_4_{mlp.hidden_layer_sizes[0]}_1",
        "hidden_layer_sizes": list(mlp.hidden_layer_sizes),
        "total_params": int(sum(W.size + b.size for W, b in zip(mlp.coefs_, mlp.intercepts_))),
        "layers": [
            {
                "layer": i,
                "weights_shape": list(W.shape),
                "weights": W.tolist(),
                "biases": b.tolist(),
            }
            for i, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_))
        ]
    }
    with open(weights_path, "w") as f:
        json.dump(raw_weights, f, indent=2)
    print(f"  Saved raw weights JSON → {weights_path}")

# ──────────────── 5. Export 6-Mode C Inference ────────────────────
def export_6mode_c(
    mlp: MLPRegressor,
    scaler: StandardScaler,
    feature_cols: list[str],
):
    """
    Generate mlp_model.h and mlp_model.c containing all 6 benchmark modes:
      - MODE_32BIT_FPU
      - MODE_32BIT_FIXED_NORMAL
      - MODE_32BIT_FIXED_POT
      - MODE_16BIT_FPU
      - MODE_16BIT_FIXED_NORMAL
      - MODE_16BIT_FIXED_POT_SIMD (with ARM SMLAD DSP assembly & zero-skipping)
    """
    print("\n" + "=" * 70)
    print("  [5/5] EXPORTING 6-MODE C FIRMWARE ENGINE (mlp_model.h & mlp_model.c)")
    print("=" * 70)

    W0 = mlp.coefs_[0]       # (4, 32)
    b0 = mlp.intercepts_[0]  # (32,)
    W1 = mlp.coefs_[1]       # (32, 1)
    b1 = mlp.intercepts_[1]  # (1,)

    S_N32 = 10000
    SHIFT_P32 = 16
    S_P32 = 1 << SHIFT_P32

    S_N16 = 2048
    SHIFT_P16 = 11
    S_P16 = 1 << SHIFT_P16

    # Floating point
    W0_f32 = W0.flatten()
    b0_f32 = b0
    W1_f32 = W1.flatten()
    b1_f32 = b1

    # 32-bit Normal Fixed
    W0_n32 = np.round(W0 * S_N32).astype(int).flatten()
    b0_n32 = np.round(b0 * S_N32).astype(int)
    W1_n32 = np.round(W1 * S_N32).astype(int).flatten()
    b1_n32 = np.round(b1 * S_N32).astype(int)

    # 32-bit PoT Fixed
    W0_p32 = np.round(W0 * S_P32).astype(int).flatten()
    b0_p32 = np.round(b0 * S_P32).astype(int)
    W1_p32 = np.round(W1 * S_P32).astype(int).flatten()
    b1_p32 = np.round(b1 * S_P32).astype(int)

    # 16-bit Float (FP16 storage)
    def f_to_u16(val):
        return int(np.float16(val).view(np.uint16))

    W0_f16 = [f_to_u16(x) for x in W0.flatten()]
    b0_f16 = [f_to_u16(x) for x in b0]
    W1_f16 = [f_to_u16(x) for x in W1.flatten()]
    b1_f16 = [f_to_u16(x) for x in b1]

    # 16-bit Normal Fixed
    W0_n16 = np.round(W0 * S_N16).astype(np.int16).flatten()
    b0_n16 = np.round(b0 * S_N16).astype(int)
    W1_n16 = np.round(W1 * S_N16).astype(np.int16).flatten()
    b1_n16 = np.round(b1 * S_N16).astype(int)

    # 16-bit PoT SIMD
    W0_p16 = np.round(W0 * S_P16).astype(np.int16)
    b0_p16 = np.round(b0 * S_P16).astype(int)
    W1_p16 = np.round(W1 * S_P16).astype(np.int16).flatten()
    b1_p16 = np.round(b1 * S_P16).astype(int)

    W0_simd_01 = []
    W0_simd_23 = []
    for j in range(32):
        w0 = int(W0_p16[0, j]) & 0xFFFF
        w1 = int(W0_p16[1, j]) & 0xFFFF
        w2 = int(W0_p16[2, j]) & 0xFFFF
        w3 = int(W0_p16[3, j]) & 0xFFFF
        W0_simd_01.append((w1 << 16) | w0)
        W0_simd_23.append((w3 << 16) | w2)

    mean_c = [int(round(x * 100)) for x in scaler.mean_]
    scale_c = [int(round(x * 100)) for x in scaler.scale_]

    # ──────────────── Write mlp_model.h ────────────────
    header_content = """/*
 * mlp_model.h — Auto-generated TinyML 6-Mode Inference Engine
 * Target: STM32 Nucleo-64 F401RE (Cortex-M4, 16 MHz HSI)
 *
 * ZERO DATA LEAKAGE GUARANTEE:
 * Weights and normalization parameters were learned STRICTLY from early/mid
 * operational telemetry (first 80% of time) and validated on unseen future time (last 20%).
 *
 * 6 BENCHMARK CONFIGURATIONS:
 * ------------------------------------------------------------------------------
 * 32-BIT SUITE:
 *   MODE_32BIT_FPU            (0) — 32-bit Float with Hardware FPU (FP32)
 *   MODE_32BIT_FIXED_NORMAL   (1) — 32-bit Int Without FPU: Normal division (/ 10000)
 *   MODE_32BIT_FIXED_POT      (2) — 32-bit Int Without FPU: Power-of-2 bit-shift (>> 16)
 *
 * 16-BIT SUITE (With DSP SIMD & Dynamic Sparsity Optimizations):
 *   MODE_16BIT_FPU            (3) — 16-bit Float with Hardware FPU (FP16 storage / FP32 compute)
 *   MODE_16BIT_FIXED_NORMAL   (4) — 16-bit Int Without FPU: Normal division (/ 2048)
 *   MODE_16BIT_FIXED_POT_SIMD (5) — 16-bit Int Without FPU: PoT shift (>> 11) + ARM SMLAD SIMD Assembly
 * ------------------------------------------------------------------------------
 */

#ifndef MLP_MODEL_H
#define MLP_MODEL_H

#include <stdint.h>

#define MODE_32BIT_FPU            0
#define MODE_32BIT_FIXED_NORMAL   1
#define MODE_32BIT_FIXED_POT      2
#define MODE_16BIT_FPU            3
#define MODE_16BIT_FIXED_NORMAL   4
#define MODE_16BIT_FIXED_POT_SIMD 5

/* Set default active benchmark mode (0 to 5) or override via compiler flag -DBENCHMARK_MODE=X */
#ifndef BENCHMARK_MODE
#define BENCHMARK_MODE            MODE_32BIT_FPU
#endif

int32_t cnn_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA,
                             int32_t amb_cC,   int32_t hs_cC);

void mlp_reset_timebase(void);

#endif /* MLP_MODEL_H */
"""
    with open(OUTPUT_DIR / "mlp_model.h", "w") as f:
        f.write(header_content)

    # Helpers
    def fmt_arr(arr, per_line=8):
        lines = []
        for i in range(0, len(arr), per_line):
            chunk = arr[i:i+per_line]
            lines.append('    ' + ', '.join(f'{x:9d}' for x in chunk) + (',' if i+per_line < len(arr) else ''))
        return '\n'.join(lines)

    def fmt_arr_hex(arr, per_line=8):
        lines = []
        for i in range(0, len(arr), per_line):
            chunk = arr[i:i+per_line]
            lines.append('    ' + ', '.join(f'0x{x:08X}U' for x in chunk) + (',' if i+per_line < len(arr) else ''))
        return '\n'.join(lines)

    def fmt_arr_f(arr, per_line=6):
        lines = []
        for i in range(0, len(arr), per_line):
            chunk = arr[i:i+per_line]
            lines.append('    ' + ', '.join(f'{x:12.7f}f' for x in chunk) + (',' if i+per_line < len(arr) else ''))
        return '\n'.join(lines)

    # ──────────────── Write mlp_model.c ────────────────
    c_source = f"""/*
 * mlp_model.c — TinyML 6-Mode Inference Engine for STM32F401RET6
 * Architecture: 4 Inputs -> 32 Neurons (ReLU) -> 1 Output (Linear)
 *
 * ZERO-LEAKAGE: Evaluated strictly on unseen future time horizon.
 */

#include "mlp_model.h"
#include <stdint.h>

static int32_t g_sample_count = 0;

void mlp_reset_timebase(void)
{{
    g_sample_count = 0;
}}

/* ==============================================================================
 * MODE 0: 32-BIT FLOAT WITH HARDWARE FPU (FP32)
 * ============================================================================== */
#if (BENCHMARK_MODE == MODE_32BIT_FPU)

static const float f32_mean[4] = {{
{fmt_arr_f(scaler.mean_)}
}};

static const float f32_scale[4] = {{
{fmt_arr_f(scaler.scale_)}
}};

static const float W0_f32[4 * 32] = {{
{fmt_arr_f(W0_f32)}
}};

static const float b0_f32[32] = {{
{fmt_arr_f(b0_f32)}
}};

static const float W1_f32[32] = {{
{fmt_arr_f(W1_f32)}
}};

static const float b1_f32[1] = {{
    {b1_f32[0]:.7f}f
}};

int32_t cnn_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC)
{{
    (void)hs_cC;
    g_sample_count++;

    float x[4];
    x[0] = ((float)ploss_cmW * 0.01f - f32_mean[0]) / f32_scale[0];
    x[1] = ((float)i_cmA * 0.01f - f32_mean[1]) / f32_scale[1];
    x[2] = ((float)amb_cC * 0.01f - f32_mean[2]) / f32_scale[2];
    x[3] = (((float)g_sample_count * (0.5f / 60.0f)) - f32_mean[3]) / f32_scale[3];

    float h[32];
    for (int j = 0; j < 32; j++) {{
        float acc = b0_f32[j];
        for (int i = 0; i < 4; i++) {{
            acc += x[i] * W0_f32[i * 32 + j];
        }}
        h[j] = (acc > 0.0f) ? acc : 0.0f;
    }}

    float out = b1_f32[0];
    for (int k = 0; k < 32; k++) {{
        out += h[k] * W1_f32[k];
    }}

    return (int32_t)(out * 100.0f);
}}

/* ==============================================================================
 * MODE 1: 32-BIT FIXED-POINT NORMAL DIVISION (Scale = 10000)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_32BIT_FIXED_NORMAL)

#define SCALE_NORMAL_32 10000

static const int32_t n32_mean_c[4] = {{
{fmt_arr(mean_c)}
}};

static const int32_t n32_scale_c[4] = {{
{fmt_arr(scale_c)}
}};

static const int32_t W0_n32[4 * 32] = {{
{fmt_arr(W0_n32)}
}};

static const int32_t b0_n32[32] = {{
{fmt_arr(b0_n32)}
}};

static const int32_t W1_n32[32] = {{
{fmt_arr(W1_n32)}
}};

static const int32_t b1_n32[1] = {{
    {b1_n32[0]}
}};

int32_t cnn_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC)
{{
    (void)hs_cC;
    g_sample_count++;

    int32_t raw_c[4];
    raw_c[0] = ploss_cmW;
    raw_c[1] = i_cmA;
    raw_c[2] = amb_cC;
    raw_c[3] = (int32_t)(((int64_t)g_sample_count * 100LL) / 120LL);

    int32_t x_q[4];
    for (int i = 0; i < 4; i++) {{
        int32_t diff = raw_c[i] - n32_mean_c[i];
        x_q[i] = (int32_t)(((int64_t)diff * SCALE_NORMAL_32) / n32_scale_c[i]);
    }}

    int32_t h_q[32];
    for (int j = 0; j < 32; j++) {{
        int64_t acc = (int64_t)b0_n32[j] * SCALE_NORMAL_32;
        for (int i = 0; i < 4; i++) {{
            acc += (int64_t)x_q[i] * (int64_t)W0_n32[i * 32 + j];
        }}
        acc /= SCALE_NORMAL_32;
        h_q[j] = (acc > 0) ? (int32_t)acc : 0;
    }}

    int64_t out_acc = (int64_t)b1_n32[0] * SCALE_NORMAL_32;
    for (int k = 0; k < 32; k++) {{
        out_acc += (int64_t)h_q[k] * (int64_t)W1_n32[k];
    }}
    out_acc /= SCALE_NORMAL_32;

    return (int32_t)((out_acc * 100LL) / SCALE_NORMAL_32);
}}

/* ==============================================================================
 * MODE 2: 32-BIT FIXED-POINT POWER-OF-2 BIT-SHIFT (Q16.16)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_32BIT_FIXED_POT)

#define SHIFT_POT_32 16

static const int32_t pot32_mean_c[4] = {{
{fmt_arr(mean_c)}
}};

static const int32_t pot32_scale_c[4] = {{
{fmt_arr(scale_c)}
}};

static const int32_t W0_p32[4 * 32] = {{
{fmt_arr(W0_p32)}
}};

static const int32_t b0_p32[32] = {{
{fmt_arr(b0_p32)}
}};

static const int32_t W1_p32[32] = {{
{fmt_arr(W1_p32)}
}};

static const int32_t b1_p32[1] = {{
    {b1_p32[0]}
}};

int32_t cnn_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC)
{{
    (void)hs_cC;
    g_sample_count++;

    int32_t raw_c[4];
    raw_c[0] = ploss_cmW;
    raw_c[1] = i_cmA;
    raw_c[2] = amb_cC;
    raw_c[3] = (int32_t)(((int64_t)g_sample_count * 100LL) / 120LL);

    int32_t x_q[4];
    for (int i = 0; i < 4; i++) {{
        int32_t diff = raw_c[i] - pot32_mean_c[i];
        x_q[i] = (int32_t)(((int64_t)diff << SHIFT_POT_32) / pot32_scale_c[i]);
    }}

    int32_t h_q[32];
    for (int j = 0; j < 32; j++) {{
        int64_t acc = (int64_t)b0_p32[j] << SHIFT_POT_32;
        for (int i = 0; i < 4; i++) {{
            acc += (int64_t)x_q[i] * (int64_t)W0_p32[i * 32 + j];
        }}
        acc >>= SHIFT_POT_32;
        h_q[j] = (acc > 0) ? (int32_t)acc : 0;
    }}

    int64_t out_acc = (int64_t)b1_p32[0] << SHIFT_POT_32;
    for (int k = 0; k < 32; k++) {{
        out_acc += (int64_t)h_q[k] * (int64_t)W1_p32[k];
    }}
    out_acc >>= SHIFT_POT_32;

    return (int32_t)(((int64_t)out_acc * 100LL) >> SHIFT_POT_32);
}}

/* ==============================================================================
 * MODE 3: 16-BIT FLOAT (FP16 STORAGE / FP32 COMPUTE WITH FPU)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_16BIT_FPU)

static inline float fp16_to_fp32(uint16_t h)
{{
    uint32_t sign = ((uint32_t)h & 0x8000U) << 16;
    uint32_t exp  = ((uint32_t)h & 0x7C00U) >> 10;
    uint32_t mant = ((uint32_t)h & 0x03FFU);
    uint32_t f;

    if (exp == 0) {{
        if (mant == 0) {{
            f = sign;
        }} else {{
            while ((mant & 0x0400U) == 0) {{
                mant <<= 1;
                exp--;
            }}
            exp++;
            mant &= ~0x0400U;
            f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }}
    }} else if (exp == 31) {{
        f = sign | 0x7F800000U | (mant << 13);
    }} else {{
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }}

    float result;
    *((uint32_t *)(void *)&result) = f;
    return result;
}}

static const float f32_mean[4] = {{
{fmt_arr_f(scaler.mean_)}
}};

static const float f32_scale[4] = {{
{fmt_arr_f(scaler.scale_)}
}};

static const uint16_t W0_f16[4 * 32] = {{
{fmt_arr_hex(W0_f16)}
}};

static const uint16_t b0_f16[32] = {{
{fmt_arr_hex(b0_f16)}
}};

static const uint16_t W1_f16[32] = {{
{fmt_arr_hex(W1_f16)}
}};

static const uint16_t b1_f16[1] = {{
    0x{b1_f16[0]:04X}U
}};

int32_t cnn_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC)
{{
    (void)hs_cC;
    g_sample_count++;

    float x[4];
    x[0] = ((float)ploss_cmW * 0.01f - f32_mean[0]) / f32_scale[0];
    x[1] = ((float)i_cmA * 0.01f - f32_mean[1]) / f32_scale[1];
    x[2] = ((float)amb_cC * 0.01f - f32_mean[2]) / f32_scale[2];
    x[3] = (((float)g_sample_count * (0.5f / 60.0f)) - f32_mean[3]) / f32_scale[3];

    float h[32];
    for (int j = 0; j < 32; j++) {{
        float acc = fp16_to_fp32(b0_f16[j]);
        for (int i = 0; i < 4; i++) {{
            acc += x[i] * fp16_to_fp32(W0_f16[i * 32 + j]);
        }}
        h[j] = (acc > 0.0f) ? acc : 0.0f;
    }}

    float out = fp16_to_fp32(b1_f16[0]);
    for (int k = 0; k < 32; k++) {{
        out += h[k] * fp16_to_fp32(W1_f16[k]);
    }}

    return (int32_t)(out * 100.0f);
}}

/* ==============================================================================
 * MODE 4: 16-BIT FIXED-POINT NORMAL DIVISION (Scale = 2048)
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_16BIT_FIXED_NORMAL)

#define SCALE_NORMAL_16 2048

static const int32_t n16_mean_c[4] = {{
{fmt_arr(mean_c)}
}};

static const int32_t n16_scale_c[4] = {{
{fmt_arr(scale_c)}
}};

static const int16_t W0_n16[4 * 32] = {{
{fmt_arr(W0_n16)}
}};

static const int32_t b0_n16[32] = {{
{fmt_arr(b0_n16)}
}};

static const int16_t W1_n16[32] = {{
{fmt_arr(W1_n16)}
}};

static const int32_t b1_n16[1] = {{
    {b1_n16[0]}
}};

int32_t cnn_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC)
{{
    (void)hs_cC;
    g_sample_count++;

    int32_t raw_c[4];
    raw_c[0] = ploss_cmW;
    raw_c[1] = i_cmA;
    raw_c[2] = amb_cC;
    raw_c[3] = (int32_t)(((int64_t)g_sample_count * 100LL) / 120LL);

    int16_t x_q[4];
    for (int i = 0; i < 4; i++) {{
        int32_t diff = raw_c[i] - n16_mean_c[i];
        x_q[i] = (int16_t)(((int32_t)diff * SCALE_NORMAL_16) / n16_scale_c[i]);
    }}

    int16_t h_q[32];
    for (int j = 0; j < 32; j++) {{
        int32_t acc = b0_n16[j] * SCALE_NORMAL_16;
        for (int i = 0; i < 4; i++) {{
            acc += (int32_t)x_q[i] * (int32_t)W0_n16[i * 32 + j];
        }}
        acc /= SCALE_NORMAL_16;
        h_q[j] = (acc > 0) ? (int16_t)acc : 0;
    }}

    int32_t out_acc = b1_n16[0] * SCALE_NORMAL_16;
    for (int k = 0; k < 32; k++) {{
        out_acc += (int32_t)h_q[k] * (int32_t)W1_n16[k];
    }}
    out_acc /= SCALE_NORMAL_16;

    return (int32_t)(((int32_t)out_acc * 100) / SCALE_NORMAL_16);
}}

/* ==============================================================================
 * MODE 5: 16-BIT FIXED-POINT POWER-OF-2 BIT-SHIFT (Q5.11) + ARM SMLAD SIMD ASSEMBLY
 * ============================================================================== */
#elif (BENCHMARK_MODE == MODE_16BIT_FIXED_POT_SIMD)

#define SHIFT_POT_16 11

static inline int32_t smlad_dual(uint32_t val1, uint32_t val2, int32_t acc)
{{
#if defined(__arm__) || defined(__thumb__)
    __asm__ ("smlad %0, %1, %2, %3" : "=r" (acc) : "r" (val1), "r" (val2), "r" (acc));
    return acc;
#else
    int16_t v1_lo = (int16_t)(val1 & 0xFFFFU);
    int16_t v1_hi = (int16_t)(val1 >> 16);
    int16_t v2_lo = (int16_t)(val2 & 0xFFFFU);
    int16_t v2_hi = (int16_t)(val2 >> 16);
    return acc + ((int32_t)v1_lo * (int32_t)v2_lo) + ((int32_t)v1_hi * (int32_t)v2_hi);
#endif
}}

static const int32_t pot_mean_c[4] = {{
{fmt_arr(mean_c)}
}};

static const int32_t pot_scale_c[4] = {{
{fmt_arr(scale_c)}
}};

static const uint32_t W0_simd_01[32] = {{
{fmt_arr_hex(W0_simd_01)}
}};

static const uint32_t W0_simd_23[32] = {{
{fmt_arr_hex(W0_simd_23)}
}};

static const int32_t b0_simd[32] = {{
{fmt_arr(b0_p16)}
}};

static const int16_t W1_simd[32] = {{
{fmt_arr(W1_p16)}
}};

static const int32_t b1_simd[1] = {{
    {b1_p16[0]}
}};

int32_t cnn_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC)
{{
    (void)hs_cC;
    g_sample_count++;

    int32_t raw_c[4];
    raw_c[0] = ploss_cmW;
    raw_c[1] = i_cmA;
    raw_c[2] = amb_cC;
    raw_c[3] = (int32_t)(((int64_t)g_sample_count * 100LL) / 120LL);

    int16_t x_q[4];
    for (int i = 0; i < 4; i++) {{
        int32_t diff = raw_c[i] - pot_mean_c[i];
        x_q[i] = (int16_t)((diff << SHIFT_POT_16) / pot_scale_c[i]);
    }}

    /* Pack 4 inputs into two 32-bit registers for ARM SIMD instructions */
    uint32_t x_pair_01 = ((uint32_t)(uint16_t)x_q[1] << 16) | ((uint32_t)(uint16_t)x_q[0]);
    uint32_t x_pair_23 = ((uint32_t)(uint16_t)x_q[3] << 16) | ((uint32_t)(uint16_t)x_q[2]);

    /* Layer 0: 4 -> 32 (Method 2: SMLAD Assembly + Method 3: Register Cached) */
    int16_t h_q[32];
    for (int j = 0; j < 32; j++) {{
        int32_t acc = b0_simd[j] << SHIFT_POT_16;
        acc = smlad_dual(x_pair_01, W0_simd_01[j], acc);
        acc = smlad_dual(x_pair_23, W0_simd_23[j], acc);
        acc >>= SHIFT_POT_16;
        h_q[j] = (acc > 0) ? (int16_t)acc : 0;
    }}

    /* Layer 1: 32 -> 1 (Method 4: Dynamic ReLU Zero-Skipping) */
    int32_t out_acc = b1_simd[0] << SHIFT_POT_16;
    for (int k = 0; k < 32; k++) {{
        if (h_q[k] != 0) {{
            out_acc += (int32_t)h_q[k] * (int32_t)W1_simd[k];
        }}
    }}
    out_acc >>= SHIFT_POT_16;

    return (int32_t)(((int32_t)out_acc * 100) >> SHIFT_POT_16);
}}

#endif
"""
    with open(OUTPUT_DIR / "mlp_model.c", "w") as f:
        f.write(c_source)

    print("  ✓ Exported mlp_model.h and mlp_model.c with all 6 benchmark modes successfully!")

# ───────────────────────────── Main ───────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train Zero-Leakage TinyML Thermal Regressor")
    parser.add_argument("--epochs", type=int, default=1000, help="Max training epochs (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--no-export", action="store_true", help="Skip C code export")
    args = parser.parse_args()

    # 1. Load data under strict zero-leakage constraints
    df_train, df_test = load_clean_datasets()

    feature_cols = ['power_loss_mW', 'current_mA', 'ambient_temp_C', 'elapsed_min']
    target_col   = 'heatsink_temp_C'

    # 2. Scaler fit STRICTLY ON TRAIN ONLY
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(df_train[feature_cols].values)
    y_train   = df_train[target_col].values

    # 3. Train MLP
    mlp = train_zero_leakage_mlp(X_train_s, y_train, hidden_dim=32, max_iter=args.epochs, seed=args.seed)

    # 4. Evaluate on holdout test set & plot
    metrics = evaluate_and_plot(mlp, scaler, df_train, df_test, feature_cols, target_col)

    # 5. Save model artifacts
    save_artifacts(mlp, scaler, feature_cols, metrics)

    # 6. Export to C
    if not args.no_export:
        export_6mode_c(mlp, scaler, feature_cols)

    print("\n" + "=" * 70)
    print("  ✓ ZERO-LEAKAGE PIPELINE EXECUTION COMPLETE!")
    print(f"  Holdout Test MAE:  {metrics['holdout_test']['MAE']:.4f} °C")
    print(f"  Holdout Test RMSE: {metrics['holdout_test']['RMSE']:.4f} °C")
    print(f"  Holdout Test R²:   {metrics['holdout_test']['R2']:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
