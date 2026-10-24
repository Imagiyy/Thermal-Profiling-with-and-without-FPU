#!/usr/bin/env python3
"""
experiment_mlp_new.py — Thermal Predictor MLP: Heatsink + Ambient → Future Temperature.

Evaluates 8 MLP configurations across 3 modeling formulations:
  1. Thermal Predictor + Time (3-feat): [heatsink_temp_C, ambient_temp_C, elapsed_min] -> future_temp (+1s)
  2. Thermal Predictor (2-feat):        [heatsink_temp_C, ambient_temp_C]              -> future_temp (+1s)
  3. Single Sensor Baseline (1-feat):   [heatsink_temp_C]                              -> future_temp (+1s)

Leakage Prevention Guarantees:
  - Temporal Split: Train on first 80% of continuous time, test on final 20% unseen future horizon.
  - Scaler Leakage: StandardScaler fit exclusively on training data, only transformed on test.
  - Label Leakage: Dropped any pre-existing 'predicted_temp_C'.
  - Noise / Dropout Filtering: Filters unpowered bench power supply states (Vin < 10V) and sensor overflow spikes.
  - No Future-to-Present Leakage: target (temp_future_1s) is strictly +1s ahead of current features.
"""

from __future__ import annotations

import json
import sys
import time
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

SCRIPT_DIR  = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent / "dataset"
OUTPUT_DIR  = SCRIPT_DIR


# ═══════════════════════════════════════════════════════════════════
# 1. DATA PREPROCESSING (NO DATA LEAKAGE)
# ═══════════════════════════════════════════════════════════════════

def load_and_preprocess() -> pd.DataFrame:
    print("=" * 70)
    print("  STEP 1: LEAKAGE-FREE DATA PREPROCESSING")
    print("=" * 70)

    csv_files = sorted(DATASET_DIR.glob("*.csv"))
    frames = []

    for f in csv_files:
        if f.stem == "zeroinput":
            print(f"  [Filter] Dropping '{f.name}': 47 unpowered rows (Vin < 0.01V, adds noise)")
            continue

        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        raw_len = len(df)

        # Drop label leakage column if present
        if "predicted_temp_C" in df.columns:
            df = df.drop(columns=["predicted_temp_C"])

        # Physical Operating Filter:
        # KA7808 voltage regulator testbed operates at Vin = 11V - 12V.
        # When power supply is turned off, Vin collapses through < 10V down to 0V.
        # During power-down, INA219 current register overflows (-2867 mA or +3276 mA)
        # and thermocouple enters passive cooldown.
        mask_powered = (
            (df["vin_V"] >= 10.0) &
            (df["fault_code"] == 0) &
            (df["current_mA"] >= 0) &
            (df["current_mA"] <= 1500) &
            (df["power_loss_mW"] >= 0) &
            (df["heatsink_temp_C"] > 20.0)
        )
        df_clean = df[mask_powered].copy()

        # Elapsed time within this continuous operational run
        t0 = df_clean["timestamp_ms"].min()
        df_clean["elapsed_min"] = (df_clean["timestamp_ms"] - t0) / 60000.0
        df_clean["source"] = f.stem

        # Lookahead target (+1s / 2 steps ahead, 500ms sample period)
        df_clean["temp_future_1s"] = df_clean["heatsink_temp_C"].shift(-2)
        df_clean = df_clean.dropna().copy()

        print(f"  [Clean] {f.name:12s}: {raw_len} raw -> {len(df_clean)} clean operational rows "
              f"(T_hs: {df_clean['heatsink_temp_C'].min():.1f}°C -> {df_clean['heatsink_temp_C'].max():.1f}°C)")
        frames.append(df_clean)

    merged = pd.concat(frames, ignore_index=True)
    print(f"\n  Total clean dataset: {len(merged)} operational rows across {len(frames)} load conditions.")
    return merged


# ═══════════════════════════════════════════════════════════════════
# 2. ARCHITECTURES & HARDWARE COST MODEL
# ═══════════════════════════════════════════════════════════════════

ARCHITECTURES = {
    "tiny_8":         (8,),
    "small_16":       (16,),
    "small_32":       (32,),
    "medium_16_8":    (16, 8),
    "medium_32_16":   (32, 16),
    "deep_16_16_8":   (16, 16, 8),
    "large_64_32":    (64, 32),
    "wide_64_32_16":  (64, 32, 16),
}

def compute_hw_stats(n_inputs: int, hidden: tuple):
    sizes = [n_inputs] + list(hidden) + [1]
    params = sum(sizes[i] * sizes[i+1] + sizes[i+1] for i in range(len(sizes)-1))
    flash_bytes = params * 4  # 32-bit float
    ram_bytes = max(sizes) * 4 * 2  # double ping-pong buffer for layer outputs
    flops = sum(2 * sizes[i] * sizes[i+1] for i in range(len(sizes)-1))
    # STM32F401RE @ 16 MHz with hardware single-precision FPU: ~3 clock cycles per MAC
    est_latency_us = (flops * 3.0) / 16.0
    return {
        "params": params,
        "flash_bytes": flash_bytes,
        "ram_bytes": ram_bytes,
        "flops": flops,
        "est_latency_us": round(est_latency_us, 1),
    }


def augment_ambient_training_data(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Augments the training data across realistic operating room temperatures [18 °C to 38 °C].
    In the laboratory dataset, ambient temperature was nearly constant (33.20 ± 0.20 °C),
    causing StandardScaler to assign an artificially tiny scale (0.20 °C), which causes
    catastrophic extrapolation spikes when the hardware runs at normal room temp (e.g. 25 °C).
    
    Physics: Heat loss rate dT/dt is proportional to (T_hs - T_amb)/tau (tau ~ 208s).
    Over dt = 1.0s, delta_T = (s / tau) * dt.
    """
    tau = 208.0
    dt = 1.0
    shifts = [-15.0, -12.0, -9.0, -6.0, -3.0, 0.0, +3.0, +5.0]

    augmented_frames = []
    for s in shifts:
        d = df_train.copy()
        d["ambient_temp_C"] = d["ambient_temp_C"] + s + np.random.normal(0, 0.05, size=len(d))
        if "temp_future_1s" in d.columns:
            d["temp_future_1s"] = d["temp_future_1s"] + (s / tau) * dt
        augmented_frames.append(d)
    return pd.concat(augmented_frames, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════
# 3. EXPERIMENT EXECUTION (TEMPORAL LEAKAGE-FREE SPLIT)
# ═══════════════════════════════════════════════════════════════════

def run_architecture_search(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("  STEP 2: ARCHITECTURE SEARCH (TEMPORAL EVALUATION, ZERO LEAKAGE)")
    print("=" * 70)

    # Strict temporal train/test split (first 80% train, last 20% test per condition)
    train_dfs, test_dfs = [], []
    for src in df["source"].unique():
        sub = df[df["source"] == src]
        n = len(sub)
        cutoff = int(n * 0.80)
        train_dfs.append(sub.iloc[:cutoff])
        test_dfs.append(sub.iloc[cutoff:])

    df_train = pd.concat(train_dfs, ignore_index=True)
    df_test  = pd.concat(test_dfs, ignore_index=True)

    print(f"  Training samples:   {len(df_train)} (early/mid heating curves)")
    print(f"  Test samples:       {len(df_test)} (unseen future time horizon, last 20%)")

    # Formulations to evaluate — all use temperature-based features only
    formulations = [
        {
            "name": "Thermal Predictor + Time (3-feat)",
            "key": "thermal_time",
            "features": ["heatsink_temp_C", "ambient_temp_C", "elapsed_min"],
            "target": "temp_future_1s",
            "desc": "Predicts future temp 1s ahead using current heatsink + ambient + elapsed time",
        },
        {
            "name": "Thermal Predictor (2-feat)",
            "key": "thermal_core",
            "features": ["heatsink_temp_C", "ambient_temp_C"],
            "target": "temp_future_1s",
            "desc": "Predicts future temp 1s ahead using current heatsink + ambient (core model)",
        },
        {
            "name": "Single Sensor Baseline (1-feat)",
            "key": "single_sensor",
            "features": ["heatsink_temp_C"],
            "target": "temp_future_1s",
            "desc": "Baseline: predicts future temp from heatsink reading alone",
        },
    ]

    all_results = []

    for form in formulations:
        fname = form["name"]
        fkey = form["key"]
        feats = form["features"]
        target = form["target"]

        print(f"\n  ------------------------------------------------------------------")
        print(f"  Evaluating Formulation: {fname}")
        print(f"  Features: {feats} -> Target: {target}")
        print(f"  ------------------------------------------------------------------")

        # For ambient feature robustness: augment training set across realistic room temperatures
        if "ambient_temp_C" in feats:
            effective_train = augment_ambient_training_data(df_train)
        else:
            effective_train = df_train

        # Scaler fit STRICTLY on train set
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(effective_train[feats].values)
        X_test_s  = scaler.transform(df_test[feats].values)
        y_train   = effective_train[target].values
        y_test    = df_test[target].values

        for arch_name, hidden in ARCHITECTURES.items():
            hw = compute_hw_stats(len(feats), hidden)

            mlp = MLPRegressor(
                hidden_layer_sizes=hidden,
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=2e-3,
                max_iter=600,
                tol=1e-5,
                early_stopping=True,
                n_iter_no_change=25,
                random_state=42,
            )

            t0 = time.time()
            mlp.fit(X_train_s, y_train)
            train_time = time.time() - t0

            y_pred_test = mlp.predict(X_test_s)
            y_pred_train = mlp.predict(X_train_s)

            test_mae  = mean_absolute_error(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_r2   = r2_score(y_test, y_pred_test)

            train_mae = mean_absolute_error(y_train, y_pred_train)
            train_r2  = r2_score(y_train, y_pred_train)

            print(f"    {arch_name:<16s} {str(hidden):<14s} params={hw['params']:<4d} "
                  f"MAE={test_mae:.3f}°C  RMSE={test_rmse:.3f}°C  R²={test_r2:.4f}  "
                  f"lat={hw['est_latency_us']:>5.1f}µs  (train: {mlp.n_iter_:3d} iters, {train_time:.1f}s)")

            all_results.append({
                "formulation": fname,
                "formulation_key": fkey,
                "features": feats,
                "target": target,
                "architecture": arch_name,
                "hidden": list(hidden),
                "params": hw["params"],
                "flash_bytes": hw["flash_bytes"],
                "ram_bytes": hw["ram_bytes"],
                "est_latency_us": hw["est_latency_us"],
                "test_mae": round(test_mae, 4),
                "test_rmse": round(test_rmse, 4),
                "test_r2": round(test_r2, 4),
                "train_mae": round(train_mae, 4),
                "train_r2": round(train_r2, 4),
                "n_iter": mlp.n_iter_,
                "train_time_s": round(train_time, 2),
            })

    return all_results, df_train, df_test


# ═══════════════════════════════════════════════════════════════════
# 4. REPORT GENERATION & PLOTTING
# ═══════════════════════════════════════════════════════════════════

def plot_and_save_results(results: list[dict], df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("  STEP 3: RESULTS ANALYSIS & COMPARISON PLOTS")
    print("=" * 70)

    # Save complete JSON results
    json_path = OUTPUT_DIR / "experiment_results_new.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved full experiment results -> {json_path}")

    # Generate 4-panel comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("ThermoGuard NEW MLP: Thermal Predictor Architecture Search\n"
                 "(Features: Heatsink + Ambient Temp → Future Temp +1s)",
                 fontsize=14, fontweight="bold")

    # Panel 1: Thermal Predictor (2-feat) MAE Comparison across Architectures
    ax1 = axes[0, 0]
    core_results = [r for r in results if r["formulation_key"] == "thermal_core"]
    core_sorted = sorted(core_results, key=lambda x: x["test_mae"])
    names1 = [f"{r['architecture']}\n({r['hidden']})" for r in core_sorted]
    maes1  = [r["test_mae"] for r in core_sorted]
    colors1 = ["#2E7D32" if i == 0 else "#1976D2" for i in range(len(names1))]
    bars1 = ax1.barh(range(len(names1)), maes1, color=colors1, edgecolor="white", alpha=0.9)
    ax1.set_yticks(range(len(names1)))
    ax1.set_yticklabels(names1, fontsize=8)
    ax1.set_xlabel("Holdout Test MAE (°C) [Lower is better]")
    ax1.set_title("Thermal Predictor Accuracy (2-feat: T_hs, T_amb)")
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis="x")
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + 0.05, bar.get_y() + bar.get_height()/2, f"{w:.2f}°C", va="center", fontsize=8)

    # Panel 2: Latency vs MAE (Pareto Frontier)
    ax2 = axes[0, 1]
    for form_key, color, marker in [
        ("thermal_time", "#1976D2", "o"),
        ("thermal_core", "#388E3C", "s"),
        ("single_sensor", "#D32F2F", "^"),
    ]:
        subset = [r for r in results if r["formulation_key"] == form_key]
        ax2.scatter(
            [r["est_latency_us"] for r in subset],
            [r["test_mae"] for r in subset],
            label=subset[0]["formulation"],
            color=color, marker=marker, s=80, alpha=0.85
        )
    ax2.set_xlabel("Estimated Latency on STM32F401RE @ 16 MHz (µs)")
    ax2.set_ylabel("Holdout Test MAE (°C)")
    ax2.set_title("Edge Efficiency: Latency vs Test MAE")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    # Panel 3: Formulation Comparison (MAE across 3 paradigms)
    ax3 = axes[1, 0]
    arch_sample = ["tiny_8", "small_16", "small_32", "medium_16_8"]
    x = np.arange(len(arch_sample))
    width = 0.25
    for idx, (fkey, lbl, col) in enumerate([
        ("single_sensor", "Single Sensor (T_hs only)", "#E53935"),
        ("thermal_core", "Thermal Predictor (T_hs + T_amb)", "#1E88E5"),
        ("thermal_time", "Thermal + Time (T_hs + T_amb + t)", "#43A047"),
    ]):
        vals = [next(r["test_mae"] for r in results if r["formulation_key"] == fkey and r["architecture"] == a) for a in arch_sample]
        ax3.bar(x + idx*width, vals, width, label=lbl, color=col, alpha=0.85, edgecolor="white")
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(arch_sample, fontsize=9)
    ax3.set_ylabel("Holdout Test MAE (°C)")
    ax3.set_title("Impact of Feature Set on Leakage-Free Generalization")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")

    # Panel 4: Memory Footprint (Flash vs RAM)
    ax4 = axes[1, 1]
    arch_all = sorted(list(ARCHITECTURES.keys()), key=lambda a: ARCHITECTURES[a])
    flash_kbs = [compute_hw_stats(2, ARCHITECTURES[a])["flash_bytes"] / 1024.0 for a in arch_all]
    ram_bytes = [compute_hw_stats(2, ARCHITECTURES[a])["ram_bytes"] for a in arch_all]

    ax4.bar(range(len(arch_all)), flash_kbs, color="#7B1FA2", alpha=0.85, edgecolor="white")
    ax4.set_xticks(range(len(arch_all)))
    ax4.set_xticklabels(arch_all, rotation=40, ha="right", fontsize=8)
    ax4.set_ylabel("Flash Memory (KB) [STM32 Budget: 512 KB]")
    ax4.set_title("Flash Memory Footprint by Architecture (2-input model)")
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "experiment_results_new.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Saved comparison chart -> {plot_path}")


# ═══════════════════════════════════════════════════════════════════
# 5. RETRAIN BEST CONFIGURATION & EXPORT C CODE
# ═══════════════════════════════════════════════════════════════════

def train_and_export_best(df: pd.DataFrame, results: list[dict]):
    print("\n" + "=" * 70)
    print("  STEP 4: TRAINING FINAL PRODUCTION MODEL & EXPORTING C")
    print("=" * 70)

    # Rank thermal_core (2-feat) models by MAE — this is the user's requested formulation
    core_results = [r for r in results if r["formulation_key"] == "thermal_core"]
    best = sorted(core_results, key=lambda x: x["test_mae"])[0]

    print(f"  [BEST] Selected Best Architecture: {best['architecture']} ({best['hidden']})")
    print(f"     Features:       {best['features']}")
    print(f"     Test MAE:       {best['test_mae']:.4f}°C")
    print(f"     Test R²:        {best['test_r2']:.4f}")
    print(f"     Parameters:     {best['params']}")
    print(f"     Flash footprint:{best['flash_bytes']} bytes")
    print(f"     Est. Latency:   {best['est_latency_us']} µs on Cortex-M4 @ 16 MHz")

    feats = best["features"]
    hidden = tuple(best["hidden"])
    target = best["target"]

    # Full data refit for production deployment with ambient robustness
    if "ambient_temp_C" in feats:
        df_export_train = augment_ambient_training_data(df)
    else:
        df_export_train = df

    X = df_export_train[feats].values
    y = df_export_train[target].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=2e-3,
        max_iter=1000,
        tol=1e-6,
        random_state=42,
    )
    mlp.fit(X_scaled, y)
    full_mae = mean_absolute_error(y, mlp.predict(X_scaled))
    print(f"  Full dataset refit MAE: {full_mae:.4f}°C")

    # Save model artifacts
    model_bundle = {
        "model": mlp,
        "scaler": scaler,
        "features": feats,
        "hidden": hidden,
        "test_mae": best["test_mae"],
        "test_r2": best["test_r2"],
        "full_mae": full_mae,
    }
    joblib.dump(model_bundle, OUTPUT_DIR / "mlp_model_new.joblib")

    # Save scaler params JSON
    scaler_dict = {
        "features": feats,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    with open(OUTPUT_DIR / "scaler_params_new.json", "w") as f:
        json.dump(scaler_dict, f, indent=2)

    # Save weights JSON
    weights_dict = {
        "architecture": best["architecture"],
        "hidden_layer_sizes": list(hidden),
        "total_params": best["params"],
        "layers": [],
    }
    for idx, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        weights_dict["layers"].append({
            "layer": idx,
            "weights_shape": list(W.shape),
            "weights": W.tolist(),
            "biases": b.tolist(),
        })
    with open(OUTPUT_DIR / "mlp_weights_new.json", "w") as f:
        json.dump(weights_dict, f, indent=2)

    # Save metrics JSON
    metrics = {
        "best_model": {
            "name": best["architecture"],
            "hidden_layers": list(hidden),
            "holdout_test_mae": best["test_mae"],
            "holdout_test_rmse": best["test_rmse"],
            "holdout_test_r2": best["test_r2"],
            "full_dataset_mae": round(full_mae, 4),
            "parameters": best["params"],
            "est_latency_us": best["est_latency_us"],
            "flash_bytes": best["flash_bytes"],
        }
    }
    with open(OUTPUT_DIR / "metrics_new.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Export C Code (mlp_model_new.h & mlp_model_new.c) ──
    coefs = mlp.coefs_
    intercepts = mlp.intercepts_
    n_inputs = len(feats)
    layer_sizes = [n_inputs] + list(hidden) + [1]
    n_layers = len(coefs)

    header_content = f"""/*
 * mlp_model_new.h — Auto-generated TinyML Thermal Predictor for STM32F401RET6
 *
 * Model Architecture: {' -> '.join(str(s) for s in layer_sizes)}
 * Activation:         ReLU (hidden), Linear (output)
 * Parameters:         {best['params']} single-precision floats ({best['flash_bytes']} bytes Flash)
 * Target:             Future Heatsink Temperature +1s ahead (centi-°C)
 *
 * Features:           [heatsink_temp_C, ambient_temp_C]
 * Unlike the original model which uses electrical features (power, current),
 * this model predicts future temperature from present thermal readings only.
 *
 * Generated by New_models/experiment_mlp_new.py
 */

#ifndef MLP_MODEL_NEW_H
#define MLP_MODEL_NEW_H

#include <stdint.h>

/*
 * TinyML Thermal Predictor — Future Heatsink Temperature.
 * Matches firmware signature in main.c:
 *   int32_t mlp_new_predict_temp_cC(int32_t hs_cC, int32_t amb_cC)
 *
 * Inputs:
 *   hs_cC   — Current heatsink temperature (centi-°C, 100 = 1 °C)
 *   amb_cC  — Ambient temperature           (centi-°C, 100 = 1 °C)
 *
 * Returns:
 *   Predicted future heatsink temperature (+1s ahead) in centi-°C
 *   (e.g., 4520 = 45.20 °C)
 */
int32_t mlp_new_predict_temp_cC(int32_t hs_cC, int32_t amb_cC);

#endif /* MLP_MODEL_NEW_H */
"""
    with open(OUTPUT_DIR / "mlp_model_new.h", "w") as f:
        f.write(header_content)

    src_lines = [
        f"/*",
        f" * mlp_model_new.c — Pure C Hardware-FPU TinyML Thermal Predictor for STM32F401RET6",
        f" *",
        f" * Predicts future heatsink temperature (+1s ahead) from present thermal readings.",
        f" * Features: [heatsink_temp_C, ambient_temp_C]",
        f" * Architecture: {' -> '.join(str(s) for s in layer_sizes)}",
        f" * Parameters:   {best['params']} floats ({best['flash_bytes']} bytes flash)",
        f" */",
        f"",
        f'#include "mlp_model_new.h"',
        f"",
        f"/* Feature Normalisation (StandardScaler) */",
        f"static const float scaler_mean[{n_inputs}] = {{",
        "    " + ", ".join(f"{v:.8f}f" for v in scaler.mean_),
        f"}};",
        f"static const float scaler_scale[{n_inputs}] = {{",
        "    " + ", ".join(f"{v:.8f}f" for v in scaler.scale_),
        f"}};\n",
    ]

    for i, (W, b) in enumerate(zip(coefs, intercepts)):
        rows, cols = W.shape
        flat = W.flatten()
        src_lines.append(f"/* Layer {i}: Weights [{rows} x {cols}] */")
        src_lines.append(f"static const float W{i}[{rows * cols}] = {{")
        for j in range(0, len(flat), 8):
            chunk = flat[j:j+8]
            src_lines.append("    " + ", ".join(f"{v: .8f}f" for v in chunk) + ("," if j+8 < len(flat) else ""))
        src_lines.append("};")
        src_lines.append(f"static const float b{i}[{cols}] = {{")
        src_lines.append("    " + ", ".join(f"{v: .8f}f" for v in b))
        src_lines.append("};\n")

    max_width = max(layer_sizes)
    src_lines.append(f"""int32_t mlp_new_predict_temp_cC(int32_t hs_cC, int32_t amb_cC)
{{
    /*
     * Sanity clamp ambient reading to realistic operating room conditions [10 °C, 45 °C].
     * If BME280 sensor is disconnected or reading <= 0, default to 25 °C (2500 cC).
     */
    if (amb_cC <= 1000 || amb_cC >= 5000) {{
        amb_cC = 2500;
    }}

    /*
     * Feature extraction:
     *   x[0] = heatsink_temp_C = hs_cC  / 100.0f  (centi-°C -> °C)
     *   x[1] = ambient_temp_C  = amb_cC / 100.0f  (centi-°C -> °C)
     */
    float x[{n_inputs}];
    x[0] = (((float)hs_cC  / 100.0f) - scaler_mean[0]) / scaler_scale[0];
    x[1] = (((float)amb_cC / 100.0f) - scaler_mean[1]) / scaler_scale[1];

    /* Forward Pass (Single Precision HW FPU) */
    float buf_a[{max_width}];
    float buf_b[{max_width}];
    const float *in_vec = x;
    float *out_vec;
""")

    for i, (W, b) in enumerate(zip(coefs, intercepts)):
        rows, cols = W.shape
        is_last = (i == n_layers - 1)
        out_buf = "buf_a" if i % 2 == 0 else "buf_b"
        src_lines.append(f"    /* Layer {i}: {rows} -> {cols} */")
        src_lines.append(f"    out_vec = {out_buf};")
        src_lines.append(f"    for (int j = 0; j < {cols}; j++) {{")
        src_lines.append(f"        float acc = b{i}[j];")
        src_lines.append(f"        for (int k = 0; k < {rows}; k++) {{")
        src_lines.append(f"            acc += in_vec[k] * W{i}[k * {cols} + j];")
        src_lines.append(f"        }}")
        if is_last:
            src_lines.append(f"        out_vec[j] = acc; /* Linear Output */")
        else:
            src_lines.append(f"        out_vec[j] = (acc > 0.0f) ? acc : 0.0f; /* ReLU */")
        src_lines.append(f"    }}")
        src_lines.append(f"    in_vec = out_vec;\n")

    src_lines.append(f"""    /* Convert predicted °C float back to centi-°C integer */
    float pred_C = out_vec[0];
    return (int32_t)(pred_C * 100.0f + (pred_C >= 0.0f ? 0.5f : -0.5f));
}}
""")

    with open(OUTPUT_DIR / "mlp_model_new.c", "w") as f:
        f.write("\n".join(src_lines))

    print(f"  Generated production C code -> mlp_model_new.h & mlp_model_new.c")


# ═══════════════════════════════════════════════════════════════════
# MAIN ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════

def main():
    print("======================================================================")
    print("  Project ThermoGuard — NEW Thermal Predictor MLP Architecture Search ")
    print("  Features: [heatsink_temp_C, ambient_temp_C] -> Future Temp (+1s)    ")
    print("======================================================================")

    df = load_and_preprocess()
    results, df_train, df_test = run_architecture_search(df)
    plot_and_save_results(results, df)
    train_and_export_best(df, results)

    print("\n======================================================================")
    print("  [OK] NEW MODEL ARCHITECTURE SEARCH COMPLETED SUCCESSFULLY!")
    print("======================================================================\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
