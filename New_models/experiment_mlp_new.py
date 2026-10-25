#!/usr/bin/env python3
"""
experiment_mlp_new.py -- Thermal Predictor MLP: Stateless 4-Feature Dual-Prediction Model.

Features: [heatsink_temp_C, ambient_temp_C, power_loss_mW, current_mA] -> [future_3s, future_10s]

Why no elapsed_time?
  The previous model used g_sample_count (elapsed minutes) as its 4th feature.
  This counter always starts at 0 when prediction is activated -- even if the
  heatsink is already at 60 degC mid-heating. This mismatch causes large prediction errors.
  By replacing elapsed_time with power_loss_mW and current_mA the model is
  STATELESS: it knows thermal state (T_hs, T_amb) AND electrical drive state
  (P_loss, I) at each call without any internal counter. Starting mid-heating
  is now fully supported with no error.

Dual Prediction Windows:
  - 3-second lookahead (6 samples ahead @ 500 ms)
  - 10-second lookahead (20 samples ahead @ 500 ms)
  Trained simultaneously via a dual-output MLP with zero data leakage.
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

## 4 stateless features -- no elapsed_time dependency
FEATURES = ["heatsink_temp_C", "ambient_temp_C", "power_loss_mW", "current_mA"]
TARGETS  = ["delta_T_3s", "delta_T_10s"]
N_INPUTS = len(FEATURES)   # 4
N_OUTPUTS = len(TARGETS)   # 2

# -----------------------------------------------------------------------
# 1. DATA PREPROCESSING
# -----------------------------------------------------------------------
def load_and_preprocess() -> pd.DataFrame:
    print("=" * 70)
    print("  STEP 1: LEAKAGE-FREE DATA PREPROCESSING (DUAL 3S & 10S HORIZONS)")
    print("=" * 70)

    frames = []
    for f in sorted(DATASET_DIR.glob("*.csv")):
        if f.stem == "zeroinput":
            print(f"  [Filter] Dropping '{f.name}': unpowered rows (Vin<0.01V)")
            continue

        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        raw_len = len(df)

        if "predicted_temp_C" in df.columns:
            df = df.drop(columns=["predicted_temp_C"])

        mask = (
            (df["vin_V"] >= 10.0) &
            (df["fault_code"] == 0) &
            (df["current_mA"] >= 0) &
            (df["current_mA"] <= 1500) &
            (df["power_loss_mW"] >= 0) &
            (df["heatsink_temp_C"] > 20.0)
        )
        df_clean = df[mask].copy()
        df_clean["source"] = f.stem

        # Lookaheads: 3s (6 samples) and 10s (20 samples) ahead
        df_clean["temp_future_3s"]  = df_clean["heatsink_temp_C"].shift(-6)
        df_clean["temp_future_10s"] = df_clean["heatsink_temp_C"].shift(-20)
        df_clean = df_clean.dropna(subset=["temp_future_3s", "temp_future_10s"]).copy()

        df_clean["delta_T_3s"]  = df_clean["temp_future_3s"] - df_clean["heatsink_temp_C"]
        df_clean["delta_T_10s"] = df_clean["temp_future_10s"] - df_clean["heatsink_temp_C"]

        print(f"  [Clean] {f.name:12s}: {raw_len} raw -> {len(df_clean)} clean "
              f"(T_hs: {df_clean['heatsink_temp_C'].min():.1f} -> "
              f"{df_clean['heatsink_temp_C'].max():.1f} degC, "
              f"P: {df_clean['power_loss_mW'].min():.0f}-{df_clean['power_loss_mW'].max():.0f} mW)")
        frames.append(df_clean)

    merged = pd.concat(frames, ignore_index=True)
    print(f"\n  Total: {len(merged)} rows across {len(frames)} load conditions.")
    return merged


# -----------------------------------------------------------------------
# 2. AUGMENTATION
# -----------------------------------------------------------------------
def augment_training_data(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Augment training data for ambient temperature robustness.
    Shifts room ambient and heatsink by s without distorting thermal gradients.
    """
    tau = 208.0  # thermal time constant (s)
    shifts = [-15.0, -12.0, -9.0, -6.0, -3.0, 0.0, +3.0, +5.0, +8.0, +10.0]

    augmented = []
    for s in shifts:
        d = df_train.copy()
        d["ambient_temp_C"]  = d["ambient_temp_C"]  + s + np.random.normal(0, 0.05, len(d))
        d["heatsink_temp_C"] = d["heatsink_temp_C"] + s
        d["temp_future_3s"]  = d["temp_future_3s"]  + s
        d["temp_future_10s"] = d["temp_future_10s"] + s
        d["delta_T_3s"]      = d["temp_future_3s"] - d["heatsink_temp_C"]
        d["delta_T_10s"]     = d["temp_future_10s"] - d["heatsink_temp_C"]
        augmented.append(d)

    rng = np.random.default_rng(42)
    standby = []
    for amb in np.linspace(15.0, 45.0, 30):
        for overheat in np.linspace(0.0, 15.0, 10):
            hs = amb + overheat + rng.normal(0, 0.05)
            cooling_delta_3s  = -(overheat) / tau * 3.0
            cooling_delta_10s = -(overheat) / tau * 10.0
            for _ in range(3):
                standby.append({
                    "heatsink_temp_C": hs,
                    "ambient_temp_C":  amb + rng.normal(0, 0.05),
                    "power_loss_mW":   0.0,
                    "current_mA":      0.0,
                    "temp_future_3s":  hs + cooling_delta_3s,
                    "temp_future_10s": hs + cooling_delta_10s,
                    "delta_T_3s":      cooling_delta_3s,
                    "delta_T_10s":     cooling_delta_10s,
                    "source":          "standby_anchor",
                })
    df_standby = pd.DataFrame(standby)

    result = pd.concat([df_standby] + augmented, ignore_index=True)
    print(f"  [Augment] {len(df_train)} train rows -> {len(result)} after augmentation")
    return result


# -----------------------------------------------------------------------
# 3. ARCHITECTURE SEARCH
# -----------------------------------------------------------------------
ARCHITECTURES = {
    "small_32": (32,),
}

def compute_hw_stats(n_inputs: int, hidden: tuple, n_outputs: int = 2) -> dict:
    sizes = [n_inputs] + list(hidden) + [n_outputs]
    params = sum(sizes[i]*sizes[i+1] + sizes[i+1] for i in range(len(sizes)-1))
    return {
        "params":        params,
        "flash_bytes":   params * 4,
        "est_latency_us": round(sum(2*sizes[i]*sizes[i+1] for i in range(len(sizes)-1)) * 3.0 / 16.0, 1),
    }


def run_architecture_search(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("  STEP 2: ARCHITECTURE SEARCH (DUAL RESIDUAL LEARNING, ZERO LEAKAGE)")
    print("=" * 70)
    print(f"  Features: {FEATURES} -> Targets: {TARGETS}")

    train_dfs, test_dfs = [], []
    for src in df["source"].unique():
        sub = df[df["source"] == src]
        n = len(sub)
        cutoff = int(n * 0.80)
        embargo = 30
        train_dfs.append(sub.iloc[:cutoff])
        test_dfs.append(sub.iloc[min(cutoff + embargo, n):])

    df_train_raw = pd.concat(train_dfs, ignore_index=True)
    df_test      = pd.concat(test_dfs,  ignore_index=True)

    print(f"  Raw train: {len(df_train_raw)} samples | Test: {len(df_test)} samples")

    df_train = augment_training_data(df_train_raw)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(df_train[FEATURES].values)
    X_test_s  = scaler.transform(df_test[FEATURES].values)
    y_train = df_train[TARGETS].values
    y_test  = df_test[TARGETS].values

    all_results = []
    for arch_name, hidden in ARCHITECTURES.items():
        hw = compute_hw_stats(N_INPUTS, hidden, N_OUTPUTS)
        mlp = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                           alpha=1e-4, learning_rate_init=2e-3, max_iter=1000,
                           tol=1e-6, random_state=42)
        t0 = time.time()
        mlp.fit(X_train_s, y_train)
        elapsed = time.time() - t0

        pred_delta = mlp.predict(X_test_s)
        pred_future_3s  = df_test["heatsink_temp_C"].values + pred_delta[:, 0]
        pred_future_10s = df_test["heatsink_temp_C"].values + pred_delta[:, 1]
        actual_future_3s  = df_test["temp_future_3s"].values
        actual_future_10s = df_test["temp_future_10s"].values

        mae_3s   = mean_absolute_error(actual_future_3s, pred_future_3s)
        rmse_3s  = np.sqrt(mean_squared_error(actual_future_3s, pred_future_3s))
        r2_3s    = r2_score(actual_future_3s, pred_future_3s)

        mae_10s  = mean_absolute_error(actual_future_10s, pred_future_10s)
        rmse_10s = np.sqrt(mean_squared_error(actual_future_10s, pred_future_10s))
        r2_10s   = r2_score(actual_future_10s, pred_future_10s)

        avg_mae = (mae_3s + mae_10s) / 2.0

        print(f"    {arch_name:<16s} {str(hidden):<14s} params={hw['params']:<5d} "
              f"3s MAE={mae_3s:.4f}C (R2={r2_3s:.4f}) | "
              f"10s MAE={mae_10s:.4f}C (R2={r2_10s:.4f}) | "
              f"lat={hw['est_latency_us']:>5.1f}us  ({mlp.n_iter_:3d} iters, {elapsed:.1f}s)")

        all_results.append({
            "architecture": arch_name, "hidden": list(hidden),
            "features": FEATURES, "targets": TARGETS,
            "params": hw["params"], "flash_bytes": hw["flash_bytes"],
            "est_latency_us": hw["est_latency_us"],
            "test_mae": round(avg_mae, 4),
            "test_mae_3s": round(mae_3s, 4), "test_rmse_3s": round(rmse_3s, 4), "test_r2_3s": round(r2_3s, 4),
            "test_mae_10s": round(mae_10s, 4), "test_rmse_10s": round(rmse_10s, 4), "test_r2_10s": round(r2_10s, 4),
            "n_iter": mlp.n_iter_, "train_time_s": round(elapsed, 2),
        })

    return all_results, df_train_raw, df_test


# -----------------------------------------------------------------------
# 4. TRAIN BEST & EXPORT
# -----------------------------------------------------------------------
def train_and_export_best(df: pd.DataFrame, results: list) -> tuple:
    print("\n" + "=" * 70)
    print("  STEP 3: TRAIN FINAL DUAL-HORIZON MODEL & EXPORT PARAMETERS")
    print("=" * 70)

    best = sorted(results, key=lambda x: x["test_mae"])[0]
    print(f"  Best: {best['architecture']} {best['hidden']} | 3s MAE={best['test_mae_3s']:.4f}C | 10s MAE={best['test_mae_10s']:.4f}C")

    hidden = tuple(best["hidden"])
    df_aug = augment_training_data(df)

    scaler = StandardScaler()
    X = scaler.fit_transform(df_aug[FEATURES].values)
    y = df_aug[TARGETS].values

    mlp = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                       alpha=1e-4, learning_rate_init=2e-3, max_iter=2000,
                       tol=1e-7, random_state=42)
    mlp.fit(X, y)

    full_pred_delta = mlp.predict(X)
    full_pred_3s  = df_aug["heatsink_temp_C"].values + full_pred_delta[:, 0]
    full_pred_10s = df_aug["heatsink_temp_C"].values + full_pred_delta[:, 1]
    full_mae_3s  = mean_absolute_error(df_aug["temp_future_3s"].values, full_pred_3s)
    full_mae_10s = mean_absolute_error(df_aug["temp_future_10s"].values, full_pred_10s)
    print(f"  Full dataset refit MAE: 3s={full_mae_3s:.4f}C, 10s={full_mae_10s:.4f}C")

    joblib.dump({"model": mlp, "scaler": scaler, "features": FEATURES, "targets": TARGETS,
                 "hidden": hidden, "test_mae_3s": best["test_mae_3s"], "test_mae_10s": best["test_mae_10s"]},
                OUTPUT_DIR / "mlp_model_new.joblib")

    with open(OUTPUT_DIR / "scaler_params_new.json", "w") as f:
        json.dump({"features": FEATURES, "mean": scaler.mean_.tolist(),
                   "scale": scaler.scale_.tolist()}, f, indent=2)

    weights_dict = {
        "architecture": best["architecture"], "hidden_layer_sizes": list(hidden),
        "total_params": best["params"],
        "layers": [{"layer": i, "weights_shape": list(W.shape),
                    "weights": W.tolist(), "biases": b.tolist()}
                   for i, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_))]
    }
    with open(OUTPUT_DIR / "mlp_weights_new.json", "w") as f:
        json.dump(weights_dict, f, indent=2)

    metrics = {"best_model": {
        "name": best["architecture"], "hidden_layers": list(hidden),
        "features": FEATURES, "targets": TARGETS,
        "holdout_test_mae_3s": best["test_mae_3s"], "holdout_test_rmse_3s": best["test_rmse_3s"], "holdout_test_r2_3s": best["test_r2_3s"],
        "holdout_test_mae_10s": best["test_mae_10s"], "holdout_test_rmse_10s": best["test_rmse_10s"], "holdout_test_r2_10s": best["test_r2_10s"],
        "parameters": best["params"], "est_latency_us": best["est_latency_us"],
        "flash_bytes": best["flash_bytes"],
    }}
    with open(OUTPUT_DIR / "metrics_new.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return mlp, scaler


# -----------------------------------------------------------------------
# 5. PLOT RESULTS
# -----------------------------------------------------------------------
def plot_results(results, df_train, df_test, best_mlp, best_scaler):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("ThermoGuard -- Dual-Horizon TinyML Thermal Predictor (3s & 10s)",
                 fontsize=14, fontweight="bold")

    # Panel 1: Bar chart comparing 3s and 10s MAE
    ax = axes[0, 0]
    names = [r["architecture"] for r in results]
    m3s  = [r["test_mae_3s"] for r in results]
    m10s = [r["test_mae_10s"] for r in results]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, m3s, width, label="3s Horizon MAE (°C)", color="#1E88E5")
    ax.bar(x + width/2, m10s, width, label="10s Horizon MAE (°C)", color="#FB8C00")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Holdout Test MAE (°C)")
    ax.set_title("Test MAE Across Horizons")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 2: Latency
    ax = axes[0, 1]
    lats = [r["est_latency_us"] for r in results]
    ax.scatter(lats, m3s, c="#1E88E5", s=90, label="3s MAE", zorder=5)
    ax.scatter(lats, m10s, c="#FB8C00", s=90, label="10s MAE", zorder=5)
    ax.set_xlabel("Estimated Latency (µs)")
    ax.set_ylabel("Holdout Test MAE (°C)")
    ax.set_title("Edge Efficiency: Latency vs Test MAE")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 3: Actual vs Predicted 3s & 10s
    ax = axes[1, 0]
    Xt = best_scaler.transform(df_test[FEATURES].values)
    pred_delta = best_mlp.predict(Xt)
    y_meas = df_test["heatsink_temp_C"].values
    yp_3s  = y_meas + pred_delta[:, 0]
    yp_10s = y_meas + pred_delta[:, 1]
    yt_3s  = df_test["temp_future_3s"].values
    yt_10s = df_test["temp_future_10s"].values

    ax.scatter(yt_3s, yp_3s, alpha=0.3, s=8, c="#1E88E5", label="3s Lookahead", edgecolors="none")
    ax.scatter(yt_10s, yp_10s, alpha=0.3, s=8, c="#FB8C00", label="10s Lookahead", edgecolors="none")
    mn = min(yt_3s.min(), yt_10s.min())
    mx = max(yt_3s.max(), yt_10s.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Ideal 1:1")
    ax.set_xlabel("Actual Future Temp (°C)")
    ax.set_ylabel("Predicted Future Temp (°C)")
    ax.set_title("Actual vs Predicted")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 4: Sequential tracking (First 400 Samples)
    ax = axes[1, 1]
    n = min(400, len(yt_3s))
    idx = np.arange(n)
    ax.plot(idx, y_meas[:n], label="Current Heatsink Temp", color="#22D3EE", lw=1.5)
    ax.plot(idx, yp_3s[:n], label="Pred (+3s)", color="#A3E635", lw=1.5, linestyle="--")
    ax.plot(idx, yp_10s[:n], label="Pred (+10s)", color="#FB923C", lw=1.5, linestyle=":")
    ax.set_xlabel("Sample Index (unseen future)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Sequential Dual-Horizon Tracking")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "experiment_results_new.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved chart -> {path}")


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  ThermoGuard -- Dual-Horizon Stateless 4-Feature Thermal Predictor")
    print(f"  Features: {FEATURES}")
    print(f"  Targets:  {TARGETS}")
    print("=" * 70)

    df = load_and_preprocess()
    results, df_train, df_test = run_architecture_search(df)

    best_mlp, best_scaler = train_and_export_best(df_train, results)
    plot_results(results, df_train, df_test, best_mlp, best_scaler)

    best = sorted(results, key=lambda x: x["test_mae"])[0]
    print("\n" + "=" * 70)
    print("  [OK] DUAL-HORIZON STATELESS MODEL TRAINING COMPLETE!")
    print(f"  Best: {best['architecture']} | 3s MAE={best['test_mae_3s']:.4f}C | 10s MAE={best['test_mae_10s']:.4f}C")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
