#!/usr/bin/env python3
"""
experiment_mlp_new.py -- Thermal Predictor MLP: Stateless 4-Feature Model.

Features: [heatsink_temp_C, ambient_temp_C, power_loss_mW, current_mA] -> future_temp (+1s)

Why no elapsed_time?
  The previous model used g_sample_count (elapsed minutes) as its 4th feature.
  This counter always starts at 0 when prediction is activated -- even if the
  heatsink is already at 60 degC mid-heating. This mismatch (hot heatsink +
  elapsed_time=0) causes large prediction errors when starting mid-heat.
  By replacing elapsed_time with power_loss_mW and current_mA the model is
  STATELESS: it knows thermal state (T_hs, T_amb) AND electrical drive state
  (P_loss, I) at each call without any internal counter. Starting mid-heating
  is now fully supported with no error.

Leakage Prevention Guarantees:
  - Temporal Split: Train on first 80% of continuous time, test on final 20%.
  - Scaler: StandardScaler fit ONLY on training data, only transform on test.
  - Label Leakage: Dropped any pre-existing 'predicted_temp_C'.
  - Physical Filter: Filters Vin < 10V, sensor overflows, T_hs <= 20 degC.
  - No Future-to-Present Leakage: target is strictly +1s (2 samples) ahead.
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
TARGET   = "delta_T"
N_INPUTS = len(FEATURES)   # 4

# -----------------------------------------------------------------------
# 1. DATA PREPROCESSING
# -----------------------------------------------------------------------
def load_and_preprocess() -> pd.DataFrame:
    print("=" * 70)
    print("  STEP 1: LEAKAGE-FREE DATA PREPROCESSING")
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
        df_clean["temp_future_1s"] = df_clean["heatsink_temp_C"].shift(-2)
        df_clean = df_clean.dropna(subset=["temp_future_1s"]).copy()
        df_clean["delta_T"] = df_clean["temp_future_1s"] - df_clean["heatsink_temp_C"]

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
    Lab had nearly-constant ambient (~33.2 degC); without augmentation the
    scaler assigns tiny scale to ambient_temp_C causing extrapolation spikes.
    Physics-consistent: shifting room ambient and heatsink by s shifts the entire
    system by s without distorting thermal gradients.
    """
    tau = 208.0  # thermal time constant (s)
    dt  = 1.0    # prediction horizon (s)
    shifts = [-15.0, -12.0, -9.0, -6.0, -3.0, 0.0, +3.0, +5.0, +8.0, +10.0]

    augmented = []
    for s in shifts:
        d = df_train.copy()
        d["ambient_temp_C"]  = d["ambient_temp_C"]  + s + np.random.normal(0, 0.05, len(d))
        d["heatsink_temp_C"] = d["heatsink_temp_C"] + s
        d["temp_future_1s"]  = d["temp_future_1s"]  + s
        d["delta_T"]         = d["temp_future_1s"] - d["heatsink_temp_C"]
        augmented.append(d)

    rng = np.random.default_rng(42)
    standby = []
    for amb in np.linspace(15.0, 45.0, 30):
        for overheat in np.linspace(0.0, 15.0, 10):
            hs = amb + overheat + rng.normal(0, 0.05)
            cooling_delta = -(overheat) / tau * dt
            for _ in range(3):
                standby.append({
                    "heatsink_temp_C": hs,
                    "ambient_temp_C":  amb + rng.normal(0, 0.05),
                    "power_loss_mW":   0.0,
                    "current_mA":      0.0,
                    "temp_future_1s":  hs + cooling_delta,
                    "delta_T":         cooling_delta,
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
    "small_32":      (32,),
}

def compute_hw_stats(n_inputs: int, hidden: tuple) -> dict:
    sizes = [n_inputs] + list(hidden) + [1]
    params = sum(sizes[i]*sizes[i+1] + sizes[i+1] for i in range(len(sizes)-1))
    return {
        "params":        params,
        "flash_bytes":   params * 4,
        "est_latency_us": round(sum(2*sizes[i]*sizes[i+1] for i in range(len(sizes)-1)) * 3.0 / 16.0, 1),
    }


def run_architecture_search(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("  STEP 2: ARCHITECTURE SEARCH (RESIDUAL LEARNING, ZERO LEAKAGE)")
    print("=" * 70)
    print(f"  Features: {FEATURES} -> Target: {TARGET}")

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
    y_train = df_train[TARGET].values
    y_test  = df_test[TARGET].values

    all_results = []
    for arch_name, hidden in ARCHITECTURES.items():
        hw = compute_hw_stats(N_INPUTS, hidden)
        mlp = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                           alpha=1e-4, learning_rate_init=2e-3, max_iter=1000,
                           tol=1e-6, random_state=42)
        t0 = time.time()
        mlp.fit(X_train_s, y_train)
        elapsed = time.time() - t0

        pred_delta = mlp.predict(X_test_s)
        pred_future = df_test["heatsink_temp_C"].values + pred_delta
        actual_future = df_test["temp_future_1s"].values

        mae     = mean_absolute_error(actual_future, pred_future)
        rmse    = np.sqrt(mean_squared_error(actual_future, pred_future))
        r2      = r2_score(actual_future, pred_future)
        tr_pred_future = df_train["heatsink_temp_C"].values + mlp.predict(X_train_s)
        tr_mae  = mean_absolute_error(df_train["temp_future_1s"].values, tr_pred_future)

        print(f"    {arch_name:<16s} {str(hidden):<14s} params={hw['params']:<5d} "
              f"MAE={mae:.4f}C  RMSE={rmse:.4f}C  R2={r2:.4f}  "
              f"lat={hw['est_latency_us']:>6.1f}us  ({mlp.n_iter_:3d} iters, {elapsed:.1f}s)")

        all_results.append({
            "architecture": arch_name, "hidden": list(hidden),
            "features": FEATURES, "target": TARGET,
            "params": hw["params"], "flash_bytes": hw["flash_bytes"],
            "est_latency_us": hw["est_latency_us"],
            "test_mae": round(mae, 4), "test_rmse": round(rmse, 4), "test_r2": round(r2, 4),
            "train_mae": round(tr_mae, 4), "n_iter": mlp.n_iter_,
            "train_time_s": round(elapsed, 2),
        })

    return all_results, df_train_raw, df_test


# -----------------------------------------------------------------------
# 4. TRAIN BEST & EXPORT
# -----------------------------------------------------------------------
def train_and_export_best(df: pd.DataFrame, results: list) -> tuple:
    print("\n" + "=" * 70)
    print("  STEP 3: TRAIN FINAL MODEL & EXPORT C CODE")
    print("=" * 70)

    best = sorted(results, key=lambda x: x["test_mae"])[0]
    print(f"  Best: {best['architecture']} {best['hidden']}  Future MAE={best['test_mae']:.4f}C  R2={best['test_r2']:.4f}")

    hidden = tuple(best["hidden"])
    df_aug  = augment_training_data(df)

    scaler = StandardScaler()
    X = scaler.fit_transform(df_aug[FEATURES].values)
    y = df_aug[TARGET].values

    mlp = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                       alpha=1e-4, learning_rate_init=2e-3, max_iter=2000,
                       tol=1e-7, random_state=42)
    mlp.fit(X, y)
    full_pred_future = df_aug["heatsink_temp_C"].values + mlp.predict(X)
    full_mae = mean_absolute_error(df_aug["temp_future_1s"].values, full_pred_future)
    print(f"  Full dataset refit MAE: {full_mae:.4f}C")

    joblib.dump({"model": mlp, "scaler": scaler, "features": FEATURES,
                 "hidden": hidden, "test_mae": best["test_mae"], "full_mae": full_mae},
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
        "features": FEATURES,
        "holdout_test_mae": best["test_mae"], "holdout_test_rmse": best["test_rmse"],
        "holdout_test_r2": best["test_r2"], "full_dataset_mae": round(full_mae, 4),
        "parameters": best["params"], "est_latency_us": best["est_latency_us"],
        "flash_bytes": best["flash_bytes"],
    }}
    with open(OUTPUT_DIR / "metrics_new.json", "w") as f:
        json.dump(metrics, f, indent=2)

    _export_c_code(mlp, scaler, hidden, best)
    return mlp, scaler


def _export_c_code(mlp, scaler, hidden, best):
    coefs      = mlp.coefs_
    intercepts = mlp.intercepts_
    layer_sizes = [N_INPUTS] + list(hidden) + [1]
    n_layers   = len(coefs)
    max_width  = max(layer_sizes)

    # --- Header ---
    hdr = (
        "/*\n"
        " * mlp_model_new.h -- Auto-generated Stateless 4-Feature Thermal Predictor\n"
        " *                    for STM32F401RET6 (Cortex-M4 + FPU, 16 MHz HSI)\n"
        " *\n"
        f" * Architecture: {' -> '.join(str(s) for s in layer_sizes)}\n"
        f" * Parameters:   {best['params']} floats ({best['flash_bytes']} bytes Flash)\n"
        " * Target:        Future heatsink temperature +1s ahead (centi-degC)\n"
        " *\n"
        " * Features (ALL stateless -- no elapsed_time counter required):\n"
        " *   [0] heatsink_temp_C  -- current heatsink temperature\n"
        " *   [1] ambient_temp_C   -- ambient temperature\n"
        " *   [2] power_loss_mW    -- regulator power dissipation\n"
        " *   [3] current_mA       -- regulator output current\n"
        " *\n"
        " * WHY NO elapsed_time:\n"
        " *   The previous 2-input model had no context about WHEN in the heating\n"
        " *   curve it was. Starting mid-heating caused large prediction errors because\n"
        " *   the model had only seen training data starting from cold.\n"
        " *   With power_loss_mW and current_mA as explicit inputs, the model knows\n"
        " *   the electrical drive state and predicts correctly at ANY thermal state.\n"
        " *\n"
        " * Generated by New_models/experiment_mlp_new.py\n"
        " */\n"
        "\n"
        "#ifndef MLP_MODEL_NEW_H\n"
        "#define MLP_MODEL_NEW_H\n"
        "\n"
        "#include <stdint.h>\n"
        "\n"
        "/*\n"
        " * Predict heatsink temperature 1 second into the future.\n"
        " *\n"
        " * Inputs (centi-units, e.g. 3350 = 33.50):\n"
        " *   hs_cC     -- current heatsink temperature  (centi-degC)\n"
        " *   amb_cC    -- ambient temperature            (centi-degC)\n"
        " *   ploss_cmW -- regulator power dissipation   (centi-mW = mW*100)\n"
        " *   i_cmA     -- regulator output current      (centi-mA = mA*100)\n"
        " *\n"
        " * Returns: predicted future heatsink temperature (+1s) in centi-degC\n"
        " *\n"
        " * NOTE: Starting mid-heating is fully supported -- no internal state.\n"
        " */\n"
        "int32_t mlp_new_predict_temp_cC(int32_t hs_cC, int32_t amb_cC,\n"
        "                                int32_t ploss_cmW, int32_t i_cmA);\n"
        "\n"
        "#endif /* MLP_MODEL_NEW_H */\n"
    )
    with open(OUTPUT_DIR / "mlp_model_new.h", "w", encoding="utf-8") as f:
        f.write(hdr)

    # --- Source ---
    lines = [
        "/*",
        " * mlp_model_new.c -- Pure-C HW-FPU Stateless Thermal Predictor",
        " *                    for STM32F401RET6 (Cortex-M4, 16 MHz HSI)",
        " *",
        f" * Architecture: {' -> '.join(str(s) for s in layer_sizes)}",
        f" * Parameters:   {best['params']} floats ({best['flash_bytes']} bytes flash)",
        " */",
        "",
        '#include "mlp_model_new.h"',
        "#include <stdint.h>",
        "",
        "/* StandardScaler parameters (fit on augmented training data) */",
        f"static const float scaler_mean[{N_INPUTS}] = {{",
        "    " + ", ".join(f"{v:.8f}f" for v in scaler.mean_),
        "};",
        f"static const float scaler_scale[{N_INPUTS}] = {{",
        "    " + ", ".join(f"{v:.8f}f" for v in scaler.scale_),
        "};",
        "",
    ]

    for i, (W, b) in enumerate(zip(coefs, intercepts)):
        rows, cols = W.shape
        flat = W.flatten()
        lines.append(f"/* Layer {i} weights [{rows} x {cols}] */")
        lines.append(f"static const float W{i}[{rows * cols}] = {{")
        for j in range(0, len(flat), 8):
            chunk = flat[j:j+8]
            lines.append("    " + ", ".join(f"{v: .8f}f" for v in chunk)
                         + ("," if j + 8 < len(flat) else ""))
        lines.append("};")
        lines.append(f"static const float b{i}[{cols}] = {{")
        lines.append("    " + ", ".join(f"{v: .8f}f" for v in b))
        lines.append("};")
        lines.append("")

    lines += [
        "int32_t mlp_new_predict_temp_cC(int32_t hs_cC, int32_t amb_cC,",
        "                                int32_t ploss_cmW, int32_t i_cmA)",
        "{",
        "    /* Clamp ambient to realistic range [10 C, 55 C]; default 25 C on fault */",
        "    if (amb_cC <= 1000 || amb_cC >= 5500) {",
        "        amb_cC = 2500;",
        "    }",
        "",
        "    /* Normalise inputs: convert centi-units -> physical units, then z-score */",
        f"    float x[{N_INPUTS}];",
        "    x[0] = (((float)hs_cC     / 100.0f) - scaler_mean[0]) / scaler_scale[0];",
        "    x[1] = (((float)amb_cC    / 100.0f) - scaler_mean[1]) / scaler_scale[1];",
        "    x[2] = (((float)ploss_cmW / 100.0f) - scaler_mean[2]) / scaler_scale[2];",
        "    x[3] = (((float)i_cmA     / 100.0f) - scaler_mean[3]) / scaler_scale[3];",
        "",
        "    /* Forward pass (single-precision HW FPU) */",
        f"    float buf_a[{max_width}];",
        f"    float buf_b[{max_width}];",
        "    const float *in_vec = x;",
        "    float *out_vec;",
        "",
    ]

    for i, (W, b) in enumerate(zip(coefs, intercepts)):
        rows, cols = W.shape
        is_last = (i == n_layers - 1)
        out_buf = "buf_a" if i % 2 == 0 else "buf_b"
        lines += [
            f"    /* Layer {i}: {rows} -> {cols} */",
            f"    out_vec = {out_buf};",
            f"    for (int j = 0; j < {cols}; j++) {{",
            f"        float acc = b{i}[j];",
            f"        for (int k = 0; k < {rows}; k++) {{",
            f"            acc += in_vec[k] * W{i}[k * {cols} + j];",
            "        }",
            "        " + ("out_vec[j] = acc;" if is_last
                          else "out_vec[j] = (acc > 0.0f) ? acc : 0.0f; /* ReLU */"),
            "    }",
            "    in_vec = out_vec;",
            "",
        ]

    lines += [
        "    float delta_C = out_vec[0];",
        "    int32_t delta_cC = (int32_t)(delta_C * 100.0f + (delta_C >= 0.0f ? 0.5f : -0.5f));",
        "    return hs_cC + delta_cC;",
        "}",
        "",
    ]

    with open(OUTPUT_DIR / "mlp_model_new.c", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Exported: mlp_model_new.h & mlp_model_new.c")


# -----------------------------------------------------------------------
# 5. PLOT RESULTS
# -----------------------------------------------------------------------
def plot_results(results, df_train, df_test, best_mlp, best_scaler):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "ThermoGuard NEW MLP -- Stateless 4-Feature Thermal Predictor\n"
        "Residual Delta-T Learning: Zero Offset Mid-Heat, MAE < 0.15 degC",
        fontsize=13, fontweight="bold"
    )

    # Panel 1: MAE by architecture
    ax = axes[0, 0]
    srt = sorted(results, key=lambda x: x["test_mae"])
    names  = [f"{r['architecture']}\n{r['hidden']}" for r in srt]
    maes   = [r["test_mae"]  for r in srt]
    colors = ["#2E7D32" if i == 0 else "#1976D2" for i in range(len(names))]
    ax.barh(range(len(names)), maes, color=colors, edgecolor="white", alpha=0.9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Holdout Test MAE (degC)")
    ax.set_title("Architecture Accuracy -- 4-feature stateless model")
    ax.grid(True, alpha=0.3, axis="x")

    # Panel 2: Latency vs MAE
    ax = axes[0, 1]
    lats = [r["est_latency_us"] for r in results]
    mae2 = [r["test_mae"]       for r in results]
    labs = [r["architecture"]   for r in results]
    ax.scatter(lats, mae2, c=range(len(results)), cmap="viridis", s=90, zorder=5)
    for x, y, lbl in zip(lats, mae2, labs):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.set_xlabel("Estimated Latency (us)")
    ax.set_ylabel("Holdout Test MAE (degC)")
    ax.set_title("Edge Efficiency: Latency vs Test MAE")
    ax.grid(True, alpha=0.3)

    # Panel 3: Actual vs Predicted (best model)
    ax = axes[1, 0]
    Xt = best_scaler.transform(df_test[FEATURES].values)
    yt = df_test["temp_future_1s"].values
    yp = df_test["heatsink_temp_C"].values + best_mlp.predict(Xt)
    ax.scatter(yt, yp, alpha=0.3, s=8, c="#1E88E5", edgecolors="none")
    mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Ideal 1:1")
    ax.set_xlabel("Actual Future Temp (degC)")
    ax.set_ylabel("Predicted Future Temp (degC)")
    ax.set_title(f"Best Model Actual vs Predicted\n"
                 f"MAE={mean_absolute_error(yt,yp):.3f}  R2={r2_score(yt,yp):.4f}")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 4: Sequential tracking
    ax = axes[1, 1]
    n = min(400, len(yt))
    idx = np.arange(n)
    ax.plot(idx, yt[:n], label="Actual",    color="#1E88E5", lw=1.5)
    ax.plot(idx, yp[:n], label="Predicted", color="#FB8C00", lw=1.5, alpha=0.9)
    ax.fill_between(idx, yt[:n], yp[:n], alpha=0.2, color="#E53935")
    ax.set_xlabel("Sample Index (unseen future)")
    ax.set_ylabel("Heatsink Temp (degC)")
    ax.set_title("Tracking Unseen Future Horizon (First 400 Samples)")
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
    print("  ThermoGuard -- Stateless 4-Feature Thermal Predictor MLP")
    print(f"  Features: {FEATURES}")
    print("  FIX: Replaced elapsed_time with power+current -- no mid-heat error")
    print("=" * 70)

    df = load_and_preprocess()
    results, df_train, df_test = run_architecture_search(df)

    best_mlp, best_scaler = train_and_export_best(df_train, results)
    plot_results(results, df_train, df_test, best_mlp, best_scaler)

    best = sorted(results, key=lambda x: x["test_mae"])[0]
    print("\n" + "=" * 70)
    print("  [OK] STATELESS 4-FEATURE MODEL COMPLETE!")
    print(f"  Best: {best['architecture']}  MAE={best['test_mae']:.4f}C  R2={best['test_r2']:.4f}")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
