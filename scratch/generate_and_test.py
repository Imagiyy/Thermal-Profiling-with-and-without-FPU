import json
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import subprocess

# 1. Load data
frames = []
for f in ['10ohm.csv', '20ohm.csv', 'noload.csv']:
    df = pd.read_csv('dataset/' + f)
    df.columns = [c.strip() for c in df.columns]
    mask = (df['vin_V'] >= 10.0) & (df['fault_code'] == 0) & (df['current_mA'] >= 0) & (df['power_loss_mW'] >= 0)
    d = df[mask].copy()
    d['source'] = f
    d['target_10s'] = d['heatsink_temp_C'].shift(-20)
    d['delta_T_10s'] = d['target_10s'] - d['heatsink_temp_C']
    d = d.dropna()
    frames.append(d)

df_all = pd.concat(frames, ignore_index=True)

# 2. Strict temporal split for evaluation
train_list, test_list = [], []
for src in ['10ohm.csv', '20ohm.csv', 'noload.csv']:
    sub = df_all[df_all['source'] == src]
    cutoff = int(len(sub) * 0.8)
    train_list.append(sub.iloc[:cutoff])
    test_list.append(sub.iloc[cutoff + 30:])

df_train = pd.concat(train_list, ignore_index=True)
df_test  = pd.concat(test_list, ignore_index=True)

# 3. Cooling & standby physics anchors
cooling_rows = []
tau = 208.0
dt = 10.0
decay_factor = 1.0 - np.exp(-dt / tau)

for amb in np.linspace(15.0, 45.0, 15):
    for hs in np.linspace(amb, 95.0, 25):
        cooling_rows.append({
            'power_loss_mW': 0.0,
            'current_mA': 0.0,
            'ambient_temp_C': amb,
            'heatsink_temp_C': hs,
            'target_10s': hs - (hs - amb) * decay_factor,
            'delta_T_10s': -(hs - amb) * decay_factor,
            'source': 'cooling_anchor'
        })
df_cooling = pd.DataFrame(cooling_rows)

aug_frames = [df_train, df_cooling]
for shift in [-12.0, -8.0, -4.0, 4.0, 8.0]:
    d = df_train.copy()
    d['ambient_temp_C'] += shift
    d['heatsink_temp_C'] += shift
    d['target_10s'] += shift
    aug_frames.append(d)

df_train_aug = pd.concat(aug_frames, ignore_index=True)

feats = ['power_loss_mW', 'current_mA', 'ambient_temp_C', 'heatsink_temp_C']
scaler = StandardScaler()
X_tr = scaler.fit_transform(df_train_aug[feats])
X_te = scaler.transform(df_test[feats])

mlp = MLPRegressor(hidden_layer_sizes=(16,), activation='relu', solver='adam', alpha=1e-4, learning_rate_init=2e-3, max_iter=1200, random_state=42)
mlp.fit(X_tr, df_train_aug['delta_T_10s'])

pred_te = df_test['heatsink_temp_C'].values + mlp.predict(X_te)
mae = mean_absolute_error(df_test['target_10s'], pred_te)
r2 = r2_score(df_test['target_10s'], pred_te)
print(f"Validation Holdout Test MAE: {mae:.4f} °C, R2: {r2:.4f}")

# Now fit on full augmented data for deployment
aug_full = [df_all, df_cooling]
for shift in [-12.0, -8.0, -4.0, 4.0, 8.0]:
    d = df_all.copy()
    d['ambient_temp_C'] += shift
    d['heatsink_temp_C'] += shift
    d['target_10s'] += shift
    aug_full.append(d)

df_deploy = pd.concat(aug_full, ignore_index=True)
scaler_deploy = StandardScaler()
X_deploy = scaler_deploy.fit_transform(df_deploy[feats])

mlp_deploy = MLPRegressor(hidden_layer_sizes=(16,), activation='relu', solver='adam', alpha=1e-4, learning_rate_init=2e-3, max_iter=1200, random_state=42)
mlp_deploy.fit(X_deploy, df_deploy['delta_T_10s'])

full_pred = df_deploy['heatsink_temp_C'].values + mlp_deploy.predict(X_deploy)
full_mae = mean_absolute_error(df_deploy['target_10s'], full_pred)
print(f"Full Dataset Refit MAE: {full_mae:.4f} °C")

# Save model bundle
joblib.dump({
    'model': mlp_deploy,
    'scaler': scaler_deploy,
    'features': feats,
    'hidden': (16,),
    'test_mae': round(float(mae), 4),
    'test_r2': round(float(r2), 4),
    'full_mae': round(float(full_mae), 4)
}, 'New_models/mlp_model_new.joblib')

# Save scaler JSON
with open('New_models/scaler_params_new.json', 'w') as f:
    json.dump({
        'features': feats,
        'mean': scaler_deploy.mean_.tolist(),
        'scale': scaler_deploy.scale_.tolist()
    }, f, indent=2)

# Save weights JSON
W0 = mlp_deploy.coefs_[0] # (4, 16)
b0 = mlp_deploy.intercepts_[0] # (16,)
W1 = mlp_deploy.coefs_[1] # (16, 1)
b1 = mlp_deploy.intercepts_[1] # (1,)

with open('New_models/mlp_weights_new.json', 'w') as f:
    json.dump({
        'architecture': 'small_16',
        'hidden_layer_sizes': [16],
        'total_params': 4*16 + 16 + 16*1 + 1,
        'layers': [
            {'layer': 0, 'weights': W0.tolist(), 'biases': b0.tolist()},
            {'layer': 1, 'weights': W1.tolist(), 'biases': b1.tolist()}
        ]
    }, f, indent=2)

# Save metrics JSON
with open('New_models/metrics_new.json', 'w') as f:
    json.dump({
        'best_model': {
            'name': 'small_16',
            'hidden_layers': [16],
            'holdout_test_mae': round(float(mae), 4),
            'holdout_test_r2': round(float(r2), 4),
            'full_dataset_mae': round(float(full_mae), 4),
            'parameters': 97,
            'est_latency_us': 15.0,
            'flash_bytes': 388
        }
    }, f, indent=2)

print("Saved all model artifacts to New_models/ successfully!")
