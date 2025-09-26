import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

# -----------------------------
# 1. Load Enhanced Models + Scaler
# -----------------------------
scaler = joblib.load("enhanced_model/enhanced_scaler.pkl")
xgb_model = joblib.load("enhanced_model/enhanced_xgb_model.pkl")
nn_model = joblib.load("enhanced_model/enhanced_nn_model.pkl")
lgb_model = joblib.load("enhanced_model/enhanced_lgb_model.pkl")
cat_model = joblib.load("enhanced_model/enhanced_catboost_model.pkl")

# Optimal threshold (من التدريب السابق)
optimal_threshold = 0.6752

# -----------------------------
# 2. Function to create features
# -----------------------------
def create_features(sample):
    period = sample['period']
    depth = sample['depth']
    duration = sample['duration']
    radius = sample['radius']
    teq = sample['teq']
    insol = sample['insol']
    steff = sample['steff']
    slogg = sample['slogg']
    srad = sample['srad']
    mag = sample['mag']

    period_depth_ratio = period / (depth + 1e-6)
    duration_period_ratio = duration / (period + 1e-6)
    radius_teq_ratio = radius / (teq + 1e-6)
    log_period = np.log1p(period)
    log_depth = np.log1p(depth)
    log_radius = np.log1p(radius)
    log_insol = np.log1p(insol + 1e-6)

    habitable_zone = 1 if 0.3 < insol < 2 and 180 < teq < 400 else 0
    hot_jupiter = 1 if radius > 5 and period < 10 and teq > 1000 else 0
    super_earth = 1 if 1.5 <= radius <= 3 else 0

    steff_srad_ratio = steff / (srad + 1e-6)
    slogg_srad_ratio = slogg / (srad + 1e-6)

    transit_efficiency = depth / (duration + 1e-6)
    signal_strength = depth * duration

    hot_zone = 1 if teq > 1000 else 0
    warm_zone = 1 if 400 <= teq <= 1000 else 0
    cold_zone = 1 if teq < 400 else 0

    features = [
        period, depth, duration, radius, teq, insol,
        steff, slogg, srad, mag,
        period_depth_ratio, duration_period_ratio, radius_teq_ratio,
        log_period, log_depth, log_radius, log_insol,
        habitable_zone, hot_jupiter, super_earth,
        steff_srad_ratio, slogg_srad_ratio,
        transit_efficiency, signal_strength,
        hot_zone, warm_zone, cold_zone
    ]

    return features

# -----------------------------
# 3. 15 Test Samples
# -----------------------------
test_samples = [
    {'name': 'Earth-like 1', 'period': 365, 'depth': 100, 'duration': 12, 'radius': 1.0, 'teq': 300, 'insol': 1, 'steff': 5800, 'slogg': 4.4, 'srad': 1.0, 'mag': 10.0, 'true':1},
    {'name': 'Hot Jupiter 1', 'period': 3.0, 'depth': 2000, 'duration': 2, 'radius': 8.0, 'teq': 1500, 'insol': 100, 'steff': 6000, 'slogg': 4.2, 'srad': 1.2, 'mag': 12.0, 'true':1},
    {'name': 'Super Earth 1', 'period': 10.5, 'depth': 500, 'duration': 3.2, 'radius': 2.1, 'teq': 800, 'insol': 50, 'steff': 5500, 'slogg': 4.5, 'srad': 1.0, 'mag': 12.0, 'true':1},
    {'name': 'Cold Gas Giant 1', 'period': 1000.0, 'depth': 800, 'duration': 8, 'radius': 5.0, 'teq': 200, 'insol': 0.1, 'steff': 5000, 'slogg': 4.6, 'srad': 0.8, 'mag': 14.0, 'true':0},
    {'name': 'False Positive 1', 'period': 0.5, 'depth': 50, 'duration': 1, 'radius': 0.5, 'teq': 1200, 'insol': 500, 'steff': 7000, 'slogg': 3.8, 'srad': 1.5, 'mag': 15.0, 'true':0},
    {'name': 'Earth-like 2', 'period': 220, 'depth': 80, 'duration': 10, 'radius': 1.1, 'teq': 250, 'insol': 0.8, 'steff': 5600, 'slogg': 4.4, 'srad': 1.0, 'mag': 11.0, 'true':1},
    {'name': 'Hot Jupiter 2', 'period': 4.0, 'depth': 2500, 'duration': 3, 'radius': 6.0, 'teq': 1400, 'insol': 120, 'steff': 6100, 'slogg': 4.1, 'srad': 1.3, 'mag': 11.5, 'true':1},
    {'name': 'Super Earth 2', 'period': 15.0, 'depth': 600, 'duration': 3.5, 'radius': 2.5, 'teq': 600, 'insol': 30, 'steff': 5400, 'slogg': 4.5, 'srad': 0.9, 'mag': 12.5, 'true':1},
    {'name': 'Cold Gas Giant 2', 'period': 800.0, 'depth': 500, 'duration': 7, 'radius': 4.5, 'teq': 180, 'insol': 0.05, 'steff': 5000, 'slogg': 4.7, 'srad': 0.8, 'mag': 14.0, 'true':0},
    {'name': 'False Positive 2', 'period': 1.0, 'depth': 20, 'duration': 0.5, 'radius': 0.3, 'teq': 2000, 'insol': 800, 'steff': 7500, 'slogg': 3.5, 'srad': 1.7, 'mag': 16.0, 'true':0},
    {'name': 'Earth-like 3', 'period': 300, 'depth': 90, 'duration': 11, 'radius': 1.05, 'teq': 280, 'insol': 1.2, 'steff': 5700, 'slogg': 4.4, 'srad': 1.0, 'mag': 10.5, 'true':1},
    {'name': 'Hot Jupiter 3', 'period': 5.0, 'depth': 1800, 'duration': 3, 'radius': 7.0, 'teq': 1300, 'insol': 90, 'steff': 5900, 'slogg': 4.2, 'srad': 1.1, 'mag': 12.0, 'true':1},
    {'name': 'Super Earth 3', 'period': 8.0, 'depth': 550, 'duration': 3.0, 'radius': 2.0, 'teq': 700, 'insol': 40, 'steff': 5600, 'slogg': 4.5, 'srad': 1.0, 'mag': 13.0, 'true':1},
    {'name': 'Cold Gas Giant 3', 'period': 900.0, 'depth': 700, 'duration': 7.5, 'radius': 5.0, 'teq': 190, 'insol': 0.08, 'steff': 5100, 'slogg': 4.6, 'srad': 0.9, 'mag': 14.5, 'true':0},
    {'name': 'False Positive 3', 'period': 0.3, 'depth': 30, 'duration': 0.3, 'radius': 0.2, 'teq': 2500, 'insol': 1000, 'steff': 8000, 'slogg': 3.0, 'srad': 2.0, 'mag': 17.0, 'true':0},
]

# -----------------------------
# 4. Predictions + Compare
# -----------------------------
correct = 0
results = []

for sample in test_samples:
    feats = np.array(create_features(sample)).reshape(1, -1)
    feats_scaled = scaler.transform(feats)

    # XGBoost
    proba_xgb = xgb_model.predict_proba(feats_scaled)[:,1]

    # Neural Net
    proba_nn = nn_model.predict(feats_scaled).ravel()

    # LightGBM Booster
    if isinstance(lgb_model, lgb.Booster):
        proba_lgb = lgb_model.predict(feats_scaled)  # booster output probability directly
    else:
        proba_lgb = lgb_model.predict_proba(feats_scaled)[:,1]

    # CatBoost
    proba_cat = cat_model.predict_proba(feats_scaled)[:,1]

    proba = (proba_xgb + proba_nn + proba_lgb + proba_cat) / 4.0
    pred_class = 1 if proba >= optimal_threshold else 0

    results.append({
        "name": sample['name'],
        "true": sample['true'],
        "pred_proba": float(proba),
        "pred_class": pred_class
    })

    if pred_class == sample['true']:
        correct += 1

df = pd.DataFrame(results)
print(df)
print(f"\nModel got {correct} / {len(test_samples)} correct ({correct/len(test_samples)*100:.1f}%)")
