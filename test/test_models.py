import pandas as pd
import numpy as np
import joblib
from load_models import load_models, predict_kepler_data, predict_toi_data

def test_kepler_model():
    print("=== TESTING KEPLER MODELS ===")
    
    models = load_models()
    if models is None:
        return
    
    df = pd.read_csv('datasets/cumulative_2025.09.25_11.23.48.csv', comment='#')
    
    df['koi_disposition'] = df['koi_disposition'].fillna('UNKNOWN')
    df['is_confirmed'] = (df['koi_disposition'] == 'CONFIRMED').astype(int)
    
    feature_cols = models['metadata']['kepler_features']
    
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=feature_cols + ['is_confirmed'])
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['is_confirmed']
    
    print(f"Testing on {len(X)} samples")
    print(f"Confirmed planets: {y.sum()}")
    
    predictions = predict_kepler_data(models, X)
    
    if predictions:
        xgb_pred = predictions['xgb_pred']
        nn_pred = predictions['nn_pred']
        ensemble_pred = predictions['ensemble_pred']
        
        print(f"\nXGBoost Predictions:")
        print(f"  Min: {xgb_pred.min():.3f}")
        print(f"  Max: {xgb_pred.max():.3f}")
        print(f"  Mean: {xgb_pred.mean():.3f}")
        
        print(f"\nNeural Network Predictions:")
        print(f"  Min: {nn_pred.min():.3f}")
        print(f"  Max: {nn_pred.max():.3f}")
        print(f"  Mean: {nn_pred.mean():.3f}")
        
        print(f"\nEnsemble Predictions:")
        print(f"  Min: {ensemble_pred.min():.3f}")
        print(f"  Max: {ensemble_pred.max():.3f}")
        print(f"  Mean: {ensemble_pred.mean():.3f}")
        
        threshold = 0.5
        xgb_classified = (xgb_pred >= threshold).sum()
        nn_classified = (nn_pred >= threshold).sum()
        ensemble_classified = (ensemble_pred >= threshold).sum()
        
        print(f"\nPredictions >= {threshold}:")
        print(f"  XGBoost: {xgb_classified} ({xgb_classified/len(X)*100:.1f}%)")
        print(f"  Neural Network: {nn_classified} ({nn_classified/len(X)*100:.1f}%)")
        print(f"  Ensemble: {ensemble_classified} ({ensemble_classified/len(X)*100:.1f}%)")

def test_toi_model():
    print("\n=== TESTING TOI MODELS ===")
    
    models = load_models()
    if models is None:
        return
    
    toi_df = pd.read_csv('datasets/TOI_2025.09.25_11.23.28.csv', comment='#')
    
    toi_df['tfopwg_disp'] = toi_df['tfopwg_disp'].fillna('UNKNOWN')
    toi_df['is_confirmed'] = (toi_df['tfopwg_disp'] == 'PC').astype(int)
    
    feature_mapping = models['metadata']['toi_feature_mapping']
    
    toi_df_clean = toi_df.dropna(subset=['is_confirmed'])
    
    X_toi = pd.DataFrame()
    for kepler_col, toi_col in feature_mapping.items():
        if toi_col and toi_col in toi_df_clean.columns:
            X_toi[kepler_col] = pd.to_numeric(toi_df_clean[toi_col], errors='coerce')
        else:
            X_toi[kepler_col] = 0
    
    X_toi = X_toi.fillna(X_toi.median())
    y_toi = toi_df_clean['is_confirmed']
    
    print(f"Testing on {len(X_toi)} samples")
    print(f"Confirmed planets (PC): {y_toi.sum()}")
    
    predictions = predict_toi_data(models, X_toi)
    
    if predictions:
        xgb_pred = predictions['xgb_pred']
        nn_pred = predictions['nn_pred']
        ensemble_pred = predictions['ensemble_pred']
        
        print(f"\nXGBoost Fine-tuned Predictions:")
        print(f"  Min: {xgb_pred.min():.3f}")
        print(f"  Max: {xgb_pred.max():.3f}")
        print(f"  Mean: {xgb_pred.mean():.3f}")
        
        print(f"\nNeural Network Fine-tuned Predictions:")
        print(f"  Min: {nn_pred.min():.3f}")
        print(f"  Max: {nn_pred.max():.3f}")
        print(f"  Mean: {nn_pred.mean():.3f}")
        
        print(f"\nEnsemble Predictions:")
        print(f"  Min: {ensemble_pred.min():.3f}")
        print(f"  Max: {ensemble_pred.max():.3f}")
        print(f"  Mean: {ensemble_pred.mean():.3f}")
        
        threshold = 0.5
        xgb_classified = (xgb_pred >= threshold).sum()
        nn_classified = (nn_pred >= threshold).sum()
        ensemble_classified = (ensemble_pred >= threshold).sum()
        
        print(f"\nPredictions >= {threshold}:")
        print(f"  XGBoost: {xgb_classified} ({xgb_classified/len(X_toi)*100:.1f}%)")
        print(f"  Neural Network: {nn_classified} ({nn_classified/len(X_toi)*100:.1f}%)")
        print(f"  Ensemble: {ensemble_classified} ({ensemble_classified/len(X_toi)*100:.1f}%)")

def test_single_sample():
    print("\n=== TESTING SINGLE SAMPLE ===")
    
    models = load_models()
    if models is None:
        return
    
    sample_data = pd.DataFrame({
        'koi_period': [10.5],
        'koi_time0bk': [100.0],
        'koi_impact': [0.3],
        'koi_duration': [3.2],
        'koi_depth': [500.0],
        'koi_prad': [2.1],
        'koi_teq': [800.0],
        'koi_insol': [50.0],
        'koi_model_snr': [15.0],
        'koi_steff': [5500.0],
        'koi_slogg': [4.5],
        'koi_srad': [1.0],
        'koi_kepmag': [12.0],
        'koi_fpflag_nt': [0],
        'koi_fpflag_ss': [0],
        'koi_fpflag_co': [0],
        'koi_fpflag_ec': [0]
    })
    
    print("Sample data:")
    print(sample_data)
    
    kepler_predictions = predict_kepler_data(models, sample_data)
    toi_predictions = predict_toi_data(models, sample_data)
    
    if kepler_predictions and toi_predictions:
        print(f"\nKepler Models Predictions:")
        print(f"  XGBoost: {kepler_predictions['xgb_pred'][0]:.3f}")
        print(f"  Neural Network: {kepler_predictions['nn_pred'][0]:.3f}")
        print(f"  Ensemble: {kepler_predictions['ensemble_pred'][0]:.3f}")
        
        print(f"\nTOI Models Predictions:")
        print(f"  XGBoost Fine-tuned: {toi_predictions['xgb_pred'][0]:.3f}")
        print(f"  Neural Network Fine-tuned: {toi_predictions['nn_pred'][0]:.3f}")
        print(f"  Ensemble: {toi_predictions['ensemble_pred'][0]:.3f}")
        
        threshold = 0.5
        print(f"\nClassification (threshold = {threshold}):")
        print(f"  Kepler Ensemble: {'CONFIRMED' if kepler_predictions['ensemble_pred'][0] >= threshold else 'NOT CONFIRMED'}")
        print(f"  TOI Ensemble: {'CONFIRMED' if toi_predictions['ensemble_pred'][0] >= threshold else 'NOT CONFIRMED'}")

def main():
    print("Testing all models on available data...")
    
    test_kepler_model()
    test_toi_model()
    test_single_sample()
    
    print("\n=== TESTING COMPLETE ===")
    print("You can now use these models for predictions!")

if __name__ == "__main__":
    main()
