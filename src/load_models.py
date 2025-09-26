import pandas as pd
import numpy as np
import joblib
import os

def load_models():
    models_dir = "saved_models"
    
    if not os.path.exists(models_dir):
        print("Models directory not found! Please train models first.")
        return None
    
    print("Loading saved models...")
    
    models = {
        'xgb_kepler': joblib.load(f"{models_dir}/xgb_kepler_model.pkl"),
        'nn_kepler': joblib.load(f"{models_dir}/nn_kepler_model.pkl"),
        'nn_kepler_scaler': joblib.load(f"{models_dir}/nn_kepler_scaler.pkl"),
        'xgb_toi': joblib.load(f"{models_dir}/xgb_toi_finetuned.pkl"),
        'nn_toi': joblib.load(f"{models_dir}/nn_toi_finetuned.pkl"),
        'nn_toi_scaler': joblib.load(f"{models_dir}/nn_toi_scaler.pkl"),
        'metadata': joblib.load(f"{models_dir}/model_metadata.pkl")
    }
    
    print("Models loaded successfully!")
    return models

def predict_kepler_data(models, data):
    if models is None:
        return None
    
    xgb_model = models['xgb_kepler']
    nn_model = models['nn_kepler']
    nn_scaler = models['nn_kepler_scaler']
    
    xgb_pred = xgb_model.predict_proba(data)[:, 1]
    nn_pred = nn_model.predict_proba(nn_scaler.transform(data))[:, 1]
    
    ensemble_pred = (xgb_pred + nn_pred) / 2
    
    return {
        'xgb_pred': xgb_pred,
        'nn_pred': nn_pred,
        'ensemble_pred': ensemble_pred
    }

def predict_toi_data(models, data):
    if models is None:
        return None
    
    xgb_model = models['xgb_toi']
    nn_model = models['nn_toi']
    nn_scaler = models['nn_toi_scaler']
    
    xgb_pred = xgb_model.predict_proba(data)[:, 1]
    nn_pred = nn_model.predict_proba(nn_scaler.transform(data))[:, 1]
    
    ensemble_pred = (xgb_pred + nn_pred) / 2
    
    return {
        'xgb_pred': xgb_pred,
        'nn_pred': nn_pred,
        'ensemble_pred': ensemble_pred
    }

def predict_single_sample(models, data, data_type='kepler'):
    if models is None:
        return None
    
    if data_type == 'kepler':
        return predict_kepler_data(models, data)
    elif data_type == 'toi':
        return predict_toi_data(models, data)
    else:
        print("Invalid data_type. Use 'kepler' or 'toi'")
        return None

def main():
    models = load_models()
    
    if models is None:
        return
    
    print("\nModel Information:")
    for key, info in models['metadata']['model_info'].items():
        print(f"  {key}: {info}")
    
    print(f"\nKepler Features: {len(models['metadata']['kepler_features'])}")
    print(f"TOI Feature Mapping: {len(models['metadata']['toi_feature_mapping'])}")
    
    print("\nModels are ready for prediction!")
    print("Use predict_kepler_data() or predict_toi_data() functions")

if __name__ == "__main__":
    main()
