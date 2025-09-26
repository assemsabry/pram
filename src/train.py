import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    print("Installing imbalanced-learn...")
    import subprocess
    subprocess.check_call(["pip", "install", "imbalanced-learn"])
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.pipeline import Pipeline as ImbPipeline

# Try to import LightGBM and CatBoost
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("Installing LightGBM...")
    import subprocess
    subprocess.check_call(["pip", "install", "lightgbm"])
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    print("Installing CatBoost...")
    import subprocess
    subprocess.check_call(["pip", "install", "catboost"])
    import catboost as cb
    CATBOOST_AVAILABLE = True

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

def load_all_datasets():
    print("=== LOADING ALL DATASETS ===")
    
    # 1. Kepler Data
    print("Loading Kepler data...")
    kepler_df = pd.read_csv('datasets/cumulative_2025.09.25_11.23.48.csv', comment='#')
    kepler_df['koi_disposition'] = kepler_df['koi_disposition'].fillna('UNKNOWN')
    kepler_df['is_confirmed'] = (kepler_df['koi_disposition'] == 'CONFIRMED').astype(int)
    kepler_df['data_source'] = 'kepler'
    
    # 2. TOI Data
    print("Loading TOI data...")
    toi_df = pd.read_csv('datasets/TOI_2025.09.25_11.23.28.csv', comment='#')
    toi_df['tfopwg_disp'] = toi_df['tfopwg_disp'].fillna('UNKNOWN')
    toi_df['is_confirmed'] = (toi_df['tfopwg_disp'] == 'PC').astype(int)
    toi_df['data_source'] = 'toi'
    
    # 3. K2 Data
    print("Loading K2 data...")
    k2_df = pd.read_csv('datasets/k2pandc_2025.09.25_11.23.21.csv', comment='#')
    k2_df['disposition'] = k2_df['disposition'].fillna('UNKNOWN')
    k2_df['is_confirmed'] = (k2_df['disposition'] == 'CONFIRMED').astype(int)
    k2_df['data_source'] = 'k2'
    
    return kepler_df, toi_df, k2_df

def create_enhanced_features(kepler_df, toi_df, k2_df):
    print("\n=== CREATING ENHANCED FEATURES ===")
    
    # Feature mapping for all datasets
    feature_mapping = {
        'period': {
            'kepler': 'koi_period',
            'toi': 'pl_orbper', 
            'k2': 'pl_orbper'
        },
        'depth': {
            'kepler': 'koi_depth',
            'toi': 'pl_trandep',
            'k2': None
        },
        'duration': {
            'kepler': 'koi_duration',
            'toi': 'pl_trandurh',
            'k2': None
        },
        'radius': {
            'kepler': 'koi_prad',
            'toi': 'pl_rade',
            'k2': 'pl_rade'
        },
        'teq': {
            'kepler': 'koi_teq',
            'toi': 'pl_eqt',
            'k2': 'pl_eqt'
        },
        'insol': {
            'kepler': 'koi_insol',
            'toi': 'pl_insol',
            'k2': 'pl_insol'
        },
        'steff': {
            'kepler': 'koi_steff',
            'toi': 'st_teff',
            'k2': 'st_teff'
        },
        'slogg': {
            'kepler': 'koi_slogg',
            'toi': 'st_logg',
            'k2': 'st_logg'
        },
        'srad': {
            'kepler': 'koi_srad',
            'toi': 'st_rad',
            'k2': 'st_rad'
        },
        'mag': {
            'kepler': 'koi_kepmag',
            'toi': 'st_tmag',
            'k2': 'sy_vmag'
        }
    }
    
    unified_data = []
    
    for dataset_name, df in [('kepler', kepler_df), ('toi', toi_df), ('k2', k2_df)]:
        print(f"Processing {dataset_name} data...")
        
        # Create unified features
        sample_data = pd.DataFrame()
        sample_data['data_source'] = df['data_source']
        sample_data['is_confirmed'] = df['is_confirmed']
        
        for feature_name, mapping in feature_mapping.items():
            if mapping[dataset_name] and mapping[dataset_name] in df.columns:
                sample_data[feature_name] = pd.to_numeric(df[mapping[dataset_name]], errors='coerce')
            else:
                sample_data[feature_name] = np.nan
        
        # Enhanced feature engineering
        sample_data = add_enhanced_features(sample_data)
        
        # Fill missing values (only for numeric columns)
        numeric_cols = sample_data.select_dtypes(include=[np.number]).columns
        sample_data[numeric_cols] = sample_data[numeric_cols].fillna(sample_data[numeric_cols].median())
        
        unified_data.append(sample_data)
    
    # Combine all datasets
    combined_df = pd.concat(unified_data, ignore_index=True)
    
    print(f"\nEnhanced dataset shape: {combined_df.shape}")
    print(f"Kepler samples: {(combined_df['data_source'] == 'kepler').sum()}")
    print(f"TOI samples: {(combined_df['data_source'] == 'toi').sum()}")
    print(f"K2 samples: {(combined_df['data_source'] == 'k2').sum()}")
    print(f"Total confirmed: {combined_df['is_confirmed'].sum()}")
    print(f"Confirmation rate: {combined_df['is_confirmed'].mean():.3f}")
    print(f"Total features: {len([col for col in combined_df.columns if col not in ['data_source', 'is_confirmed']])}")
    
    return combined_df

def add_enhanced_features(df):
    """Add enhanced features for better model performance"""
    
    # Basic ratios
    df['period_depth_ratio'] = np.where(
        (df['depth'] > 0) & (df['depth'].notna()),
        df['period'] / df['depth'],
        np.nan
    )
    df['duration_period_ratio'] = np.where(
        (df['period'] > 0) & (df['period'].notna()),
        df['duration'] / df['period'],
        np.nan
    )
    df['radius_teq_ratio'] = np.where(
        (df['teq'] > 0) & (df['teq'].notna()),
        df['radius'] / df['teq'],
        np.nan
    )
    
    # Logarithmic transformations for large values
    df['log_period'] = np.log1p(df['period'])
    df['log_depth'] = np.log1p(df['depth'])
    df['log_radius'] = np.log1p(df['radius'])
    df['log_insol'] = np.log1p(df['insol'])
    
    # Physical relationships
    df['habitable_zone'] = np.where(
        (df['insol'] >= 0.25) & (df['insol'] <= 2.0), 1, 0
    )
    df['hot_jupiter'] = np.where(
        (df['period'] < 10) & (df['radius'] > 6), 1, 0
    )
    df['super_earth'] = np.where(
        (df['radius'] >= 1.25) & (df['radius'] <= 2.0), 1, 0
    )
    
    # Stellar properties ratios
    df['steff_srad_ratio'] = np.where(
        (df['srad'] > 0) & (df['srad'].notna()),
        df['steff'] / df['srad'],
        np.nan
    )
    df['slogg_srad_ratio'] = np.where(
        (df['srad'] > 0) & (df['srad'].notna()),
        df['slogg'] / df['srad'],
        np.nan
    )
    
    # Transit characteristics
    df['transit_efficiency'] = np.where(
        (df['duration'] > 0) & (df['period'] > 0) & (df['duration'].notna()) & (df['period'].notna()),
        df['duration'] / (df['period'] * 24),  # Duration as fraction of orbital period
        np.nan
    )
    
    # Signal strength indicators
    df['signal_strength'] = np.where(
        (df['depth'] > 0) & (df['mag'].notna()),
        df['depth'] / (10 ** (df['mag'] / 2.5)),  # Normalized by stellar brightness
        np.nan
    )
    
    # Temperature zones
    df['hot_zone'] = np.where(df['teq'] > 1000, 1, 0)
    df['warm_zone'] = np.where((df['teq'] >= 300) & (df['teq'] <= 1000), 1, 0)
    df['cold_zone'] = np.where(df['teq'] < 300, 1, 0)
    
    return df

def handle_class_imbalance(X, y):
    """Handle class imbalance using balanced SMOTE + Undersampling and class weights"""
    print("\n=== HANDLING CLASS IMBALANCE ===")
    
    print(f"Original class distribution:")
    print(f"  Class 0: {(y == 0).sum()} ({(y == 0).mean():.3f})")
    print(f"  Class 1: {(y == 1).sum()} ({(y == 1).mean():.3f})")
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    print(f"Class weights: {class_weight_dict}")
    
    # Use SMOTEENN (SMOTE + Edited Nearest Neighbours) for better balance
    # This combines oversampling minority class with undersampling majority class
    smoteenn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smoteenn.fit_resample(X, y)
    
    print(f"After SMOTEENN:")
    print(f"  Class 0: {(y_resampled == 0).sum()} ({(y_resampled == 0).mean():.3f})")
    print(f"  Class 1: {(y_resampled == 1).sum()} ({(y_resampled == 1).mean():.3f})")
    
    return X_resampled, y_resampled, class_weight_dict

def hyperparameter_tuning_xgb(X_train, y_train, class_weight_dict):
    """Hyperparameter tuning for XGBoost with class weights"""
    print("\n=== XGBOOST HYPERPARAMETER TUNING ===")
    
    # Define parameter grid with higher regularization
    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [800, 1200, 1500],
        'subsample': [0.6, 0.7, 0.8],
        'colsample_bytree': [0.6, 0.7, 0.8],
        'reg_alpha': [0.1, 0.5, 1.0],
        'reg_lambda': [0.1, 0.5, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }
    
    # Use RandomizedSearchCV for efficiency
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=class_weight_dict[1]/class_weight_dict[0]  # Apply class weights
    )
    
    random_search = RandomizedSearchCV(
        xgb_model, param_grid, n_iter=50, cv=5, 
        scoring='roc_auc', random_state=42, n_jobs=-1
    )
    
    print("Running hyperparameter tuning...")
    random_search.fit(X_train, y_train)
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best CV score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def train_enhanced_neural_network(X_train, y_train, class_weight_dict):
    """Train enhanced neural network with class weights and higher regularization"""
    print("\n=== TRAINING ENHANCED NEURAL NETWORK ===")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Enhanced neural network with higher regularization and class weights
    nn_model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.01,  # Higher regularization
        batch_size=32,  # Smaller batch size
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=3000,  # More iterations
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    )
    
    # Apply class weights by using sample_weight
    sample_weights = np.array([class_weight_dict[label] for label in y_train])
    nn_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    
    return nn_model, scaler

def train_lightgbm_model(X_train, y_train, class_weight_dict):
    """Train LightGBM model with class weights"""
    print("\n=== TRAINING LIGHTGBM MODEL ===")
    
    # LightGBM parameters with class weights
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'class_weight': 'balanced',
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'min_child_samples': 20,
        'min_child_weight': 0.001,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    
    # Create dataset (ensure numpy arrays)
    train_data = lgb.Dataset(np.array(X_train), label=np.array(y_train))
    
    # Train model
    lgb_model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    return lgb_model

def train_catboost_model(X_train, y_train, class_weight_dict):
    """Train CatBoost model with class weights"""
    print("\n=== TRAINING CATBOOST MODEL ===")
    
    # CatBoost parameters
    cb_model = cb.CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,  # Use only l2_leaf_reg, not reg_lambda
        random_seed=42,
        class_weights=[class_weight_dict[0], class_weight_dict[1]],
        eval_metric='AUC',
        verbose=False,
        early_stopping_rounds=100,
        subsample=0.8,
        colsample_bylevel=0.8
    )
    
    cb_model.fit(X_train, y_train)
    
    return cb_model

def find_optimal_threshold(y_true, y_pred_proba):
    """Find optimal threshold using precision-recall curve"""
    print("\n=== FINDING OPTIMAL THRESHOLD ===")
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Calculate F1 score for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Find threshold with maximum F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"Max F1 score: {f1_scores[optimal_idx]:.4f}")
    
    return optimal_threshold

def train_enhanced_model(combined_df):
    print("\n=== TRAINING ENHANCED UNIFIED MODEL ===")
    
    # Prepare features
    feature_cols = [col for col in combined_df.columns if col not in ['data_source', 'is_confirmed']]
    
    X = combined_df[feature_cols].copy()
    y = combined_df['is_confirmed']
    
    # Handle any remaining NaN values
    X = X.fillna(X.median())
    
    # Check for infinite values and replace them
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    print(f"Total features: {len(feature_cols)}")
    print(f"Feature names: {feature_cols}")
    
    # Handle class imbalance
    X_resampled, y_resampled, class_weight_dict = handle_class_imbalance(X, y)
    
    # Split data with stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train multiple models
    models = {}
    
    # 1. XGBoost with hyperparameter tuning
    xgb_model = hyperparameter_tuning_xgb(X_train, y_train, class_weight_dict)
    models['xgb'] = xgb_model
    
    # 2. Enhanced Neural Network
    nn_model, scaler = train_enhanced_neural_network(X_train, y_train, class_weight_dict)
    models['nn'] = nn_model
    models['scaler'] = scaler
    
    # 3. LightGBM (if available)
    if LIGHTGBM_AVAILABLE:
        lgb_model = train_lightgbm_model(X_train, y_train, class_weight_dict)
        models['lgb'] = lgb_model
    
    # 4. CatBoost (if available)
    if CATBOOST_AVAILABLE:
        cb_model = train_catboost_model(X_train, y_train, class_weight_dict)
        models['catboost'] = cb_model
    
    return models, X_test, y_test, feature_cols, class_weight_dict

def evaluate_enhanced_model(models, X_test, y_test, feature_cols):
    print("\n=== EVALUATING ENHANCED MODEL ===")
    
    predictions = {}
    
    # XGBoost predictions
    xgb_pred_proba = models['xgb'].predict_proba(X_test)[:, 1]
    predictions['xgb'] = xgb_pred_proba
    
    # Neural Network predictions
    X_test_scaled = models['scaler'].transform(X_test)
    nn_pred_proba = models['nn'].predict_proba(X_test_scaled)[:, 1]
    predictions['nn'] = nn_pred_proba
    
    # LightGBM predictions (if available)
    if 'lgb' in models:
        lgb_pred_proba = models['lgb'].predict(np.array(X_test))
        predictions['lgb'] = lgb_pred_proba
    
    # CatBoost predictions (if available)
    if 'catboost' in models:
        cb_pred_proba = models['catboost'].predict_proba(X_test)[:, 1]
        predictions['catboost'] = cb_pred_proba
    
    # Calculate individual model metrics
    model_metrics = {}
    for model_name, pred_proba in predictions.items():
        auc = roc_auc_score(y_test, pred_proba)
        ap = average_precision_score(y_test, pred_proba)
        pred = (pred_proba >= 0.5).astype(int)
        accuracy = (pred == y_test).mean()
        
        model_metrics[model_name] = {
            'auc': auc,
            'ap': ap,
            'accuracy': accuracy
        }
        
        print(f"\n{model_name.upper()} Results:")
        print(f"  AUC: {auc:.4f}")
        print(f"  Average Precision: {ap:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
    
    # Create ensemble predictions
    ensemble_pred_proba = np.mean(list(predictions.values()), axis=0)
    
    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(y_test, ensemble_pred_proba)
    
    # Calculate ensemble metrics with optimal threshold
    ensemble_pred = (ensemble_pred_proba >= optimal_threshold).astype(int)
    ensemble_auc = roc_auc_score(y_test, ensemble_pred_proba)
    ensemble_ap = average_precision_score(y_test, ensemble_pred_proba)
    ensemble_accuracy = (ensemble_pred == y_test).mean()
    
    # Calculate additional metrics
    precision = precision_score(y_test, ensemble_pred)
    recall = recall_score(y_test, ensemble_pred)
    f1 = f1_score(y_test, ensemble_pred)
    
    print(f"\nENSEMBLE Results (Threshold: {optimal_threshold:.4f}):")
    print(f"  AUC: {ensemble_auc:.4f}")
    print(f"  Average Precision: {ensemble_ap:.4f}")
    print(f"  Accuracy: {ensemble_accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # Feature importance (XGBoost)
    print(f"\nTop 10 XGBoost Feature Importances:")
    feature_importance = models['xgb'].feature_importances_
    importance_df = pd.DataFrame({
        'feature': [f"feature_{i}" for i in range(len(feature_importance))],
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return {
        'models': models,
        'feature_cols': [f"feature_{i}" for i in range(len(feature_importance))],
        'optimal_threshold': optimal_threshold,
        'metrics': {
            'ensemble_auc': ensemble_auc,
            'ensemble_ap': ensemble_ap,
            'ensemble_accuracy': ensemble_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'individual_metrics': model_metrics
        }
    }

def save_enhanced_model(model_data):
    print("\n=== SAVING ENHANCED MODEL ===")
    
    models_dir = "enhanced_model"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Save all models
    models = model_data['models']
    for model_name, model in models.items():
        if model_name != 'scaler':  # Handle scaler separately
            joblib.dump(model, f"{models_dir}/enhanced_{model_name}_model.pkl")
        else:
            joblib.dump(model, f"{models_dir}/enhanced_scaler.pkl")
    
    # Save metadata
    metadata = {
        'feature_cols': model_data['feature_cols'],
        'optimal_threshold': model_data['optimal_threshold'],
        'metrics': model_data['metrics'],
        'model_info': {
            'description': 'Enhanced unified model with multiple algorithms, class weights, and threshold optimization',
            'datasets': ['kepler', 'toi', 'k2'],
            'features': len(model_data['feature_cols']),
            'models': list(models.keys()),
            'enhancements': [
                'Class weights implementation',
                'Threshold calibration',
                'SMOTEENN balanced sampling',
                'Stratified cross-validation',
                'Higher regularization',
                'LightGBM and CatBoost integration',
                'Multi-model ensemble',
                'Optimal threshold finding'
            ],
            'created': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    joblib.dump(metadata, f"{models_dir}/enhanced_metadata.pkl")
    
    print(f"Enhanced model saved to '{models_dir}' directory!")
    print(f"Model performance: AUC = {model_data['metrics']['ensemble_auc']:.4f}")
    print(f"Optimal threshold: {model_data['optimal_threshold']:.4f}")
    print(f"F1 Score: {model_data['metrics']['f1']:.4f}")

def main():
    print("=== ENHANCED UNIFIED EXOPLANET DETECTION MODEL ===")
    print("Training on Kepler + TOI + K2 data with advanced enhancements")
    
    # Load all datasets
    kepler_df, toi_df, k2_df = load_all_datasets()
    
    # Create enhanced features
    combined_df = create_enhanced_features(kepler_df, toi_df, k2_df)
    
    # Train enhanced model
    models, X_test, y_test, feature_cols, class_weight_dict = train_enhanced_model(combined_df)
    
    # Evaluate model
    model_data = evaluate_enhanced_model(models, X_test, y_test, feature_cols)
    
    # Save model
    save_enhanced_model(model_data)
    
    print("\n=== ENHANCED TRAINING COMPLETE ===")
    print("You now have an ENHANCED unified model with:")
    print("✓ Class weights implementation")
    print("✓ Threshold calibration")
    print("✓ SMOTEENN balanced sampling")
    print("✓ Stratified cross-validation")
    print("✓ Higher regularization")
    print("✓ LightGBM and CatBoost integration")
    print("✓ Multi-model ensemble")
    print("✓ Optimal threshold finding")

if __name__ == "__main__":
    main()
