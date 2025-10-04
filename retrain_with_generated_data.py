#!/usr/bin/env python3
"""
XGBoost Retraining with Generated Data + SMOTE
Optimized for better precision/recall balance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("XGBOOST RETRAINING WITH GENERATED DATA + SMOTE")
print("=" * 80)

# Load all available data
print("\nLoading data from data/ folder...")

datasets = []
dataset_names = []

# Try to load all available datasets
try:
    real_combined = pd.read_csv('data/combined_real_data_with_features.csv')
    datasets.append(real_combined)
    dataset_names.append(f"Real Combined ({len(real_combined)})")
    print(f"[OK] Loaded combined_real_data_with_features.csv: {len(real_combined)} rows")
except:
    print("[SKIP] combined_real_data_with_features.csv not found")

try:
    synthetic_combined = pd.read_csv('data/combined_synthetic_data.csv')
    datasets.append(synthetic_combined)
    dataset_names.append(f"Synthetic Combined ({len(synthetic_combined)})")
    print(f"[OK] Loaded combined_synthetic_data.csv: {len(synthetic_combined)} rows")
except:
    print("[SKIP] combined_synthetic_data.csv not found")

try:
    real_tahweel = pd.read_csv('data/real_tahweel_with_features.csv')
    datasets.append(real_tahweel)
    dataset_names.append(f"Real Tahweel ({len(real_tahweel)})")
    print(f"[OK] Loaded real_tahweel_with_features.csv: {len(real_tahweel)} rows")
except:
    print("[SKIP] real_tahweel_with_features.csv not found")

try:
    real_virement = pd.read_csv('data/real_virement_with_features.csv')
    datasets.append(real_virement)
    dataset_names.append(f"Real Virement ({len(real_virement)})")
    print(f"[OK] Loaded real_virement_with_features.csv: {len(real_virement)} rows")
except:
    print("[SKIP] real_virement_with_features.csv not found")

try:
    synthetic_tahweel = pd.read_csv('data/synthetic_tahweel_generated.csv')
    datasets.append(synthetic_tahweel)
    dataset_names.append(f"Synthetic Tahweel ({len(synthetic_tahweel)})")
    print(f"[OK] Loaded synthetic_tahweel_generated.csv: {len(synthetic_tahweel)} rows")
except:
    print("[SKIP] synthetic_tahweel_generated.csv not found")

try:
    synthetic_virement = pd.read_csv('data/synthetic_virement_generated.csv')
    datasets.append(synthetic_virement)
    dataset_names.append(f"Synthetic Virement ({len(synthetic_virement)})")
    print(f"[OK] Loaded synthetic_virement_generated.csv: {len(synthetic_virement)} rows")
except:
    print("[SKIP] synthetic_virement_generated.csv not found")

if not datasets:
    print("[ERROR] No data files found! Please check data/ folder.")
    exit(1)

print(f"\nTotal datasets loaded: {len(datasets)}")
for name in dataset_names:
    print(f"  - {name}")

# Combine all datasets
print("\nCombining all datasets...")
combined_data = pd.concat(datasets, ignore_index=True)
print(f"[OK] Total combined data: {len(combined_data)} transactions")

# Define the 25 required features
required_features = [
    'montant', 'amount_log', 'hour', 'day_of_week', 'day_of_month', 'month',
    'is_weekend', 'is_night', 'is_business_hours', 'amount_zscore',
    'is_large_transaction', 'is_very_large_transaction',
    'user_montant_count', 'user_montant_sum', 'user_montant_mean', 'user_montant_std',
    'amount_vs_user_avg', 'temporal_risk_score', 'amount_risk_score',
    'behavioral_risk_score', 'total_risk_score',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
]

# Check and create missing features
print("\nChecking features...")

# Create user behavioral features if missing
if 'user_montant_count' not in combined_data.columns:
    print("[INFO] Creating user behavioral features...")
    
    # Ensure required base columns exist
    if 'emetteur' in combined_data.columns and 'montant' in combined_data.columns:
        user_agg = combined_data.groupby('emetteur').agg({
            'montant': ['count', 'sum', 'mean', 'std']
        }).round(2)
        
        user_agg.columns = [f'user_{col[0]}_{col[1]}' for col in user_agg.columns]
        user_agg = user_agg.reset_index()
        
        combined_data = combined_data.merge(user_agg, on='emetteur', how='left')
        
        # Fill NaN
        for col in ['user_montant_count', 'user_montant_sum', 'user_montant_mean', 'user_montant_std']:
            if col in combined_data.columns:
                combined_data[col] = combined_data[col].fillna(combined_data[col].median())
    else:
        # Fallback values
        combined_data['user_montant_count'] = 1
        combined_data['user_montant_sum'] = combined_data.get('montant', 0)
        combined_data['user_montant_mean'] = combined_data.get('montant', 0)
        combined_data['user_montant_std'] = 0

# Create cyclical features if missing
if 'hour_sin' not in combined_data.columns and 'hour' in combined_data.columns:
    print("[INFO] Creating cyclical temporal features...")
    combined_data['hour_sin'] = np.sin(2 * np.pi * combined_data['hour'] / 24)
    combined_data['hour_cos'] = np.cos(2 * np.pi * combined_data['hour'] / 24)

if 'day_sin' not in combined_data.columns and 'day_of_week' in combined_data.columns:
    combined_data['day_sin'] = np.sin(2 * np.pi * combined_data['day_of_week'] / 7)
    combined_data['day_cos'] = np.cos(2 * np.pi * combined_data['day_of_week'] / 7)

# Verify all features now exist
missing_features = [f for f in required_features if f not in combined_data.columns]
if missing_features:
    print(f"[ERROR] Still missing features: {missing_features}")
    print("Available columns:", combined_data.columns.tolist())
    exit(1)

print("[OK] All required features are present")

# Get fraud labels
if 'fraud' in combined_data.columns:
    fraud_col = 'fraud'
elif 'is_fraud' in combined_data.columns:
    fraud_col = 'is_fraud'
elif 'fraud_prediction' in combined_data.columns:
    fraud_col = 'fraud_prediction'
else:
    print("\n[ERROR] No fraud label column found!")
    print("Available columns:", combined_data.columns.tolist())
    exit(1)

print(f"\n[OK] Using fraud label column: '{fraud_col}'")

# Prepare data
X = combined_data[required_features].fillna(0)
y = combined_data[fraud_col].fillna(0).astype(int)

fraud_rate = y.mean()
print(f"\nDataset Statistics:")
print(f"  Total transactions: {len(X):,}")
print(f"  Fraud cases: {y.sum():,}")
print(f"  Fraud rate: {fraud_rate:.2%}")
print(f"  Non-fraud cases: {(y == 0).sum():,}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain/Test Split:")
print(f"  Training: {len(X_train):,} ({y_train.mean():.2%} fraud)")
print(f"  Testing: {len(X_test):,} ({y_test.mean():.2%} fraud)")

# Apply SMOTE
print("\nApplying SMOTE for class balancing...")

smote = SMOTE(
    random_state=42,
    k_neighbors=min(5, (y_train == 1).sum() - 1),  # Adaptive k_neighbors
    sampling_strategy='auto'
)

X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"[OK] Balanced training set:")
print(f"  Total: {len(X_train_balanced):,}")
print(f"  Fraud rate: {y_train_balanced.mean():.2%}")
print(f"  Fraud cases: {(y_train_balanced == 1).sum():,}")
print(f"  Non-fraud cases: {(y_train_balanced == 0).sum():,}")

# Optimized XGBoost Training
print("\nTraining Optimized XGBoost Model...")
print("Testing multiple hyperparameter combinations...")

# Calculate scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Scale pos weight: {scale_pos_weight:.2f}")

# Hyperparameter grid
param_grid = {
    'n_estimators': [300, 400, 500],
    'max_depth': [6, 7, 8],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
    'min_child_weight': [1, 3],
    'gamma': [0, 0.1]
}

xgb_model = xgb.XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight,
    tree_method='hist',
    enable_categorical=False
)

# Grid search with F1 optimization
grid_search = GridSearchCV(
    xgb_model,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

print("\n[RUNNING] Training in progress (this will take 5-15 minutes)...")
print("Testing 144 different parameter combinations with 5-fold CV...")

grid_search.fit(X_train_balanced, y_train_balanced)

print("\n" + "=" * 80)
print("[SUCCESS] TRAINING COMPLETED!")
print("=" * 80)

best_model = grid_search.best_estimator_

print(f"\nBest Parameters Found:")
for param, value in grid_search.best_params_.items():
    print(f"  - {param}: {value}")

print(f"\nBest CV F1-Score: {grid_search.best_score_:.4f}")

# Evaluate on test set
print("\n" + "=" * 80)
print("EVALUATION ON TEST SET")
print("=" * 80)

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Detailed metrics
auc = roc_auc_score(y_test, y_pred_proba)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nPerformance Metrics:")
print(f"  AUC Score:  {auc:.4f}")
print(f"  Precision:  {precision:.2%}")
print(f"  Recall:     {recall:.2%}")
print(f"  F1-Score:   {f1:.2%}")

print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {tn:,}")
print(f"  False Positives: {fp:,}")
print(f"  False Negatives: {fn:,}")
print(f"  True Positives:  {tp:,}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': required_features,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

# Save model
print("\nSaving optimized model...")

joblib.dump(best_model, 'best_xgboost_model_retrained.pkl')
joblib.dump(required_features, 'xgboost_features_retrained.pkl')

print("\n[OK] Model saved successfully!")
print("\nFiles created:")
print("  - best_xgboost_model_retrained.pkl")
print("  - xgboost_features_retrained.pkl")

# Performance summary
print("\n" + "=" * 80)
print("RETRAINING SUMMARY")
print("=" * 80)

print(f"\nData Used:")
for name in dataset_names:
    print(f"  - {name}")
print(f"  - Total: {len(combined_data):,} transactions")

print(f"\nFinal Model Performance:")
print(f"  [OK] AUC Score:  {auc:.4f}")
print(f"  [OK] Precision:  {precision:.2%}")
print(f"  [OK] Recall:     {recall:.2%}")
print(f"  [OK] F1-Score:   {f1:.2%}")

print(f"\nImprovement Expected:")
if precision > 0.75 and recall > 0.75:
    print("  [EXCELLENT] Both precision and recall >75%")
    print("  [OK] Ready for production use")
    print("  [OK] Can use 'Pure ML' mode in dashboard")
elif precision > 0.70 and recall > 0.70:
    print("  [GOOD] Both metrics >70%")
    print("  [OK] Use 'Enhanced ML' mode with thresholds 55%/30%")
elif f1 > 0.65:
    print("  [FAIR] F1-Score >65%")
    print("  [OK] Use 'Hybrid' mode with thresholds 45%/25%")
else:
    print("  [WARNING] Consider adding more features or data")

print(f"\nNext Steps:")
print("1. Backup current model:")
print("   copy best_xgboost_model.pkl best_xgboost_model_backup.pkl")
print("\n2. Replace with new model:")
print("   move /Y best_xgboost_model_retrained.pkl best_xgboost_model.pkl")
print("   move /Y xgboost_features_retrained.pkl xgboost_features.pkl")
print("\n3. Refresh dashboard and test")
print(f"   Recommended mode: {'Pure ML' if precision > 0.75 and recall > 0.75 else 'Enhanced ML'}")
print(f"   Recommended thresholds: {'65%/40%' if precision > 0.75 else '50%/30%'}")

print("\n" + "=" * 80)
print("[SUCCESS] RETRAINING COMPLETE!")
print("=" * 80)

