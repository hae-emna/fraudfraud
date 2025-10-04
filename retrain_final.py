#!/usr/bin/env python3
"""
Final XGBoost Retraining - Works with Your Actual Data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("XGBOOST RETRAINING WITH YOUR DATA + SMOTE")
print("="*80)

# Load data
print("\nLoading data...")
df = pd.read_csv('data/combined_real_data_with_features.csv', low_memory=False)
print(f"Loaded: {len(df)} transactions")

# Map your feature names to dashboard feature names
print("\nMapping features...")

# Create mappings
if 'user_count' in df.columns and 'user_montant_count' not in df.columns:
    df['user_montant_count'] = df['user_count']

if 'user_avg_amount' in df.columns and 'user_montant_mean' not in df.columns:
    df['user_montant_mean'] = df['user_avg_amount']
    df['user_montant_sum'] = df['user_avg_amount'] * df.get('user_count', 1)
    df['user_montant_std'] = df.get('user_std_amount', 0)

# Create cyclical features
if 'hour' in df.columns:
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

if 'day_of_week' in df.columns:
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Final feature list (dashboard features)
features = [
    'montant', 'amount_log', 'hour', 'day_of_week', 'day_of_month', 'month',
    'is_weekend', 'is_night', 'is_business_hours', 'amount_zscore',
    'is_large_transaction', 'is_very_large_transaction',
    'user_montant_count', 'user_montant_sum', 'user_montant_mean', 'user_montant_std',
    'amount_vs_user_avg', 'temporal_risk_score', 'amount_risk_score',
    'behavioral_risk_score', 'total_risk_score',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
]

# Verify features exist
for f in features:
    if f not in df.columns:
        print(f"WARNING: Missing {f}, filling with 0")
        df[f] = 0

X = df[features].fillna(0)
y = df['fraud_prediction'].fillna(0).astype(int)

print(f"Features: {len(features)}")
print(f"Fraud rate: {y.mean():.1%} ({y.sum()} fraud)")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# SMOTE
print("Applying SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_sm, y_sm = smote.fit_resample(X_train, y_train)
print(f"Balanced: {len(X_sm)} samples ({y_sm.mean():.0%} fraud)")

# Train XGBoost with best parameters
print("\nTraining XGBoost (this takes 2-5 minutes)...")

model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=3,
    gamma=0.1,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

model.fit(X_sm, y_sm)

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_proba)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

prec = tp / (tp + fp) if (tp + fp) > 0 else 0
rec = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"\nAUC:       {auc:.4f}")
print(f"Precision: {prec:.1%}")
print(f"Recall:    {rec:.1%}")
print(f"F1-Score:  {f1:.1%}")
print(f"\nTP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")

# Feature importance
fi = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
fi = fi.sort_values('Importance', ascending=False)
print("\nTop 10 Features:")
print(fi.head(10).to_string(index=False))

# Save
print("\nSaving...")
joblib.dump(model, 'best_xgboost_model.pkl')
joblib.dump(features, 'xgboost_features.pkl')

print("\n[SUCCESS] COMPLETE!")
print("\nRefresh dashboard - set to 'Pure ML' mode with 60%/35% thresholds")

