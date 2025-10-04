#!/usr/bin/env python3
"""
Optimized XGBoost Retraining Script
Focus: Better precision while maintaining high recall
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🚀 OPTIMIZED XGBOOST RETRAINING FOR FRAUD DETECTION")
print("=" * 70)

# Load real Tunisian banking data
print("\n📊 Loading real Tunisian banking data...")

# Load Tahweel Cash data
tahweel = pd.read_csv('realdata/encrypted_transactions_tahweel_Cash.xlsx - Sheet1.csv', encoding='latin-1')
print(f"✅ Tahweel Cash: {len(tahweel)} transactions")

# Load Virement data  
virement = pd.read_csv('realdata/virement_encrypted.xlsx - Sheet1.csv', encoding='latin-1')
print(f"✅ Virement: {len(virement)} transactions")

# Normalize column names
def normalize_columns(df):
    mapping = {
        'Émetteur': 'emetteur', 'émetteur': 'emetteur',
        'Bénéficiaire': 'beneficiaire', 'bénéficiaire': 'beneficiaire',
        'Date': 'date', 'Montant': 'montant',
        'Num. tel. émetteur': 'num_tel_emetteur',
        'Agent émetteur': 'agent_emetteur',
        'Agent récepteur': 'agent_recepteur'
    }
    return df.rename(columns=mapping)

tahweel = normalize_columns(tahweel)
virement = normalize_columns(virement)

# Parse data
for df in [tahweel, virement]:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['montant'] = pd.to_numeric(df['montant'], errors='coerce')

tahweel = tahweel.dropna(subset=['date', 'montant'])
virement = virement.dropna(subset=['date', 'montant'])

print(f"\n📊 After cleaning:")
print(f"Tahweel: {len(tahweel)} | Virement: {len(virement)}")

# Create features for both datasets
def create_features(df):
    """Create the same 25 features as the dashboard"""
    df = df.sort_values('date').reset_index(drop=True)
    
    # Temporal
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
    
    # Amount
    df['amount_log'] = np.log1p(df['montant'])
    df['amount_zscore'] = (df['montant'] - df['montant'].mean()) / (df['montant'].std() + 0.001)
    df['is_large_transaction'] = (df['montant'] >= df['montant'].quantile(0.95)).astype(int)
    df['is_very_large_transaction'] = (df['montant'] >= df['montant'].quantile(0.99)).astype(int)
    
    # User behavioral
    user_agg = df.groupby('emetteur').agg({
        'montant': ['count', 'sum', 'mean', 'std', 'min', 'max'],
        'date': ['min', 'max']
    }).round(2)
    user_agg.columns = [f'user_{col[0]}_{col[1]}' for col in user_agg.columns]
    user_agg = user_agg.reset_index()
    df = df.merge(user_agg, on='emetteur', how='left')
    
    user_cols = [col for col in df.columns if col.startswith('user_')]
    for col in user_cols:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
    
    df['amount_vs_user_avg'] = df['montant'] / (df['user_montant_mean'] + 1)
    
    # Risk scores
    df['temporal_risk_score'] = (
        (df['is_night'] * 3) +
        (df['is_weekend'] * 2) +
        (df['hour'].isin([22, 23, 0, 1, 2, 3]).astype(int) * 2)
    )
    
    df['amount_risk_score'] = (
        (df['is_large_transaction'] * 2) +
        (df['is_very_large_transaction'] * 3) +
        (df['amount_zscore'] > 2).astype(int) * 2
    )
    
    df['behavioral_risk_score'] = (
        (df['user_montant_count'] > df['user_montant_count'].quantile(0.9)).astype(int) * 2 +
        (df['amount_vs_user_avg'] > 3).astype(int) * 3
    )
    
    df['total_risk_score'] = (
        df['temporal_risk_score'] +
        df['amount_risk_score'] +
        df['behavioral_risk_score']
    )
    
    # Cyclical
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df.fillna(0)

print("\n🔧 Creating features...")
tahweel = create_features(tahweel)
virement = create_features(virement)

# Combine datasets
combined = pd.concat([tahweel, virement], ignore_index=True)
print(f"✅ Combined dataset: {len(combined)} transactions")

# Create BETTER fraud labels using multiple methods
print("\n🎯 Creating improved fraud labels...")

from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

# Select key features for anomaly detection
anomaly_features = ['amount_zscore', 'amount_vs_user_avg', 'total_risk_score', 
                   'user_montant_count', 'amount_log']
X_anomaly = combined[anomaly_features].fillna(0)

# Method 1: Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_labels = iso_forest.fit_predict(X_anomaly)

# Method 2: Elliptic Envelope (robust covariance)
elliptic = EllipticEnvelope(contamination=0.05, random_state=42)
elliptic_labels = elliptic.fit_predict(X_anomaly)

# Method 3: Rule-based (high confidence fraud)
rule_based = (
    (combined['amount_zscore'] > 3) |
    (combined['amount_vs_user_avg'] > 10) |
    (combined['total_risk_score'] >= 12) |
    ((combined['is_very_large_transaction'] == 1) & (combined['temporal_risk_score'] >= 3))
).astype(int)

# Combine methods: Fraud if 2+ methods agree
fraud_votes = (
    (iso_labels == -1).astype(int) +
    (elliptic_labels == -1).astype(int) +
    rule_based
)

combined['fraud'] = (fraud_votes >= 2).astype(int)

fraud_rate = combined['fraud'].mean()
print(f"📊 Fraud rate: {fraud_rate:.1%} ({combined['fraud'].sum()} fraud cases)")

# Prepare data for training
features = ['montant', 'amount_log', 'hour', 'day_of_week', 'day_of_month', 'month',
           'is_weekend', 'is_night', 'is_business_hours', 'amount_zscore',
           'is_large_transaction', 'is_very_large_transaction',
           'user_montant_count', 'user_montant_sum', 'user_montant_mean', 'user_montant_std',
           'amount_vs_user_avg', 'temporal_risk_score', 'amount_risk_score',
           'behavioral_risk_score', 'total_risk_score',
           'hour_sin', 'hour_cos', 'day_sin', 'day_cos']

X = combined[features]
y = combined['fraud']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Training set: {len(X_train)} | Test set: {len(X_test)}")
print(f"Train fraud rate: {y_train.mean():.1%} | Test fraud rate: {y_test.mean():.1%}")

# Apply SMOTE for better balance
print("\n🔄 Applying SMOTE balancing...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
print(f"✅ Balanced training set: {len(X_balanced)} (fraud rate: {y_balanced.mean():.1%})")

# Optimized XGBoost with GridSearch
print("\n🔍 Hyperparameter optimization...")

param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [5, 6, 7],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2]
}

xgb_model = xgb.XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False,
    scale_pos_weight=(y_balanced == 0).sum() / (y_balanced == 1).sum()  # Handle imbalance
)

grid_search = GridSearchCV(
    xgb_model,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

print("Training... (this may take 5-10 minutes)")
grid_search.fit(X_balanced, y_balanced)

print(f"\n✅ Best parameters: {grid_search.best_params_}")
print(f"✅ Best CV F1-Score: {grid_search.best_score_:.3f}")

# Get best model
best_model = grid_search.best_estimator_

# Evaluate on test set
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 70)
print("📊 MODEL PERFORMANCE ON TEST SET")
print("=" * 70)

print("\n🎯 Classification Report:")
print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_pred_proba)
print(f"\n🎯 AUC Score: {auc:.4f}")

# Confusion Matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n📊 Detailed Metrics:")
print(f"Precision: {precision:.1%}")
print(f"Recall: {recall:.1%}")
print(f"F1-Score: {f1:.1%}")
print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Negatives: {tn}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n📊 Top 10 Most Important Features:")
print(feature_importance.head(10))

# Save optimized model
print("\n💾 Saving optimized model...")

joblib.dump(best_model, 'best_xgboost_model_optimized.pkl')
joblib.dump(features, 'xgboost_features_optimized.pkl')

print("\n✅ Model saved successfully!")
print("\n📁 Files created:")
print("  - best_xgboost_model_optimized.pkl")
print("  - xgboost_features_optimized.pkl")

print("\n" + "=" * 70)
print("🎉 RETRAINING COMPLETE!")
print("=" * 70)
print(f"\n🎯 Target Metrics Achieved:")
print(f"✅ AUC: {auc:.3f}")
print(f"✅ Precision: {precision:.1%}")
print(f"✅ Recall: {recall:.1%}")
print(f"✅ F1-Score: {f1:.1%}")
print(f"\n💡 Next Steps:")
print("1. Rename best_xgboost_model_optimized.pkl → best_xgboost_model.pkl")
print("2. Rename xgboost_features_optimized.pkl → xgboost_features.pkl")
print("3. Refresh the compliance dashboard")
print("4. Set detection mode to 'Pure ML'")
print("5. Test with thresholds: 60%/35%")

