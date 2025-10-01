import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import pickle

# --- Prepare data ---
with open("second_model/game_features_2020_to_2025.pkl", "rb") as f:
    feature_df = pickle.load(f)
print(feature_df.isna().any)
print(feature_df.columns)
print(feature_df.dtypes)
X = feature_df.drop(columns=["game_id", "home_win"])
y = feature_df["home_win"]

# train/validation split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Define models ---
rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)
gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5)

# --- Ensemble (soft voting) ---
ensemble = VotingClassifier(
    estimators=[("rf", rf), ("xgb", xgb), ("gb", gb)],
    voting="soft"  # uses predicted probabilities
)

# --- Train ---
ensemble.fit(X_train, y_train)

# --- Evaluate ---
y_pred = ensemble.predict(X_test)
y_proba = ensemble.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

with open("second_model/home_win_model.pkl", "wb") as f:
    pickle.dump(ensemble, f)
print("Model saved as home_win_model.pkl")


rf_importances = ensemble.estimators_[0].feature_importances_
xgb_importances = ensemble.estimators_[1].feature_importances_
gb_importances = ensemble.estimators_[2].feature_importances_

importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "RandomForest": rf_importances,
    "XGBoost": xgb_importances,
    "GradientBoosting": gb_importances
})
print(importance_df)