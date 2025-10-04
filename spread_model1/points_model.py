import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# --- Prepare data ---
with open("spread_model1/game_features_2020_to_2025.pkl", "rb") as f:
    feature_df = pickle.load(f)
print(feature_df.isna().any)
print(feature_df.columns)
print(feature_df.dtypes)
X = feature_df.drop(columns=["game_id", "home_win", "spread_result", "total_result"])
home_points_features = [
    "points_per_drive_home",
    "total_points_home",
    "yards_per_play_home",
    "epa_home",
    "redzone_drive_count_home",
    "total_tds_home",
    "total_yards_home",
    "points_per_drive_home_away_off_diff",
    "total_points_home_away_off_diff",
    "epa_home_away_off_diff",
    "total_tds_home_away_off_diff",
    "yards_per_play_home_away_off_diff",
    "total_yards_home_away_off_diff",
    "redzone_drive_count_home_away_off_diff",
    "redzone_touchdowns_home_away_off_diff"
]

away_points_features = [
    "epa_away",
    "points_per_drive_away",
    "total_points_away",
    "total_yards_away",
    "yards_per_play_away",
    "total_tds_away",
    "win_prob_added_away",
    "passing_yards_per_attempt_away",
    "redzone_drive_count_away",
    "redzone_touchdowns_away",
    "epa_away_away_diff",
    "total_points_away_away_diff",
    "points_per_drive_away_away_diff",
    "total_yards_away_away_diff",
    "total_points_home_away_off_diff",
    "points_per_drive_home_away_off_diff",
    "epa_home_away_off_diff"
]


X_home =X[home_points_features]
X_away = X[away_points_features]
y_home = feature_df["home_points"]
y_away = feature_df["away_points"]

# train/validation split
X_train_home, X_test_home, y_train_home, y_test_home = train_test_split(
    X_home, y_home, test_size=0.2, random_state=42
)
X_train_away, X_test_away, y_train_away, y_test_away = train_test_split(
    X_away, y_away, test_size=0.2, random_state=42
)

# --- Define models ---
rf_home = RandomForestRegressor(n_estimators=300, random_state=42)
xgb_home = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
gb_home = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5)

rf_away = RandomForestRegressor(n_estimators=300, random_state=42)
xgb_away = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
gb_away = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5)

# --- Ensemble (Voting Regressor) ---
ensemble_home = VotingRegressor(
    estimators=[("rf", rf_home), ("xgb", xgb_home), ("gb", gb_home)]
)
ensemble_away = VotingRegressor(
    estimators=[("rf", rf_away), ("xgb", xgb_away), ("gb", gb_away)]
)

# --- Train ---
ensemble_home.fit(X_train_home, y_train_home)
ensemble_away.fit(X_train_away, y_train_away)

# --- Evaluate ---
y_pred_home = ensemble_home.predict(X_test_home)
y_pred_away = ensemble_away.predict(X_test_away)
#y_proba = ensemble_home.predict_proba(X_test_home)[:, 1]

print("MAE Home:", mean_absolute_error(y_test_home, y_pred_home))
print("MSE Home:", mean_squared_error(y_test_home, y_pred_home))
print("R² Score Home:", r2_score(y_test_home, y_pred_home))
print("\n")
print("MAE Away:", mean_absolute_error(y_test_away, y_pred_away))
print("MSE Away:", mean_squared_error(y_test_away, y_pred_away))
print("R² Score Away:", r2_score(y_test_away, y_pred_away))

with open("spread_model1/home_points_model.pkl", "wb") as f:
    pickle.dump(ensemble_home, f)
print("Model saved as home_win_model.pkl")
with open("spread_model1/away_points_model.pkl", "wb") as f:
    pickle.dump(ensemble_away, f)
print("Model saved as away_points_model.pkl")


rf_importances = ensemble_home.estimators_[0].feature_importances_
xgb_importances = ensemble_home.estimators_[1].feature_importances_
gb_importances = ensemble_home.estimators_[2].feature_importances_

importance_df = pd.DataFrame({
    "feature": X_train_home.columns,
    "RandomForest": rf_importances,
    "XGBoost": xgb_importances,
    "GradientBoosting": gb_importances
})
print(importance_df)