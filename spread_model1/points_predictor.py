import pickle
import pandas as pd

with open("spread_model1/home_team_stats_2020_to_2025.pkl", "rb") as f:
    home_stats_dict = pickle.load(f)
with open("spread_model1/away_team_stats_2020_to_2025.pkl", "rb") as f:
    away_stats_dict = pickle.load(f)
with open("spread_model1/home_points_model.pkl", "rb") as f:
    model_home = pickle.load(f)
with open("spread_model1/away_points_model.pkl", "rb") as f:
    model_away = pickle.load(f)

#print(away_stats_dict["DEN"].columns)
# Example dictionaries
# home_team_stats_df = {"ARI": pd.DataFrame(...), ...}
# away_team_stats_df = {"ARI": pd.DataFrame(...), ...}

# Load schedule CSV
schedule_df = pd.read_csv("nfl_schedule_formatted.csv")

# Week to process
week_num = 1
week_df = schedule_df[schedule_df['week #'] == week_num]
print(week_df)
#print(home_stats_dict.keys())

# Lookback configuration
lookback_start = 4  # 0 = last 10 games, 1 = 11th-to-last 20th, etc.
lookback_length = 10  # number of games to average

home_rows = []
away_rows = []

home_points_features = [
    "points_per_drive_home",
    "total_points_home",
    "yards_per_play_home",
    "epa_home",
    "redzone_drive_count_home",
    "total_tds_home",
    "total_yards_home",
]
home_away_featurers =  [
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
]
away_away_features = [
    "epa_away_away_diff",
    "total_points_away_away_diff",
    "points_per_drive_away_away_diff",
    "total_yards_away_away_diff",
]
away_home_features = [
    "total_points_home_away_off_diff",
    "points_per_drive_home_away_off_diff",
    "epa_home_away_off_diff"
]

for idx, game in week_df.iterrows():
    home_team = game['home_team']
    away_team = game['away_team']

    home_stats = home_stats_dict[home_team]
    away_stats = away_stats_dict[away_team]

    # Determine indices for slicing
    home_slice = home_stats.iloc[-(lookback_start + lookback_length):-lookback_start if lookback_start != 0 else None]
    away_slice = away_stats.iloc[-(lookback_start + lookback_length):-lookback_start if lookback_start != 0 else None]

    # Compute average stats
    home_avg = home_slice.mean(numeric_only=True)
    away_avg = away_slice.mean(numeric_only=True)
    home_avg = home_avg.drop(["home", "win"])
    away_avg = away_avg.drop(["home", "win"])
    # Subtract away from home

    print(home_avg.index)

    home_stats = pd.Series([home_avg[col[:-5]] for col in home_points_features], index=home_points_features)
    home_away_stats = pd.Series([home_avg[col[:-19]] - away_avg[col[:-19]] for col in home_away_featurers], index=home_away_featurers)
    feature_row_home = pd.concat([home_stats, home_away_stats])

    away_stats = pd.Series([away_avg[col[:-5]] for col in away_points_features], index=away_points_features)
    away_away_stats = pd.Series([away_avg[col[:-15]] - away_avg[f"{col[:-15]}_allowed"] for col in away_away_features], index=away_away_features)
    away_home_stats = pd.Series([away_avg[col[:-19]] for col in away_home_features], index=away_home_features)
    feature_row_away = pd.concat([away_stats, away_away_stats, away_home_stats])

    # feature_row.index = [f"{col}_home_away_off_diff" for col in feature_row.index]
    # if "sacks_allowed_home_away_off_diff" in feature_row:
    #     feature_row["sacks_allowed_home_away_def_diff"] = feature_row.pop("sacks_allowed_home_away_off_diff")
    # home_avg.index = [f"{col}_home" for col in home_avg.index]
    # away_avg.index = [f"{col}_away" for col in away_avg.index]

    #feature_row = pd.concat([home_avg, away_avg, feature_row])
    # Add meta info
    feature_row_home['week #'] = week_num
    feature_row_home['home_team'] = home_team
    feature_row_home['away_team'] = away_team

    home_rows.append(feature_row_home)
    away_rows.append(feature_row_away)

# Combine into DataFrame
features_df_home = pd.DataFrame(home_rows)
features_df_away = pd.DataFrame(away_rows)

# Save CSV
# features_df_home.to_csv(f"week_{week_num}_features.csv", index=False)
# print(f"Week {week_num} features saved as week_{week_num}_features.csv")

X_all_home = features_df_home.drop(columns=["home_diff", "win_diff", "week #" ,"home_team","away_team"], errors='ignore')
X_all_away = features_df_away.drop(columns=["home_diff", "win_diff", "week #" ,"home_team","away_team"], errors='ignore')

#print(X_all.columns)

# Predict probabilities for all games
#home_win_probs = model_home.predict_proba(X_all)[:, 1]
home_score_preds = model_home.predict(X_all_home).round().astype(int)
away_score_preds = model_away.predict(X_all_away).round().astype(int)

results = pd.DataFrame({
    "home_team": features_df_home["home_team"],
    "away_team": features_df_home["away_team"],
    "pred_home_score": home_score_preds,
    "pred_away_score": away_score_preds,
})
#features_df_home["pred_home_score"] = home_score_preds
#features_df_home["pred_away_score"] = away_score_preds
#features_df_home["home_win_prob"] = home_win_probs

print(results)