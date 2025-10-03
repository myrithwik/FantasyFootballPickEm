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
week_num = 5
week_df = schedule_df[schedule_df['week #'] == week_num]
print(week_df)
#print(home_stats_dict.keys())

# Lookback configuration
lookback_start = 0  # 0 = last 10 games, 1 = 11th-to-last 20th, etc.
lookback_length = 10  # number of games to average

rows = []
home_points_features = [
    "points_per_drive_home_away_off_diff",
    "total_points_home_away_off_diff",
    "points_per_drive_home",
    "total_points_home",
    "qb_epa_home_away_off_diff",
    "yards_per_play_home",
    "epa_home_away_off_diff",
    "total_tds_home_away_off_diff",
    "qb_epa_home",
    "total_tds_home",
    "total_yards_home",
    "yards_per_play_home_away_off_diff",
    "total_yards_home_away_off_diff",
    "epa_home",
    "redzone_drive_count_home_away_off_diff",
    "redzone_touchdowns_home_away_off_diff",
    "redzone_drive_count_home"
]

away_points_features = [
    "epa_away",
    "qb_epa_away",
    "points_per_drive_away",
    "total_points_away",
    "total_yards_away",
    "yards_per_play_away",
    "total_tds_away",
    "win_prob_added_away",
    "passing_yards_per_attempt_away",
    "epa_away_away_diff",
    "total_points_away_away_diff",
    "points_per_drive_away_away_diff",
    "qb_epa_away_away_diff",
    "redzone_drive_count_away",
    "redzone_touchdowns_away",
    "total_yards_away_away_diff",
    "total_points_home_away_off_diff",
    "points_per_drive_home_away_off_diff",
    "qb_epa_home_away_off_diff",
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
    feature_row = pd.Series([home_avg[col] - away_avg[col] for col in derived_stats], index=derived_stats)
    feature_row.index = [f"{col}_home_away_off_diff" for col in feature_row.index]
    if "sacks_allowed_home_away_off_diff" in feature_row:
        feature_row["sacks_allowed_home_away_def_diff"] = feature_row.pop("sacks_allowed_home_away_off_diff")
    # home_avg.index = [f"{col}_home" for col in home_avg.index]
    # away_avg.index = [f"{col}_away" for col in away_avg.index]

    #feature_row = pd.concat([home_avg, away_avg, feature_row])
    # Add meta info
    feature_row['week #'] = week_num
    feature_row['home_team'] = home_team
    feature_row['away_team'] = away_team

    rows.append(feature_row)

# Combine into DataFrame
features_df = pd.DataFrame(rows)

# Save CSV
features_df.to_csv(f"week_{week_num}_features.csv", index=False)
print(f"Week {week_num} features saved as week_{week_num}_features.csv")

X_all = features_df.drop(columns=["home_diff", "win_diff", "week #" ,"home_team","away_team"], errors='ignore')

print(X_all.columns)

# Predict probabilities for all games
home_win_probs = model.predict_proba(X_all)[:, 1]
home_win_preds = model.predict(X_all)

features_df["pred_home_win"] = home_win_preds
features_df["home_win_prob"] = home_win_probs

print(features_df)