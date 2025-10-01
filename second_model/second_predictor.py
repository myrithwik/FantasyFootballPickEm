import pickle
import pandas as pd

with open("second_model/home_team_stats_2020_to_2025.pkl", "rb") as f:
    home_stats_dict = pickle.load(f)
with open("second_model/away_team_stats_2020_to_2025.pkl", "rb") as f:
    away_stats_dict = pickle.load(f)
with open("second_model/home_win_model.pkl", "rb") as f:
    model = pickle.load(f)

print(away_stats_dict["DEN"].columns)
# Example dictionaries
# home_team_stats_df = {"ARI": pd.DataFrame(...), ...}
# away_team_stats_df = {"ARI": pd.DataFrame(...), ...}

# Load schedule CSV
schedule_df = pd.read_csv("nfl_schedule_formatted.csv")

# Week to process
week_num = 4
week_df = schedule_df[schedule_df['week #'] == week_num]
print(week_df)
print(home_stats_dict.keys())

# Lookback configuration
lookback_start = 1  # 0 = last 10 games, 1 = 11th-to-last 20th, etc.
lookback_length = 10  # number of games to average

rows = []

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
    feature_row = home_avg - away_avg
    feature_row.index = [f"{col}_diff" for col in feature_row.index]
    home_avg.index = [f"{col}_home" for col in home_avg.index]
    away_avg.index = [f"{col}_away" for col in away_avg.index]

    feature_row = pd.concat([home_avg, away_avg, feature_row])
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

# Predict probabilities for all games
home_win_probs = model.predict_proba(X_all)[:, 1]
home_win_preds = model.predict(X_all)

features_df["pred_home_win"] = home_win_preds
features_df["home_win_prob"] = home_win_probs

print(features_df)