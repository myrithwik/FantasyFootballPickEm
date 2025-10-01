import nfl_data_py as nfl
import pandas as pd
import pickle
import numpy as np

# stats to track
stats = [
    "passing_yards", "rushing_yards", "total_yards", "total_points",
    "passing_tds", "rushing_tds", "qb_epa", "total_tds", "sacks_allowed", 'qb_hits_allowed', 'tfl_allowed',
    'win_prob_added', 'epa', 'air_yards', 'yac', "turnovers",
    "passing_yards_allowed", "rushing_yards_allowed", "total_yards_allowed",
    "total_points_allowed", "passing_tds_allowed", "rushing_tds_allowed",
    'qb_epa_allowed', 'total_tds_allowed', 'sacks', 'qb_hits', 'tfl',
       'win_prob_added_allowed', 'epa_allowed', 'air_yards_allowed',
       'yac_allowed', "turnovers_forced"
]

# df_games: should have game_id, home_team, away_team, home_score, away_score
# home_stats_dict, away_stats_dict: prebuilt from your earlier step

rows = []
with open("second_model/game_data_2020_to_2025.pkl", "rb") as f:
    df_games = pickle.load(f)
with open("second_model/home_team_stats_2020_to_2025.pkl", "rb") as f:
    home_stats_dict = pickle.load(f)
with open("second_model/away_team_stats_2020_to_2025.pkl", "rb") as f:
    away_stats_dict = pickle.load(f)

for idx, game in df_games.iterrows():
    game_id = game["game_id"]
    home_team = game["home_team"]
    away_team = game["away_team"]
    home_win = game["home_win"]
    # if pd.isna(game["Temp"]):
    #     print("TEMP is NA")

    # --- Get home team last 10 home games before this game ---
    home_df = home_stats_dict[home_team]
    past_home = home_df[home_df["game_id"] < game_id].sort_values("game_id").tail(10)
    home_avgs = past_home[stats].mean()

    # --- Get away team last 10 away games before this game ---
    away_df = away_stats_dict[away_team]
    past_away = away_df[away_df["game_id"] < game_id].sort_values("game_id").tail(10)
    away_avgs = past_away[stats].mean()
    # Skip if not enough past data
    if len(past_home) < 10 or len(past_away) < 10:
        continue
    # --- Compute feature differences ---
    home_features = {f"{stat}_home": home_avgs[stat] for stat in stats}
    away_features = {f"{stat}_away": away_avgs[stat] for stat in stats}
    features = {f"{stat}_diff": home_avgs[stat] - away_avgs[stat] for stat in stats}

    # --- Build row ---
    row = {"game_id": game_id, "home_win": home_win} 
           #"wind": np.float32(game["Wind"]), "temp": np.float32(game["Temp"]), "spread_line": np.float32(game["spread_line"]), "total_line": np.float32(game["total_line"])}
    row.update(home_features)
    row.update(away_features)
    row.update(features)
    rows.append(row)
# --- Final dataset ---
feature_df = pd.DataFrame(rows)
diff_cols = [col for col in feature_df.columns if col.endswith("_diff")]

# Downcast to float32
for i, val in feature_df["qb_hits_diff"].items():
    if not isinstance(val, (float, int, np.float32, np.float64)):
        print(f"Row {i} - type: {type(val)}, value: {val}")
#feature_df["qb_hits_diff"] = feature_df["qb_hits_diff"].astype(np.float32)

print(feature_df.head())
print(len(feature_df))
print(feature_df.dtypes)

print(feature_df.isna().sum())
with open("second_model/game_features_2020_to_2025.pkl", "wb") as f:
    pickle.dump(feature_df, f)
