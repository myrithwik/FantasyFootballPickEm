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
with open("spread_model1/game_data_2020_to_2025.pkl", "rb") as f:
    df_games = pickle.load(f)
with open("spread_model1/home_team_stats_2020_to_2025.pkl", "rb") as f:
    home_stats_dict = pickle.load(f)
with open("spread_model1/away_team_stats_2020_to_2025.pkl", "rb") as f:
    away_stats_dict = pickle.load(f)

stats = [col for col in home_stats_dict["DEN"].columns if col not in ["game_id", "win"]]
print(stats)

# Columns that should appear in both offense and defense DataFrames
shared_columns = ['game_id', 'home']

# Defensive columns have '_allowed', '_forced', or '_allowed_per_' or other known suffixes
defensive_keywords = ['_allowed', '_forced', '_per_drive_allowed', '_per_pass_attempt_allowed',
                      '_per_rush_attempt_allowed', '_per_*_allowed']

# Separate offensive and defensive columns based on suffixes
def is_defensive(col):
    return any(keyword in col for keyword in defensive_keywords)

def parse_game_id(game_id):
    """
    Convert a game_id like '2024_13_CLE_DEN' into a sortable integer: 202413
    """
    try:
        parts = game_id.split("_")
        year = int(parts[0])
        week = int(parts[1])
        return year * 100 + week  # e.g., 2024 * 100 + 13 = 202413
    except:
        return None

for idx, game in df_games.iterrows():
    game_id = game["game_id"]
    home_team = game["home_team"]
    away_team = game["away_team"]
    home_win = game["home_win"]
    spread = game["spread_result"]
    total = game["total_result"]
    home_points = game["home_score"]
    away_points = game["away_score"]
    # if pd.isna(game["Temp"]):
    #     print("TEMP is NA")

    # --- Get home team last 10 home games before this game ---
    home_df = home_stats_dict[home_team]
    away_df = away_stats_dict[away_team]

    # Separate offense and defense columns
    offensive_columns = [col for col in home_df.columns if not is_defensive(col) and col not in ['game_id', 'home', 'win']]
    defensive_columns = [col for col in home_df.columns if is_defensive(col)]

    # Add shared columns to each
    offense_df_home = home_df[shared_columns + offensive_columns]
    defense_df_home = home_df[shared_columns + defensive_columns]
    offense_df_away = away_df[shared_columns + offensive_columns]
    defense_df_away = away_df[shared_columns + defensive_columns]

    past_offense_home = offense_df_home[offense_df_home["game_id"] < game_id].sort_values("game_id").tail(10)
    past_defense_home = defense_df_home[defense_df_home["game_id"] < game_id].sort_values("game_id").tail(10)
    home_avgs_offense = past_offense_home[offensive_columns].mean()
    home_avgs_defense = past_defense_home[defensive_columns].mean()

    past_offense_away = offense_df_away[offense_df_away["game_id"] < game_id].sort_values("game_id").tail(10)
    past_defense_away = defense_df_away[defense_df_away["game_id"] < game_id].sort_values("game_id").tail(10)
    away_avgs_offense = past_offense_away[offensive_columns].mean()
    away_avgs_defense = past_defense_away[defensive_columns].mean()

    # past_home = home_df[home_df["game_id"] < game_id].sort_values("game_id").tail(10)
    # home_avgs = past_home[stats].mean()

    # --- Get away team last 10 away games before this game ---
    #away_df = away_stats_dict[away_team]
    #past_away = away_df[away_df["game_id"] < game_id].sort_values("game_id").tail(10)
    #away_avgs = past_away[stats].mean()
    # Skip if not enough past data
    if len(past_offense_home) < 10 or len(past_offense_away) < 10 or len(past_defense_home) < 10 or len(past_defense_away) < 10:
        continue
    # for col in home_df.columns:
    #     print(col)

    home_features_offense = {f"{stat}_home": home_avgs_offense[stat] for stat in offensive_columns}
    away_features_offense = {f"{stat}_away": away_avgs_offense[stat] for stat in offensive_columns}
    home_features_defense = {f"{stat}_home": home_avgs_defense[stat] for stat in defensive_columns}
    away_features_defense = {f"{stat}_away": away_avgs_defense[stat] for stat in defensive_columns}
    home_features = {**home_features_offense, **home_features_defense}
    away_features = {**away_features_offense, **away_features_defense}
    # features = {f"{stat}_diff": home_avgs[stat] - away_avgs[stat] for stat in stats}

    ############# Calculate features #############
    
    ########## Get home vs home features ##########
    home_home = {
        f"{stat}_home_home_diff": home_avgs_offense[stat] - home_avgs_defense.get(f"{stat}_allowed", 0)
        for stat in offensive_columns
        if f"{stat}_allowed" in defensive_columns
    }

    # Manual mappings for exceptions
    manual_stat_mappings = {
        "turnovers": "turnovers_forced",
        "sacks_allowed": "sacks",
        "tfl_allowed": "tfl",
        "qb_hits_allowed": "qb_hits",
        "turnovers_per_drive": "turnovers_forced_per_drive",
    }

    manual_home_home = {
        f"{off}_home_home_diff": home_avgs_offense[off] - home_avgs_defense.get(defn, 0)
        for off, defn in manual_stat_mappings.items()
        if off in home_avgs_offense and defn in home_avgs_defense
    }

    # Merge the two
    home_home.update(manual_home_home)

    ########## Get away vs away features ##########
    away_away = {
        f"{stat}_away_away_diff": away_avgs_offense[stat] - away_avgs_defense.get(f"{stat}_allowed", 0)
        for stat in offensive_columns
        if f"{stat}_allowed" in defensive_columns
    }  
    manual_away_away = {
        f"{off}_away_away_diff": away_avgs_offense[off] - away_avgs_defense.get(defn, 0)
        for off, defn in manual_stat_mappings.items()
        if off in away_avgs_offense and defn in away_avgs_defense
    }  
    away_away.update(manual_away_away)

    ########## Get home vs away features ##########
    home_away_off_def = {
        f"{stat}_home_away_off_def_diff": home_avgs_offense[stat] - away_avgs_defense.get(f"{stat}_allowed", 0)
        for stat in offensive_columns
        if f"{stat}_allowed" in defensive_columns
    }
    manual_home_away_off_def = {
        f"{off}_home_away_off_def_diff": home_avgs_offense[off] - away_avgs_defense.get(defn, 0)
        for off, defn in manual_stat_mappings.items()
        if off in home_avgs_offense and defn in away_avgs_defense
    }
    home_away_off_def.update(manual_home_away_off_def)  

    ########## Get away vs home features ##########
    home_away_def_off = {
        f"{stat}_home_away_def_off_diff": -1*(home_avgs_offense[stat] - home_avgs_defense.get(f"{stat}_allowed", 0))
        for stat in offensive_columns
        if f"{stat}_allowed" in defensive_columns
    }
    manual_away_home_off_def = {
        f"{off}_home_away_def_off_diff": -1 * (away_avgs_offense[off] - home_avgs_defense.get(defn, 0))
        for off, defn in manual_stat_mappings.items()
        if off in away_avgs_offense and defn in home_avgs_defense
    }
    home_away_def_off.update(manual_away_home_off_def)  

    ########## Get home vs away offense features ##########
    home_away_off = {
        f"{stat}_home_away_off_diff": home_avgs_offense[stat] - away_avgs_offense.get(stat, 0)
        for stat in offensive_columns
        if stat in away_avgs_offense
    }
    
    ######## Get away vs home defense features ##########
    home_away_def = {
        f"{stat}_home_away_def_diff": home_avgs_defense[stat] - away_avgs_defense.get(stat, 0)
        for stat in defensive_columns
        if stat in home_avgs_defense
    }

    # --- Build row ---
    row = {"game_id": game_id, "home_win": home_win, "spread_result": spread, "total_result": total, "home_points": home_points, "away_points": away_points} 
           #"wind": np.float32(game["Wind"]), "temp": np.float32(game["Temp"]), "spread_line": np.float32(game["spread_line"]), "total_line": np.float32(game["total_line"])}
    row.update(home_features)
    row.update(away_features)
    row.update(home_home)
    row.update(away_away)
    row.update(home_away_off_def)
    row.update(home_away_def_off)
    row.update(home_away_off)
    row.update(home_away_def)
    #row.update(features)
    rows.append(row)
# --- Final dataset ---
feature_df = pd.DataFrame(rows)
diff_cols = [col for col in feature_df.columns if col.endswith("_diff")]

# Downcast to float32
# for i, val in feature_df["qb_hits_diff"].items():
#     if not isinstance(val, (float, int, np.float32, np.float64)):
#         print(f"Row {i} - type: {type(val)}, value: {val}")
#feature_df["qb_hits_diff"] = feature_df["qb_hits_diff"].astype(np.float32)

print(feature_df.head())
print(len(feature_df))
print(feature_df.dtypes)
for col in feature_df.columns:
    #print(feature_df[col].dtype, col)
    if feature_df[col].dtype == bool:
        print(col)

#print(feature_df.isna().sum())
with open("spread_model1/game_features_2020_to_2025.pkl", "wb") as f:
    pickle.dump(feature_df, f)
