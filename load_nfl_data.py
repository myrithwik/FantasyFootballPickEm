import nfl_data_py as nfl
import pandas as pd
import pickle

#pbp_data = nfl.import_pbp_data([2025])
pbp_data = nfl.import_pbp_data([2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015])

# for col in nfl.see_pbp_cols():
#     print(col)

print(pbp_data[["pass_attempt", "rush_attempt", "third_down_converted", "third_down_failed", "drive_time_of_possession"]])

print("*********")

print(pbp_data[["drive", "series", "series_success", "series_result", "fixed_drive", "fixed_drive_result", "drive_play_count"]])

print("**********")

print(pbp_data[pbp_data["drive_inside20"] == 1][["yardline_100", "yrdln", "drive_inside20", "drive_ended_with_score", "fixed_drive_result"]])
print(pbp_data["fixed_drive_result"].unique())

teams = pbp_data["posteam"].unique()
teams = teams[teams != None]

# Filter once
pbp_data = pbp_data[(pbp_data["season_type"] == "REG") & (pbp_data["posteam"].notna())]

pbp_data["Temp"] = pbp_data["weather"].str.extract(r"Temp:\s*(\d+)").astype(float)
# Extract Wind
pbp_data["Wind"] = pbp_data["weather"].str.extract(r"Wind:\s*[A-Z]*\s*(\d+)").astype(float)
# Aggregations for offensive stats
agg_funcs = {
    "passing_yards": "sum",
    "rushing_yards": "sum",
    "yards_gained": "sum",
    "fumble_lost": "sum",
    "interception": "sum",
    "posteam_score": "max",    # max score reached
    "pass_touchdown": "sum",
    "rush_touchdown": "sum",
    "qb_epa": "sum",
    "touchdown": "sum",
    "sack": "sum",
    "qb_hit": "sum",
    "tackled_for_loss": "sum",
    "wpa": "sum",
    "epa": "sum",
    "air_yards": "sum",
    "yards_after_catch": "sum",
    "pass_attempt": "sum",
    "rush_attempt": "sum",
    "third_down_converted": "sum",
    "third_down_failed": "sum",
    "fixed_drive": "nunique",
    "drive_inside20": "sum"
}

# --- Step 1: Aggregate offensive stats ---
offense = (
    pbp_data
    .groupby(["game_id", "posteam", "defteam", "home_team"], as_index=False)
    .agg(agg_funcs)
)

offense = offense.rename(columns={
    "posteam": "team",
    "defteam": "opponent",
    "yards_gained": "total_yards",
    "posteam_score": "total_points",
    "pass_touchdown": "passing_tds",
    "rush_touchdown": "rushing_tds",
    "touchdown": "total_tds",
    "sack": "sacks_allowed",
    "tackled_for_loss": "tfl_allowed",
    "wpa": "win_prob_added",
    "yards_after_catch": "yac",
    "qb_hit": "qb_hits_allowed",
    "fixed_drive": "number_of_drives"
})

# Derived stats
offense["turnovers"] = offense["fumble_lost"] + offense["interception"]
offense["home"] = (offense["home_team"] == offense["team"]).astype(int)
offense["pass_yards_per_atempt"] = offense["passing_yards"] / offense["pass_attempt"]
offense["rush_yards_per_atempt"] = offense["rushing_yards"] / offense["rush_attempt"]
offense["third_down_efficiency"] = offense["third_down_converted"] / (offense["third_down_converted"] + offense["third_down_failed"])

# Drop helper cols no longer needed
offense = offense.drop(columns=["fumble_lost", "interception", "home_team", "pass_attempt", "rush_attempt", "third_down_converted", "third_down_failed"])

drive_data = (
    pbp_data
    .groupby(["game_id", "posteam", "fixed_drive"], as_index=False)
    .agg({
        "fixed_drive_result": "first",
        "drive_time_of_possession": "first",
        "drive_inside20": "max"
    })
)

def time_to_seconds(t):
    if pd.isna(t):  # check for NaN
        return pd.NA
    minutes, seconds = map(int, t.split(":"))
    return minutes * 60 + seconds
drive_data["drive_time_of_possession"] = drive_data["drive_time_of_possession"].apply(time_to_seconds)

drive_time = (
    drive_data
    .groupby(["game_id", "posteam"], as_index=False)
    .agg({
        "drive_time_of_possession": "sum",
    })
)

drive_redzone = (
    drive_data[drive_data["drive_inside20"] == 1]  # Filter first
    .groupby(["game_id", "posteam"], as_index=False)
    .agg(
        redzone_touchdowns=("fixed_drive_result", lambda x: (x == "Touchdown").sum()),
        redzone_drive_count=("fixed_drive_result", "count")  # Optional: total red zone drives
    )
)

drive_redzone["redzone_efficiency"] = drive_redzone["redzone_touchdowns"] / drive_redzone["redzone_drive_count"]
drive_time = drive_time.rename(columns={"drive_time_of_possession": "time_of_possession", "posteam": "team"})
drive_redzone = drive_redzone.rename(columns={"posteam": "team"})

print(drive_redzone)
print(drive_time)
offense = offense.merge(
    drive_time,
    on=["game_id", "team"],
    how="left"
).merge(
    drive_redzone,
    on=["game_id", "team"],
    how="left"
)
offense = offense.drop(columns=["drive_inside20"])
print(offense)

# --- Step 2: Merge in opponent stats (defense) ---
# We'll suffix opponent columns with `_allowed`
defense = offense.copy()
defense = defense.rename(columns={col: col + "_allowed" for col in [
    "passing_yards", "rushing_yards", "total_yards", "total_points", "passing_tds", "rushing_tds", "total_tds", "qb_epa", "win_prob_added", "epa", "air_yards", "yac", "number_of_drives", "turnovers", "sacks_allowed", "tfl_allowed", "qb_hits_allowed", "third_down_efficiency", "pass_yards_per_atempt", "rush_yards_per_atempt", "redzone_touchdowns", "redzone_drive_count", "redzone_efficiency", "time_of_possession"
]})

# Flip perspective: opponent in offense table = team in defense
defense = defense.rename(columns={"team": "opponent", "opponent": "team", "sacks_allowed": "sacks", "tfl_allowed": "tfl", "qb_hits_allowed": "qb_hits", "turnovers": "turnovers_forced"})
defense = defense.drop(columns=["home"])

# Merge on game_id + team
full_stats = pd.merge(
    offense,
    defense,
    on=["game_id", "team", "opponent"],
    how="left"
)

print(full_stats.columns)
# full_stats = full_stats.drop(columns=["home_y"])
# full_stats = full_stats.rename(columns={"home_x": "home"})
full_stats["win"] = (full_stats["total_points"] > full_stats["total_points_allowed"]).astype(int)

full_stats.sort_values(by="game_id", ascending=True, inplace=True)
# --- Step 3: Build dict of DataFrames per team ---
team_stats_dict = {
    team: df.drop(columns=["team", "opponent"]).reset_index(drop=True)
    for team, df in full_stats.groupby("team")
}

# Example: Denver’s per-game stats
# print(len(team_stats_dict.keys()))
# for team in team_stats_dict.keys():
#     print(len(team_stats_dict[team]))

home_stats_dict = {}
away_stats_dict = {}

for team, df in team_stats_dict.items():
    home_stats_dict[team] = df[df["home"] == 1].reset_index(drop=True).copy()
    away_stats_dict[team] = df[df["home"] == 0].reset_index(drop=True).copy()

print(home_stats_dict["DEN"])
# with open("second_model/home_team_stats_2020_to_2025.pkl", "wb") as f:
#     pickle.dump(home_stats_dict, f)

# with open("second_model/away_team_stats_2020_to_2025.pkl", "wb") as f:
#     pickle.dump(away_stats_dict, f)
agg_funcs = {
    "home_score": "max",    # max score reached
    "away_score": "max",
    # "Wind": "mean",
    # "Temp": "mean",
    # "spread_line": "first",
    # "total_line": "first"
}

game_data = (
    pbp_data
    .groupby(["game_id", "home_team", "away_team"], as_index=False)
    .agg(agg_funcs)
)
game_data["home_win"] = (game_data["home_score"] > game_data["away_score"]).astype(int)
game_data = game_data.drop(columns=["home_score", "away_score"])
# game_data["Wind"] = game_data["Wind"].fillna(0)
# game_data["Temp"] = game_data["Temp"].fillna(68)

with open("spread_model1/game_data_2020_to_2025.pkl", "wb") as f:
    pickle.dump(game_data, f)