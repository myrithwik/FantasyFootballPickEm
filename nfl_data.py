import nfl_data_py as nfl
import pandas as pd

#print(nfl.import_seasonal_data([2024]))
#pbp_data = nfl.import_pbp_data([2024])
# pbp_cols = nfl.see_pbp_cols()
# for col in pbp_cols:
#     print(col)

pbp_2024 = nfl.import_pbp_data([2022])

teams = pbp_2024["posteam"].unique()
teams = teams[teams != None]
print(teams)

team_stats_dict = {}

for team in teams:
    team_stats_dict[team] = pd.DataFrame(columns=["game_id", "passing_yards"])

#overall_passing_data = pd.DataFrame(columns=["team", "game_id", "passing_yards"])

for team in teams[:2]:
    #Filter for the teams offensive pass plays
    team_plays = pbp_2024[
        (pbp_2024['posteam'] == team) &
        (pbp_2024['season_type'] == "REG")
    ]

    # Sum the passing yards, excluding NaNs
    #total_passing_yards = team_plays['rush_touchdown'].dropna().sum()

    #print(f"Total Passing Yards for the {team} in 2024: {total_passing_yards}")

    # Drop rows with missing passing_yards
    team_pass_plays = team_pass_plays.dropna(subset=['passing_yards'])

    # Group by week and sum passing yards
    weekly_passing_yards = (
        team_pass_plays
        .groupby('game_id')['passing_yards']
        .sum()
        .reset_index()
        .sort_values('game_id')
    )

    # Optional: Rename columns for clarity
    weekly_passing_yards.columns = ['game_id', 'passing_yards']
    weekly_passing_yards["team"] = team
    weekly_passing_yards = weekly_passing_yards[["team", "game_id", "passing_yards"]]
    overall_passing_data = pd.concat([overall_passing_data, weekly_passing_yards])

    #print(weekly_passing_yards)
print(overall_passing_data)