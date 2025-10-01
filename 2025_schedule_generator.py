import pandas as pd

# Load your CSV
df = pd.read_csv("/Users/rithwikmylavarapu/Downloads/2025_raw_nfl_schedule.csv")

weeks = [col for col in df.columns if col.startswith("W")]
rows = []

for week_num, week_col in enumerate(weeks, start=1):
    for idx, row in df.iterrows():
        team = row['Team']
        matchup = str(row[week_col])
        if matchup.upper() == 'BYE' or matchup.upper() == 'NAN':
            continue

        if matchup.startswith('@'):
            # Away game: this team is away, opponent is home
            away_team = team
            home_team = matchup[1:]
        else:
            # Home game: this team is home, opponent is away
            home_team = team
            away_team = matchup

        # Normalize team names to abbreviations if needed (already in abbreviation format)
        rows.append({
            'week #': week_num,
            'home_team': home_team.strip(),
            'away_team': away_team.strip()
        })

# Remove duplicates (since each game appears twice in your CSV)
df_out = pd.DataFrame(rows)
df_out = df_out.drop_duplicates(subset=['week #','home_team','away_team'])

# Save to CSV
df_out.to_csv("nfl_schedule_formatted.csv", index=False)
print("CSV saved as nfl_schedule_formatted.csv")
