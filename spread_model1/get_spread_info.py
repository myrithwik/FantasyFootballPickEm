import requests
import pandas as pd

def fetch_nfl_spreads(api_key: str, region: str = "us") -> pd.DataFrame:
    """
    Fetch NFL game spreads and totals from The Odds API.
    
    Parameters
    ----------
    api_key : str
        Your API key from the-odds-api.com
    region : str
        Betting region ('us', 'uk', etc.), default='us'

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        [home_team, away_team, spread, total_line, bookmaker, commence_time]
    """
    TEAM_NAME_TO_ABBR = {
        "Arizona Cardinals": "ARI",
        "Atlanta Falcons": "ATL",
        "Baltimore Ravens": "BAL",
        "Buffalo Bills": "BUF",
        "Carolina Panthers": "CAR",
        "Chicago Bears": "CHI",
        "Cincinnati Bengals": "CIN",
        "Cleveland Browns": "CLE",
        "Dallas Cowboys": "DAL",
        "Denver Broncos": "DEN",
        "Detroit Lions": "DET",
        "Green Bay Packers": "GB",
        "Houston Texans": "HOU",
        "Indianapolis Colts": "IND",
        "Jacksonville Jaguars": "JAX",
        "Kansas City Chiefs": "KC",
        "Las Vegas Raiders": "LV",
        "Los Angeles Chargers": "LAC",
        "Los Angeles Rams": "LA",
        "Miami Dolphins": "MIA",
        "Minnesota Vikings": "MIN",
        "New England Patriots": "NE",
        "New Orleans Saints": "NO",
        "New York Giants": "NYG",
        "New York Jets": "NYJ",
        "Philadelphia Eagles": "PHI",
        "Pittsburgh Steelers": "PIT",
        "San Francisco 49ers": "SF",
        "Seattle Seahawks": "SEA",
        "Tampa Bay Buccaneers": "TB",
        "Tennessee Titans": "TEN",
        "Washington Commanders": "WAS"
    }


    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    params = {
        "regions": region,
        "markets": "spreads,totals",
        "oddsFormat": "american",
        "apiKey": api_key
    }

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    games = []
    for game in data:
        home = game["home_team"]
        away = game["away_team"]
        commence = game["commence_time"]

        # take the first bookmaker (or choose consensus later)
        if not game.get("bookmakers"):
            continue  # Skip if no bookmakers at all

        first_bookmaker = game["bookmakers"][0]
        book = first_bookmaker.get("title", "Unknown")

        # Get spreads market safely
        spread_market = next((m for m in first_bookmaker.get("markets", []) if m.get("key") == "spreads"), None)
        if not spread_market:
            continue  # Skip if no spreads

        # Find home spread safely
        home_spread = next((o.get("point") for o in spread_market["outcomes"] if o.get("name") == home), None)
        if home_spread is None:
            continue  # Skip if home team spread not found

        # Get totals market safely
        total_market = next((m for m in first_bookmaker.get("markets", []) if m.get("key") == "totals"), None)
        if not total_market or not total_market.get("outcomes"):
            continue  # Skip if no totals

        total_line = total_market["outcomes"][0].get("point")

        games.append({
            "home_team": home,
            "away_team": away,
            "spread": home_spread,
            "total_line": total_line,
            "bookmaker": book,
            "commence_time": commence
        })

    df = pd.DataFrame(games)
    df["home_team"] = df["home_team"].map(TEAM_NAME_TO_ABBR)
    df["away_team"] = df["away_team"].map(TEAM_NAME_TO_ABBR)
    return df

# Example usage
if __name__ == "__main__":
    API_KEY = "45d6d7361ebb182ab03624f977439b8f"
    df = fetch_nfl_spreads(API_KEY)
    print(df.head())
    df.to_csv("nfl_spreads_try1.csv", index=False)
