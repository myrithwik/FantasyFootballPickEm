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
        book = game["bookmakers"][0]["title"]

        # spreads
        spread_market = next(m for m in game["bookmakers"][0]["markets"] if m["key"] == "spreads")
        home_spread = next(o for o in spread_market["outcomes"] if o["name"] == home)["point"]

        # totals (same number for Over/Under, just take one)
        total_market = next(m for m in game["bookmakers"][0]["markets"] if m["key"] == "totals")
        total_line = total_market["outcomes"][0]["point"]

        games.append({
            "home_team": home,
            "away_team": away,
            "spread": home_spread,
            "total_line": total_line,
            "bookmaker": book,
            "commence_time": commence
        })

    return pd.DataFrame(games)


# Example usage
if __name__ == "__main__":
    API_KEY = "45d6d7361ebb182ab03624f977439b8f"
    df = fetch_nfl_spreads(API_KEY)
    print(df.head())
