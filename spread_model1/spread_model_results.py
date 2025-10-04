import pickle
import pandas as pd
from get_spread_info import fetch_nfl_spreads

def load_models_and_stats():
    with open("spread_model1/home_team_stats_2020_to_2025.pkl", "rb") as f:
        home_stats_dict = pickle.load(f)
    with open("spread_model1/away_team_stats_2020_to_2025.pkl", "rb") as f:
        away_stats_dict = pickle.load(f)
    with open("spread_model1/game_data_2020_to_2025.pkl", "rb") as f:
        game_data = pickle.load(f)
        game_data = game_data[game_data["game_id"].str.startswith("2025_")]
    with open("spread_model1/home_points_model.pkl", "rb") as f:
        model_home = pickle.load(f)
    with open("spread_model1/away_points_model.pkl", "rb") as f:
        model_away = pickle.load(f)
    return home_stats_dict, away_stats_dict, model_home, model_away, game_data

def get_predictions(curr_week, week_num, home_stats_dict, away_stats_dict, model_home, model_away, schedule_df):
    week_df = schedule_df[schedule_df['week #'] == week_num]

    lookback_start = curr_week - week_num # 0 = last 10 games, 1 = 11th-to-last 20th, etc.
    lookback_length = 10  # number of games to average


    home_rows = []
    away_rows = []

    home_points_features = [
        "points_per_drive_home", "total_points_home", "yards_per_play_home",
        "epa_home", "redzone_drive_count_home", "total_tds_home", "total_yards_home",
    ]
    home_away_featurers =  [
        "points_per_drive_home_away_off_diff", "total_points_home_away_off_diff",
        "epa_home_away_off_diff", "total_tds_home_away_off_diff", "yards_per_play_home_away_off_diff",
        "total_yards_home_away_off_diff", "redzone_drive_count_home_away_off_diff",
        "redzone_touchdowns_home_away_off_diff"
    ]

    away_points_features = [
        "epa_away", "points_per_drive_away", "total_points_away", "total_yards_away",
        "yards_per_play_away", "total_tds_away", "win_prob_added_away",
        "passing_yards_per_attempt_away", "redzone_drive_count_away", "redzone_touchdowns_away",
    ]
    away_away_features = [
        "epa_away_away_diff", "total_points_away_away_diff",
        "points_per_drive_away_away_diff", "total_yards_away_away_diff",
    ]
    away_home_features = [
        "total_points_home_away_off_diff", "points_per_drive_home_away_off_diff",
        "epa_home_away_off_diff"
    ]

    for _, game in week_df.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']
        home_stats = home_stats_dict[home_team]
        away_stats = away_stats_dict[away_team]

        home_slice = home_stats.iloc[-(lookback_start + lookback_length):-lookback_start if lookback_start != 0 else None]
        away_slice = away_stats.iloc[-(lookback_start + lookback_length):-lookback_start if lookback_start != 0 else None]

        home_avg = home_slice.mean(numeric_only=True).drop(["home", "win"])
        away_avg = away_slice.mean(numeric_only=True).drop(["home", "win"])

        home_stats_features = pd.Series([home_avg[col[:-5]] for col in home_points_features], index=home_points_features)
        home_away_stats = pd.Series([home_avg[col[:-19]] - away_avg[col[:-19]] for col in home_away_featurers], index=home_away_featurers)
        feature_row_home = pd.concat([home_stats_features, home_away_stats])

        away_stats_features = pd.Series([away_avg[col[:-5]] for col in away_points_features], index=away_points_features)
        away_away_stats = pd.Series([away_avg[col[:-15]] - away_avg[f"{col[:-15]}_allowed"] for col in away_away_features], index=away_away_features)
        away_home_stats = pd.Series([away_avg[col[:-19]] for col in away_home_features], index=away_home_features)
        feature_row_away = pd.concat([away_stats_features, away_away_stats, away_home_stats])

        feature_row_home['home_team'] = home_team
        feature_row_home['away_team'] = away_team
        home_rows.append(feature_row_home)
        away_rows.append(feature_row_away)

    features_df_home = pd.DataFrame(home_rows)
    features_df_away = pd.DataFrame(away_rows)

    X_all_home = features_df_home.drop(columns=["home_team", "away_team"], errors='ignore')
    X_all_away = features_df_away.drop(columns=["home_team", "away_team"], errors='ignore')

    home_score_preds = model_home.predict(X_all_home).round().astype(int)
    away_score_preds = model_away.predict(X_all_away).round().astype(int)

    results = pd.DataFrame({
        "home_team": features_df_home["home_team"],
        "away_team": features_df_home["away_team"],
        "pred_home_score": home_score_preds,
        "pred_away_score": away_score_preds,
    })

    return results

def merge_with_spread(pred_df, spread_df, evaluation):
    if not evaluation:
        spread_df = spread_df[["home_team", "away_team", "spread"]]
    merged = pred_df.merge(spread_df, on=["home_team", "away_team"], how="left")

    merged["pred_spread"] = merged["pred_home_score"] - merged["pred_away_score"]
    merged["predicted_winner"] = merged.apply(lambda row: row["home_team"] if row["pred_spread"] > 0 else row["away_team"], axis=1)
    merged["predicted_covers"] = (merged["pred_spread"] > -1*merged["spread"]).astype(int)
    if evaluation:
        merged["actual_winner"] = merged.apply(lambda row: row["home_team"] if row["actual_home_score"] > row["actual_away_score"] else row["away_team"], axis=1)
        merged["actual_covers"] = ((merged["actual_home_score"] - merged["actual_away_score"]) > -1*merged["spread"]).astype(int)
        merged["correct_winner_pred"] = (merged["predicted_winner"] == merged["actual_winner"]).astype(int)
        merged["correct_cover_pred"] = (merged["predicted_covers"] == merged["actual_covers"]).astype(int)
    return merged

def main():
    API_KEY = "your_api_key_here"
    curr_week = 5
    week_num = 4
    evaluation = True

    home_stats_dict, away_stats_dict, model_home, model_away, game_data = load_models_and_stats()
    schedule_df = pd.read_csv("nfl_schedule_formatted.csv")
    final_df = pd.DataFrame()
    weekly_accuracies = []

    for week_num in range(1, curr_week + 1):
        if week_num < curr_week:
            prediction_df = get_predictions(curr_week, week_num, home_stats_dict, away_stats_dict, model_home, model_away, schedule_df)
            spread_df = game_data[["home_team", "away_team", "spread_line", "home_score", "away_score"]].rename(columns={"spread_line": "spread", "home_score": "actual_home_score", "away_score": "actual_away_score"})
        else:
            prediction_df = get_predictions(curr_week, week_num, home_stats_dict, away_stats_dict, model_home, model_away, schedule_df)
            spread_df = fetch_nfl_spreads(API_KEY)
        
        weekly_df = merge_with_spread(prediction_df, spread_df, week_num < curr_week)
        final_df = pd.concat([final_df, weekly_df], ignore_index=True)

        if week_num < curr_week:
            winner_acc = weekly_df["correct_winner_pred"].mean()
            cover_acc = weekly_df["correct_cover_pred"].mean()
            weekly_accuracies.append({
                "week": week_num,
                "winner_accuracy": winner_acc,
                "cover_accuracy": cover_acc
            })

    print(final_df)
    final_df.to_csv(f"spread_predictions.csv", index=False)

    overall_winner_acc = final_df["correct_winner_pred"].dropna().mean()
    overall_cover_acc = final_df["correct_cover_pred"].dropna().mean()
    for acc in weekly_accuracies:
        print(f"Week {acc['week']} - Winner Accuracy: {acc['winner_accuracy']:.2%}, Cover Accuracy: {acc['cover_accuracy']:.2%}")
    print(f"Overall Winner Prediction Accuracy: {overall_winner_acc:.2%}")
    print(f"Overall Cover Prediction Accuracy: {overall_cover_acc:.2%}")

if __name__ == "__main__":
    main()
