# 🏈 NFL Score & Spread Prediction (2015–2025)

This project builds and evaluates a machine learning pipeline for **predicting NFL game scores, winners, and spread outcomes**. It combines historical team performance data, predictive modeling, and live betting spread data to forecast the outcome of NFL games during the 2025 season.

---

## 📊 Project Overview

Using team-level stats from the past **10 NFL seasons (2015–2024)**, this project:

- Trains ML models to predict **home and away team scores**.
- Uses those predicted scores to:
  - Determine the **predicted game winner**.
  - Evaluate whether the team is expected to **cover the betting spread**.
- Tracks **weekly** and **overall prediction accuracy** for both winner and spread.

---

## 🔍 Features

- 🧠 **ML Modeling**  
  Trains separate models for predicting home and away team points using engineered statistical features.

- 📈 **Score Predictions**  
  Generates realistic score predictions using team historical performance.

- 🏆 **Outcome Evaluation**  
  Determines:
  - Predicted Winner
  - Predicted Cover (against the spread)
  - Accuracy of both (per week & overall)

- 🧮 **Spread Integration**  
  Pulls current and historical betting spread data from [The Odds API](https://the-odds-api.com/).

- 🗓️ **Weekly Tracking**  
  Aggregates and stores prediction performance by week.

---

## ⚙️ How It Works

1. **Data Collection**  
   - Pulls historical play by play stats and aggregates them into derived team level stats (e.g. points per drive, EPA, red zone efficiency).
   - Loads spreads and results for past games.

2. **Model Training**  
   - Two separate tree based ensemble regression models are trained:
     - One for predicting **home team points**.
     - One for **away team points**.

3. **Prediction Pipeline**  
   - For each week in the 2025 NFL season:
     - Predicts scores using the last 10 games of team stats.
     - Compares predicted spread vs. Vegas spread.
     - Determines predicted winner & cover.

4. **Evaluation**  
   - Tracks:
     - ✅ Correct winner predictions
     - ✅ Correct spread predictions
     - 📈 Weekly and overall prediction accuracy

---

## 📈 Example Output
| week | correct_winner_accuracy | correct_spread_accuracy |
|------|--------------------------|--------------------------|
| 1    | 62.5%                   | 56.3%                   |
| 2    | 68.8%                   | 62.5%                   |
| ...  | ...                     | ...                     |
| All  | 65.4%                   | 59.8%                   |

Each game's predictions include:
- Predicted Scores
- Vegas Spread
- Predicted Winner
- Predicted Cover
- Actual Winner & Cover (for completed games)
- Binary correctness indicators

---

## 📦 Requirements

- Python 3.8+
- pandas
- scikit-learn
- requests
- pickle

Install dependencies:
```bash
pip install -r requirements.txt
