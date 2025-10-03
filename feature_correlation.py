import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import nfl_data_py as nfl


with open("spread_model1/game_features_2020_to_2025.pkl", "rb") as f:
    feature_df = pickle.load(f)

y = feature_df.loc[:, "away_points"]
features = feature_df.drop(columns=["home_win", "game_id", "spread_result", "total_result", "home_points", "away_points"])
print(y.value_counts(normalize=True))

#Understand the Data
print(features.shape)

# Select only numeric columns (to be safe)
numeric_df = feature_df.select_dtypes(include="number")

# Compute correlations with home_win
correlations = numeric_df.corr()["away_points"].sort_values(ascending=False)

for col in correlations.index:
    if abs(correlations[col]) > 0.185 and col != "away_points":
        print(f"{col}: {correlations[col]:.3f}")
#print(correlations)

# for col in features.columns:
#     plt.figure(figsize=(6,4))
#     sns.kdeplot(
#         data=feature_df, 
#         x=col, 
#         hue="home_win",  # color by outcome
#         common_norm=False,
#         fill=True,
#         alpha=0.5
#     )
#     plt.title(f"KDE of {col} by Game Outcome")
#     plt.xlabel(col)
#     plt.ylabel("Density")
#     plt.legend(title="Home Win", labels=["Loss (0)", "Win (1)"])
#     plt.tight_layout()
#     plt.show()
