#analasis.py
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split

import lightgbm as lgb

df = pd.read_csv("f1_processed_dataset(2021_2024).csv")

print("Dataset shape:", df.shape)
print(df.head())

numeric_df = df.select_dtypes(include=[np.number])

print("Numeric features:", numeric_df.columns)

corr_matrix = numeric_df.corr()

plt.figure(figsize=(16,12))

sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)

plt.title("Feature Correlation Heatmap")
plt.show()

lap_corr = corr_matrix["LapTime"].sort_values(ascending=False)

print("\nCorrelation with LapTime")
print(lap_corr)

target = "LapTime"

X = numeric_df.drop(columns=[target])
y = numeric_df[target]

mi = mutual_info_regression(X, y)

mi_scores = pd.Series(mi, index=X.columns)
mi_scores = mi_scores.sort_values(ascending=False)

print("\nMutual Information Scores")
print(mi_scores)

plt.figure(figsize=(10,6))

mi_scores.head(15).plot(
    kind="bar"
)

plt.title("Top Mutual Information Features")
plt.ylabel("MI Score")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(ascending=False)

print("\nLightGBM Feature Importance")
print(importance)

plt.figure(figsize=(10,6))

importance.head(15).plot(
    kind="bar"
)

plt.title("Top Feature Importance (LightGBM)")
plt.ylabel("Importance Score")

plt.show()

plt.figure(figsize=(6,4))

sns.scatterplot(
    x=df["TyreLife"],
    y=df["LapTime"],
    alpha=0.3
)

plt.xlabel("Tyre Life")
plt.ylabel("Lap Time (seconds)")
plt.title("Tyre Degradation")

plt.show()

plt.figure(figsize=(6,4))

sns.scatterplot(
    x=df["LapProgress"],
    y=df["LapTime"],
    alpha=0.3
)

plt.xlabel("Race Progress")
plt.ylabel("Lap Time")

plt.title("Lap Time vs Race Progress")

plt.show()