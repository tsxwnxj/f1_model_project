import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import lightgbm as lgb
import joblib

df = pd.read_csv("f1_feature_selected_dataset(2021-2024).csv")

print(df.shape)
print(df.head())

target = "LapTime"

X = df.drop(columns = [target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)

model = lgb.LGBMRegressor(
    n_estimators = 1000,
    learning_rate = 0.03,
    max_depth = 10,
    num_leaves = 64,
    subsample = 0.8,
    colsample_bytree = 0.8,
    random_state = 42
)

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE: ", rmse)
print("R2: ", r2)

importance = pd.Series(
    model.feature_importances_,
    index = X.columns
).sort_values(ascending=False)

print(importance)

joblib.dump(model, "F1_lap_time_model.pkl")

print("Model saved")