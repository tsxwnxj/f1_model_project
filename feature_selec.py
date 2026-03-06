import pandas as pd
import numpy as np

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split

import lightgbm as lgb

df = pd.read_csv("f1_processed_dataset(2021_2024).csv")

numeric_df = df.select_dtypes(include = [np.number])

target = "LapTime"

X = numeric_df.drop(columns = [target])
y = numeric_df[target]

X = X.fillna(0)

selector = VarianceThreshold(threshold = 0.01)

X_var = selector.fit_transform(X)

selected_features_var = X.columns[selector.get_support()]

X_var = pd.DataFrame(X_var, columns=selected_features_var)

mi = mutual_info_regression(X_var, y)
mi_scores = pd.Series(mi, index = X_var.columns)

mi_selected = mi_scores.sort_values(ascending=False).head(30).index

X_mi = X_var[mi_selected]

X_train, X_test, y_train, y_test = train_test_split(
    X_mi,
    y,
    test_size = 0.2,
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
    index=X_mi.columns
)

final_features = importance.sort_values(ascending=False).head(15).index

X_selected = X_mi[final_features]
final_dataset = pd.concat([X_selected, y], axis = 1)

final_dataset.to_csv(
    "f1_feature_selected_dataset(2021-2024).csv",
    index = False
)

print("Feature selection complete")
print("Selected feature count: ", len(final_features))