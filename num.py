# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns


# %%
df_original = pd.read_csv("merged_data.csv")

df = df_original.copy()
df

# %%
cols_to_drop = [
    "tip_amount",
    "tolls_amount",
    "total_amount",
    "payment_type",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "store_and_fwd_flag",
    "pickup_datetime",
    "dropoff_datetime",
    "pickup_date",
]
df = df.drop(columns=cols_to_drop)
df

# %%
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()
df["pickup_day_of_week"] = encoder.fit_transform(df[["pickup_day_of_week"]])
df

# %%
X = df.drop(columns=["fare_amount"])
y = df["fare_amount"]

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# %%
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

col = 

arr = []
print("significance of the attribute")
for f, idx in enumerate(indices):
    arr.append([round(importances[idx], 4), col[idx]])

arr.sort(reverse=True)
arr


# %%
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)


# %%
mae

# %%
mse

# %%
rmse
