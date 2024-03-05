import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
df = pd.read_csv("for knn_paymentmethod_numerical.csv")

# ___________________________________________________________________________________________________

# df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])


# df['pickup_date'] = df["pickup_datetime"].dt.date
# df["pickup_day"] = df["pickup_datetime"].apply(lambda x:x.day)
# df["pickup_hour"] = df["pickup_datetime"].apply(lambda x:x.hour)

# df = df[((df['pickup_longitude'] > -78) &
#         (df['pickup_longitude'] < -70)) &
#         ((df['dropoff_longitude'] > -78) &
#         (df['dropoff_longitude'] < -70)) &
#         ((df['pickup_latitude'] > 37) &
#         (df['pickup_latitude'] < 45)) &
#         ((df['dropoff_latitude'] > 37) &
#         (df['dropoff_latitude'] < 45)) &
#         (df['passenger_count'] > 0) &
#         (df['trip_distance'] > 0) &
#         (df['fare_amount'] >= 2.5)]

# # List of columns to delete
# columns_to_delete = ['store_and_fwd_flag', 'pickup_longitude', 'dropoff_longitude', 'pickup_latitude', 'dropoff_latitude', 'pickup_datetime', 'dropoff_datetime', 'tolls_amount', 'pickup_date']

# # Perform one-hot encoding for the 'payment_method' column
# df = pd.get_dummies(df, columns=['payment_type'])
# Now 'payment_method' has been encoded into binary columns

# # Move the 'total_amount' column to the end
# desired_columns_order = list(df.columns.drop('total_amount')) + ['total_amount']

# Reindex the DataFrame with the desired column order
# df = df.reindex(columns=desired_columns_order)

# # Remove the desired columns
# for column in columns_to_delete:
#     if column in df.columns:
#         df.drop(column, axis=1, inplace=True)
#     else:
#         print(f"Column '{column}' not found.")


# Save the DataFrame back to a new CSV file
# df.to_csv('C:\\Users\\User\\OneDrive - American University of Armenia\\Desktop\\NY Taxi\\NY Taxi\\for knn_paymentmethod_numerical.csv', index=False)

# ________________________________________________________________________________________________________________________________
# df.info()

df.drop("surcharge", axis=1, inplace=True)
df.drop("mta_tax", axis=1, inplace=True)
df.drop("fare_amount", axis=1, inplace=True)
df.drop("total_amount", axis=1, inplace=True)
# Move the 'total_amount' column to the end
desired_columns_order = list(df.columns.drop("tip_amount")) + ["tip_amount"]
df.info()
# df['trip_time_in_secs'] //= 60
# df['trip_distance'] //= 1

# Reindex the DataFrame with the desired column order
df = df.reindex(columns=desired_columns_order)


# Step 5: Split the Data
X = df.drop("tip_amount", axis=1)
y = df["tip_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Train the kNN Model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Choose an appropriate value for k
k = 20
knn_model = KNeighborsRegressor(n_neighbors=k)
knn_model.fit(X_train_scaled, y_train)

# Step 7: Model Evaluation
y_pred = knn_model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

# Define the threshold (e.g., ± $1.00)
threshold = 1.0

# Calculate the absolute errors
absolute_errors = np.abs(y_pred - y_test)

# Calculate the percentage of predictions within the threshold
within_threshold = (absolute_errors <= threshold).mean() * 100

print(f"Accuracy within ±${threshold:.2f}: {within_threshold:.2f}%")
