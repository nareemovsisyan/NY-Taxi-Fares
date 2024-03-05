import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load your DataFrame
# Replace 'your_data.csv' with the actual path or filename of your dataset
df = pd.read_csv('for knn.csv')

# Extract relevant features for modeling (excluding payment-related columns)
features = ['rate_code', 'passenger_count', 'trip_time_in_secs', 'trip_distance', 'pickup_day', 'pickup_hour']
target = 'fare_amount'

X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Regressor model with hyperparameters
random_forest_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)

# Train the model
random_forest_model.fit(X_train, y_train)

# Make predictions on the test set
rf_predictions = random_forest_model.predict(X_test)

# Evaluate the model
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)

print(f'Random Forest Mean Squared Error: {rf_mse}')
print(f'Random Forest Mean Absolute Error: {rf_mae}')

# Now you can use the trained Random Forest model to make predictions on new data
# For example:
new_data = pd.DataFrame({
    'rate_code': [1],
    'passenger_count': [1],
    'trip_time_in_secs': [600],
    'trip_distance': [2.5],
    'pickup_day': [1],
    'pickup_hour': [12]
})

rf_fare_prediction = random_forest_model.predict(new_data)
print(f'Random Forest Predicted Fare Amount: {rf_fare_prediction[0]}')
