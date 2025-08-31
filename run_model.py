import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the trained model
model = joblib.load('./models/trained_model.pkl')

# Load the original dataset with the Date column
data = pd.read_csv('./data/raw/city_day.csv')

# Ensure the Date column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Sort the data by Date
data = data.sort_values(by='Date')

# Prepare the historical data
historical_data = data[['Date', 'AQI']].dropna()

# Generate future timestamps
last_date = historical_data['Date'].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')  # Predict next 30 days

# Load the processed data
processed_data = pd.read_csv('./data/processed/processed_data.csv')

# Split into features and target
X = processed_data.drop('AQI', axis=1)
y = processed_data['AQI']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

# Generate future features by adding a trend and random noise
last_row = processed_data.drop('AQI', axis=1).iloc[-1]
trend = np.linspace(0, 5, 30)  # Simulate a gradual increase or decrease
future_features = pd.DataFrame(
    [last_row + trend[i] + np.random.normal(0, 1, len(last_row)) for i in range(30)],
    columns=processed_data.drop('AQI', axis=1).columns
)

# Predict future AQI values
future_predictions = model.predict(future_features)

# Combine future dates and predictions into a DataFrame
future_data = pd.DataFrame({'Date': future_dates, 'Predicted_AQI': future_predictions})

# Visualization: Historical and Future AQI
fig, ax = plt.subplots(2, 1, figsize=(16, 12))

# Historical AQI
ax[0].plot(historical_data['Date'], historical_data['AQI'], label='Historical AQI', color='blue', marker='o')
ax[0].set_title('Historical AQI')
ax[0].set_xlabel('Date')
ax[0].set_ylabel('AQI')
ax[0].grid(True, linestyle='--', alpha=0.6)

# Future Predictions
ax[1].plot(future_data['Date'], future_data['Predicted_AQI'], label='Future Predictions', linestyle='--', marker='x', color='orange')
ax[1].set_title('Future AQI Predictions')
ax[1].set_xlabel('Date')
ax[1].set_ylabel('AQI')
ax[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()