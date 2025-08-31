import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

def evaluate_model(data_path, model_path):
    data = pd.read_csv(data_path)
    X = data.drop('AQI', axis=1)
    y = data['AQI']
    model = joblib.load(model_path)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print(f"Evaluation Mean Squared Error: {mse}")