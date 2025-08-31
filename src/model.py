from sklearn.ensemble import RandomForestRegressor

def get_model():
    return RandomForestRegressor(random_state=42)