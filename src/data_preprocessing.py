import pandas as pd
from sklearn.impute import SimpleImputer

def preprocess_data(input_path, output_path):
    data = pd.read_csv(input_path)
    data = data.drop(['City', 'Date', 'AQI_Bucket'], axis=1)
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    data_imputed.to_csv(output_path, index=False)