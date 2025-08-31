# Air Quality Prediction

## Overview
This project predicts air quality using historical pollution data, weather conditions, and traffic data. It uses a machine learning model to forecast AQI (Air Quality Index) for future dates, helping individuals and governments plan outdoor activities and take preventive measures.

## Features
- **Data Preprocessing**: Handles missing values and prepares the dataset for modeling.
- **Model Training**: Trains a Random Forest Regressor to predict AQI values.
- **Future Predictions**: Predicts AQI for the next 30 days based on placeholder features.
- **Visualization**: Plots historical AQI values and future predictions.

---

## Project Structure
```
air-quality-prediction/
├── data/
│   ├── raw/
│   │   └── city_day.csv          # Raw dataset
│   ├── processed/
│   │   ├── processed_data.csv    # Processed dataset
│   │   └── predictions_with_actual.csv # Predictions with actual values
├── models/
│   └── trained_model.pkl         # Trained Random Forest model
├── notebooks/
│   ├── data_exploration.ipynb    # Jupyter notebook for data exploration
│   ├── data_preprocessing.ipynb  # Jupyter notebook for data preprocessing
│   └── model_training.ipynb      # Jupyter notebook for model training
├── run_model.py                  # Script for future predictions and visualization
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Libraries listed in `requirements.txt`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Air-Quality-Prediction.git
   cd Air-Quality-Prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. **Data Preprocessing**
Run the `data_preprocessing.ipynb` notebook to preprocess the raw dataset (`city_day.csv`). This will generate `processed_data.csv` in the `data/processed` folder.

### 2. **Model Training**
Run the `model_training.ipynb` notebook to train the Random Forest model. The trained model will be saved as `trained_model.pkl` in the `models` folder.

### 3. **Future Predictions**
Run the `run_model.py` script to predict AQI for the next 30 days and visualize the results:
```bash
python run_model.py
```
The script will:
- Save predictions to `data/processed/predictions_with_actual.csv`.
- Display a plot comparing historical AQI values and future predictions.

---

## Example Output

### Visualization
![Historical and Future AQI Predictions](https://via.placeholder.com/800x400.png)  
*Historical AQI values (blue) and future predictions (orange).*

### Metrics
- **Mean Absolute Error (MAE)**: 21.48
- **Mean Squared Error (MSE)**: 1736.91
- **R² Score**: 89.47%

---

## Dataset
The dataset used for this project is `city_day.csv`, which contains air quality data for various cities over time. Columns include:
- **Date**: The date of observation.
- **AQI**: Air Quality Index.
- **Pollutants**: PM2.5, PM10, NO, NO2, etc.

---

## Dependencies
The project uses the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `joblib`

Install them using:
```bash
pip install -r requirements.txt
```

---

## Future Improvements
- **Hyperparameter Tuning**: Optimize the Random Forest model for better accuracy.
- **Feature Engineering**: Add more dynamic features for future predictions.
- **Deployment**: Create a web application using Flask or FastAPI to make predictions accessible online.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Author
Developed by Piyush Phule (https://github.com/NilfGuardian).

---

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

---

## Acknowledgments
- Dataset source: [Central Pollution Control Board (CPCB)](https://cpcb.nic.in/)
- Libraries: [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/)
