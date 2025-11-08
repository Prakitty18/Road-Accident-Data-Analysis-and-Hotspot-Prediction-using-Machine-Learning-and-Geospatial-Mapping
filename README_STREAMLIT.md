# Road Accident Probability Predictor

This application predicts the probability of road accidents based on various factors using a trained Random Forest model.

## Files

- `train_model.py` - Script to train the machine learning model
- `streamlit_app.py` - Streamlit web application for predictions
- `accident_prediction_model.pkl` - Trained model file (generated after training)
- `feature_importance.csv` - Feature importance scores (generated after training)
- `final_dataframe.csv` - Training dataset

## Setup

1. Install required packages:
```bash
pip install streamlit pandas numpy scikit-learn joblib
```

## Training the Model

Before running the Streamlit app, make sure the model is trained:

```bash
python train_model.py
```

This will:
- Load data from `final_dataframe.csv`
- Train a Random Forest Classifier
- Save the model to `accident_prediction_model.pkl`
- Generate feature importance to `feature_importance.csv`
- Display model accuracy and performance metrics

**Model Performance:**
- Training Accuracy: ~99.96%
- Test Accuracy: ~95.5%

## Running the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Features

The Streamlit app allows you to input:

1. **Location**: State and City
2. **Date & Time**: Year, Month, Day of Week, Time
3. **Weather**: Weather conditions, Rainfall, Temperature
4. **Road Conditions**: Road Type, Condition, Lighting, Traffic Control
5. **Driver Information**: Age, Gender, License Status, Alcohol Involvement
6. **Traffic**: Number of Vehicles, Traffic Congestion Level

## Output

The app provides:
- **Risk Level**: High Risk or Low Risk
- **Accident Probability**: Percentage probability (0-100%)
- **Safety Score**: Inverse of accident probability
- **Visualization**: Progress bar showing probability
- **Interpretation**: Risk assessment with recommendations

## Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: 22 input features
- **Target**: Binary classification (High Risk / Low Risk)
- **Top Features**:
  1. Speed Limit
  2. Minimum Temperature
  3. Rainfall
  4. Maximum Temperature
  5. City Name

## Notes

- The model predicts the probability of a high-risk accident occurring
- All predictions are based on the trained model and historical data patterns
- Results should be used as guidance, not absolute predictions

