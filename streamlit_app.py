import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import os

# Page configuration
st.set_page_config(
    page_title="Road Accident Probability Predictor",
    page_icon="🚗",
    layout="wide"
)

# Load model function
@st.cache_resource
def load_model():
    """Load the trained model and encoders"""
    try:
        model_data = joblib.load('accident_prediction_model.pkl')
        return model_data['model'], model_data['label_encoders'], model_data['feature_columns']
    except FileNotFoundError:
        st.error("Model file not found! Please run train_model.py first.")
        st.stop()

# Load data to get unique values for dropdowns
@st.cache_data
def load_data_for_options():
    """Load data to get unique values for dropdown options"""
    try:
        df = pd.read_csv('final_dataframe.csv')
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        return df
    except:
        return None

def extract_hour(time_str):
    """Extract hour from time string"""
    try:
        if ':' in str(time_str):
            hour = int(str(time_str).split(':')[0])
            return hour
        return 12
    except:
        return 12

def prepare_input_features(user_input, label_encoders, feature_columns):
    """Prepare user input for prediction"""
    # Create a dataframe with the input features
    input_data = {}
    
    # Map user inputs to feature columns
    feature_mapping = {
        'State Name': user_input['state'],
        'City Name': user_input['city'],
        'Year': user_input['year'],
        'Month': user_input['month'],
        'Day of Week': user_input['day_of_week'],
        'Time of Day': user_input['time_of_day'],
        'Weather Conditions': user_input['weather'],
        'Road Type': user_input['road_type'],
        'Road Condition': user_input['road_condition'],
        'Lighting Conditions': user_input['lighting'],
        'Traffic Control Presence': user_input['traffic_control'],
        'Speed Limit (km/h)': user_input['speed_limit'],
        'Driver Age': user_input['driver_age'],
        'Driver Gender': user_input['driver_gender'],
        'Driver License Status': user_input['license_status'],
        'Alcohol Involvement': user_input['alcohol'],
        'Accident Location Details': user_input['location'],
        'Number of Vehicles Involved': user_input['vehicles'],
        'Rainfall_mm': user_input['rainfall'],
        'Max_temp_celsius': user_input['max_temp'],
        'Min_temp_celsius': user_input['min_temp'],
        'Traffic Congestion': user_input['traffic_congestion']
    }
    
    # Handle Time of Day - convert to Hour first (before creating dataframe)
    # Check if Hour is in feature_columns (model expects Hour, not Time of Day)
    if 'Hour' in feature_columns:
        # Model expects Hour, so create it directly
        input_data['Hour'] = [extract_hour(user_input['time_of_day'])]
        # Don't include Time of Day if Hour is expected
        if 'Time of Day' in feature_mapping:
            del feature_mapping['Time of Day']
    elif 'Time of Day' in feature_columns:
        # Old model format - keep Time of Day
        input_data['Time of Day'] = [user_input['time_of_day']]
    
    # Only include features that are in feature_columns
    for col in feature_columns:
        if col in feature_mapping:
            input_data[col] = [feature_mapping[col]]
    
    df_input = pd.DataFrame(input_data)
    
    # Encode categorical features
    for col in df_input.columns:
        if col in label_encoders:
            le = label_encoders[col]
            # Handle unseen values
            value = str(df_input[col].iloc[0])
            # Handle empty strings and NaN
            if value == 'nan' or value == '' or value == 'None':
                value = 'Unknown'
            if value not in le.classes_:
                # Use the most common class as fallback
                value = le.classes_[0]
            try:
                df_input[col] = le.transform([value])[0]
            except:
                df_input[col] = 0
    
    # If Time of Day is still in df_input, convert it to Hour
    if 'Time of Day' in df_input.columns and 'Hour' in feature_columns:
        df_input['Hour'] = extract_hour(user_input['time_of_day'])
        df_input = df_input.drop('Time of Day', axis=1)
    
    # Ensure all feature columns are present (use feature_columns which has the actual model features)
    for col in feature_columns:
        if col not in df_input.columns:
            # Default values for missing features
            if col == 'Hour':
                df_input[col] = extract_hour(user_input['time_of_day'])
            else:
                df_input[col] = 0
    
    # Reorder columns to match training data exactly
    df_input = df_input[feature_columns]
    
    return df_input

def main():
    """Main Streamlit application"""
    st.title("🚗 Road Accident Probability Predictor")
    st.markdown("### Predict the probability of a road accident based on various factors")
    
    # Load model
    try:
        model, label_encoders, feature_columns = load_model()
        st.sidebar.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    # Load data for dropdown options
    df_data = load_data_for_options()
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application predicts the probability of a road accident 
        based on various factors including:
        - Location (State, City)
        - Weather conditions
        - Road conditions
        - Driver information
        - Traffic conditions
        - Environmental factors
        """)
        st.markdown("---")
        st.markdown("**Model:** Random Forest Classifier")
        st.markdown("**Accuracy:** ~95.5%")
    
    # Main input form
    st.header("📋 Input Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Location
        st.subheader("📍 Location")
        if df_data is not None:
            if 'State Name' in df_data.columns:
                states = sorted([str(s) for s in df_data['State Name'].dropna().unique()])
                state = st.selectbox("State", states, index=0 if states else None)
            else:
                states = []
                state = None
            
            if state and 'City Name' in df_data.columns:
                cities = sorted([str(c) for c in df_data[df_data['State Name'] == state]['City Name'].dropna().unique()])
                city = st.selectbox("City", cities, index=0 if cities else None)
            else:
                city = st.text_input("City", "Hyderabad")
        else:
            state = st.text_input("State", "Telangana")
            city = st.text_input("City", "Hyderabad")
        
        # Date and Time
        st.subheader("📅 Date & Time")
        year = st.number_input("Year", min_value=2018, max_value=2025, value=2023)
        month = st.selectbox("Month", 
            ['January', 'February', 'March', 'April', 'May', 'June',
             'July', 'August', 'September', 'October', 'November', 'December'],
            index=4)
        day_of_week = st.selectbox("Day of Week",
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            index=0)
        time_of_day = st.text_input("Time of Day (HH:MM)", "12:00")
        
        # Weather
        st.subheader("🌤️ Weather Conditions")
        if df_data is not None and 'Weather Conditions' in df_data.columns:
            weather_options = sorted([str(w) for w in df_data['Weather Conditions'].dropna().unique()])
            weather = st.selectbox("Weather Conditions", weather_options, index=0 if weather_options else None)
        else:
            weather = st.selectbox("Weather Conditions",
                ['Clear', 'Rainy', 'Foggy', 'Hazy', 'Stormy'],
                index=0)
        
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=0.0, step=0.1)
        max_temp = st.number_input("Maximum Temperature (°C)", min_value=-20.0, max_value=50.0, value=30.0, step=0.1)
        min_temp = st.number_input("Minimum Temperature (°C)", min_value=-20.0, max_value=50.0, value=20.0, step=0.1)
    
    with col2:
        # Road Conditions
        st.subheader("🛣️ Road Conditions")
        if df_data is not None and 'Road Type' in df_data.columns:
            road_type_options = sorted([str(r) for r in df_data['Road Type'].dropna().unique()])
            road_type = st.selectbox("Road Type", road_type_options, index=0 if road_type_options else None)
        else:
            road_type = st.selectbox("Road Type",
                ['National Highway', 'State Highway', 'Urban Road', 'Village Road'],
                index=0)
        
        if df_data is not None and 'Road Condition' in df_data.columns:
            road_condition_options = sorted([str(r) for r in df_data['Road Condition'].dropna().unique()])
            road_condition = st.selectbox("Road Condition", road_condition_options, index=0 if road_condition_options else None)
        else:
            road_condition = st.selectbox("Road Condition",
                ['Dry', 'Wet', 'Damaged', 'Under Construction'],
                index=0)
        
        if df_data is not None and 'Lighting Conditions' in df_data.columns:
            lighting_options = sorted([str(l) for l in df_data['Lighting Conditions'].dropna().unique()])
            lighting = st.selectbox("Lighting Conditions", lighting_options, index=0 if lighting_options else None)
        else:
            lighting = st.selectbox("Lighting Conditions",
                ['Daylight', 'Dusk', 'Dawn', 'Dark'],
                index=0)
        
        if df_data is not None and 'Traffic Control Presence' in df_data.columns:
            # Get unique values, filter out NaN and empty strings, convert to strings
            unique_values = df_data['Traffic Control Presence'].dropna().unique()
            traffic_control_options = sorted([str(opt) for opt in unique_values if pd.notna(opt) and str(opt).strip() != ''])
            if traffic_control_options:
                traffic_control = st.selectbox("Traffic Control Presence", 
                    [''] + traffic_control_options, index=0)
            else:
                traffic_control = st.selectbox("Traffic Control Presence",
                    ['', 'Signs', 'Signals', 'Police Checkpost'],
                    index=0)
        else:
            traffic_control = st.selectbox("Traffic Control Presence",
                ['', 'Signs', 'Signals', 'Police Checkpost'],
                index=0)
        
        speed_limit = st.number_input("Speed Limit (km/h)", min_value=0, max_value=200, value=60)
        
        # Driver Information
        st.subheader("👤 Driver Information")
        driver_age = st.number_input("Driver Age", min_value=18, max_value=100, value=35)
        driver_gender = st.selectbox("Driver Gender", ['Male', 'Female'], index=0)
        license_status = st.selectbox("Driver License Status",
            ['Valid', 'Expired', 'No License'], index=0)
        alcohol = st.selectbox("Alcohol Involvement", ['Yes', 'No'], index=1)
        
        # Accident Location
        st.subheader("📍 Accident Location Details")
        if df_data is not None and 'Accident Location Details' in df_data.columns:
            location_options = sorted([str(l) for l in df_data['Accident Location Details'].dropna().unique()])
            location = st.selectbox("Location Type", location_options, index=0 if location_options else None)
        else:
            location = st.selectbox("Location Type",
                ['Straight Road', 'Curve', 'Intersection', 'Bridge'],
                index=0)
        
        # Traffic
        st.subheader("🚦 Traffic Conditions")
        vehicles = st.number_input("Number of Vehicles Involved", min_value=1, max_value=10, value=2)
        traffic_congestion = st.slider("Traffic Congestion (0.0 - 1.0)", 
            min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    
    # Prediction button
    st.markdown("---")
    if st.button("🔮 Predict Accident Probability", type="primary", use_container_width=True):
        # Prepare input
        user_input = {
            'state': state,
            'city': city,
            'year': year,
            'month': month,
            'day_of_week': day_of_week,
            'time_of_day': time_of_day,
            'weather': weather,
            'road_type': road_type,
            'road_condition': road_condition,
            'lighting': lighting,
            'traffic_control': traffic_control if traffic_control else '',
            'speed_limit': speed_limit,
            'driver_age': driver_age,
            'driver_gender': driver_gender,
            'license_status': license_status,
            'alcohol': alcohol,
            'location': location,
            'vehicles': vehicles,
            'rainfall': rainfall,
            'max_temp': max_temp,
            'min_temp': min_temp,
            'traffic_congestion': traffic_congestion
        }
        
        try:
            # Prepare features
            input_features = prepare_input_features(user_input, label_encoders, feature_columns)
            
            # Make prediction
            prediction = model.predict(input_features)[0]
            probability = model.predict_proba(input_features)[0]
            
            # Display results
            st.markdown("---")
            st.header("📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Risk Level", "High Risk" if prediction == 1 else "Low Risk")
            
            with col2:
                accident_prob = probability[1] if len(probability) > 1 else probability[0]
                st.metric("Accident Probability", f"{accident_prob * 100:.2f}%")
            
            with col3:
                st.metric("Safety Score", f"{(1 - accident_prob) * 100:.2f}%")
            
            # Progress bar
            st.markdown("### Probability Visualization")
            st.progress(accident_prob)
            
            # Interpretation
            st.markdown("---")
            if accident_prob >= 0.7:
                st.error("⚠️ **High Risk**: There is a high probability of an accident. Please exercise extreme caution!")
            elif accident_prob >= 0.4:
                st.warning("⚡ **Medium Risk**: Moderate probability of an accident. Drive carefully.")
            else:
                st.success("✅ **Low Risk**: Low probability of an accident. Safe driving conditions.")
            
            # Feature importance (if available)
            try:
                feature_importance = pd.read_csv('feature_importance.csv')
                st.markdown("---")
                st.markdown("### 📈 Top Contributing Factors")
                top_factors = feature_importance.head(5)
                for idx, row in top_factors.iterrows():
                    st.markdown(f"- **{row['feature']}**: {row['importance']:.4f}")
            except:
                pass
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.exception(e)

if __name__ == "__main__":
    main()

