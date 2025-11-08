import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import joblib
import os

def load_and_prepare_data(csv_path='final_dataframe.csv'):
    """
    Load and prepare data for model training
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Drop the index column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df

def create_target_variable(df):
    """
    Create target variable for accident probability prediction.
    We'll predict accident severity and convert it to probability.
    """
    # Create severity categories
    def categorize_severity(casualties, fatalities):
        if casualties == 0 and fatalities == 0:
            return 0  # No accident risk
        elif casualties <= 2 and fatalities == 0:
            return 1  # Low risk
        elif casualties <= 5 or fatalities <= 2:
            return 2  # Medium risk
        else:
            return 3  # High risk
    
    df['Accident_Risk_Level'] = df.apply(
        lambda x: categorize_severity(x['Number of Casualties'], x['Number of Fatalities']), 
        axis=1
    )
    
    # Also create a binary target: High Risk (1) vs Low Risk (0)
    df['High_Risk'] = (df['Accident_Risk_Level'] >= 2).astype(int)
    
    # Create probability score (0-1) based on severity
    df['Accident_Probability'] = (
        (df['Number of Casualties'] / df['Number of Casualties'].max()) * 0.6 +
        (df['Number of Fatalities'] / df['Number of Fatalities'].max()) * 0.4
    ).clip(0, 1)
    
    return df

def encode_categorical_features(df, categorical_columns, label_encoders=None, fit=True):
    """
    Encode categorical features using LabelEncoder
    """
    if label_encoders is None:
        label_encoders = {}
    
    df_encoded = df.copy()
    
    for col in categorical_columns:
        if col in df_encoded.columns:
            if fit:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str).fillna('Unknown'))
                label_encoders[col] = le
            else:
                if col in label_encoders:
                    le = label_encoders[col]
                    # Handle unseen values
                    unique_values = set(df_encoded[col].astype(str).fillna('Unknown').unique())
                    known_values = set(le.classes_)
                    for val in unique_values:
                        if val not in known_values:
                            # Add to encoder
                            le.classes_ = np.append(le.classes_, val)
                    df_encoded[col] = le.transform(df_encoded[col].astype(str).fillna('Unknown'))
    
    return df_encoded, label_encoders

def prepare_features(df, label_encoders=None, fit=True):
    """
    Prepare features for model training
    """
    # Select relevant features
    feature_columns = [
        'State Name',
        'City Name',
        'Year',
        'Month',
        'Day of Week',
        'Time of Day',
        'Weather Conditions',
        'Road Type',
        'Road Condition',
        'Lighting Conditions',
        'Traffic Control Presence',
        'Speed Limit (km/h)',
        'Driver Age',
        'Driver Gender',
        'Driver License Status',
        'Alcohol Involvement',
        'Accident Location Details',
        'Number of Vehicles Involved',
        'Rainfall_mm',
        'Max_temp_celsius',
        'Min_temp_celsius',
        'Traffic Congestion'
    ]
    
    # Filter to only columns that exist
    available_features = [col for col in feature_columns if col in df.columns]
    
    # Categorical columns
    categorical_cols = [
        'State Name', 'City Name', 'Month', 'Day of Week', 'Time of Day',
        'Weather Conditions', 'Road Type', 'Road Condition', 'Lighting Conditions',
        'Traffic Control Presence', 'Driver Gender', 'Driver License Status',
        'Alcohol Involvement', 'Accident Location Details'
    ]
    
    categorical_cols = [col for col in categorical_cols if col in available_features]
    
    # Encode categorical features
    df_encoded, label_encoders = encode_categorical_features(
        df, categorical_cols, label_encoders, fit=fit
    )
    
    # Extract features
    X = df_encoded[available_features].copy()
    
    # Handle missing values
    X = X.fillna(X.median() if fit else 0)
    
    # Convert Time of Day to numeric (extract hour)
    if 'Time of Day' in X.columns:
        def extract_hour(time_str):
            try:
                if ':' in str(time_str):
                    hour = int(str(time_str).split(':')[0])
                    return hour
                return 12  # Default to noon
            except:
                return 12
        X['Hour'] = X['Time of Day'].apply(extract_hour)
        X = X.drop('Time of Day', axis=1)
        # Update available_features to reflect the change
        if 'Time of Day' in available_features:
            available_features = [col if col != 'Time of Day' else 'Hour' for col in available_features]
    
    # Return the actual column names from X (which the model will be trained on)
    actual_feature_columns = list(X.columns)
    return X, label_encoders, actual_feature_columns

def train_model(X, y, test_size=0.2, random_state=42):
    """
    Train Random Forest model
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train Random Forest Classifier
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\n=== Model Performance ===")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Classification report
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_test_pred))
    
    # Confusion matrix
    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== Top 10 Feature Importance ===")
    print(feature_importance.head(10))
    
    # ROC AUC for binary classification
    if len(np.unique(y)) == 2:
        try:
            roc_auc = roc_auc_score(y_test, y_test_proba[:, 1])
            print(f"\nROC AUC Score: {roc_auc:.4f}")
        except:
            pass
    
    return model, X_train, X_test, y_train, y_test, feature_importance

def save_model(model, label_encoders, feature_columns, model_path='accident_prediction_model.pkl'):
    """
    Save the trained model and encoders
    """
    model_data = {
        'model': model,
        'label_encoders': label_encoders,
        'feature_columns': feature_columns
    }
    
    joblib.dump(model_data, model_path)
    print(f"\nModel saved to {model_path}")

def main():
    """
    Main training function
    """
    # Load data
    df = load_and_prepare_data('final_dataframe.csv')
    
    # Create target variable
    df = create_target_variable(df)
    
    # Prepare features
    X, label_encoders, feature_columns = prepare_features(df, fit=True)
    
    # Use High_Risk as target (binary classification)
    y = df['High_Risk']
    
    print(f"\nTarget distribution:")
    print(y.value_counts())
    
    # Train model
    model, X_train, X_test, y_train, y_test, feature_importance = train_model(X, y)
    
    # Save model
    save_model(model, label_encoders, feature_columns, 'accident_prediction_model.pkl')
    
    # Also save feature importance
    feature_importance.to_csv('feature_importance.csv', index=False)
    print("Feature importance saved to feature_importance.csv")
    
    print("\n=== Training Complete ===")
    return model, label_encoders, feature_columns

if __name__ == "__main__":
    model, label_encoders, feature_columns = main()

