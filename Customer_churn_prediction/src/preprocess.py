import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(data_path):
    """Load and preprocess the telecom customer churn dataset"""
    # Load data
    df = pd.read_csv(data_path)

    # Convert TotalCharges to numeric, handling any spaces
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    df["TotalCharges"].fillna(0, inplace=True)  # Fill any NaN values with 0

    # Separate features and target
    X = df.drop(["Churn", "customerID"], axis=1, errors="ignore")
    y = df["Churn"].map({"Yes": 1, "No": 0}) if "Churn" in df.columns else None

    # Identify numerical and categorical columns
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_cols = [col for col in X.columns if col not in numerical_cols]

    # Handle categorical variables
    label_encoders = {}
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        X[col] = label_encoders[col].fit_transform(X[col])

    # Scale all features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y, label_encoders, scaler


def prepare_single_prediction(data, label_encoders, scaler):
    """Prepare a single customer's data for prediction"""
    # Convert to DataFrame if dict
    if isinstance(data, dict):
        data = pd.DataFrame([data])

    # Identify numerical and categorical columns
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_cols = [col for col in data.columns if col not in numerical_cols]

    # Encode categorical variables
    for col in categorical_cols:
        if col in label_encoders:
            data[col] = label_encoders[col].transform(data[col])

    # Scale features
    data = pd.DataFrame(scaler.transform(data), columns=data.columns)

    return data


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
