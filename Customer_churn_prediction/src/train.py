import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import joblib
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.custom_model import CustomLogisticRegression
from src.preprocess import load_and_preprocess_data, split_data


def train_models(data_path):
    """Train both custom and scikit-learn models"""
    print("Loading and preprocessing data...")
    # Load and preprocess data
    X, y, label_encoders, scaler = load_and_preprocess_data(data_path)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training custom model...")
    # Train custom model
    custom_model = CustomLogisticRegression(learning_rate=0.01, n_iterations=1000)
    custom_model.fit(X_train.values, y_train.values)

    print("Training scikit-learn model...")
    # Train scikit-learn model
    sklearn_model = LogisticRegression(random_state=42, max_iter=1000)
    sklearn_model.fit(X_train, y_train)

    # Evaluate models
    custom_pred = custom_model.predict(X_test.values)
    sklearn_pred = sklearn_model.predict(X_test)

    print("Custom Model Performance:")
    print(classification_report(y_test, custom_pred))
    print("\nScikit-learn Model Performance:")
    print(classification_report(y_test, sklearn_pred))

    print("Saving models...")
    # Save models and preprocessing objects
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    # Save the custom model attributes instead of the whole object
    custom_model_state = {
        "weights": custom_model.weights,
        "bias": custom_model.bias,
        "learning_rate": custom_model.learning_rate,
        "n_iterations": custom_model.n_iterations,
    }
    joblib.dump(custom_model_state, os.path.join(models_dir, "custom_model.pkl"))
    joblib.dump(sklearn_model, os.path.join(models_dir, "sklearn_model.pkl"))
    joblib.dump(label_encoders, os.path.join(models_dir, "label_encoders.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

    print("Models saved successfully!")
    return custom_model, sklearn_model, label_encoders, scaler


if __name__ == "__main__":
    data_path = "data/customer_churn.csv"
    train_models(data_path)
