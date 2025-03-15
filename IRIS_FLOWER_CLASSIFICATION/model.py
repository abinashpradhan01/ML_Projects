import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import seaborn as sns
import matplotlib.pyplot as plt


def load_and_preprocess_data():
    # Load the Iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="species")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler
    joblib.dump(scaler, "models/scaler.pkl")

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_and_evaluate_models():
    # Create models directory if it doesn't exist
    import os

    os.makedirs("models", exist_ok=True)

    # Load and preprocess data
    X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess_data()

    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    knn_pred = knn.predict(X_test_scaled)

    # Train Decision Tree model
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_scaled, y_train)
    dt_pred = dt.predict(X_test_scaled)

    # Save models
    joblib.dump(knn, "models/knn_model.pkl")
    joblib.dump(dt, "models/dt_model.pkl")

    # Evaluate models
    models = {"KNN": (knn, knn_pred), "Decision Tree": (dt, dt_pred)}

    results = {}
    for name, (model, predictions) in models.items():
        accuracy = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions)

        results[name] = {
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix,
            "classification_report": report,
        }

        # Plot and save confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.savefig(f'models/{name.lower().replace(" ", "_")}_confusion_matrix.png')
        plt.close()

    return results


if __name__ == "__main__":
    results = train_and_evaluate_models()

    # Print results
    for model_name, metrics in results.items():
        print(f"\n{'-'*50}")
        print(f"{model_name} Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("\nClassification Report:")
        print(metrics["classification_report"])
