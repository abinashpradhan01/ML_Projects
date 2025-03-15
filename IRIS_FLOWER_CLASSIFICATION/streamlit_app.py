import streamlit as st
import joblib
import numpy as np
import pandas as pd


# Load the models and scaler
@st.cache_resource
def load_models():
    knn = joblib.load("models/knn_model.pkl")
    dt = joblib.load("models/dt_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return knn, dt, scaler


def main():
    st.title("🌸 Iris Flower Classification")
    st.write(
        """
    This app predicts the species of Iris flowers using both K-Nearest Neighbors (KNN) 
    and Decision Tree algorithms. Enter the measurements below to get predictions!
    """
    )

    try:
        knn, dt, scaler = load_models()

        # Create input fields
        st.subheader("Enter Flower Measurements")
        col1, col2 = st.columns(2)

        with col1:
            sepal_length = st.number_input(
                "Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0
            )
            sepal_width = st.number_input(
                "Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5
            )

        with col2:
            petal_length = st.number_input(
                "Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4
            )
            petal_width = st.number_input(
                "Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2
            )

        # Create a feature array
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Scale the features
        features_scaled = scaler.transform(features)

        # Make predictions
        if st.button("Predict Species"):
            # Get predictions from both models
            knn_pred = knn.predict(features_scaled)[0]
            dt_pred = dt.predict(features_scaled)[0]

            # Map predictions to species names
            species_names = ["Setosa", "Versicolor", "Virginica"]

            # Display predictions
            st.subheader("Predictions")
            col1, col2 = st.columns(2)

            with col1:
                st.info(f"KNN Prediction: **{species_names[knn_pred]}**")
                st.write(f"Probability Distribution:")
                knn_probs = knn.predict_proba(features_scaled)[0]
                prob_df = pd.DataFrame(
                    {"Species": species_names, "Probability": knn_probs}
                )
                st.dataframe(prob_df)

            with col2:
                st.info(f"Decision Tree Prediction: **{species_names[dt_pred]}**")
                st.write(f"Probability Distribution:")
                dt_probs = dt.predict_proba(features_scaled)[0]
                prob_df = pd.DataFrame(
                    {"Species": species_names, "Probability": dt_probs}
                )
                st.dataframe(prob_df)

    except FileNotFoundError:
        st.error(
            """
        Models not found! Please run the model training script first:
        ```
        python model.py
        ```
        """
        )


if __name__ == "__main__":
    main()
