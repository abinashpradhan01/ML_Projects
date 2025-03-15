import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


class SimpleLogisticRegression:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        probabilities = self.sigmoid(linear_model)
        # Return probabilities for both classes in the format [[1-p, p], ...]
        return np.column_stack([1 - probabilities, probabilities])


def load_models():
    """Load trained models and preprocessing objects"""
    try:
        models_dir = "models"

        # Load custom model state and create a simple model
        custom_model_state = joblib.load(os.path.join(models_dir, "custom_model.pkl"))
        custom_model = SimpleLogisticRegression(
            weights=custom_model_state["weights"], bias=custom_model_state["bias"]
        )

        sklearn_model = joblib.load(os.path.join(models_dir, "sklearn_model.pkl"))
        label_encoders = joblib.load(os.path.join(models_dir, "label_encoders.pkl"))
        scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        return custom_model, sklearn_model, label_encoders, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error("Please make sure you've run 'python src/train.py' first")
        return None, None, None, None


def main():
    st.title("Customer Churn Prediction")
    st.write("Enter customer details to predict churn probability")

    models = load_models()
    if not all(models):
        return

    custom_model, sklearn_model, label_encoders, scaler = models

    # Create input form
    with st.form("prediction_form"):
        # Create three columns for better layout
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Basic Info")
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0)

            st.subheader("Contract Details")
            contract = st.selectbox(
                "Contract Type", ["Month-to-month", "One year", "Two year"]
            )
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        with col2:
            st.subheader("Internet Services")
            internet_service = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )
            online_security = st.selectbox(
                "Online Security", ["Yes", "No", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online Backup", ["Yes", "No", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device Protection", ["Yes", "No", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech Support", ["Yes", "No", "No internet service"]
            )

        with col3:
            st.subheader("Additional Services")
            multiple_lines = st.selectbox(
                "Multiple Lines", ["Yes", "No", "No phone service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", ["Yes", "No", "No internet service"]
            )

            st.subheader("Personal Info")
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            gender = st.selectbox("Gender", ["Female", "Male"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

        submit_button = st.form_submit_button("Predict Churn")

        if submit_button:
            # Prepare input data
            input_data = {
                # Numerical features
                "tenure": tenure,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
                # Categorical features
                "Contract": contract,
                "PaymentMethod": payment_method,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "MultipleLines": multiple_lines,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
                "Dependents": dependents,
                "Partner": partner,
                "PaperlessBilling": paperless_billing,
                "PhoneService": phone_service,
                "gender": gender,
            }

            # Create DataFrame
            input_df = pd.DataFrame([input_data])

            # Ensure the column order matches what was used during training
            # This is critical to avoid the ValueError
            if hasattr(scaler, "feature_names_in_"):
                # Get the feature names used during training
                train_features = scaler.feature_names_in_.tolist()

                # Transform categorical features first
                for col in input_df.columns:
                    if col in label_encoders:
                        input_df[col] = label_encoders[col].transform(input_df[col])

                # Create a new DataFrame with the exact same columns in the same order
                ordered_df = pd.DataFrame(columns=train_features)
                for col in train_features:
                    if col in input_df.columns:
                        ordered_df[col] = input_df[col]
                    else:
                        st.error(f"Missing feature: {col}")
                        return

                # Now scale the correctly ordered features
                input_scaled = scaler.transform(ordered_df)
            else:
                st.error(
                    "Scaler doesn't have feature_names_in_ attribute. Please retrain the model."
                )
                return

            # Make predictions
            custom_prob = custom_model.predict_proba(input_scaled)[0][
                1
            ]  # Get probability for class 1
            sklearn_prob = sklearn_model.predict_proba(input_scaled)[0][1]

            # Display results
            st.subheader("Churn Probability Predictions")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Custom Model", f"{custom_prob:.2%}")
            with col2:
                st.metric("Scikit-learn Model", f"{sklearn_prob:.2%}")

            # Interpretation
            risk_level = "High" if (custom_prob + sklearn_prob) / 2 > 0.5 else "Low"
            st.write(f"**Churn Risk Level:** {risk_level}")

            if risk_level == "High":
                st.warning(
                    "This customer has a high risk of churning. Consider implementing retention strategies."
                )

                # Add retention recommendations
                st.subheader("Recommended Retention Strategies")
                if monthly_charges > 70:
                    st.write("- Consider offering a discount on monthly charges")
                if contract == "Month-to-month":
                    st.write("- Propose a long-term contract with better rates")
                if internet_service == "Fiber optic" and tech_support == "No":
                    st.write("- Offer complimentary tech support")
                if not any(
                    [
                        online_security == "Yes",
                        online_backup == "Yes",
                        device_protection == "Yes",
                    ]
                ):
                    st.write("- Suggest a security and protection package")
            else:
                st.success("This customer has a low risk of churning.")


if __name__ == "__main__":
    main()
