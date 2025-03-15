import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import traceback
from PIL import Image
import base64


# Set page configuration
st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


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
        models_dir = os.path.join(os.path.dirname(__file__), "models")

        # Check if models directory exists
        if not os.path.exists(models_dir):
            st.error(f"Models directory not found: {models_dir}")
            st.error("Please run 'python src/train.py' first to train and save models")
            return None, None, None, None

        # Load custom model state and create a simple model
        custom_model_path = os.path.join(models_dir, "custom_model.pkl")
        if not os.path.exists(custom_model_path):
            st.error(f"Custom model file not found: {custom_model_path}")
            return None, None, None, None

        custom_model_state = joblib.load(custom_model_path)
        custom_model = SimpleLogisticRegression(
            weights=custom_model_state["weights"], bias=custom_model_state["bias"]
        )

        # Load scikit-learn model
        sklearn_model_path = os.path.join(models_dir, "sklearn_model.pkl")
        if not os.path.exists(sklearn_model_path):
            st.error(f"Scikit-learn model file not found: {sklearn_model_path}")
            return None, None, None, None

        sklearn_model = joblib.load(sklearn_model_path)

        # Load label encoders
        label_encoders_path = os.path.join(models_dir, "label_encoders.pkl")
        if not os.path.exists(label_encoders_path):
            st.error(f"Label encoders file not found: {label_encoders_path}")
            return None, None, None, None

        label_encoders = joblib.load(label_encoders_path)

        # Load scaler
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        if not os.path.exists(scaler_path):
            st.error(f"Scaler file not found: {scaler_path}")
            return None, None, None, None

        scaler = joblib.load(scaler_path)

        return custom_model, sklearn_model, label_encoders, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error(traceback.format_exc())
        st.error("Please make sure you've run 'python src/train.py' first")
        return None, None, None, None


def display_sidebar_info():
    """Display information in the sidebar"""
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This app predicts customer churn for a telecom company using 
        both custom-implemented and scikit-learn logistic regression models.
        
        **Data Source:** 
        [Kaggle: Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
        
        **Features Used:**
        - Customer demographics
        - Account information
        - Services subscribed
        - Charges
        """
    )

    st.sidebar.title("Prediction Details")
    st.sidebar.info(
        """
        Enter customer details in the form and click "Predict Churn" to:
        
        1. View churn probability from both models
        2. See risk assessment
        3. Get retention recommendations for high-risk customers
        """
    )

    st.sidebar.title("Model Performance")

    metrics_custom = {
        "Precision (Class 1)": "0.65",
        "Recall (Class 1)": "0.59",
        "F1-score (Class 1)": "0.62",
        "Accuracy": "0.81",
    }

    metrics_sklearn = {
        "Precision (Class 1)": "0.68",
        "Recall (Class 1)": "0.58",
        "F1-score (Class 1)": "0.62",
        "Accuracy": "0.82",
    }

    st.sidebar.subheader("Custom Model")
    for metric, value in metrics_custom.items():
        st.sidebar.text(f"{metric}: {value}")

    st.sidebar.subheader("Scikit-learn Model")
    for metric, value in metrics_sklearn.items():
        st.sidebar.text(f"{metric}: {value}")


def prepare_sample_data():
    """Prepare sample data for quick testing"""
    high_risk_sample = {
        "tenure": 1,
        "MonthlyCharges": 85.0,
        "TotalCharges": 85.0,
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "MultipleLines": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "SeniorCitizen": 0,
        "gender": "Male",
        "Dependents": "No",
        "Partner": "No",
        "PhoneService": "Yes",
        "PaperlessBilling": "Yes",
    }

    low_risk_sample = {
        "tenure": 72,
        "MonthlyCharges": 105.8,
        "TotalCharges": 7608.0,
        "Contract": "Two year",
        "PaymentMethod": "Bank transfer (automatic)",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "MultipleLines": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "SeniorCitizen": 0,
        "gender": "Female",
        "Dependents": "Yes",
        "Partner": "Yes",
        "PhoneService": "Yes",
        "PaperlessBilling": "Yes",
    }

    return {"High Churn Risk": high_risk_sample, "Low Churn Risk": low_risk_sample}


def main():
    # Display header with logo/title
    st.title("📊 Telecom Customer Churn Prediction")
    st.markdown(
        """
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
    }
    .stAlert > div {
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
    }
    .css-eh5xgm {
        max-width: 900px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.write(
        """
    This application helps predict whether a telecom customer is likely to churn based on
    various customer attributes. The predictions are made using both a custom-implemented 
    logistic regression model and scikit-learn's implementation.
    """
    )

    # Display sidebar information
    display_sidebar_info()

    # Load models
    models = load_models()
    if not all(models):
        st.error("⚠️ Models failed to load. Please check the error messages above.")
        return

    custom_model, sklearn_model, label_encoders, scaler = models
    st.success("✅ Models loaded successfully!")

    # Prepare sample data
    samples = prepare_sample_data()

    # Create tabs for different modes
    tab1, tab2 = st.tabs(["📋 Enter Customer Details", "⚡ Quick Test with Samples"])

    with tab1:
        # Create input form
        with st.form("prediction_form"):
            # Create three columns for better layout
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Basic Info")
                tenure = st.number_input(
                    "Tenure (months)", min_value=0, max_value=100, value=1
                )
                monthly_charges = st.number_input(
                    "Monthly Charges ($)", min_value=0.0, value=70.0
                )
                total_charges = st.number_input(
                    "Total Charges ($)", min_value=0.0, value=70.0
                )

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

            submit_button = st.form_submit_button(
                "Predict Churn", use_container_width=True
            )

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

                # Process and predict
                process_input_and_predict(
                    input_data, custom_model, sklearn_model, label_encoders, scaler
                )

    with tab2:
        st.subheader("Quick Test with Sample Profiles")
        st.write("Choose a pre-configured customer profile to see predictions:")

        sample_choice = st.selectbox("Select Sample Profile", list(samples.keys()))

        if st.button("Run Prediction", use_container_width=True):
            selected_sample = samples[sample_choice]
            process_input_and_predict(
                selected_sample, custom_model, sklearn_model, label_encoders, scaler
            )


def process_input_and_predict(
    input_data, custom_model, sklearn_model, label_encoders, scaler
):
    """Process input data and make predictions"""
    try:
        with st.spinner("Processing..."):
            # Create DataFrame
            input_df = pd.DataFrame([input_data])

            # Ensure the column order matches what was used during training
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

            # Display results with improved UI
            st.markdown("### 📊 Churn Prediction Results")

            col1, col2 = st.columns(2)

            with col1:
                custom_prob_percentage = custom_prob * 100
                color = "red" if custom_prob_percentage > 50 else "green"
                st.markdown(
                    f"""
                    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center;">
                        <h4 style="margin-bottom: 10px;">Custom Model</h4>
                        <p style="font-size: 28px; font-weight: bold; color: {color};">{custom_prob_percentage:.1f}%</p>
                        <p>Churn Probability</p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                sklearn_prob_percentage = sklearn_prob * 100
                color = "red" if sklearn_prob_percentage > 50 else "green"
                st.markdown(
                    f"""
                    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center;">
                        <h4 style="margin-bottom: 10px;">Scikit-learn Model</h4>
                        <p style="font-size: 28px; font-weight: bold; color: {color};">{sklearn_prob_percentage:.1f}%</p>
                        <p>Churn Probability</p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

            # Calculate average probability and determine risk level
            avg_prob = (custom_prob + sklearn_prob) / 2
            risk_level = "High" if avg_prob > 0.5 else "Low"

            # Display risk level with appropriate styling
            risk_color = "red" if risk_level == "High" else "green"
            st.markdown(
                f"""
                <div style="margin-top: 20px; background-color: #f8f9fa; padding: 15px; border-radius: 10px;">
                    <h4>Risk Assessment</h4>
                    <p>Based on both models, this customer has a <span style="color: {risk_color}; font-weight: bold;">{risk_level} Risk</span> of churning.</p>
                </div>
            """,
                unsafe_allow_html=True,
            )

            # Provide retention recommendations for high-risk customers
            if risk_level == "High":
                st.markdown("### 🛑 Recommended Retention Strategies")

                recommendations = []

                if input_data["MonthlyCharges"] > 70:
                    recommendations.append(
                        {
                            "title": "Offer a Discount",
                            "description": "Customer has high monthly charges. Consider offering a 10-15% discount for a 6-month commitment.",
                        }
                    )

                if input_data["Contract"] == "Month-to-month":
                    recommendations.append(
                        {
                            "title": "Suggest Long-term Contract",
                            "description": "Customer is on a month-to-month plan. Propose a yearly contract with better rates or added benefits.",
                        }
                    )

                if (
                    input_data["InternetService"] == "Fiber optic"
                    and input_data["TechSupport"] == "No"
                ):
                    recommendations.append(
                        {
                            "title": "Add Tech Support",
                            "description": "Customer has fiber internet without tech support. Offer complimentary tech support for 3 months.",
                        }
                    )

                if (
                    input_data["OnlineSecurity"] == "No"
                    and input_data["OnlineBackup"] == "No"
                    and input_data["DeviceProtection"] == "No"
                ):
                    recommendations.append(
                        {
                            "title": "Security Package",
                            "description": "Customer lacks security features. Suggest a discounted security and protection package.",
                        }
                    )

                if input_data["tenure"] < 12:
                    recommendations.append(
                        {
                            "title": "New Customer Loyalty Program",
                            "description": "This is a relatively new customer. Enroll them in a special new customer loyalty program with rewards.",
                        }
                    )

                # Display recommendations in an attractive format
                cols = st.columns(min(3, len(recommendations)))
                for i, rec in enumerate(recommendations):
                    with cols[i % 3]:
                        st.markdown(
                            f"""
                        <div style="background-color: #e8f4f8; padding: 15px; border-radius: 8px; height: 200px; margin-bottom: 10px;">
                            <h5 style="color: #2c3e50;">{rec['title']}</h5>
                            <p style="color: #34495e; font-size: 0.9rem;">{rec['description']}</p>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                if not recommendations:
                    st.info(
                        "No specific retention strategies identified for this customer profile."
                    )
            else:
                st.markdown("### ✅ Low Churn Risk Assessment")
                st.markdown(
                    """
                    <div style="background-color: #e8f8e8; padding: 15px; border-radius: 8px;">
                        <h5 style="color: #2c3e50;">Positive Customer Profile</h5>
                        <p>This customer has a low risk of churning. Consider the following actions:</p>
                        <ul>
                            <li>Monitor satisfaction periodically</li>
                            <li>Introduce loyalty rewards to maintain engagement</li>
                            <li>Consider opportunities for service upgrades</li>
                        </ul>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

    except Exception as e:
        st.error(f"Error processing prediction: {str(e)}")
        st.error(traceback.format_exc())


if __name__ == "__main__":
    main()
