import streamlit as st
import pandas as pd
import numpy as np
import traceback
import time
from preprocess import load_and_clean_data, handle_missing_values, prepare_data
from model import HousePriceModel

# Set page config
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    menu_items={
        "Get Help": "https://github.com/your-username/house-price-prediction/issues",
        "Report a bug": "https://github.com/your-username/house-price-prediction/issues",
        "About": "# House Price Prediction\nA machine learning app to predict house prices using linear regression.",
    },
)

# App title and description
st.title("🏠 House Price Prediction")
st.markdown(
    """
This application predicts house prices based on various features. Enter the details of a house below to get an estimated price.
"""
)


# Function to load and prepare data with caching for performance
@st.cache_data(ttl=3600, show_spinner="Loading and preparing model data...")
def load_model_data():
    """Load and prepare the model data"""
    try:
        # Load data
        data = load_and_clean_data("house_prices.csv")
        data = handle_missing_values(data)

        # Prepare data
        X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(
            data, target_column="SalePrice"
        )

        # Train model
        model = HousePriceModel(use_custom=False)
        model.fit(X_train, y_train)
        model.save_model_info(scaler, feature_names)

        # Get model evaluation metrics
        evaluation = model.evaluate(X_test, y_test)

        return model, feature_names, evaluation
    except Exception as e:
        st.error(f"Error while loading model data: {str(e)}")
        raise e


# Main app interface
try:
    # Load model and get features
    with st.spinner("Preparing the model..."):
        model, feature_names, evaluation = load_model_data()

    # Display model metrics in an expander
    with st.expander("Model Performance Metrics"):
        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", f"{evaluation['r2']:.3f}")
        col2.metric("RMSE", f"{evaluation['rmse']:.2f}")
        col3.metric("MSE", f"{evaluation['mse']:.2f}")

    # Create input form
    st.subheader("House Features")

    # Create two columns for input fields
    col1, col2 = st.columns(2)

    with col1:
        # Numerical inputs
        lot_area = st.number_input(
            "Lot Area (sqft)",
            min_value=1000,
            max_value=100000,
            value=8000,
            help="Size of the lot in square feet",
        )
        living_area = st.number_input(
            "Living Area (sqft)",
            min_value=500,
            max_value=10000,
            value=1500,
            help="Above ground living area in square feet",
        )
        bedrooms = st.number_input(
            "Number of Bedrooms",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Number of bedrooms",
        )
        bathrooms = st.number_input(
            "Number of Bathrooms",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help="Number of bathrooms",
        )

    with col2:
        # Additional features
        current_year = 2024
        year_built = st.number_input(
            "Year Built",
            min_value=1900,
            max_value=current_year,
            value=2000,
            help="Year the house was built",
        )
        garage_cars = st.number_input(
            "Garage Capacity (cars)",
            min_value=0,
            max_value=4,
            value=2,
            step=1,
            help="How many cars the garage can fit",
        )
        overall_quality = st.slider(
            "Overall Quality",
            min_value=1,
            max_value=10,
            value=5,
            help="Overall material and finish quality (1=Poor, 10=Excellent)",
        )
        condition = st.slider(
            "Overall Condition",
            min_value=1,
            max_value=10,
            value=5,
            help="Overall condition rating (1=Poor, 10=Excellent)",
        )

    # Add a submit button with loading state
    with st.form(key="prediction_form"):
        submit_button = st.form_submit_button(
            label="Predict Price", use_container_width=True
        )

        if submit_button:
            with st.spinner("Calculating house price..."):
                # Add a small artificial delay to improve UX
                time.sleep(0.5)

                # Create a feature dictionary based on the expected feature names
                features = {}

                # Map user inputs to the expected feature names
                feature_mapping = {
                    "LotArea": lot_area,
                    "GrLivArea": living_area,
                    "BedroomAbvGr": bedrooms,
                    "FullBath": bathrooms,
                    "YearBuilt": year_built,
                    "GarageCars": garage_cars,
                    "OverallQual": overall_quality,
                    "OverallCond": condition,
                }

                # Create a DataFrame with all expected features, filling with defaults for any missing
                for feature in feature_names:
                    if feature in feature_mapping:
                        features[feature] = feature_mapping[feature]
                    else:
                        # Use a default value (0 or the mean) for any features not provided in the UI
                        features[feature] = 0

                # Convert to DataFrame with columns in the expected order
                input_df = pd.DataFrame([features])

                # Ensure column order matches what the model expects
                input_df = input_df[feature_names]

                # Make prediction using the model's predict method
                prediction = model.predict(input_df)

                if isinstance(prediction, np.ndarray) and len(prediction) > 0:
                    prediction_value = prediction[0]
                else:
                    prediction_value = prediction

                # Display prediction with formatting and style
                st.success(f"### Predicted House Price: ${prediction_value:,.2f}")

                # Display a note about the prediction
                st.info(
                    "Note: This prediction is based on the California Housing dataset and should be used as a reference only."
                )

                # Display feature importance
                st.subheader("Feature Importance")
                if hasattr(model.model, "feature_importances_"):
                    feature_importance = pd.DataFrame(
                        {
                            "Feature": feature_names,
                            "Importance": model.model.feature_importances_,
                        }
                    ).sort_values("Importance", ascending=False)

                    # Display in bar chart
                    st.bar_chart(feature_importance.set_index("Feature"))

                elif hasattr(model.model, "coef_"):
                    coefficients = model.model.coef_
                    feature_importance = pd.DataFrame(
                        {"Feature": feature_names, "Importance": np.abs(coefficients)}
                    ).sort_values("Importance", ascending=False)

                    # Display in bar chart
                    st.bar_chart(feature_importance.set_index("Feature"))

                    # Display the top 3 most important features
                    top_features = feature_importance.head(3)
                    st.write("##### Top 3 Most Important Features:")
                    for i, (feature, importance) in enumerate(
                        zip(top_features["Feature"], top_features["Importance"])
                    ):
                        st.write(f"{i+1}. **{feature}**: {importance:.4f}")
                else:
                    st.warning("Feature importance not available for this model.")

    # Sample data section
    with st.expander("See sample house data"):
        st.write(
            """
        Here are some sample house data points that you can use:
        
        | Feature | Economy | Mid-Range | Luxury |
        | --- | --- | --- | --- |
        | Lot Area (sqft) | 5000 | 8000 | 15000 |
        | Living Area (sqft) | 1200 | 2000 | 3500 |
        | Bedrooms | 2 | 3 | 5 |
        | Bathrooms | 1 | 2 | 3.5 |
        | Year Built | 1975 | 2000 | 2020 |
        | Garage Capacity | 1 | 2 | 3 |
        | Overall Quality | 3 | 6 | 9 |
        | Overall Condition | 4 | 6 | 8 |
        """
        )

except FileNotFoundError:
    st.error(
        """
        ### Error: House prices dataset not found!
        
        Please make sure to:
        1. Download the house prices dataset
        2. Save it as 'house_prices.csv' in the project directory
        3. Restart the application
        
        The dataset will be automatically downloaded when the application runs for the first time.
        """
    )
except Exception as e:
    st.error(f"### An unexpected error occurred")
    st.error(f"Error type: {type(e).__name__}")
    st.error(f"Error details: {str(e)}")

    # Show detailed error in an expander (only in development)
    with st.expander("See detailed error (for developers)"):
        st.code(traceback.format_exc())

    # Recovery suggestions
    st.info(
        """
    #### Possible solutions:
    - Refresh the page and try again
    - Check if the dataset is properly formatted
    - Ensure all required packages are installed
    - Contact support if the issue persists
    """
    )

# Add footer with version info
st.markdown("---")
st.markdown(
    """
<div style="display: flex; justify-content: space-between; align-items: center;">
    <span>Built with ❤️ using Streamlit</span>
    <span>Version 1.0.0</span>
</div>
""",
    unsafe_allow_html=True,
)


def run_app():
    """Entry point for the application when used as a package."""
    import sys
    import streamlit.web.cli as stcli

    sys.argv = ["streamlit", "run", __file__, "--global.developmentMode=false"]
    sys.exit(stcli.main())


if __name__ == "__main__":
    # The app is already running via Streamlit's CLI
    pass
