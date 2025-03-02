import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ✅ Define DummyCO2Model before loading the pickle file
class DummyCO2Model:
    def predict(self, X):
        return [100] * len(X)  # Always returns 100 as a placeholder
# Load trained model
with open("co2_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load dataset for insights (optional, change path accordingly)
df = pd.read_csv("CO2_Emissions_Canada.csv")

# --- HEADER SECTION ---
st.markdown("# 🌍 CO₂ Emission Predictor")
st.write(
    "Predict vehicle CO₂ emissions and get recommendations to reduce your carbon footprint."
)

# --- USER INPUT SECTION ---
st.header("🔢 Enter Vehicle Details")

vehicle_class = st.selectbox(
    "Vehicle Class",
    df["Vehicle Class"].unique().tolist(),
    help="Choose the vehicle category.",
)
engine_size = st.number_input(
    "Engine Size (L)",
    min_value=0.5,
    max_value=10.0,
    step=0.1,
    help="Engine displacement in liters.",
)
cylinders = st.number_input(
    "Cylinders",
    min_value=2,
    max_value=16,
    step=1,
    help="Number of cylinders in the engine.",
)
transmission = st.selectbox(
    "Transmission Type",
    ["Automatic", "Manual"],
    help="Choose between automatic or manual transmission.",
)
fuel_type = st.selectbox(
    "Fuel Type",
    ["Zero-emission vehicle", "Diesel", "Ethanol", "CNG/LNG"],
    help="Select your vehicle's fuel type.",
)
fuel_consumption_city = st.number_input(
    "Fuel Consumption City",
    min_value=0.5,
    step=0.1,
    help="Fuel Consumption in city(L/100km)",
)
fuel_consumption_Hwy = st.number_input(
    "Fuel Consumption Highway",
    min_value=0.5,
    step=0.1,
    help="Fuel Consumption in Highway(L/100km)",
)
fuel_consumption_comb = st.number_input(
    "Fuel Consumption Combined",
    min_value=0.5,
    step=0.1,
    help="Fuel Consumption considering both city and highway (L/100km)",
)
fuel_consumption_comb_mpeg = st.number_input(
    "Fuel Consumption Combined(MPG)",
    min_value=0.5,
    step=0.1,
    help="Fuel Consumption combined measured in miles per gallon (mpg)",
)

# --- PREDICTION LOGIC ---
if st.button("🚀 Predict CO₂ Emission"):
    # Convert categorical values to numerical representation
    vehicle_class_map = {
        "COMPACT": 0,
        "SUV - SMALL": 1,
        "MID-SIZE": 2,
        "TWO-SEATER": 3,
        "MINICOMPACT": 4,
        "SUBCOMPACT": 5,
        "FULL-SIZE": 6,
        "STATION WAGON - SMALL": 7,
        "SUV - STANDARD": 8,
        "VAN - CARGO": 9,
        "VAN - PASSENGER": 10,
        "PICKUP TRUCK - STANDARD": 11,
        "MINIVAN": 12,
        "SPECIAL PURPOSE VEHICLE": 13,
        "STATION WAGON - MID-SIZE": 14,
        "PICKUP TRUCK - SMALL": 15,
    }
    transmission_map = {"Automatic": 0, "Manual": 1}
    fuel_map = {"Zero-emission vehicle": 0, "Diesel": 1, "Ethanol": 2, "CNG/LNG": 3}

    input_data = np.array(
        [
            vehicle_class_map[vehicle_class],
            cylinders,
            engine_size,
            transmission_map[transmission],
            fuel_map[fuel_type],
            fuel_consumption_city,
            fuel_consumption_Hwy,
            fuel_consumption_comb,
            fuel_consumption_comb_mpeg,
        ]
    ).reshape(1, -1)

    predicted_co2 = model.predict(input_data)[0]

    st.success(f"🚗 Estimated CO₂ Emission: {predicted_co2:.2f} g/km")

    # Emission comparison
    st.header("📊 CO₂ Emission Comparison")
    labels = ["Electric Car", "Hybrid", "Your Car", "High-Emission"]
    values = [0, 80, predicted_co2, 250]
    fig, ax = plt.subplots()
    ax.bar(labels, values, color=["green", "blue", "red", "black"])
    ax.set_ylabel("CO₂ Emissions (g/km)")
    st.pyplot(fig)

    # Advice based on emission level
    st.header("💡 Emission Reduction Tips")
    if predicted_co2 < 100:
        st.success(
            "✅ Your vehicle has low emissions. Keep up the good eco-friendly driving!"
        )
    elif predicted_co2 < 180:
        st.warning(
            "⚠️ Moderate emissions. Consider carpooling or using public transport."
        )
    else:
        st.error(
            "❌ High emissions! Reduce engine idling, drive smoothly, and maintain tire pressure."
        )

    # Carbon footprint calculator
    km_per_year = st.number_input(
        "🌍 Enter yearly driving distance (km):", min_value=0, step=100
    )
    yearly_co2 = predicted_co2 * km_per_year / 1000  # Convert g/km to kg
    st.write(f"🌱 Your estimated yearly CO₂ emissions: **{yearly_co2:.2f} kg**")
    trees_needed = yearly_co2 / 22
    st.write(
        f"🌳 You need to plant **{trees_needed:.1f} trees per year** to offset your emissions."
    )

# --- FOOTER SECTION ---
st.markdown("---")
st.write("© 2025. Abinash Pradhan. All Rights Reserved.")
st.markdown("[🔗 GitHub Repository](https://github.com/your_repo)")
