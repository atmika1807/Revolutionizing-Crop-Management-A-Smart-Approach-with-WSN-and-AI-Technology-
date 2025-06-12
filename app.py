import streamlit as st
import pandas as pd
import joblib
import os

# Load model
model = joblib.load(r"C:\Users\atpt1\New folder\Revolutionizing Crop Management\crop_model.pkl")
history_file = r"C:\Users\atpt1\New folder\Revolutionizing Crop Management\crop data.csv"

# Streamlit page setup
st.set_page_config(page_title="🌾 Smart Crop Dashboard", layout="wide")

# Custom CSS Styling
st.markdown("""
    <style>
        .main {
            background-color: #f5f8fa;
        }
        .block-container {
            max-width: 1000px;
            margin: auto;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton>button {
            background-color: #228B22;
            color: white;
            padding: 0.6em 1.2em;
            border-radius: 8px;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #1e7e34;
        }
        .card {
            background-color: #ffffff;
            padding: 1.5rem 2rem;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>🌾 Smart Crop Management Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem;'>Get actionable crop advice based on real-time soil and climate inputs.</p>", unsafe_allow_html=True)

# --- Input Form ---
with st.form("predict_form"):
    with st.container():
        st.markdown("### 🌿 Sensor Inputs")
        col1, col2 = st.columns(2)

        with col1:
            crop_type = st.selectbox("Crop Type", ["Wheat", "Rice", "Corn", "Soybean", "Sugarcane"])
            soil_type = st.selectbox("Soil Type", ["Clay", "Sandy", "Loamy", "Silty", "Peaty", "Chalky"])
            soil_moisture = st.slider("Soil Moisture (%)", 10, 100, 44)
            temperature = st.slider("Temperature (°C)", 10, 45, 33)

        with col2:
            humidity = st.slider("Humidity (%)", 20, 100, 60)
            nutrient_level = st.selectbox("Nutrient Level (1=Low, 5=High)", [1, 2, 3, 4, 5])
            soil_pH = st.slider("Soil pH", 3.5, 9.5, 6.5)

        submit = st.form_submit_button("🔍 Predict Action")

# --- Prediction and Output ---
if submit:
    input_df = pd.DataFrame([{
        "crop_type": crop_type,
        "soil_type": soil_type,
        "soil_moisture": soil_moisture,
        "temperature": temperature,
        "humidity": humidity,
        "nutrient_level": nutrient_level,
        "soil_pH": soil_pH
    }])

    try:
        prediction = model.predict(input_df)[0]
        st.markdown(f"""
            <div class='card'>
                <h4>✅ Prediction Result</h4>
                <p style='font-size: 1.2rem;'>🌱 <strong>{prediction.capitalize()}</strong></p>
            </div>
        """, unsafe_allow_html=True)

        input_df.to_csv(history_file, mode='a', header=not os.path.exists(history_file), index=False)

    except Exception as e:
        st.error(f"Prediction error: {e}")
