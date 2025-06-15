import streamlit as st
import pandas as pd
import joblib
import os
import json
from datetime import datetime

# === Page Config ===
st.set_page_config(page_title="Smart Crop Management", layout="wide")

# === File Paths ===
MODEL_PATH = "crop_model.pkl"
HISTORY_PATH = "crop_prediction_history.csv"
METRICS_PATH = "metrics.json"
FEATURE_PLOT = "feature_importance.png"

# === Load Model ===
model = joblib.load(MODEL_PATH)

# === Load Metrics ===
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    model_accuracy = metrics.get("accuracy", 0.0)
    macro_f1 = metrics.get("classification_report", {}).get("macro avg", {}).get("f1-score", 0.0)
else:
    model_accuracy = 0.0
    macro_f1 = 0.0

# === App Header ===
st.title("\U0001F33F Smart Crop Management")
st.caption("Leverage AI + IoT sensor data to optimize irrigation and fertilization decisions.")

# === Two-Column Layout ===
col1, col2 = st.columns(2)

with col1:
    st.header("\U0001F9EA Sensor Inputs")
    crop_type = st.selectbox("Crop Type", ["Wheat", "Rice", "Corn", "Soybean", "Sugarcane"])
    soil_type = st.selectbox("Soil Type", ["Clay", "Sandy", "Loamy", "Silty", "Peaty", "Chalky"])
    soil_moisture = st.slider("Soil Moisture (%)", 10, 100, 50)
    temperature = st.slider("Temperature (°C)", 10, 45, 25)
    humidity = st.slider("Humidity (%)", 20, 100, 60)
    nutrient_level = st.selectbox("Nutrient Level (1=Low, 5=High)", [1, 2, 3, 4, 5])
    soil_pH = st.slider("Soil pH", 4.5, 9.0, 6.5)

with col2:
    st.header("\U0001F50E Prediction & Results")
    if st.button("Predict Recommended Action"):
        input_data = pd.DataFrame([{
            "crop_type": crop_type,
            "soil_type": soil_type,
            "soil_moisture": soil_moisture,
            "temperature": temperature,
            "humidity": humidity,
            "nutrient_level": nutrient_level,
            "soil_pH": soil_pH
        }])

        prediction = model.predict(input_data)[0]
        st.success(f"\u2705 Action: **{prediction.capitalize()}**")


        input_data["predicted_action"] = prediction
        input_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if not os.path.exists(HISTORY_PATH):
            input_data.to_csv(HISTORY_PATH, index=False)
        else:
            input_data.to_csv(HISTORY_PATH, mode="a", header=False, index=False)

    with st.expander("\U0001F4CA View Model Metrics"):
        st.markdown(f"""
        - **Model Accuracy:** `{model_accuracy * 100:.2f}%`
        - **Macro F1-Score:** `{macro_f1:.2f}`
        - **Model:** `Random Forest`
        """)
        if os.path.exists(FEATURE_PLOT):
            st.image(FEATURE_PLOT, caption="Top Features", use_column_width=True)

# === Full-Width History Section ===
st.divider()
st.subheader("\U0001F4C8 Prediction History + Filters")

if os.path.exists(HISTORY_PATH):
    try:
        df_hist = pd.read_csv(HISTORY_PATH)

        if "timestamp" not in df_hist.columns:
            st.warning("⚠️ 'timestamp' column is missing in the prediction log. Please delete or fix the file.")
            st.stop()

        df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])
        df_hist["date"] = df_hist["timestamp"].dt.date

        col_hist1, col_hist2 = st.columns(2)
        with col_hist1:
            crop_filter = st.selectbox("🌱 Filter by Crop", options=["All"] + sorted(df_hist["crop_type"].unique().tolist()))
        with col_hist2:
            date_filter = st.selectbox("📅 Filter by Date", options=["All"] + sorted(df_hist["date"].astype(str).unique().tolist()))

        if crop_filter != "All":
            df_hist = df_hist[df_hist["crop_type"] == crop_filter]
        if date_filter != "All":
            df_hist = df_hist[df_hist["date"] == pd.to_datetime(date_filter)]

        st.dataframe(df_hist.tail(10))
        st.line_chart(df_hist[["soil_moisture", "temperature", "humidity"]])
        st.bar_chart(df_hist["predicted_action"].value_counts())

        st.download_button("📥 Download Filtered Logs", data=df_hist.to_csv(index=False),
                           file_name="filtered_crop_prediction_history.csv")

    except pd.errors.ParserError:
        st.error("⚠️ Could not read the prediction history file. It may be corrupted.")
else:
    st.info("No prediction history found yet.")