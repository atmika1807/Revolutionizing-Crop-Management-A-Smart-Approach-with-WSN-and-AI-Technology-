import streamlit as st
import pandas as pd
import joblib
import os
import json
from datetime import datetime
from PIL import Image

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

# === App Title ===
st.title("🌾 Smart Crop Management Dashboard")
st.write("Use AI + Sensor Data for real-time crop action recommendations.")

# === Display Model Evaluation ===
with st.expander("📊 Model Evaluation Metrics"):
    st.markdown(f"""
    - **Model Accuracy:** `{model_accuracy * 100:.2f}%`
    - **Macro F1-Score:** `{macro_f1:.2f}`
    - **Model Type:** `Random Forest (Balanced)`
    """)
    if os.path.exists(FEATURE_PLOT):
        st.image(FEATURE_PLOT, caption="🌿 Feature Importance", use_column_width=True)
    else:
        st.warning("⚠️ Feature importance plot not found.")

# === Sidebar Inputs ===
st.sidebar.header("🧪 Sensor Inputs")
crop_type = st.sidebar.selectbox("Crop Type", ["Wheat", "Rice", "Corn", "Soybean", "Sugarcane"])
soil_type = st.sidebar.selectbox("Soil Type", ["Clay", "Sandy", "Loamy", "Silty", "Peaty", "Chalky"])
soil_moisture = st.sidebar.slider("Soil Moisture (%)", 10, 100, 50)
temperature = st.sidebar.slider("Temperature (°C)", 10, 45, 25)
humidity = st.sidebar.slider("Humidity (%)", 20, 100, 60)
nutrient_level = st.sidebar.selectbox("Nutrient Level (1=Low, 5=High)", [1, 2, 3, 4, 5])
soil_pH = st.sidebar.slider("Soil pH", 4.5, 9.0, 6.5)

# === Predict and Log ===
if st.button("🔍 Predict Recommended Action"):
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
    st.success(f"✅ Recommended Action: **{prediction.capitalize()}**")

    # Log data
    input_data["predicted_action"] = prediction
    input_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not os.path.exists(HISTORY_PATH):
        input_data.to_csv(HISTORY_PATH, index=False)
    else:
        input_data.to_csv(HISTORY_PATH, mode="a", header=False, index=False)

# === History Viewer with Filters and Safeguards ===
st.subheader("📈 Prediction History")

if os.path.exists(HISTORY_PATH):
    try:
        df_hist = pd.read_csv(HISTORY_PATH)

        # Check if 'timestamp' exists
        if "timestamp" not in df_hist.columns:
            st.warning("⚠️ 'timestamp' column is missing in the history file.")
            st.stop()

        df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])
        df_hist["date"] = df_hist["timestamp"].dt.date

        # Filters
        crops = df_hist["crop_type"].unique()
        selected_crop = st.selectbox("🌱 Filter by Crop", options=["All"] + sorted(crops.tolist()))
        if selected_crop != "All":
            df_hist = df_hist[df_hist["crop_type"] == selected_crop]

        dates = df_hist["date"].unique()
        selected_date = st.selectbox("📅 Filter by Date", options=["All"] + sorted(dates.tolist()))
        if selected_date != "All":
            df_hist = df_hist[df_hist["date"] == pd.to_datetime(selected_date)]

        # Display Data
        st.dataframe(df_hist.tail(10))

        # Charts
        if not df_hist.empty:
            st.line_chart(df_hist[["soil_moisture", "temperature", "humidity"]])
            action_counts = df_hist["predicted_action"].value_counts()
            st.bar_chart(action_counts)

        # Download
        st.download_button("📥 Download Filtered Logs", data=df_hist.to_csv(index=False),
                           file_name="filtered_crop_prediction_history.csv")

    except pd.errors.ParserError:
        st.error("⚠️ Could not read the prediction history file. It may be corrupted.")
else:
    st.info("No predictions made yet.")
