import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# Load model
model = joblib.load(r'C:\Users\atpt1\OneDrive\Desktop\New folder\crop_model.pkl')

# File to store input history
history_file = "crop_history.csv"

st.title("🌾 Smart Crop Management Dashboard")
st.write("Get real-time crop recommendations based on sensor readings.")

# --- User Inputs ---
soil_moisture = st.slider("Soil Moisture (%)", 10, 100, 45)
temperature = st.slider("Temperature (°C)", 10, 45, 25)
humidity = st.slider("Humidity (%)", 10, 100, 60)
nutrient_level = st.selectbox("Nutrient Level (1=Low, 5=High)", [1, 2, 3, 4, 5])

# --- Predict ---
if st.button("Predict Action"):
    input_df = pd.DataFrame([[soil_moisture, temperature, humidity, nutrient_level]],
                            columns=['soil_moisture', 'temperature', 'humidity', 'nutrient_level'])
    prediction = model.predict(input_df)[0]
    
    # Show result
    st.success(f"🌿 Suggested Action: **{prediction.capitalize()}**")

    # Log input
    input_df["action"] = prediction
    input_df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists(history_file):
        input_df.to_csv(history_file, index=False)
    else:
        input_df.to_csv(history_file, mode='a', header=False, index=False)

# --- View Trend Charts ---
st.subheader("📊 Sensor Data Trends")
if os.path.exists(history_file):
    df_history = pd.read_csv(history_file)

    # Show raw data
    with st.expander("View Data Log"):
        st.dataframe(df_history.tail(10))

    # Trend chart
    st.line_chart(df_history[['soil_moisture', 'temperature', 'humidity']])
else:
    st.info("No history yet. Submit a prediction to start logging.")
