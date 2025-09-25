# recordar que el servicio web de streamlit debe levantarse como:
# streamlit run .\streamlit_crop_predictor_v2.py
# streamlit_crop_predictor_v2.py es este código

import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo entrenado
model = joblib.load("knn_crop_model.joblib")

# Título de la app
st.title("Crop Recommendation Predictor")

# Entradas del usuario
N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=85)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=140, value=55)
K = st.number_input("Potassium (K)", min_value=0, max_value=140, value=40)
temperature = st.number_input("Temperature (°C)", value=25.0)
humidity = st.number_input("Humidity (%)", value=80.0)
ph = st.number_input("Soil pH", value=6.5)
rainfall = st.number_input("Rainfall (mm)", value=200.0)

# Botón de predicción
if st.button("Predict Crop"):
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                              columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
    prediction = model.predict(input_data)
    st.success(f"Recommended crop: {prediction[0]}")
