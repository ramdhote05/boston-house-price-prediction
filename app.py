import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Boston House Price Predictor", layout="wide")

st.title("🏡 Boston House Price Predictor")
st.markdown("---")

# Organized inputs using columns
col1, col2, col3 = st.columns(3)

with col1:
    crim = st.number_input("Crime Rate (CRIM)", value=0.1)
    zn = st.number_input("Res. Land Zone (ZN)", value=18.0)
    indus = st.number_input("Non-retail Business (INDUS)", value=2.3)
    chas = st.selectbox("Near Charles River? (CHAS)", [0, 1])
    nox = st.number_input("Nitric Oxides (NOX)", value=0.5)

with col2:
    rm = st.number_input("Avg Rooms (RM)", value=6.0)
    age = st.number_input("Built before 1940 (AGE)", value=65.0)
    dis = st.number_input("Distance to Centers (DIS)", value=4.0)
    rad = st.number_input("Highway Accessibility (RAD)", value=1.0)

with col3:
    tax = st.number_input("Property Tax (TAX)", value=300.0)
    ptratio = st.number_input("Pupil-Teacher Ratio", value=15.0)
    b = st.number_input("Black Population Index (B)", value=396.0)
    lstat = st.number_input("% Lower Status (LSTAT)", value=5.0)

st.markdown("---")

if st.button("Calculate Estimated Price", use_container_width=True):
    # Construct the full feature array (13 features)
    features = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]])
    
    # Predict
    prediction = model.predict(features)
    
    # Display Result
    st.balloons()
    st.success(f"### The estimated market value is: ${prediction[0]:.2f}k")
