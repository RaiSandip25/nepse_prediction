import joblib
import streamlit as st
import numpy as np

model = joblib.load("nepse_prediction_model.pkl")
st.title("Welcome to Nepse Stock Prediction App")

st.image("nepsepredict_logo.png", use_column_width="auto")

st.info("In this app, the opening, low, and high price should be known.")

open = st.number_input("Enter the opening value: ")
high = st.number_input("Enter the high value: ")
low = st.number_input("Enter the low value: ")

if st.button("Predict"):
    features = np.array([[open, high, low]])
    close_price = model.predict(features)
    st.success(close_price[0])
