import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('classifier_model.pkl')
scaler = joblib.load('scaler.pkl')

# App title and description
st.title("Social Network Ads Prediction App 🚀")
st.write("This app predicts whether a customer will purchase the product based on their age and estimated salary.")

# User input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
salary = st.number_input("Estimated Salary", min_value=10000, max_value=200000, value=50000)

# Prediction button
if st.button("Predict"):
    # Preprocess user data using the saved scaler
    user_data = np.array([[age, salary]])
    user_data_scaled = scaler.transform(user_data)
    
    # Make prediction
    prediction = model.predict(user_data_scaled)
    
    # Display the result
    if prediction[0] == 1:
        st.success("🎉 Result: This customer is likely to **BUY** the product!")
    else:
        st.warning("⚠️ Result: This customer is likely to **NOT BUY** the product.")