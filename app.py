import streamlit as st
import joblib
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="Social Network Ads Predictor",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Load Model and Scaler
@st.cache_resource
def load_assets():
    model = joblib.load('classifier_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# 3. Header Design
st.title("🚀 Smart Customer Purchase Predictor")
st.markdown("""
An advanced machine learning web application (SVM RBF) designed to predict customer purchasing behavior based on demographic features.
""")
st.divider()

# 4. Sidebar for User Inputs
st.sidebar.header("🎛️ Control Panel & Inputs")
st.sidebar.markdown("Adjust the values to predict customer behavior:")

age = st.sidebar.slider("Customer Age", min_value=18, max_value=70, value=30, step=1)
salary = st.sidebar.number_input("Estimated Salary ($)", min_value=10000, max_value=150000, value=50000, step=5000)

# 5. Display Selected Metrics in Main Area
col1, col2 = st.columns(2)
with col1:
    st.metric(label="👤 Selected Age", value=f"{age} Years")
with col2:
    st.metric(label="💰 Selected Salary", value=f"${salary:,.0f}")

st.markdown("<br>", unsafe_allow_html=True)

# 6. Prediction Button and Results
if st.button("🔍 Analyze & Predict", use_container_width=True):
    with st.spinner('Analyzing data using the model...'):
        # Preprocess data
        user_data = np.array([[age, salary]])
        user_data_scaled = scaler.transform(user_data)
        
        # Make prediction
        prediction = model.predict(user_data_scaled)
        
        # Display Results
        st.divider()
        if prediction[0] == 1:
            st.success("### 🎉 Positive Result: This customer is highly likely to **PURCHASE** the product!")
            st.balloons()
        else:
            st.error("### ⚠️ Negative Result: This customer is unlikely to make a purchase.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Built with ❤️ using Python, Scikit-Learn & Streamlit</p>", unsafe_allow_html=True)
