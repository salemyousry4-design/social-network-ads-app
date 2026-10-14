# 🚀 Smart Customer Purchase Predictor (SVM RBF)

An end-to-end Machine Learning web application deployed on Streamlit Cloud, designed to predict customer purchasing behavior based on demographic features (Age and Estimated Salary).

## 🌟 Live Demo
Try out the live web application here: [Streamlit App Link](https://social-network-ads-app.streamlit.app)

---

## 🛠️ Tech Stack & Libraries
* **Python**
* **Scikit-Learn** (SVM Machine Learning Model & StandardScaler)
* **Streamlit** (Interactive Web UI & Cloud Deployment)
* **Joblib** (Model Serialization)
* **NumPy** (Data Processing)

---

## 📂 Project Structure
```text
social-network-ads-app/
│
├── app.py                 # Main Streamlit web application script
├── classifier_model.pkl   # Trained SVM model artifact
├── scaler.pkl             # Fitted StandardScaler
├── requirements.txt       # Project dependencies
└── Social_Network_Ads.csv # Dataset used for training
