import streamlit as st
import pickle
import numpy as np

# Load trained models and preprocessors
rf_model = pickle.load(open("crop_model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Streamlit UI
st.title("🌱 Crop Recommendation System")
st.markdown("Enter soil and weather details to get the best crop recommendation.")

# Input fields
N = st.slider("Nitrogen (N)", 0, 140, 70)
P = st.slider("Phosphorus (P)", 5, 150, 38)
K = st.slider("Potassium (K)", 5, 200, 50)
temperature = st.slider("Temperature (°C)", 0, 50, 25)
humidity = st.slider("Humidity (%)", 10, 100, 60)
pH = st.slider("Soil pH Level", 3.5, 9.5, 6.5)
rainfall = st.slider("Rainfall (mm)", 20, 300, 100)

# Convert input to numpy array
input_features = np.array([[N, P, K, temperature, humidity, pH, rainfall]])

# Apply scaling
input_scaled = scaler.transform(input_features)

# Predict crop
predicted_crop_index = rf_model.predict(input_scaled)[0]
predicted_crop = label_encoder.inverse_transform([predicted_crop_index])[0]

# Display results
if st.button("Recommend Crop 🌾"):
    st.success(f"✅ Recommended Crop: **{predicted_crop}**")

# Add developer credits
st.markdown("---")
st.markdown("Developed by Aravindprakash | Powered by Machine Learning")
