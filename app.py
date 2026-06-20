import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Diabetes Prediction System")

st.write("Enter patient details below:")

preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

if st.button("Predict"):

    data = np.array([[preg, glucose, bp, skin,
                      insulin, bmi, dpf, age]])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)
    probability = model.predict_proba(data_scaled)[0][1]

    if prediction[0] == 1:
        st.error("Patient is likely Diabetic")
    else:
        st.success("Patient is likely Not Diabetic")

    st.write(f"Diabetes Probability: {probability:.2%}")
