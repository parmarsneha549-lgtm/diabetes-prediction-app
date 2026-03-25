import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="Diabetes AI System", layout="centered")

# Title
st.title("🩺 AI-Based Diabetes Risk Prediction System")

# Sidebar
st.sidebar.title("About")
st.sidebar.write("Advanced ML-based health prediction system")
st.sidebar.write("Developed by Sneha Parmar")

# Load model
with open("diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)

# Initialize session history
if "history" not in st.session_state:
    st.session_state.history = []

# Image
st.image("https://cdn-icons-png.flaticon.com/512/2966/2966484.png", width=100)

st.subheader("📋 Enter Patient Details")
name = st.text_input("Patient Name")
gender = st.radio("Gender", ["Male", "Female", "Other"])

# Inputs
preg = st.slider("Pregnancies", 0, 20, 1)
glucose = st.slider("Glucose", 0, 200, 100)
bp = st.slider("Blood Pressure", 0, 150, 70)
skin = st.slider("Skin Thickness", 0, 100, 20)
insulin = st.slider("Insulin", 0, 900, 80)
bmi = st.slider("BMI", 0.0, 50.0, 25.0)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.slider("Age", 0, 100, 30)

# BMI Category
st.subheader("⚖️ BMI Category")
if bmi < 18.5:
    st.info("Underweight")
elif 18.5 <= bmi < 25:
    st.success("Normal Weight")
elif 25 <= bmi < 30:
    st.warning("Overweight")
else:
    st.error("Obese")

# Predict button
if st.button("🔍 Predict Risk"):

    data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    result = model.predict(data)
    prob = model.predict_proba(data)

    risk_score = prob[0][1] * 100

    st.subheader("🧾 Prediction Result")

    # Risk classification
    if risk_score < 30:
        st.success("🟢 Low Risk")
    elif 30 <= risk_score < 70:
        st.warning("🟡 Moderate Risk")
    else:
        st.error("🔴 High Risk")

    # Prediction
    st.subheader("🧾 Patient Details")
    st.write(f"Name: {name}")
    st.write(f"Gender: {gender}")
    st.write(f"Age: {age}")

    if result[0] == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

    # Probability
    st.subheader("📊 Prediction Confidence")
    st.progress(int(risk_score))
    st.write(f"Probability: {risk_score:.2f}%")

    # Graph
    st.subheader("📈 Risk Visualization")
    labels = ['No Diabetes', 'Diabetes']
    values = prob[0]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    st.pyplot(fig)

    # Health Summary
    st.subheader("🧾 Health Summary")

    if glucose > 140:
        st.warning("High Glucose Level")
    if bp > 90:
        st.warning("High Blood Pressure")
    if bmi > 30:
        st.warning("High BMI")

    # Smart Recommendations
    st.subheader("💡 Personalized Advice")

    if bmi > 30:
        st.write("👉 Start weight loss and regular exercise")
    if glucose > 140:
        st.write("👉 Avoid sugar-rich foods")
    if age > 45:
        st.write("👉 Regular medical checkups required")

    if result[0] == 1:
        st.write("👉 Consult a doctor immediately")
    else:
        st.write("👉 Maintain healthy lifestyle")

    # Emergency Alert
    if glucose > 200:
        st.error("🚨 Critical Condition! Seek medical help immediately")

    # Save history
    record = {
    "Name": name,
    "Gender": gender,
    "Age": age,
    "Glucose": glucose,
    "BMI": bmi,
    "Risk %": risk_score,
    "Result": "High Risk" if result[0]==1 else "Low Risk"
}

    st.session_state.history.append(record)

    # Download report
    df = pd.DataFrame([record])
    st.download_button("📥 Download Report", df.to_csv(index=False), "report.csv")

# Show history
st.subheader("📂 Patient History")

if st.session_state.history:
    st.write(pd.DataFrame(st.session_state.history))
else:
    st.write("No records yet")
