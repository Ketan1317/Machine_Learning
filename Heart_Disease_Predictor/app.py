import streamlit as st
import pandas as pd
import joblib

model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
cols = joblib.load("columns.pkl")

st.title("Heart Disease Prediction")
st.markdown("Provide the following details")

age = st.number_input("Age", min_value=1, max_value=100, value=50)

sex = st.selectbox("Sex", ["M", "F"])

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "ASY", "TA"]
)

resting_bp = st.number_input(
    "Resting Blood Pressure",
    min_value=0,
    max_value=250,
    value=120
)

cholesterol = st.number_input(
    "Cholesterol",
    min_value=0,
    max_value=700,
    value=200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    [0, 1]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

max_hr = st.number_input(
    "Maximum Heart Rate",
    min_value=40,
    max_value=220,
    value=140
)

exercise_angina = st.selectbox(
    "Exercise Angina",
    ["N", "Y"]
)

oldpeak = st.number_input(
    "Oldpeak",
    min_value=-5.0,
    max_value=10.0,
    value=0.0
)

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

if st.button("Predict"):

    data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest_pain],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "RestingECG": [resting_ecg],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope]
    })

    data = pd.get_dummies(data, drop_first=True)
    data = data.reindex(columns=cols, fill_value=0)

    numerical_cols = [
        "Age",
        "RestingBP",
        "Cholesterol",
        "MaxHR",
        "Oldpeak"
    ]

    data[numerical_cols] = scaler.transform(data[numerical_cols])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("Higher likelihood of heart disease")
    else:
        st.success("Lower likelihood of heart disease")