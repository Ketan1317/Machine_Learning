import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import time

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent

model = joblib.load(BASE_DIR / "KNN_heart.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
cols = joblib.load(BASE_DIR / "columns.pkl")

st.markdown(
    """
    <style>
    .main {
        padding-top: 2rem;
    }

    .hero {
        padding: 2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #ff4b4b, #ff758c);
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(255, 75, 75, 0.2);
    }

    .hero h1 {
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }

    .hero p {
        font-size: 1.1rem;
        opacity: 0.95;
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .info-card {
        padding: 1.2rem;
        border-radius: 15px;
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.15);
        margin-bottom: 1rem;
    }

    .result-card {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-top: 2rem;
        border: 1px solid rgba(128, 128, 128, 0.15);
    }

    .result-icon {
        font-size: 4rem;
    }

    .result-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }

    .result-text {
        font-size: 1.1rem;
    }

    div.stButton > button {
        width: 100%;
        height: 3.5rem;
        border-radius: 12px;
        font-size: 1.15rem;
        font-weight: 700;
        border: none;
    }

    div[data-testid="stMetric"] {
        padding: 1rem;
        border-radius: 12px;
        background-color: rgba(128, 128, 128, 0.08);
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero">
        <h1>❤️ Heart Disease Predictor</h1>
        <p>Machine Learning powered heart disease risk prediction</p>
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("About the Model")

    st.markdown(
        """
        This application uses a **K-Nearest Neighbors (KNN)** 
        machine learning model to predict the likelihood of 
        heart disease based on the provided health information.
        """
    )

    st.divider()

    st.metric("Model", "KNN")
    st.metric("Features", "16")

    st.divider()

    st.caption(
        "⚠️ This application is for educational and demonstration "
        "purposes only. It is not a medical diagnostic tool."
    )

st.markdown(
    '<div class="section-title">👤 Personal Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=50,
        help="Enter your age."
    )

with col2:
    sex = st.selectbox(
        "Sex",
        ["M", "F"],
        help="Select biological sex."
    )

st.markdown(
    '<div class="section-title">🫀 Heart Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "ASY", "TA"],
        help="Type of chest pain experienced."
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure",
        min_value=0,
        max_value=250,
        value=120,
        help="Resting blood pressure in mm Hg."
    )

    cholesterol = st.number_input(
        "Cholesterol",
        min_value=0,
        max_value=700,
        value=200,
        help="Serum cholesterol level."
    )

with col2:
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    max_hr = st.number_input(
        "Maximum Heart Rate",
        min_value=40,
        max_value=220,
        value=140,
        help="Maximum heart rate achieved."
    )

st.markdown(
    '<div class="section-title">🏃 Exercise & ECG Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    exercise_angina = st.selectbox(
        "Exercise Induced Angina",
        ["N", "Y"],
        format_func=lambda x: "Yes" if x == "Y" else "No"
    )

    oldpeak = st.number_input(
        "Oldpeak",
        min_value=-5.0,
        max_value=10.0,
        value=0.0,
        step=0.1,
        help="ST depression induced by exercise."
    )

with col2:
    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

st.divider()

st.markdown(
    '<div class="section-title">🔍 Prediction</div>',
    unsafe_allow_html=True
)

predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])

with predict_col2:
    predict = st.button(
        "❤️ Predict Heart Disease Risk",
        use_container_width=True
    )

if predict:

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

    data = pd.get_dummies(
        data,
        drop_first=True
    )

    data = data.reindex(
        columns=cols,
        fill_value=0
    )

    numerical_cols = [
        "Age",
        "RestingBP",
        "Cholesterol",
        "MaxHR",
        "Oldpeak"
    ]

    data[numerical_cols] = scaler.transform(
        data[numerical_cols]
    )

    with st.spinner("🧠 Analyzing your health information..."):
        time.sleep(1.5)
        prediction = model.predict(data)

    st.divider()

    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.error(
            """
            ### ❤️ Higher Likelihood of Heart Disease

            The model predicts a higher likelihood of heart disease
            based on the information provided.
            """
        )

    else:

        st.success(
            """
            ### 💚 Lower Likelihood of Heart Disease

            The model predicts a lower likelihood of heart disease
            based on the information provided.
            """
        )

    st.markdown("### 📋 Your Input Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric("Age", age)
        st.metric("Resting BP", resting_bp)

    with summary_col2:
        st.metric("Cholesterol", cholesterol)
        st.metric("Maximum HR", max_hr)

    with summary_col3:
        st.metric("Oldpeak", oldpeak)
        st.metric(
            "Exercise Angina",
            "Yes" if exercise_angina == "Y" else "No"
        )

st.divider()

st.caption(
    "❤️ Heart Disease Prediction • Machine Learning Project • "
    "Educational Use Only"
)