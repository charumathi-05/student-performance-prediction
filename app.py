from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "student_performance_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"

st.set_page_config(
    page_title="Student Performance Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.8rem; padding-bottom: 2rem; max-width: 1180px;}
    .hero {padding: 1.5rem 1.7rem; border-radius: 18px;
           background: linear-gradient(120deg, #102a43 0%, #243b53 55%, #486581 100%);
           color: white; margin-bottom: 1.2rem;}
    .hero h1 {color: white; margin-bottom: .35rem;}
    .hero p {color: #e6eef5; font-size: 1.03rem; margin-bottom: 0;}
    .small-note {font-size: .88rem; color: #64748b;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained model is missing. Run `python train.py` before starting the app."
        )
    model = joblib.load(MODEL_PATH)
    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            metadata = json.load(file)
    return model, metadata


st.markdown(
    """
    <div class="hero">
      <h1>🎓 Student Performance Intelligence</h1>
      <p>Estimate a student's final mathematics grade using a machine-learning model
      trained on the UCI Student Performance dataset.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("About this project")
    st.write(
        "This application demonstrates a complete ML workflow: data preparation, "
        "model comparison, MLflow experiment tracking, and interactive prediction."
    )
    st.divider()
    st.caption("Model: selected from three regression algorithms using held-out RMSE.")
    st.caption("Grade scale: 0–20, based on the source dataset.")
    st.caption("Educational demonstration only. Predictions are estimates, not guarantees.")

try:
    model, metadata = load_artifacts()
except Exception as error:
    st.error(str(error))
    st.stop()

selected_model = metadata.get("selected_model", "Trained regression model")
metrics = metadata.get("best_test_metrics", {})

overview = st.columns(3)
overview[0].metric("Selected model", selected_model)
overview[1].metric("Test MAE", f"{metrics.get('mae', float('nan')):.2f}" if metrics else "See MLflow")
overview[2].metric("Test RMSE", f"{metrics.get('rmse', float('nan')):.2f}" if metrics else "See MLflow")

st.subheader("Enter student details")
st.write("Use the previous-grade information and study-related factors to generate an estimate.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        studytime = st.select_slider(
            "Weekly study time",
            options=[1, 2, 3, 4],
            value=2,
            format_func=lambda x: {
                1: "1 — Less than 2 hours",
                2: "2 — 2 to 5 hours",
                3: "3 — 5 to 10 hours",
                4: "4 — More than 10 hours",
            }[x],
        )
        failures = st.selectbox("Past class failures", options=[0, 1, 2, 3], index=0)
        absences = st.number_input("School absences", min_value=0, max_value=93, value=4, step=1)
        g1 = st.slider("First-period grade (G1)", min_value=0, max_value=20, value=11)
        g2 = st.slider("Second-period grade (G2)", min_value=0, max_value=20, value=12)
    with col2:
        schoolsup = st.selectbox("Extra educational support", ["no", "yes"], index=0)
        famsup = st.selectbox("Family educational support", ["yes", "no"], index=0)
        internet = st.selectbox("Internet access at home", ["yes", "no"], index=0)
        st.info(
            "The source dataset records study time on a 1–4 scale and grades from 0 to 20. "
            "G1 and G2 are previous-period grades, not the grade being predicted."
        )
    submitted = st.form_submit_button("Predict final grade", type="primary", use_container_width=True)

if submitted:
    input_data = pd.DataFrame(
        [{
            "studytime": int(studytime),
            "failures": int(failures),
            "absences": int(absences),
            "G1": int(g1),
            "G2": int(g2),
            "schoolsup": schoolsup,
            "famsup": famsup,
            "internet": internet,
        }]
    )
    try:
        prediction = float(model.predict(input_data)[0])
        prediction = max(0.0, min(20.0, prediction))
        rounded = round(prediction, 1)

        if prediction < 10:
            category = "Needs additional support"
            advice = "Consider regular revision, targeted support, and a discussion with the teacher."
        elif prediction < 14:
            category = "Developing"
            advice = "A consistent study routine and practice on weaker topics may help."
        elif prediction < 17:
            category = "Good"
            advice = "Maintain the current routine and focus on topics where mistakes occur."
        else:
            category = "Strong"
            advice = "Continue the effective study habits and keep reviewing key concepts."

        st.divider()
        st.subheader("Prediction result")
        left, right = st.columns([1, 2])
        with left:
            st.metric("Estimated final grade", f"{rounded} / 20")
            st.progress(min(max(prediction / 20, 0.0), 1.0))
        with right:
            st.success(f"Performance band: **{category}**")
            st.write(advice)
            st.caption(
                "This is a statistical estimate based on historical data. It should not be "
                "used as the sole basis for academic decisions."
            )

        with st.expander("View submitted model inputs"):
            st.dataframe(input_data, use_container_width=True, hide_index=True)
    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()
st.subheader("How the system works")
step1, step2, step3 = st.columns(3)
step1.markdown("**1. Prepare data**\n\nClean the public dataset and select relevant features.")
step2.markdown("**2. Train and compare**\n\nEvaluate Random Forest, Extra Trees, and Gradient Boosting.")
step3.markdown("**3. Predict**\n\nUse the selected preprocessing-and-model pipeline for new inputs.")

with st.expander("Model evaluation details"):
    if metrics:
        st.write(f"Selected model: **{selected_model}**")
        st.write(f"Test RMSE: **{metrics.get('rmse', 0):.3f}**")
        st.write(f"Test MAE: **{metrics.get('mae', 0):.3f}**")
        st.write(f"Test R²: **{metrics.get('r2', 0):.3f}**")
        st.caption("These are held-out test-set results from the local training run.")
    else:
        st.write("Run training locally to generate model evaluation metadata.")

st.caption("CPPE Mini Project | Python • Scikit-learn • MLflow • Streamlit • Hugging Face Spaces")
