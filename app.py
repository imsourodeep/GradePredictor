# app.py
# This is the main file — run this to launch the web app
# Command: streamlit run app.py

import streamlit as st
import numpy as np
import joblib
import os
import requests
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Grade Predictor by Sourodeep Guin",
    page_icon="🎓",
    layout="centered"
)

# ── Header ─────────────────────────────────────────────────────────
st.title("🎓 Student Grade Predictor")
st.markdown("Fill in your details below to predict your final exam grade and get personalized advice.")
st.markdown(
    "<div style='text-align:right; color:gray; font-size:13px;'>Built by <b>Sourodeep Guin</b> | ML + Groq AI</div>",
    unsafe_allow_html=True
)
st.divider()

# ── Sidebar ─────────────────────────────────────────────────────────
st.sidebar.image("https://avatars.githubusercontent.com/imsourodeep", width=80)
st.sidebar.markdown("## About")
st.sidebar.markdown("This app predicts your final exam grade using a **Machine Learning** model and provides personalized study advice powered by **Groq AI (LLaMA 3.1)**.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Made by Sourodeep Guin**")
st.sidebar.markdown("🔗 [GitHub](https://github.com/imsourodeep)")

# ── Load ML model (train if not found) ─────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists("grade_model.pkl"):
        st.info("🔧 Training model for the first time... please wait.")
        from model import train_model
        return train_model()
    return joblib.load("grade_model.pkl")

model = load_model()

# ── Input Form ─────────────────────────────────────────────────────
st.subheader("📋 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider(
        "📚 Study Hours per Day",
        min_value=0.0, max_value=10.0, value=4.0, step=0.5,
        help="Average hours you study each day"
    )
    attendance = st.slider(
        "🏫 Attendance (%)",
        min_value=40, max_value=100, value=75,
        help="Percentage of classes attended"
    )
    sleep_hours = st.slider(
        "😴 Sleep Hours per Night",
        min_value=3.0, max_value=10.0, value=7.0, step=0.5,
        help="Average hours of sleep per night"
    )

with col2:
    prev_score = st.number_input(
        "📝 Previous Exam Score (out of 100)",
        min_value=0, max_value=100, value=65,
        help="Your score in the last exam"
    )
    extracurricular = st.radio(
        "⚽ Extracurricular Activities?",
        options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Do you participate in sports, clubs, etc.?"
    )

st.divider()

# ── Predict Button ──────────────────────────────────────────────────
if st.button("🔮 Predict My Grade", use_container_width=True, type="primary"):

    # ── Step 1: ML Model Prediction ─────────────────────────────────
    input_data = np.array([[study_hours, attendance, sleep_hours, prev_score, extracurricular]])
    predicted_grade = model.predict(input_data)[0]
    predicted_grade = round(float(np.clip(predicted_grade, 0, 100)), 1)

    # ── Step 2: Display Predicted Grade ─────────────────────────────
    st.subheader("📊 Prediction Result")

    if predicted_grade >= 75:
        color = "green"
        emoji = "🟢"
        label = "Great!"
    elif predicted_grade >= 50:
        color = "orange"
        emoji = "🟡"
        label = "Average"
    else:
        color = "red"
        emoji = "🔴"
        label = "Needs Improvement"

    st.markdown(
        f"<h2 style='text-align:center; color:{color};'>{emoji} Predicted Grade: {predicted_grade}/100 — {label}</h2>",
        unsafe_allow_html=True
    )

    # ── Step 3: LLM Advice via Groq API ───────────────────────────
    st.subheader("🤖 Personalized Advice from Groq AI")

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.warning("⚠️ No API key found. Check your .env file.")
    else:
        with st.spinner("Getting personalized advice from Groq AI..."):
            try:
                prompt = f"""You are a friendly and supportive academic advisor. A student has the following profile:

- Study hours per day: {study_hours} hours
- Class attendance: {attendance}%
- Sleep hours per night: {sleep_hours} hours
- Previous exam score: {prev_score}/100
- Participates in extracurricular activities: {"Yes" if extracurricular else "No"}
- Predicted final grade (by ML model): {predicted_grade}/100

Please give this student:
1. A brief analysis of their strongest and weakest habits (2-3 sentences)
2. 3 specific, actionable tips to improve their grade
3. One encouraging closing sentence

Keep the tone warm, motivating, and practical."""

                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 600
                }

                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload
                )

                result = response.json()
                advice = result["choices"][0]["message"]["content"]
                st.markdown(advice)

            except Exception as e:
                st.error(f"❌ Groq API error: {str(e)}")

    # ── Step 4: Show input summary ───────────────────────────────────
    with st.expander("📋 See your input summary"):
        st.json({
            "Study Hours/Day": study_hours,
            "Attendance (%)": attendance,
            "Sleep Hours/Night": sleep_hours,
            "Previous Score": prev_score,
            "Extracurricular": "Yes" if extracurricular else "No",
            "Predicted Grade": predicted_grade
        })

# ── Footer ──────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:gray; font-size:12px;'>© 2026 Sourodeep Guin| Student Grade Predictor | Powered by ML + Groq AI</div>",
    unsafe_allow_html=True
)
