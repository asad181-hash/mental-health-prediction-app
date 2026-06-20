import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Mental Health Risk Predictor", page_icon="🧠", layout="wide")

st.title("🧠 Early Mental Health Risk Prediction")
st.caption("Final Year Project — Predicting depression risk from academic, financial, and lifestyle factors")

@st.cache_resource
def load_info():
    return joblib.load("models/deployment_info.pkl")

info = load_info()

st.markdown("---")
st.subheader("Model Performance")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{info['accuracy']*100:.1f}%")
col2.metric("Precision", f"{info['precision']*100:.0f}%")
col3.metric("Recall", f"{info['recall']*100:.0f}%")
col4.metric("F1 Score", f"{info['f1']*100:.0f}%")

st.markdown("---")
st.subheader("Key Findings")

st.markdown("""
1. **Suicidal thoughts** is by far the strongest predictor of depression risk in this dataset.
2. **Stress_Score**, an engineered feature combining academic, work, and financial stress, is the second most important predictor — proof that feature engineering meaningfully improved the model.
3. **Academic Pressure** and **Financial Stress** follow as strong, intuitive predictors.
4. **Sleep duration** shows the expected negative relationship with depression, but its individual importance is diluted because its signal overlaps with stress-related features already in the model.
5. **Dietary habits** has the strongest negative correlation with depression — healthier diet, lower risk.
""")

st.markdown("---")
st.subheader("Top 5 Predictors")

importance_data = pd.DataFrame({
    "Feature": ["Suicidal Thoughts", "Stress Score", "Academic Pressure", "Financial Stress", "Dietary Habits"],
    "Importance": [0.472, 0.203, 0.102, 0.039, 0.024]
})
st.bar_chart(importance_data.set_index("Feature"))

st.markdown("---")
st.info("👈 Use the sidebar to explore the data analysis or try the live prediction tool.")