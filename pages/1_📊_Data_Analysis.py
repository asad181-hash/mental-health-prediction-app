import streamlit as st
import pandas as pd

st.set_page_config(page_title="Data Analysis", page_icon="📊", layout="wide")
st.title("📊 Exploratory Data Analysis")

@st.cache_data
def load_data():
    return pd.read_csv("data/mental_health_preprocessed.csv")

df = load_data()
st.caption(f"Dataset: {df.shape[0]:,} students, {df.shape[1]} features")

st.markdown("---")
st.subheader("😴 Sleep Deep-Dive")
st.write("This directly answers the question: does sleep actually matter for depression risk?")

sleep_labels = {
    0: "Insufficient (<5h)",
    1: "Below Recommended (6-7h)",
    2: "Recommended (7-9h)",
    3: "Excellent"
}

sleep_summary = (
    df.dropna(subset=["sleep_category_encoded"])
    .groupby("sleep_category_encoded")["Depression"]
    .mean()
    .rename("Depression Rate")
    .reset_index()
)
sleep_summary["Sleep Category"] = sleep_summary["sleep_category_encoded"].map(sleep_labels)
sleep_summary["Sleep Category"] = sleep_summary["Sleep Category"].fillna("Unknown")

col1, col2 = st.columns([2, 1])
with col1:
    st.bar_chart(sleep_summary.set_index("Sleep Category")["Depression Rate"])
with col2:
    st.write("**Depression rate by sleep category:**")
    for _, row in sleep_summary.iterrows():
        st.metric(row["Sleep Category"], f"{row['Depression Rate']*100:.1f}%")

st.caption("Sleep shows a real relationship with depression rate, even though its standalone model importance is low — its signal overlaps with Stress Score and Academic Pressure.")

st.markdown("---")
st.subheader("🔍 Explore Any Feature's Relationship with Depression")

feature = st.selectbox(
    "Pick a feature to compare against Depression",
    ["Academic Pressure", "Financial Stress", "Work/Study Hours", "Stress_Score", "CGPA_category_encoded"]
)

chart_data = df.groupby(feature)["Depression"].mean().reset_index()
st.bar_chart(chart_data.set_index(feature))

st.markdown("---")
st.subheader("📈 Feature Correlation with Depression")

corr = df.corr(numeric_only=True)["Depression"].drop("Depression").sort_values()
st.bar_chart(corr)