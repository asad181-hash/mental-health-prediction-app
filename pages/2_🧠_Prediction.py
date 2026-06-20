import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Risk Prediction", page_icon="🧠", layout="wide")
st.title("🧠 Live Risk Prediction")

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/xgb_mental_health_model.pkl")
    columns = joblib.load("models/training_columns.pkl")
    template = joblib.load("models/row_template.pkl")
    return model, columns, template

model, training_columns, template_row = load_artifacts()

def categorize_sleep_hours(hours):
    if hours < 5: return 0
    elif hours < 6: return 1
    else: return 2

st.subheader("Enter Student Details")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("Age", 15, 40, 22)
    academic_pressure = st.slider("Academic Pressure", 0, 5, 3)
    work_pressure = st.slider("Work Pressure (0 if student only)", 0, 5, 0)
    study_satisfaction = st.slider("Study Satisfaction", 0, 5, 3)
with col2:
    job_satisfaction = st.slider("Job Satisfaction (0 if no job)", 0, 5, 0)
    work_study_hours = st.slider("Work/Study Hours per day", 0, 12, 6)
    financial_stress_raw = st.slider("Financial Stress", 1, 5, 2)
    sleep_hours = st.slider("Sleep Duration (hours)", 0.0, 12.0, 7.0, step=0.5)
with col3:
    cgpa = st.slider("CGPA (0-4 scale)", 0.0, 4.0, 3.0, step=0.1)
    suicidal_thoughts = st.selectbox("Suicidal Thoughts?", ["No", "Yes"]) == "Yes"
    family_history = st.selectbox("Family History of Mental Illness?", ["No", "Yes"]) == "Yes"
    gender = st.selectbox("Gender", ["Male", "Female"])

diet = st.selectbox("Dietary Habits", ["Unhealthy", "Others", "Moderate", "Healthy"])
degree_group = st.selectbox("Degree", ["HighSchool", "Bachelors", "Masters", "Doctorate", "Professional", "Other"])
profession_group = "Student"  # dataset population is overwhelmingly students

if st.button("🔍 Predict Risk", type="primary"):
    row = template_row.copy()

    row["gender_encoded"] = 0 if gender == "Male" else 1
    row["Academic Pressure"] = academic_pressure
    row["Work Pressure"] = work_pressure
    row["Study Satisfaction"] = study_satisfaction
    row["Job Satisfaction"] = job_satisfaction
    row["Work/Study Hours"] = work_study_hours
    row["Financial Stress"] = financial_stress_raw
    row["suicidal thoughts"] = int(suicidal_thoughts)
    row["Family History of Mental Illness_encoded"] = int(family_history)
    row["Financial_Stress_encoded"] = 1 if financial_stress_raw >= 3 else 0
    row["sleep_category_encoded"] = categorize_sleep_hours(sleep_hours)

    diet_order = ["Unhealthy", "Others", "Moderate", "Healthy"]
    row["dietary_habits_encoded"] = diet_order.index(diet)

    cgpa_cat = 0 if cgpa <= 2.0 else (1 if cgpa <= 3.0 else 2)
    row["CGPA_category_encoded"] = cgpa_cat

    if age <= 20: age_cat = 0
    elif age <= 25: age_cat = 1
    elif age <= 30: age_cat = 2
    else: age_cat = 3
    row["age_group_encoded"] = age_cat

    degree_cols = ["Degree_Bachelors","Degree_Doctorate","Degree_HighSchool",
                   "Degree_Masters","Degree_Other","Degree_Professional"]
    for col in degree_cols:
        if col in row.columns:
            row[col] = 0
    target_col = f"Degree_{degree_group}"
    if target_col in row.columns:
        row[target_col] = 1

    prof_cols = ["Profession_Creative","Profession_Education","Profession_Healthcare",
                 "Profession_Other","Profession_Student","Profession_Technology"]
    for col in prof_cols:
        if col in row.columns:
            row[col] = 0
    target_prof_col = f"Profession_{profession_group}"
    if target_prof_col in row.columns:
        row[target_prof_col] = 1

    row["Academic_Work_Stress"] = academic_pressure * work_pressure
    row["Satisfaction_Gap"] = study_satisfaction - job_satisfaction
    row["Total_Sleep_Impact"] = row["sleep_category_encoded"].values[0] * financial_stress_raw
    row["Stress_Score"] = (academic_pressure + work_pressure + financial_stress_raw) / 3

    for col in training_columns:
        if template_row[col].dtype.name == "category":
            row[col] = pd.Categorical([row[col].values[0]], categories=template_row[col].cat.categories)
        else:
            row[col] = row[col].astype(template_row[col].dtype)

    row = row[training_columns]
    probability = model.predict_proba(row)[0, 1]

    st.markdown("---")
    st.subheader("Result")

    if probability >= 0.65:
        st.error(f"⚠️ HIGH RISK — {probability:.1%} probability")
    elif probability >= 0.35:
        st.warning(f"🔶 MEDIUM RISK — {probability:.1%} probability")
    else:
        st.success(f"✅ LOW RISK — {probability:.1%} probability")

    st.progress(float(probability))