import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib

# ─────────────────────────────────────────────
# @st.cache_resource
def load_model():
    with open("lightGBM_bayes_auc.pkl", "rb") as f:
         lgbm=joblib.load(f)
    with open("Catboost_bayes_auc.pkl", "rb") as f:
         cbt=joblib.load(f)
        
    return lgbm, cbt



def soft_vote_proba(models, X):
    probs = [m.predict_proba(X)[:, 1] for m in models]
    return sum(probs) / len(probs)

# ─────────────────────────────────────────────
def input_values():
    st.subheader("Basic Information")
    sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
    sex = 1 if sex == "Male" else 2

    age = st.number_input("Age", value=30)
    region = st.radio("Region", ["Urban", "Rural"], horizontal=True)
    region = 1 if region == "Urban" else 2
    income = st.selectbox("Income level (1=Low ~ 4=High)", [1, 2, 3, 4])

    st.subheader(" Diagnosis History")
    m_str = st.radio("Stroke diagnosis", ["No", "Yes"], horizontal=True)
    m_str = 1 if m_str == "Yes" else 0
    m_htn = st.radio("Hypertension diagnosis", ["No", "Yes"], horizontal=True)
    m_htn = 1 if m_htn == "Yes" else 0
    m_dia = st.radio("Diabetes diagnosis", ["No", "Yes"], horizontal=True)
    m_dia = 1 if m_dia == "Yes" else 0

    st.subheader("Lifestyle")
    smk = st.selectbox("Smoking status", [
        "0: Non-smoker", "1: Past smoker", "2: Occasional smoker", "3: Regular smoker"
    ])
    smk = int(smk.split(":")[0])

    drnk = st.selectbox("Alcohol consumption", [
        "0: None", "1: Rarely (≤ once/month)", "2: Sometimes (1–4/month)", "3: Frequently (≥ 2/week)"
    ])
    drnk = int(drnk.split(":")[0])

    exercise = st.selectbox("Physical activity frequency", [
        "0: None", "1: Rarely", "2: 1–2 times/week", "3: ≥ 3 times/week"
    ])
    exercise = int(exercise.split(":")[0])

    st.subheader("Health Screening Results")
    waist = st.number_input("Waist circumference (cm)", value=80.0)
    bp_high = st.number_input("Systolic blood pressure", value=120.0)
    bp_low = st.number_input("Diastolic blood pressure", value=80.0)
    blds = st.number_input("Fasting blood sugar", value=100.0)
    tot_chole = st.number_input("Total cholesterol", value=180.0)
    triglyceride = st.number_input("Triglyceride", value=150.0)
    hdl = st.number_input("HDL cholesterol", value=50.0)
    ldl = st.number_input("LDL cholesterol", value=100.0)
    hmg = st.number_input("Hemoglobin", value=13.0)
    sgot = st.number_input("SGOT (AST)", value=25.0)
    sgpt = st.number_input("SGPT (ALT)", value=25.0)
    gamma_gtp = st.number_input("Gamma-GTP", value=30.0)
    bmi = st.number_input("BMI", value=22.0)

    inputs = {
        "SEX": sex, "Age": age, "Region": region, "Income": income,
        "M_STR": m_str, "M_HTN": m_htn, "M_DIA": m_dia,
        "SMK": smk, "DRNK": drnk, "EXERCISE": exercise,
        "WAIST": waist, "BP_HIGH": bp_high, "BP_LWST": bp_low,
        "BLDS": blds, "TOT_CHOLE": tot_chole, "TRIGLYCERIDE": triglyceride,
        "HDL_CHOLE": hdl, "LDL_CHOLE": ldl, "HMG": hmg,
        "SGOT_AST": sgot, "SGPT_ALT": sgpt, "GAMMA_GTP": gamma_gtp,
        "BMI": bmi
    }

    model_columns = [
        "WAIST", "BP_HIGH", "BP_LWST", "BLDS", "TOT_CHOLE", "TRIGLYCERIDE",
        "HDL_CHOLE", "LDL_CHOLE", "HMG", "SGOT_AST", "SGPT_ALT", "GAMMA_GTP",
        "SMK", "DRNK", "EXERCISE", "M_STR", "M_HTN", "M_DIA",
        "BMI", "SEX", "Age", "Region", "Income"
    ]

    X_input = pd.DataFrame([[inputs[col] for col in model_columns]], columns=model_columns)
    return X_input

# ─────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Health Risk Predictor", layout="centered")
    st.title("Dyslipidemia Disease Risk Prediction (Ensemble Model)")
    st.caption("Estimate your disease risk by entering your health and lifestyle information.")

    X_input = input_values()
    lgbm_model, cat_model = load_model()
    y_proba = soft_vote_proba((lgbm_model, cat_model), X_input)

    st.markdown("---")
    st.subheader("Prediction Result")
    st.metric("Predicted Probability", f"{y_proba[0]*100:.2f}%")

# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
