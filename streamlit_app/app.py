import streamlit as st
import requests
import json

st.title("🏦 Loan Approval Prediction")

st.write("Lütfen kredi başvuru bilgilerinizi girin:")

# Kullanıcı girişi
gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
married = st.selectbox("Married", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
dependents = st.number_input("Dependents", min_value=0, max_value=5, value=0)
education = st.selectbox("Education", [0, 1], format_func=lambda x: "Not Graduate" if x == 0 else "Graduate")
self_employed = st.selectbox("Self Employed", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
loan_amount_log = st.number_input("LoanAmount_log", value=4.5)
loan_amount_term = st.number_input("LoanAmount Term", value=360)
credit_history = st.selectbox("Credit History", [0, 1])
total_income_log = st.number_input("TotalIncome_log", value=10.5)
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Tahmin düğmesi
if st.button("Tahmin Et"):
    data = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "LoanAmount_log": loan_amount_log,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "TotalIncome_log": total_income_log,
        "Property_Area": property_area,
    }
    response = requests.post("http://loan_api:8000/predict", json=data)
    if response.status_code == 200:
        result = response.json()
        st.success(f"Prediction: {result['meaning']} (Loan Status: {result['loan_status_prediction']})")
    else:
        st.error(f"API Error: {response.status_code}")
