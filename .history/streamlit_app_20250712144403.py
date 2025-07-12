import streamlit as st
import joblib
import numpy as np
st.title("Loan Approval Prediction system")
st.write("Enter the details below to predict loan approval status.")

# Input fields for user data
with st.form(key="loan_form"):
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    income_annum = st.number_input("Annual Income", min_value=0.0, step=1000.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, step=1000.0)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, step=1)

    if st.form_submit_button("Predict"):
        # Load the model and scaler
        rf_model = joblib.load('rf_model.pkl')
        scaler = joblib.load('scaler.pkl')

        # Preprocess input data
        education_numeric = 1 if education == "Graduate" else 0
        self_employed_numeric = 1 if self_employed == "Yes" else 0
        input_data = np.array([[education_numeric, self_employed_numeric, income_annum, loan_amount, cibil_score]])

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = rf_model.predict(input_data_scaled)
        loan_status = "Approved" if prediction[0] == 1 else "Rejected"

        st.write(f"Loan Status: {loan_status}")