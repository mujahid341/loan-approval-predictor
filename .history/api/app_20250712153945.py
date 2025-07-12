from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the pre-trained model and scaler
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

class LoanApplication(BaseModel):
    education: str  # "Graduate" or "Not Graduate"
    self_employed: str  # "Yes" or "No"
    income_annum: float
    loan_amount: float
    cibil_score: float

#Preprocessing function to convert string to numeric
def preprocess(application: LoanApplication):
    education = 1 if application.education.strip().lower() == "graduate" else 0
    self_employed = 1 if application.self_employed.strip().lower() == "yes" else 0
    return np.array([[education, self_employed, application.income_annum,
                      application.loan_amount, application.cibil_score]])

@app.post("/predict")
def predict_loan_status(application: LoanApplication):
    # Convert readable input to numeric
    input_data = preprocess(application)
    
    # Scale the input
    input_data_scaled = scaler.transform(input_data)
    
    # Predict using trained model
    prediction = rf_model.predict(input_data_scaled)
    
    # Interpret prediction
    loan_status = "Approved" if prediction[0] == 1 else "Rejected"
    
    return {"loan_status": loan_status}
