from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
# Load the pre-trained model and scaler
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

class LoanApplication(BaseModel):
    education: int  # 1 for Graduate, 0 for Not Graduate
    self_employed: int  # 1 for Yes, 0 for No
    income_annum: float
    loan_amount: float
    cibil_score: float

@app.post("/predict")
def predict_loan_status(application: LoanApplication):
    # Prepare the input data
    input_data = np.array([
            [application.education,
            application.self_employed,
            application.income_annum, application.loan_amount,
            application.cibil_score]
        ])
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = rf_model.predict(input_data_scaled)
    
    # Convert prediction to human-readable format
    loan_status = "Approved" if prediction[0] == 1 else "Rejected"
    
    # Returning the prediction result in JSON format
    return {"loan_status": loan_status}