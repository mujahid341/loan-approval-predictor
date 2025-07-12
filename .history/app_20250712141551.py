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