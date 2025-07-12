import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
from sklearn.preprocessing import StandardScaler

# load the dataset from a CSV file
df = pd.read_csv('loan_approval_dataset.csv')

df = df[['graduate', 'self_employed','income_annum', 'loan_amount','cibil_score', 'loan_status']]