import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
from sklearn.preprocessing import StandardScaler

# load the dataset from a CSV file
df = pd.read_csv('loan_approval_dataset.csv')

# preprocess the dataset
df = df[['education', 'self_employed','income_annum', 'loan_amount','cibil_score', 'loan_status']]

# convert categorical variables to numerical
df['education'] = df['education'].map({'Graduate': 1, 'Not Graduate': 0})
df['self_employed'] = df['self_employed'].map({'Yes': 1, 'No': 0})
df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})

# clean the dataset
df = df.dropna()
df = df.drop_duplicates()

# saperate features and target variable
X = df.drop('loan_status', axis=1) # It contains all columns except 'loan_status'
y = df['loan_status']

# scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=True)    