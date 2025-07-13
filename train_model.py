import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# load the dataset from a CSV file
df = pd.read_csv('loan_approval_dataset.csv')

# preprocess the dataset
df = df[['education', 'self_employed','income_annum', 'loan_amount','cibil_score', 'loan_status']]

# convert categorical variables to numerical
df['education'] = df['education'].str.strip().map({'Graduate': 1, 'Not Graduate': 0})
df['self_employed'] = df['self_employed'].str.strip().map({'Yes': 1, 'No': 0})
df['loan_status'] = df['loan_status'].str.strip().map({'Approved': 1, 'Rejected': 0})


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

#train model using Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=400, random_state=42)

# fit the model on the training data
rf_model.fit(X_train, y_train)  
# Evaluate the model on the test data
rf_y_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_conf_matrix = confusion_matrix(y_test, rf_y_pred)
print(f'Random Forest Accuracy: {rf_accuracy}')
print(f'Random Forest Confusion Matrix:\n{rf_conf_matrix}')

# save the using joblib
# joblib.dump(rf_model, 'rf_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')
