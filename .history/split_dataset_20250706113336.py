from sklearn.model_selection import train_test_split
from load_dataset import load_dataset

def split_dataset():
    # Load the dataset
    dataset = load_dataset()
    print(dataset)
    
    if dataset is None:
        return None
    
    # Split the dataset into features and target variable
    X = dataset[["income", "credit_score", "loan_amount"]] #features
    y = dataset['approval'] #target variable
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Dataset split into training and testing sets successfully.")
    
    return X_train, X_test, y_train, y_test