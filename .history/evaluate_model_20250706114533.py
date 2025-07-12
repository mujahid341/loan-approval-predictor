from sklearn.metrics import accuracy_score, confusion_matrix
from train_model import train_model
from split_dataset import split_dataset

def evaluate_model_performance(x_train, x_test, y_train, y_test):
    # Train the model
    model = train_model(x_train, y_train)

    # Make predictions
    prediction = model.predict(x_test)
    
    # Evaluate accuracy
    accuracy = accuracy_score(y_test, prediction)
    print("Accuracy:", accuracy)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, prediction, labels=[0, 1])
    print("Confusion Matrix:\n", conf_matrix)

# Testing block
if __name__ == "__main__":
    x_train, x_test, y_train, y_test = split_dataset()
    evaluate_model_performance(x_train, x_test, y_train, y_test)
