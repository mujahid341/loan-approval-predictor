from sklearn.metrics import accuracy_score, confusion_matrix
from train_model import train_model
from split_dataset import split_dataset

def evaluate_model_performace(x_test, y_test):
    # train the model
    model = train_model()

    # make predictions
    prediction = model.predict(x_test)
    
    # evaluate accuracy
    accuracy = accuracy_score(y_test, prediction)
    print("Accuracy:", accuracy)
    
    # confusion matrix
    conf_metrix = confusion_matrix(y_test, prediction)
    print("Confusion Matrix:\n", conf_metrix)

# Testing block
if __name__ == "__main__":
    x_train, x_test, y_train, y_test = split_dataset()
    evaluate_model_performace(x_test, y_test)
