from sklearn.metrics import accuracy_score , confusion_matrix
from train_model import train_model
from split_dataset import split_dataset

def evaluate_model_performace(x_test, y_test):
    # train the model
    # split the dataset
    
    model = train_model()

    prediction = model.predict(x_test)
    accuracy = accuracy_score(y_test, prediction)
    print("accuracy : ", accuracy)
    conf_metrix = confusion_matrix(y_test, prediction)
    print("confusion metrix: ",conf_metrix)

