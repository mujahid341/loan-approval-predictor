from sklearn.metrics import accuracy_score , confusion_matrix
from train_model import train_model
from split_dataset import split_dataset

def evaluate_model_performace():
    # train the model
    # split the dataset
    _, x_test, _, y_test = split_dataset()
    
    model = train_model()
    prediction = model.predict()

