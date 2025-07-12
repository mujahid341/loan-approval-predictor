from sklearn.linear_model import LogisticRegression
from split_dataset import split_dataset

def train_model(x_train, y_train):

    # train the model
    model = LogisticRegression()
    model.fit(x_train, y_train) 
    print("Model trained successfully.")
    return model