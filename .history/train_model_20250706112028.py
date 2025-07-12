from sklearn.linear_model import LogisticRegression
from split_dataset import split_dataset

def train_model(x, y):

    # spit dataset into training and testing sets

    # train the model
    model = LogisticRegression()
    model.fit(x, y) 
    print("Model trained successfully.")
    return model