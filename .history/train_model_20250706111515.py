from sklearn.linear_model import LogisticRegression
from split_dataset import split_dataset

def train_model():

    # spit dataset into training and testing sets

    x_train, x_test, y_train, y_test = split_dataset()