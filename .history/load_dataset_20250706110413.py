import pandas as pd

# load the dataset

def load_dataset():
    
    # load the dataset from CSV file

    pathOfDataset = "loan_approval_training_data.csv"

    try:
        dataset = pd.read_csv(pathOfDataset)
        print("Dataset loaded successfully.")
        return dataset
    except FileNotFoundError:
        print(f"Error: The file {pathOfDataset} was not found.")
        return None

