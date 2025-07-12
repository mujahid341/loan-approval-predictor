import pandas as pds
from load_dataset import load_dataset

def clean_dataset():
    # Load the dataset
    dataset = load_dataset()
    
    if dataset is None:
        return None
    
    

