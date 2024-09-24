"""
    This is the main module
"""
# Local Imports
from preprocess import Preprocess

# General Imports
import warnings

warnings.filterwarnings("ignore")


def main():
    """
    This method is the starting point for the project. It performs the following tasks -
        1. Preprocess the datasets
            -> Loading the datasets, filtering, null checks, drop duplicate rows, selecting required columns, renaming columns, datasets seggregation, column values conversion.
        2. Extract Embedding from the datasets
        3.
        4.
    """
    # 1. Preprocess the datasets
    datasets = Preprocess().main()
    return datasets

if __name__ == "__main__":
    main()
