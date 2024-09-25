"""
    This is the main module
"""
# Local Imports
from preprocess import Preprocess

# General Imports
import warnings

warnings.filterwarnings("ignore")


def main(debug : bool = False, preprocess : bool = True):
    """
    This method is the starting point for the project. It performs the following tasks -
        1. Preprocess the datasets
            -> Loading the datasets, filtering, null checks, drop duplicate rows, selecting required columns, renaming columns, datasets seggregation, column values conversion.
        2. Extract Embedding from the datasets.
        3. Run the based models on the extracted embeddings.
        4.
    """
    # 1. Preprocess the datasets
    if preprocess:
        datasets = Preprocess(debug).preprocess()
        return datasets


if __name__ == "__main__":
    main()
