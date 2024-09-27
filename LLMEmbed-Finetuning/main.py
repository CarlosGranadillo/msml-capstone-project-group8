"""
    This is the main module
"""

# Local Imports
from preprocess import Preprocess
from embeddings import Embeddings
from dataset import Data

# General Imports
import os
import warnings

warnings.filterwarnings("ignore")


def main(debug: bool = False, preprocess: bool = True, extract: bool = True):
    """
    This method is the starting point for the project. It performs the following tasks -
        1. Preprocess the datasets.
            -> Loading the datasets, filtering, null checks, drop duplicate rows, selecting required columns, renaming columns, datasets seggregation, column values conversion.
        2. Extract Embedding from the datasets using the LLM's.
        3. Run the base downstream model on the extracted embeddings.
        4.
    """
    # 1. Preprocess the datasets
    if preprocess:
        datasets = Preprocess(debug).preprocess()
    else:
        print(
            "----------------------------------Skipping Preprocessing--------------------------------------------------------"
        )
    # 2. Extract the embeddings
    if extract:
        embeddings = Embeddings(debug).extract(datasets=datasets)
    else:
        print(
            "----------------------------------Skipping Embeddings Extraction-----------------------------------------------"
        )

    # 3. Run the downstream model on the extracted embeddings
    data = Data(debug).extract_data()

if __name__ == "__main__":
    main(debug=False, preprocess=False, extract=False)
