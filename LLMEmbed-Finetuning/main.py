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


def main(
    debug: bool,
    preprocess: bool,
    extract: bool,
    save_data_in_local: bool,
    read_data_from_local: bool,
):
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
        datasets = Preprocess(debug).preprocess(
            save_data_in_local=save_data_in_local,
            read_data_from_local=read_data_from_local,
        )
    else:
        print(
            "\n----------------------------------Skipping Preprocessing--------------------------------------------------------"
        )
    # 2. Extract the embeddings
    if extract:
        embeddings = Embeddings(debug).extract(datasets=datasets)
    else:
        print(
            "\n----------------------------------Skipping Embeddings Extraction-----------------------------------------------"
        )

    # 3. Run the downstream model on the extracted embeddings
    #data = Data(debug).extract_data()


if __name__ == "__main__":
    main(
        debug=True,
        preprocess=True,
        extract=False,
        save_data_in_local=True,
        read_data_from_local=False,
    )
