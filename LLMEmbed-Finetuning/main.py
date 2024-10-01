"""
    This is the main module
"""

# Local Imports
from preprocess import Preprocess
from embeddings import Embeddings
from models.execute import Execute
from dataset import Data

# General Imports
import os
import warnings
import torch

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
    # Clear CUDA Cache
    print("Clearing CUDA Cache")
    torch.cuda.empty_cache()
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
        datasets_to_extract_embeddings = {
            "sentiment_analysis": datasets["sentiment_analysis"],
            "yes_no_question": datasets["yes_no_question"],
        }
        embeddings = Embeddings(debug).extract(datasets=datasets_to_extract_embeddings)
    else:
        print(
            "\n----------------------------------Skipping Embeddings Extraction-----------------------------------------------"
        )

    # 3. Run the downstream model on the extracted embeddings
    data = Data(debug).extract_data()
    metrics = Execute(debug).execute(data=data)
    print(metrics)


if __name__ == "__main__":
    main(
        debug=True,
        preprocess=False,
        extract=False,
        save_data_in_local=False,
        read_data_from_local=True,
    )
