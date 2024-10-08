"""
    This is the main module
"""

# Local Imports
from preprocess import Preprocess
from embeddings import Embeddings
from finetuning.finetune import FineTune
from models.execute import Execute
from dataset import Data
from helpers import Helpers

# General Imports
import os
import warnings
import torch
import pandas as pd
import shutil

warnings.filterwarnings("ignore")


def main(
    debug: bool,
    preprocess: bool,
    extract: bool,
    save_data_in_local: bool,
    read_data_from_local: bool,
    use_finetuned_model: bool,
    use_finetuned_embdes: bool,
):
    """
    This method is the starting point for the project. It performs the following tasks -
        1. Preprocess the datasets.
            -> Loading the datasets, filtering, null checks, drop duplicate rows, selecting required columns, renaming columns, datasets seggregation, column values conversion.
        2. Extract embeddings from the datasets using the LLM's.
        3. Run the base downstream model on the extracted embeddings for the results.
        4. Finetune the LLM models.
        5. Extract embeddings from the datasets using the finetuned LLM's.
        6. Run the base downstream model on the extracted finetuned embeddings for the results.
        7. Compare the results
    """
    # Clear CUDA Cache
    print("Clearing CUDA Cache")
    torch.cuda.empty_cache()

    # Clear huggingface Cache
    print("Clearing Hugging Face Cache")
    Helpers().clear_huggingface_cache()

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
        embeddings = Embeddings(debug, use_finetuned_model=use_finetuned_model).extract(
            datasets=datasets_to_extract_embeddings
        )
    else:
        print(
            "\n----------------------------------Skipping Embeddings Extraction-----------------------------------------------"
        )

    # 3. Run the downstream model on the extracted embeddings
    metrics = Execute(debug).execute(use_finetuned_embdes)
    df = pd.DataFrame.from_dict(metrics, orient="index")
    print(df)

    # 4. Fine tune the LLM models
    FineTune(enable_logging = True).finetune()


if __name__ == "__main__":
    main(
        debug=True,  # True, if we want to enable debugging, else False.
        preprocess=True,  # True, if we want to preprocess the data from hugging face, else False.
        save_data_in_local=False,  # True, if we want save the huggingface datasets in local, else False.
        read_data_from_local=True,  # True, if we want to read the data saved in local, else False.
        extract=True,  # True, if we want to extract the embeddings and save it in local, False, if we want to load the embeddings saved in the local
        use_finetuned_model=True,  # True, if we want to use the fine tuned models to extract embeddings, else False.
        use_finetuned_embdes=False,  # True, if we want to use the fine tuned embeddings else False.
    )
    
    
