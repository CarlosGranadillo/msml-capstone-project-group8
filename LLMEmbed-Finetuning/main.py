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
from config import Config

# General Imports
import os
import warnings
import torch
import pandas as pd
import numpy as np


warnings.filterwarnings("ignore")


def main(
    debug: bool = False,
    extract: bool = False,
    save_data_in_local: bool = True,
    finetune: bool = False,
):
    """
    This method is the starting point for the project. It performs the following tasks -
        1. Preprocess the datasets.
            -> Loading the datasets, filtering, null checks, drop duplicate rows, selecting required columns, renaming columns, datasets seggregation, column values conversion.
        2. Extract embeddings from the datasets using the base LLM's.
        3. Finetune the LLM models.
        4. Extract embeddings from the datasets using the finetuned LLM's.
        5. Run the base downstream model on the extracted finetuned embeddings for the results.
        6. Compare the results.
    """
    # Clear CUDA Cache
    print("Clearing CUDA Cache")
    torch.cuda.empty_cache()

    # Clear huggingface Cache
    Helpers().clear_huggingface_cache()

    # 1. Preprocess the datasets
    if extract:
        datasets = Preprocess(debug).preprocess(
            save_data_in_local=save_data_in_local,
            read_data_from_local=False,
        )
        # 2. Extract the embeddings
        datasets_to_extract_embeddings = {
            "sentiment_analysis": datasets["sentiment_analysis"],
            "yes_no_question": datasets["yes_no_question"],
        }
        Embeddings(debug, use_finetuned_model=False).extract(
            datasets=datasets_to_extract_embeddings,
            bert=True,
            roberta=True,
            llama2=True,
        )
    else:
        print("\n","-" * 30,"Skipping Preprocessing and Embeddings extraction using base LLM's","-" * 10)

    # 3. Fine tune the base LLM models
    if finetune:
        FineTune(debug).finetune(bert=True, roberta=True, llama2=True)
    else:
        print("\n", "-" * 30, "Skipping finetuning of base LLM's", "-" * 40)

    # 1. Preprocess the datasets
    if extract:
        datasets = Preprocess(debug).preprocess(
            save_data_in_local=save_data_in_local,
            read_data_from_local=True,
        )
        # 2. Extract the embeddings
        datasets_to_extract_embeddings = {
            "sentiment_analysis": datasets["sentiment_analysis"],
            "yes_no_question": datasets["yes_no_question"],
        }
        Embeddings(debug, use_finetuned_model=True).extract(
            datasets=datasets_to_extract_embeddings,
            bert=True,
            roberta=True,
            llama2=True,
        )
    else:
        print("\n","-" * 30,"Skipping Preprocessing and Embeddings extraction using finetuned LLM's","-" * 10)

    # 3. Run the downstream model on the extracted embeddings for the tasks
    # sentiment paraemeters - epochs = 10,15,20,50,75

    task = "sentiment_analysis"  # "yes_no_question"
    print(f"\n[Started] - Running downstream model on {task}")
    epochs = 10
    learning_rate = 0.001  # 0.002 is making a huge difference
    for SIGMA in np.arange(0.1, 0.6, 0.1):
        metrics_base = Execute(
            debug, epochs=epochs, SIGMA=SIGMA, learning_rate=learning_rate
        ).execute(use_finetuned_embeddings=False, task=task)
        metrics_base_df = pd.DataFrame.from_dict(metrics_base, orient="index")

        metrics_finetuned = Execute(
            debug, epochs=epochs, SIGMA=SIGMA, learning_rate=learning_rate
        ).execute(use_finetuned_embeddings=True, task=task)
        metrics_finetuned_df = pd.DataFrame.from_dict(metrics_finetuned, orient="index")

        filename = f"results_SIGMA={SIGMA}_LR={learning_rate}_EPOCHS={epochs}"
        Helpers().save_model_results(
            df=metrics_base_df, finetuned=False, filename=filename, task=task
        )
        Helpers().save_model_results(
            df=metrics_finetuned_df, finetuned=True, filename=filename, task=task
        )
        print("SIGMA :", SIGMA)
        print("Metrics for Base LLM's :")
        print(metrics_base_df)
        print("-" * 100)
        print("Metrics for Finetuned LLM's :")
        print(metrics_finetuned_df)
        print(f"[Completed] - Running downstream model on {task}")


if __name__ == "__main__":
    main(debug=True, extract=False, save_data_in_local=True, finetune=False)
