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
        print(
            "\n","-" * 30,"Skipping Preprocessing and Embeddings extraction using base LLM's","-" * 10)

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
        # 4. Extract the embeddings using fine tuned LLM's
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

    # 5. Run the downstream model on the extracted embeddings for the tasks
    learning_rates = {
        "sentiment_analysis": [0.001, 0.002, 0.0001],
        "yes_no_question": [0.001, 0.0001, 0.0002],
    }
    tasks = ["sentiment_analysis", "yes_no_question"]
    epochs = [10, 25, 50, 75, 100]
    SIGMA_values = np.arange(0.1, 0.6, 0.1)
    for task in tasks:
        for epoch in epochs:
            for lr in learning_rates[task]:
                for SIGMA in SIGMA_values:
                    SIGMA = round(SIGMA, 1)
                    print(
                        f"\n[Started] - Running downstream model on {task} with parameters : [epoch : {epoch}, lr : {lr}, SIGMA : {SIGMA}]"
                    )
                    folder = f"EPOCH={epoch}/LR={lr}/SIGMA={SIGMA}"
                    filename = f"/metrics"
                    metrics_base = Execute(
                        debug, epochs=epoch, SIGMA=SIGMA, learning_rate=lr
                    ).execute(use_finetuned_embeddings=False, task=task)
                    metrics_base_df = pd.DataFrame.from_dict(
                        metrics_base, orient="index"
                    )

                    metrics_finetuned = Execute(
                        debug, epochs=epoch, SIGMA=SIGMA, learning_rate=lr
                    ).execute(use_finetuned_embeddings=True, task=task)
                    metrics_finetuned_df = pd.DataFrame.from_dict(
                        metrics_finetuned, orient="index"
                    )

                    Helpers().save_model_results(
                        df=metrics_base_df,
                        finetuned=False,
                        filename=filename,
                        folder=folder,
                        task=task,
                    )
                    Helpers().save_model_results(
                        df=metrics_finetuned_df,
                        finetuned=True,
                        filename=filename,
                        folder=folder,
                        task=task,
                    )
                    # 6. Compare the results.
                    print("Metrics for Base LLM's :")
                    print(metrics_base_df)
                    print("-" * 100)
                    print("Metrics for Finetuned LLM's :")
                    print(metrics_finetuned_df)
                    print(f"\n[Started] - Running downstream model on {task} with parameters : [epoch : {epoch}, lr : {lr}, SIGMA : {SIGMA}]")


if __name__ == "__main__":
    main(debug=True, extract=True, save_data_in_local=True, finetune=True)
