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


warnings.filterwarnings("ignore")


def main(
    debug : bool = False,
    extract : bool = False,
    save_data_in_local : bool = False,
    read_data_from_local : bool = False,
    use_finetuned_model : bool = False,
    finetune : bool = False
):
    """
    This method is the starting point for the project. It performs the following tasks -
        1. Preprocess the datasets.
            -> Loading the datasets, filtering, null checks, drop duplicate rows, selecting required columns, renaming columns, datasets seggregation, column values conversion.
        2. Extract embeddings from the datasets using the base LLM's.
        3. Finetune the LLM models.
        4. Extract embeddings from the datasets using the finetuned LLM's.
        5. Run the base downstream model on the extracted finetuned embeddings for the results.
        6. Compare the results
    """
    # Clear CUDA Cache
    print("Clearing CUDA Cache")
    torch.cuda.empty_cache()

    # Clear huggingface Cache
    Helpers().clear_huggingface_cache()

    if extract:
        # 1. Preprocess the datasets
        datasets = Preprocess(debug).preprocess(
            save_data_in_local=save_data_in_local,
            read_data_from_local=read_data_from_local,
        )

        # 2. Extract the embeddings
        datasets_to_extract_embeddings = {
            "sentiment_analysis": datasets["sentiment_analysis"],
            "yes_no_question": datasets["yes_no_question"],
        }
        Embeddings(debug, use_finetuned_model=use_finetuned_model).extract(
            datasets=datasets_to_extract_embeddings,
            bert = True,
            roberta = True,
            llama2 = True
        )
    else:
        print("\n","-"*30,"Skipping Embeddings Extraction","-"*30)
    
    # 3. Fine tune the base LLM models
    if finetune:
        FineTune(debug).finetune(
            bert=True, 
            roberta=True, 
            llama2=True
        )

    else:
        print("\n","-"*30,"Skipping Finetuning","-"*30)

    # 3. Run the downstream model on the extracted embeddings
    epochs = 50
    SIGMA = 10
    learning_rate = 0.001

    metrics_base = Execute(
        debug, epochs=epochs, SIGMA=SIGMA, learning_rate=learning_rate
    ).execute(use_finetuned_embeddings=False)
    metrics_base_df = pd.DataFrame.from_dict(metrics_base, orient="index")
    metrics_finetuned = Execute(
        debug, epochs=epochs, SIGMA=SIGMA, learning_rate=learning_rate
    ).execute(use_finetuned_embeddings=True)
    metrics_finetuned_df = pd.DataFrame.from_dict(metrics_finetuned, orient="index")
    filename = f"results_SIGMA={SIGMA}_LR={learning_rate}_EPOCHS={epochs}"
    # Helpers().save_model_results(df=metrics_base_df, finetuned=False, filename=filename)
    # Helpers().save_model_results(df=metrics_finetuned_df, finetuned=True, filename=filename)

    print("Metrics for Base LLM's :")
    print(metrics_base_df)
    print("-" * 100)
    print("Metrics for Finetuned LLM's :")
    print(metrics_finetuned_df)


if __name__ == "__main__":
    main(
        debug=True, 
        extract=False,
        save_data_in_local=True,  
        read_data_from_local=True,  
        use_finetuned_model=True,
        finetune=False  
    )


