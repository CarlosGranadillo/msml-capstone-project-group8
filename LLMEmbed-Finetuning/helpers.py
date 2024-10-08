"""
    This module contains the Helpers class
"""

# General Imports
import os
import shutil
import torch
from datasets import Dataset, concatenate_datasets, load_from_disk

# Local Imports
from config import Config


class Helpers:
    """
    This class contains all the helper methods that are used in this project
    """

    @classmethod
    def __init__(cls):
        cls.config = Config()

    @classmethod
    def convert_column_to_lowercase(cls, example, column_to_lowercase: str) -> str:
        """
        This method converts the column values into lowercase.
        """
        example[column_to_lowercase] = example[column_to_lowercase].lower()
        return example

    @classmethod
    def replace_string_with_int(cls, example, mapping: dict, column_to_modify: str):
        """
        This function converts string values to integers in a specific column.
        """
        example[column_to_modify] = mapping.get(example[column_to_modify], -1)
        return example

    @classmethod
    def replace_int_with_string(cls, example, mapping: dict, column_to_modify: str):
        """
        This function converts integer values to string in a specific column.
        """
        example[column_to_modify] = mapping.get(example[column_to_modify], "None")
        return example

    @classmethod
    def concatenate_sentimental_analysis_datasets(cls, dataset1, dataset2):
        """
        This method will concatenate the two datasets with same structure.
        """
        print("\n[Started] - Concatenation of the datasets")
        dataset1_df = dataset1["train"].to_pandas()
        dataset2_df = dataset2["train"].to_pandas()
        temp1 = Dataset.from_pandas(dataset1_df)
        temp2 = Dataset.from_pandas(dataset2_df)
        concanted_dataset = concatenate_datasets([temp1, temp2])
        print("[Completed] - Concatenation of the datasets")
        return concanted_dataset

    @classmethod
    def save_embeddings(
        cls, sentences_embeds: list, labels: list, file_path: str, mode: str
    ):
        """
        This method saves the embeddings in a local folder
        """
        save_path = cls.config.get_base_path() + "embeddings/"
        if not os.path.exists(save_path + file_path):
            os.makedirs(save_path + file_path)
        torch.save(
            sentences_embeds.to("cpu"), save_path + file_path + f"{mode}_texts.pt"
        )
        torch.save(labels, save_path + file_path + f"{mode}_labels.pt")
        print(f"Saved at :", save_path + file_path)

    @classmethod
    def save_dataset(cls, dataset, file_name: str):
        """
        This method will save the dataset to a local folder inorder to reuse.
        """
        save_path = cls.config.get_base_path() + "data/"
        file_path = save_path + file_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        elif os.path.exists(file_path):
            shutil.rmtree(file_path)
            dataset.save_to_disk(file_path)
        else:
            dataset.save_to_disk(file_path)

    @classmethod
    def save_finetuned_model(cls, trainer, model_name: str):
        """
        This method will save the dataset to a local folder inorder to reuse.
        """
        save_path = cls.config.get_base_path() + "finetuned_models/"
        model_path = save_path + model_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        elif os.path.exists(model_path):
            shutil.rmtree(model_path)
            trainer.model.save_pretrained(model_path)
        else:
            trainer.model.save_pretrained(model_path)

    @classmethod
    def read_dataset_from_local(cls, dataset_name: str):
        """
        This function loads the data from a local directory.
        """
        load_path = cls.config.get_base_path() + "data/" + dataset_name
        if not os.path.exists(load_path):
            raise Exception(
                f"{load_path} does not exists. Please save the data to local again."
            )
            return
        dataset = load_from_disk(load_path)
        return dataset

    @classmethod
    def clear_huggingface_cache(cls):
        """
        This function clears the hugging face cache
        """
        print("Clearing Hugging Face Cache")
        cache_dir = os.path.expanduser("~/.cache/huggingface/")

        # Check if the directory exists and remove it
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)  # Recursively delete the directory
            print(f"Cache directory: '{cache_dir}' cleared successfully.")
        else:
            print(f"Cache directory: '{cache_dir}' does not exist.")
