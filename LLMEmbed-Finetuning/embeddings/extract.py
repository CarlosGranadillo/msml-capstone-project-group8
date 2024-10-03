"""
    This module is the starting point for th embeddings extraction.
"""

# Local Imports
from config import Config
from helpers import Helpers
from logger import Logger
from .bert_embeddings import Bert
from .roberta_embeddings import Roberta
from .llama2_embeddings import Llama2

# General Imports
from sklearn.model_selection import train_test_split
from collections import defaultdict


class Embeddings:
    """
    This class perform the following tasks.
        1. Extract embeddings for the Sujet-Finance-Instruct-177k using Bert Model and save the embeddings as .pt file.
        2. Extract embeddings for the Sujet-Finance-Instruct-177k using Llama Model and save the embeddings as .pt file.
        3. Extract embeddings for the Sujet-Finance-Instruct-177k using Roberta Model and save the embeddings as .pt file.
    """

    @classmethod
    def __init__(cls, enable_logging):
        """
        This method initializes the dictionary to save the datasets.
        """
        cls.config = Config()
        cls.helpers = Helpers()
        cls.log = Logger()
        cls.embeddings = defaultdict(dict)
        cls.bert = Bert(enable_logging)
        cls.roberta = Roberta(enable_logging)
        cls.llama2 = Llama2(enable_logging)
        cls.enable_logging = enable_logging
        cls.device = cls.config.get_device()

    @classmethod
    def extract(cls, datasets) -> dict:
        """
        This method extracts the embeddings using the LLM's.
        Extracting embeddings for only sentiment_analysis and yes_no_question from the dataset.
        Extraction will not be performed on the fine tuning datasets.
        """
        print("\n[Started] -  Embeddings extraction")
        tasks = cls.config.get_selected_task_types()

        for dataset_name, dataset in datasets.items():
            sentences = dataset["text"]
            labels = dataset["label"]
            # Train-test split
            (
                sentences_train,
                sentences_test,
                labels_train,
                labels_test,
            ) = train_test_split(sentences, labels, test_size=0.2, random_state=42)

            # Bert Training data emebeddings extraction
            return_dict = cls.bert.extract_bert_embeddings(
                mode="train",
                device=cls.device,
                sentences=sentences_train,
                labels=labels_train,
                task=dataset_name,
            )
            cls.embeddings[f"bert_{dataset_name}_train_embeddings"]["sentences"] = (
                return_dict[0]
            )
            cls.embeddings[f"bert_{dataset_name}_train_embeddings"]["labels"] = (
                return_dict[1]
            )

            # Bert Testing data emebeddings extraction
            return_dict = cls.bert.extract_bert_embeddings(
                mode="test",
                device=cls.device,
                sentences=sentences_test,
                labels=labels_test,
                task=dataset_name,
            )
            cls.embeddings[f"bert_{dataset_name}_test_embeddings"]["sentences"] = (
                return_dict[0]
            )
            cls.embeddings[f"bert_{dataset_name}_test_embeddings"]["labels"] = (
                return_dict[1]
            )

            # Roberta Training emebeddings extraction
            return_dict = cls.roberta.extract_roberta_embeddings(
                mode="train",
                device=cls.device,
                sentences=sentences_train,
                labels=labels_train,
                task=dataset_name,
            )
            cls.embeddings[f"roberta_{dataset_name}_train_embeddings"]["sentences"] = (
                return_dict[0]
            )
            cls.embeddings[f"roberta_{dataset_name}_train_embeddings"]["labels"] = (
                return_dict[1]
            )

            # Roberta Testing emebeddings extraction
            return_dict = cls.roberta.extract_roberta_embeddings(
                mode="test",
                device=cls.device,
                sentences=sentences_test,
                labels=labels_test,
                task=dataset_name,
            )
            cls.embeddings[f"roberta_{dataset_name}_test_embeddings"]["sentences"] = (
                return_dict[0]
            )
            cls.embeddings[f"roberta_{dataset_name}_test_embeddings"]["labels"] = (
                return_dict[1]
            )

            # Llama Training emebeddings extraction
            return_dict = cls.llama2.extract_llama2_embeddings(
                mode="train",
                device=cls.device,
                sentences=sentences_train,
                labels=labels_train,
                task=dataset_name,
            )
            cls.embeddings[f"llama2_{dataset_name}_train_embeddings"]["sentences"] = (
                return_dict[0]
            )
            cls.embeddings[f"llama2_{dataset_name}_train_embeddings"]["labels"] = (
                return_dict[1]
            )

            # Llama Testing emebeddings extraction
            return_dict = cls.llama2.extract_llama2_embeddings(
                mode="test",
                device=cls.device,
                sentences=sentences_test,
                labels=labels_test,
                task=dataset_name,
            )
            cls.embeddings[f"llama_{dataset_name}_test_embeddings"]["sentences"] = (
                return_dict[0]
            )
            cls.embeddings[f"llama_{dataset_name}_test_embeddings"]["labels"] = (
                return_dict[1]
            )

        print("[Completed] -  Embeddings extraction.")

        return cls.embeddings
