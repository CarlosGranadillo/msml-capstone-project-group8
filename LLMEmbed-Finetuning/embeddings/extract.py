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
        cls.embeddings = {}
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
            if dataset_name in tasks:
                sentences = dataset["text"]
                labels = dataset["label"]
                (
                    sentences_train,
                    sentences_test,
                    labels_train,
                    labels_test,
                ) = train_test_split(sentences, labels, test_size=0.2, random_state=42)

                for task in tasks:
                    # Bert Training data emebeddings extraction
                    cls.embeddings[f"bert_{task}_train_embeddings"] = (
                        cls.bert.extract_bert_embeddings(
                            mode="train",
                            device=cls.device,
                            sentences=sentences_train,
                            labels=labels_train,
                            task=task,
                        )
                    )
                    # Bert Testing data emebeddings extraction
                    cls.embeddings[f"bert_{task}_test_embeddings"] = (
                        cls.bert.extract_bert_embeddings(
                            mode="test",
                            device=cls.device,
                            sentences=sentences_test,
                            labels=labels_test,
                            task=task,
                        )
                    )

                for task in tasks:
                    # Roberta Training emebeddings extraction
                    cls.embeddings[f"roberta_{task}_train_embeddings"] = (
                        cls.roberta.extract_roberta_embeddings(
                            mode="train",
                            device=cls.device,
                            sentences=sentences_train,
                            labels=labels_train,
                            task=task,
                        )
                    )
                    # Roberta Testing emebeddings extraction
                    cls.embeddings[f"roberta_{task}_test_embeddings"] = (
                        cls.roberta.extract_roberta_embeddings(
                            mode="test",
                            device=cls.device,
                            sentences=sentences_test,
                            labels=labels_test,
                            task=task,
                        )
                    )
                for task in tasks:
                    # Llama Training emebeddings extraction
                    cls.embeddings[f"llama_{task}_train_embeddings"] = (
                        cls.llama2.extract_llama2_embeddings(
                            mode="train",
                            device=cls.device,
                            sentences=sentences_train,
                            labels=labels_train,
                            task=task,
                        )
                    )
                    # Llama Testing emebeddings extraction
                    cls.embeddings[f"llama_{task}_test_embeddings"] = (
                        cls.llama2.extract_llama2_embeddings(
                            mode="test",
                            device=cls.device,
                            sentences=sentences_test,
                            labels=labels_test,
                            task=task,
                        )
                    )

        print("[Completed] -  Embeddings extraction.")

        return cls.embeddings
