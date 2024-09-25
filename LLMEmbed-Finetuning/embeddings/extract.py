"""
    This module is the starting point for th embeddings extraction
"""
# Local Imports
from config import Config
from helpers import Helpers
from logger import Logger
from preprocess import Preprocess
from sklearn.model_selection import train_test_split

from .bert_embeddings import Bert


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
        cls.bert = Bert()
        cls.enable_logging = enable_logging
        cls.device = cls.config.get_device()

    @classmethod
    def extract(cls, datasets) -> dict:
        """
        This method extracts the embeddings using the LLM's
        """
        print("\n[Started] -  Embeddings extraction")
        # Load the logger
        logger = cls.log
        tasks = cls.config.get_selected_task_types()

        for dataset_name, dataset in datasets.items():
            print(dataset_name, dataset)
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
                    # Training emebeddings
                    logger.log(
                        message=f"\n[Started] - Extract embeddings using Bert LLM for {task} train dataset.",
                        enable_logging=cls.enable_logging,
                    )
                    cls.embeddings[
                        f"bert_{task}_train_embeddings"
                    ] = cls.bert.extract_bert_embeddings(
                        mode="train",
                        device=cls.device,
                        sentences=sentences_train,
                        labels=labels_train,
                        task=task,
                    )
                    logger.log(
                        message=f"[Completed] - Extract embeddings using Bert LLM for {task} train dataset",
                        enable_logging=cls.enable_logging,
                    )

                    # Testing emebeddings
                    logger.log(
                        message=f"\n[Started] - Extract embeddings using Bert LLM for {task} test dataset.",
                        enable_logging=cls.enable_logging,
                    )
                    cls.embeddings[
                        f"bert_{task}_test_embeddings"
                    ] = cls.bert.extract_bert_embeddings(
                        mode="test",
                        device=cls.device,
                        sentences=sentences_test,
                        labels=labels_test,
                        task=task,
                    )
                    logger.log(
                        message=f"[Completed] - Extract embeddings using Bert LLM for {task} test dataset",
                        enable_logging=cls.enable_logging,
                    )

                # for task in tasks:
                #     # Training emebeddings
                #     logger.log(message=f"\n[Started] - Extract embeddings using Roberta LLM for {task} train dataset.",enable_logging=cls.enable_logging)
                #     cls.embeddings[f"Roberta_{task}_train_embeddings"] = cls.bert.extract_roberta_embeddings(mode = "train", device = cls.device, sentences = sentences_train, labels = labels_train, task = task)
                #     logger.log(message=f"[Completed] - Extract embeddings using Roberta LLM for {task} train dataset",enable_logging=cls.enable_logging)

                #     # Testing emebeddings
                #     logger.log(message=f"\n[Started] - Extract embeddings using Roberta LLM for {task} test dataset.",enable_logging=cls.enable_logging)
                #     cls.embeddings[f"Roberta_{task}_test_embeddings"] = cls.bert.extract_roberta_embeddings(mode = "test", device = cls.device, sentences = sentences_test, labels = labels_test, task = task)
                #     logger.log(message=f"[Completed] - Extract embeddings using Roberta LLM for {task} test dataset",enable_logging=cls.enable_logging)

                # for task in tasks:
                #     # Training emebeddings
                #     logger.log(message=f"\n[Started] - Extract embeddings using Llama LLM for {task} train dataset.",enable_logging=cls.enable_logging)
                #     cls.embeddings[f"Llama_{task}_train_embeddings"] = cls.bert.extract_llama_embeddings(mode = "train", device = cls.device, sentences = sentences_train, labels = labels_train, task = task)
                #     logger.log(message=f"[Completed] - Extract embeddings using Llama LLM for {task} train dataset",enable_logging=cls.enable_logging)

                #     # Testing emebeddings
                #     logger.log(message=f"\n[Started] - Extract embeddings using Llama LLM for {task} test dataset.",enable_logging=cls.enable_logging)
                #     cls.embeddings[f"Llama_{task}_test_embeddings"] = cls.bert.extract_llama_embeddings(mode = "test", device = cls.device, sentences = sentences_test, labels = labels_test, task = task)
                #     logger.log(message=f"[Completed] - Extract embeddings using Llama LLM for {task} test dataset",enable_logging=cls.enable_logging)

        print("[Completed] -  Embeddings extraction")

        return cls.embeddings
