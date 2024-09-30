"""
    This module contains the class to read the data (embeddings)
"""

# General Imports
import torch
from torch.utils.data import Dataset, DataLoader

# Local Imports
from config import Config
from logger import Logger


class Data(Dataset):
    """
    This class loads the extracted embeddings.
    """

    @classmethod
    def __init__(cls, enable_logging: bool):
        """
        This method inializes the instances and other variable
        """
        cls.config = Config()
        cls.log = Logger()
        cls.enable_logging = enable_logging
        cls.data = {}

    @classmethod
    def load_data(cls, task: str, mode: str) -> tuple:
        """
        This method loads the embedding extracted based on task and mode (train/test)
        """
        (
            l_sentences_path,
            b_sentences_path,
            r_sentences_path,
            labels_path,
        ) = cls.config.get_embeddings_path(task=task, mode=mode)
        l_sentences_reps = torch.load(l_sentences_path)
        b_sentences_reps = torch.load(b_sentences_path)
        r_sentences_reps = torch.load(r_sentences_path)
        labels = torch.load(labels_path)
        return l_sentences_reps, b_sentences_reps, r_sentences_reps, labels

    @classmethod
    def extract_data(cls) -> dict:
        """
        This method returns the train and test datasets for tasks and modes
        """
        for task in cls.config.get_selected_task_types():
            for mode in cls.config.get_modes():
                cls.log.log(
                    message=f"\n[Started] - Load the {task} {mode} data emebeddings.",
                    enable_logging=cls.enable_logging,
                )
                cls.data[f"{task}_{mode}_data"] = cls.load_data(task=task, mode=mode)
                cls.log.log(
                    message=f"[Completed] - Load the {task} {mode} data emebeddings.",
                    enable_logging=cls.enable_logging,
                )

        return cls.data
