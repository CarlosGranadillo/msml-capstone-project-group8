"""
    This module contain the Config class.
"""

# General Imports
import torch


class Config:
    """
    This class will load the configuration details and also save the default values.
    """

    @classmethod
    def get_selected_task_types(cls) -> list:
        """
        This method returns the selected task types for the sujet_finance dataset.
        """
        tasks = ["sentiment_analysis", "yes_no_question"]
        return tasks

    @classmethod
    def get_selected_columns(cls) -> list:
        """
        This method returns the selected columns for the sujet_finance dataset.
        """
        cols = ["answer", "user_prompt", "task_type"]
        return cols

    @classmethod
    def get_rename_column_names_mapping(cls) -> dict:
        """
        This method returns the mappings for the column names rename.
        """
        sujet_finance_column_renamings = {"user_prompt": "text", "answer": "label"}
        rename_dict = {
            "sujet": sujet_finance_column_renamings,
        }
        return rename_dict

    @classmethod
    def get_columns_order(cls) -> list:
        """
        This function returns the order of the columns.
        """
        cols_order = ["text", "label"]
        return cols_order

    @classmethod
    def get_sentiment_mapping(cls) -> dict:
        """
        This function returns the sentiment string to integer mapping of the class labels.
        """
        sentiment_mapping = {
            "negative": 0,
            "neutral": 1,
            "positive": 2,
            "bearish": 3,
            "bullish": 4,
            "unknown": 5,
        }
        return sentiment_mapping

    @classmethod
    def get_yes_no_mapping(cls) -> dict:
        """
        This function returns the sentiment string to integer mapping of the class labels.
        """
        yes_no_mapping = {"yes": 0, "no": 1}
        return yes_no_mapping

    @classmethod
    def get_device(cls) -> str:
        """
        This function returns the device available.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return device
