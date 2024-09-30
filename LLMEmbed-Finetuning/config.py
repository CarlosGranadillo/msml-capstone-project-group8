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

    @classmethod
    def get_base_path(cls) -> str:
        """
            This function returns the path for the embeddings to be stored.
        """
        path = "/home/exouser/Desktop/msml-group8-code-base/"  # change it to a desired path
        return path

    @classmethod
    def get_embeddings_path(cls, task: str, mode: str) -> str:
        """
            This function returns the paths where the embeddings are stored for the LLM's.
        """
        base_path = cls.get_base_path() + "embeddings/"
        l_path = base_path + f"llama2_embeddings/{task}/dataset_tensors/{mode}_texts.pt"
        b_path = base_path + f"bert_embeddings/{task}/dataset_tensors/{mode}_texts.pt"
        r_path = (
            base_path + f"roberta_embeddings/{task}/dataset_tensors/{mode}_texts.pt"
        )
        # labels will remain same for all the extractions
        labels_path = (
            base_path + f"llama2_embeddings/{task}/dataset_tensors/{mode}_labels.pt"
        )
        return l_path, b_path, r_path, labels_path

    @classmethod
    def get_hugging_face_token(cls) -> str:
        """
            This function returns the path for the embeddings to be stored.
        """
        token = "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"  # change it to the user token
        return token

    @classmethod
    def get_modes(cls) -> list:
        """
            This function returns the modes.
        """
        modes = ["train", "test"]
        return modes
    
    @classmethod
    def get_local_datasets_names(cls) -> list:
        """
            This function returns the local dataset names.
        """
        names = ["sentiment_analysis","sentiment_analysis_fine_tuning","sujet_finance","sujet_finance_fine_tuning","yes_no_question","yes_no_question_fine_tuning"]
        return names