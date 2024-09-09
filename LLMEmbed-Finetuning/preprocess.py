"""
    This module contains the Preprocess class
"""
# Local Imports
from config import Config
from helpers import Helpers

# General Imports
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
import warnings

warnings.filterwarnings("ignore")


class Preprocess:
    """
    This class performs the following tasks :
        1. Load the financial_phrasebank and Sujet-Finance-Instruct-177k from hugging face.
        2. Filter out the selected task types from the datasets.
        3. Convert the labels, texts into lowercase in both the datasets.
        4. Rename column names to maintain consistency between the two datasets.
        5. Reorder the dataset columns to maintain consistency between the two datasets.
        6. Seggregate the sujet_finance datasets according to the task types.
        7. Replace the string labels with integers in Sujet-Finance-Instruct-177k dataset.
        8. Concantenate the Sujet-Finance-Instruct-177k and financial_phrasebank datasets based on the task types

    """

    @classmethod
    def __init__(cls):
        """
        This method initializes the dictionary to save the datasets
        """
        cls.config = Config()
        cls.helpers = Helpers()
        cls.datasets = {}
        cls.datasets_df = {}

    @classmethod
    def load_datasets(cls) -> dict:
        """
        This method will load the two datasets and save in a dictionary as key value pair
        """
        datasets = {}
        datasets["financial_phrasebank"] = load_dataset(
            "financial_phrasebank", "sentences_allagree"
        )
        datasets["sujet_finance"] = load_dataset("sujet-ai/Sujet-Finance-Instruct-177k")
        return datasets

    @classmethod
    def convert_to_pandas_df(cls) -> dict:
        """
        This method will convert the hugging face datasets into pandas dataframe for better visualization
        """
        df = {}
        df["financial_phrasebank"] = cls.datasets["financial_phrasebank"][
            "train"
        ].to_pandas()
        df["sujet_finance"] = cls.datasets["sujet_finance"]["train"].to_pandas()

        return df

    @classmethod
    def select_task_types_columns(cls, dataset) -> dict:
        """
        This method will filter out the task types, and also select only few columns from the sujet_finance dataset.
        The selected task types : ["sentiment_analysis","yes_no_question", "ner_sentiment_analysis"]
        """
        dataset = dataset.filter(
            lambda x: x["task_type"] in cls.config.get_selected_task_types()
        )
        dataset = dataset.select_columns(cls.config.get_selected_columns())
        return dataset

    @classmethod
    def convert_lower_case(cls, datasets) -> dict:
        """
        This method will convert the column values to lowercase in the datasets
        """
        datasets["sujet_finance"] = datasets["sujet_finance"].map(
            cls.helpers.convert_column_to_lowercase,
            fn_kwargs={"column_to_lowercase": "answer"},
        )
        return datasets

    @classmethod
    def rename_column_names(cls, datasets) -> dict:
        """
        This method will rename the columns names to maintain the consistency between the two datasets
        """
        rename_map_dict = cls.config.get_rename_column_names_mapping()
        for old_name, new_name in rename_map_dict["sujet"].items():
            datasets["sujet_finance"] = datasets["sujet_finance"].rename_column(
                old_name, new_name
            )

        for old_name, new_name in rename_map_dict["phrasebank"].items():
            datasets["financial_phrasebank"] = datasets[
                "financial_phrasebank"
            ].rename_column(old_name, new_name)
        return datasets

    @classmethod
    def reorder_columns(cls, datasets) -> dict:
        """
        This method will reoder the columns to maintain the consistency between the two datasets
        """
        columns_order = cls.config.get_columns_order()
        datasets["financial_phrasebank"] = datasets[
            "financial_phrasebank"
        ].select_columns(columns_order)
        return datasets

    @classmethod
    def seggregate_sujet_task_types(cls, datasets) -> dict:
        """
        This method seggregates the datasets based on the task types and order the columns in the sujet_finance datasets.
        """
        task_types = cls.config.get_selected_task_types()
        columns_order = cls.config.get_columns_order()
        for task_type in task_types:
            datasets[f"sujet_finance_{task_type}"] = datasets["sujet_finance"].filter(
                lambda x: x["task_type"] == task_type
            ).select_columns(columns_order)
        return datasets

    @classmethod
    def convert_string_labels_to_integers(cls, datasets) -> dict:
        """
        This method converts the string lables in the sujet_finance dataset to integer labels
        """
        sentiment_mapping = cls.config.get_sentiment_mapping()
        yes_no_mapping = cls.config.get_yes_no_mapping()
        datasets["sujet_finance_sentiment_analysis_new"] = datasets[
            "sujet_finance_sentiment_analysis"
        ].map(
            cls.helpers.replace_string_with_int,
            fn_kwargs={"mapping": sentiment_mapping, "column_to_modify": "label"},
        )
        datasets["sujet_finance_yes_no_question_new"] = datasets[
            "sujet_finance_yes_no_question"
        ].map(
            cls.helpers.replace_string_with_int,
            fn_kwargs={"mapping": yes_no_mapping, "column_to_modify": "label"},
        )
        datasets["sujet_finance_ner_sentiment_analysis_new"] = datasets[
            "sujet_finance_ner_sentiment_analysis"
        ].map(
            cls.helpers.ner_extract_label,
            fn_kwargs={"mapping": sentiment_mapping, "column_to_modify": "label"},
        )
        return datasets

    @classmethod
    def concatenate_two_datasets(cls, datasets) -> dict:
        """
        This method will concatenate the datasets based on task types.
        """
        # Sentimental analysis
        sujet_finance_senti_new_df = datasets["sujet_finance_sentiment_analysis_new"][
            "train"
        ].to_pandas()
        financial_phrasebank_df = datasets["financial_phrasebank"]["train"].to_pandas()
        sujet_finance_senti_temp = Dataset.from_pandas(sujet_finance_senti_new_df)
        financial_phrasebank_temp = Dataset.from_pandas(financial_phrasebank_df)
        datasets["sentimental_analysis"] = concatenate_datasets(
            [sujet_finance_senti_temp, financial_phrasebank_temp]
        )
        # Yes/No and NER sentimental analysis are already seperated in cls.datasets
        return datasets

    @classmethod
    def main(cls) -> dict:
        """
        This function in the starting point for the preprocessing of the datasets
        """
        print("\n Preprocessing the datasets - [Started]")

        # 1. Load the financial_phrasebank and Sujet-Finance-Instruct-177k from hugging face
        cls.datasets = cls.load_datasets()
        # return cls.datasets

        ## Convert the datasets into pandas dataframe
        cls.datasets_df = cls.convert_to_pandas_df()
        # return cls.datasets_df

        # 2. Filter out the selected task types from the Sujet-Finance-Instruct-177k dataset.
        cls.datasets["sujet_finance"] = cls.select_task_types_columns(
            dataset=cls.datasets["sujet_finance"]
        )
        # return cls.datasets

        # 3. Convert the labels, texts into lowercase in both the datasets
        cls.datasets = cls.convert_lower_case(datasets=cls.datasets)

        # 4. Rename column names to maintain consistency between the two datasets.
        cls.datasets = cls.rename_column_names(datasets=cls.datasets)

        # 5. Reorder the dataset columns to maintain consistency between the two datasets.
        cls.datasets = cls.reorder_columns(datasets=cls.datasets)

        # 6. Seggregate the sujet_finance dataset according to the task types.
        cls.datasets = cls.seggregate_sujet_task_types(datasets=cls.datasets)

        # 7. Replace the string labels with integers in Sujet-Finance-Instruct-177k dataset.
        cls.datasets = cls.convert_string_labels_to_integers(datasets=cls.datasets)

        # 8. Concantenate the Sujet-Finance-Instruct-177k and financial_phrasebank datasets based on the task types
        cls.datasets = cls.concatenate_two_datasets(datasets=cls.datasets)

        print("\n Preprocessing the datasets - [Completed]")

        return cls.datasets
