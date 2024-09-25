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
import numpy as np

warnings.filterwarnings("ignore")


class Preprocess:
    """
    This class performs the following tasks :
        1. Load the financial_phrasebank and Sujet-Finance-Instruct-177k from hugging face
        2. Filter out the selected task types from the Sujet-Finance-Instruct-177k dataset.
        3. Convert the labels, texts into lowercase in Sujet-Finance-Instruct-177k dataset.
        4. Rename column names in Sujet-Finance-Instruct-177k dataset.
        5. Seggregate the Sujet-Finance-Instruct-177k dataset according to the task types.
        6. Check for the null rows and drop them if required.
        7. Check for the duplicate rows and drop them if required.
        8. Replace the string labels with integers in Sujet-Finance-Instruct-177k dataset.
        9. Concatenate the sentimental analysis and NER sentimental analysis into a single dataset.
        10. Return the final required datasets.

    """

    @classmethod
    def __init__(cls):
        """
        This method initializes the dictionary to save the datasets
        """
        cls.config = Config()
        cls.helpers = Helpers()
        cls.temp_datasets = {}
        cls.datasets_df = {}

    @classmethod
    def load_datasets(cls) -> dict:
        """
        This method will load the datasets and save in a dictionary as key value pair.
        """
        datasets = {}
        print("\n[Started] - Load the Sujet-Finance-Instruct-177k dataset.")
        datasets["sujet_finance"] = load_dataset("sujet-ai/Sujet-Finance-Instruct-177k")
        print("[Completed] - Load the Sujet-Finance-Instruct-177k dataset.")
        return datasets

    @classmethod
    def convert_to_pandas_df(cls) -> dict:
        """
        This method will convert the hugging face datasets into pandas dataframe for better visualization.
        """
        df = {}
        print(
            "\n[Started] - Convert the Sujet-Finance-Instruct-177k dataset to pandas dataframe."
        )
        df["sujet_finance"] = cls.temp_datasets["sujet_finance"]["train"].to_pandas()
        print(
            "[Completed] - Convert the Sujet-Finance-Instruct-177k dataset to pandas dataframe."
        )
        return df

    @classmethod
    def select_task_types_columns(cls, dataset) -> dict:
        """
        This method will filter out the task types, and also select only few columns from the sujet_finance dataset.
        The selected task types : ["sentiment_analysis","yes_no_question", "ner_sentiment_analysis"]
        """
        print(
            "\n[Started] - Select the task types in the Sujet-Finance-Instruct-177k dataset."
        )
        selected_task_types = cls.config.get_selected_task_types()
        selected_columns = cls.config.get_selected_columns()
        dataset = dataset.filter(lambda x: x["task_type"] in selected_task_types)
        dataset = dataset.select_columns(selected_columns)
        print(
            "[Completed] - Select the task types in the Sujet-Finance-Instruct-177k dataset."
        )
        return dataset

    @classmethod
    def convert_lower_case(cls, datasets) -> dict:
        """
        This method will convert the column values to lowercase in the datasets.
        """
        print(
            "\n[Started] - Convert the column vaules to lower case in the Sujet-Finance-Instruct-177k dataset."
        )
        datasets["sujet_finance"] = datasets["sujet_finance"].map(
            cls.helpers.convert_column_to_lowercase,
            fn_kwargs={"column_to_lowercase": "answer"},
        )
        print(
            "[Completed] - Convert the column vaules to lower case in the Sujet-Finance-Instruct-177k dataset."
        )
        return datasets

    @classmethod
    def rename_column_names(cls, datasets) -> dict:
        """
        This method will rename the columns names.
        """
        print(
            "\n[Started] - Rename the column names in the Sujet-Finance-Instruct-177k dataset."
        )
        rename_map_dict = cls.config.get_rename_column_names_mapping()
        for old_name, new_name in rename_map_dict["sujet"].items():
            datasets["sujet_finance"] = datasets["sujet_finance"].rename_column(
                old_name, new_name
            )
        print(
            "[Completed] - Rename the column names in the Sujet-Finance-Instruct-177k dataset."
        )
        return datasets

    @classmethod
    def seggregate_sujet_task_types(cls, datasets) -> dict:
        """
        This method seggregates the datasets based on the task types and order the columns in the sujet_finance datasets.
        """
        print(
            "\n[Started] - Seggregate the data sets based on task types in the Sujet-Finance-Instruct-177k dataset."
        )
        task_types = cls.config.get_selected_task_types()
        columns_order = cls.config.get_columns_order()
        for task_type in task_types:
            datasets[f"sujet_finance_{task_type}"] = (
                datasets["sujet_finance"]
                .filter(lambda x: x["task_type"] == task_type)
                .select_columns(columns_order)
            )
        print(
            "[Completed] - Seggregate the data sets based on task types in the Sujet-Finance-Instruct-177k dataset."
        )
        return datasets

    @classmethod
    def drop_null_rows(cls, datasets) -> dict:
        """
        This method will drop the null rows in the datasets.
        """
        print("\n[Started] - Remove null rows from the datasets.")
        for dataset_name, dataset in datasets.items():
            print(f"\t[Started] - Null rows removal for {dataset_name}.")
            column_names = dataset["train"].column_names
            print(f"\t\tNo.of rows in {dataset_name} : {len(dataset['train'])}")
            filtered_dataset = dataset["train"].filter(
                lambda example: all(
                    example[col] is not None
                    and not (isinstance(example[col], float) and np.isnan(example[col]))
                    for col in column_names
                )
            )
            no_of_null_rows = len(dataset["train"]) - len(filtered_dataset)
            print(f"\t\tRemoved {no_of_null_rows} null rows in {dataset_name}")
            datasets[dataset_name]["train"] = filtered_dataset
            print(f"\t[Completed] - Null rows removal for {dataset_name}.")
        print("[Completed] - Remove null rows from the datasets.")
        return datasets

    @classmethod
    def drop_duplicate_rows(cls, datasets) -> dict:
        """
        This method will drop the duplicate rows in the datasets.
        """
        print("\n[Started] - Remove duplicate rows from the datasets.")
        for dataset_name, dataset in datasets.items():
            print(f"\t[Started] - Duplicate rows removal for {dataset_name}.")
            unique = set()
            column_names = dataset["train"].column_names
            print(f"\t\tNo.of rows in {dataset_name} : {len(dataset['train'])}")
            unique_dataset = dataset["train"].filter(
                lambda example: not tuple(example[col] for col in column_names)
                in unique
                and not unique.add(tuple(example[col] for col in column_names))
            )
            no_of_duplicate_rows = len(dataset["train"]) - len(unique_dataset)
            print(
                f"\t\tRemoved {no_of_duplicate_rows} duplicate rows in {dataset_name}"
            )
            datasets[dataset_name]["train"] = unique_dataset
            print(f"\t[Completed] - Duplicate rows removal for {dataset_name}.")
        print("[Completed] - Remove duplicate rows from the datasets.")
        return datasets

    @classmethod
    def convert_string_labels_to_integers(cls, datasets) -> dict:
        """
        This method converts the string lables in the sujet_finance dataset to integer labels.
        """
        print(
            "\n[Started] - Convert the string labels to integers in the Sujet-Finance-Instruct-177k dataset."
        )
        sentiment_mapping = cls.config.get_sentiment_mapping()
        yes_no_mapping = cls.config.get_yes_no_mapping()

        print("\t[Started] - Sentimental Analysis conversion.")
        datasets["sujet_finance_sentiment_analysis"] = datasets[
            "sujet_finance_sentiment_analysis"
        ].map(
            cls.helpers.replace_string_with_int,
            fn_kwargs={"mapping": sentiment_mapping, "column_to_modify": "label"},
        )
        print("\t[Completed] - Sentimental Analysis conversion.")

        print("\t[Started] - Yes/No question conversion.")
        datasets["sujet_finance_yes_no_question"] = datasets[
            "sujet_finance_yes_no_question"
        ].map(
            cls.helpers.replace_string_with_int,
            fn_kwargs={"mapping": yes_no_mapping, "column_to_modify": "label"},
        )
        print("\t[Completed] - Yes/No question conversion.")

        print("\t[Started] - NER Sentimental Analysis conversion.")
        datasets["sujet_finance_ner_sentiment_analysis"] = datasets[
            "sujet_finance_ner_sentiment_analysis"
        ].map(
            cls.helpers.ner_extract_label,
            fn_kwargs={"mapping": sentiment_mapping, "column_to_modify": "label"},
        )
        print("\t[Completed] - NER Sentimental Analysis conversion.")
        print(
            "[Completed] - Convert the string labels to integers in the Sujet-Finance-Instruct-177k dataset."
        )
        return datasets

    @classmethod
    def concatenate_sentimental_analysis_datasets(cls, datasets) -> dict:
        """
        This method will concatenate the datasets based on task types.
        """
        print(
            "\n[Started] - Concatenation of setimental analysis and NER sentimental analysis into a single dataset."
        )
        sujet_finance_senti_df = datasets["sujet_finance_sentiment_analysis"][
            "train"
        ].to_pandas()
        sujet_finance_ner_senti_df = datasets["sujet_finance_ner_sentiment_analysis"][
            "train"
        ].to_pandas()
        sujet_finance_senti_temp = Dataset.from_pandas(sujet_finance_senti_df)
        sujet_finance_ner_senti_temp = Dataset.from_pandas(sujet_finance_ner_senti_df)
        datasets["sentimental_analysis"] = concatenate_datasets(
            [sujet_finance_senti_temp, sujet_finance_ner_senti_temp]
        )
        print(
            "[Completed] - Concatenation of setimental analysis and NER sentimental analysis into a single dataset."
        )
        return datasets

    @classmethod
    def main(cls) -> dict:
        """
        This function in the starting point for the preprocessing of the datasets
        """
        print("\n[Started] - Preprocessing the datasets.")

        # 1. Load the financial_phrasebank and Sujet-Finance-Instruct-177k from hugging face.
        cls.temp_datasets = cls.load_datasets()

        ## Convert the datasets into pandas dataframe
        cls.datasets_df = cls.convert_to_pandas_df()

        # 2. Filter out the selected task types from the Sujet-Finance-Instruct-177k dataset.
        cls.temp_datasets["sujet_finance"] = cls.select_task_types_columns(
            dataset=cls.temp_datasets["sujet_finance"]
        )

        # 3. Convert the labels, texts into lowercase in Sujet-Finance-Instruct-177k dataset.
        cls.temp_datasets = cls.convert_lower_case(datasets=cls.temp_datasets)

        # 4. Rename column names in Sujet-Finance-Instruct-177k dataset.
        cls.temp_datasets = cls.rename_column_names(datasets=cls.temp_datasets)

        # 5. Seggregate the Sujet-Finance-Instruct-177k dataset according to the task types.
        cls.temp_datasets = cls.seggregate_sujet_task_types(datasets=cls.temp_datasets)

        # 6. Check for the null rows and drop them if required.
        cls.temp_datasets = cls.drop_null_rows(datasets=cls.temp_datasets)

        # 7. Check for the duplicate rows and drop them if required.
        cls.temp_datasets = cls.drop_duplicate_rows(datasets=cls.temp_datasets)

        # 8. Replace the string labels with integers in Sujet-Finance-Instruct-177k dataset.
        cls.temp_datasets = cls.convert_string_labels_to_integers(
            datasets=cls.temp_datasets
        )

        # 9. Concatenate the sentimental analysis and NER sentimental analysis into a single dataset.
        cls.temp_datasets = cls.concatenate_sentimental_analysis_datasets(
            datasets=cls.temp_datasets
        )

        # 10. Return the final required datasets.
        datasets = {
            "sujet_finance": cls.temp_datasets["sujet_finance"]["train"],
            "sentimental_analysis": cls.temp_datasets["sentimental_analysis"],
            "yes_no_question": cls.temp_datasets["sujet_finance_yes_no_question"]["train"],
        }

        print("\n[Completed] - Preprocessing the datasets.")

        return datasets
