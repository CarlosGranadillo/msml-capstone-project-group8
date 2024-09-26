"""
    This module contains the Helpers class
"""
# General Imports
from datasets import Dataset, concatenate_datasets


class Helpers:
    """
    This class contains all the helper methods that are used in this project
    """

    @classmethod
    def convert_column_to_lowercase(cls, example, column_to_lowercase: str) -> str:
        """
        This method converts the column values into lowercase.
        """
        example[column_to_lowercase] = example[column_to_lowercase].lower()
        return example

    @classmethod
    def replace_string_with_int(
        cls, example, mapping: dict, column_to_modify: str
    ) -> int:
        """
        This function converts string values to integers in a specific column
        """
        example[column_to_modify] = mapping.get(example[column_to_modify], -1)
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
