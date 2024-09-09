"""
    This module contains the Helpers class
"""


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
    def ner_extract_label(cls, example, mapping: dict, column_to_modify: str) -> int:
        """
        This function checks if the specific column has
        positive, negative and neutral in it and assigns the appropriate integer label
        Example -
            text : the sentiment in this text about levi strauss & co is positive
            sentiment : positive
        """
        text = example[column_to_modify].lower()
        if "positive" in text:
            example["sentiment"] = mapping["positive"]
        elif "neutral" in text:
            example["sentiment"] = mapping["neutral"]
        elif "negative" in text:
            example["sentiment"] = mapping["negative"]
        else:
            example["sentiment"] = mapping["unknown"]
        return example
