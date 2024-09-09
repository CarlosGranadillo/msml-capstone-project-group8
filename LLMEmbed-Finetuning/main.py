"""
    This is the main module
"""
# Local Imports
from preprocess import Preprocess

# General Imports
import warnings

warnings.filterwarnings("ignore")


def main():
    """
    This method is the starting point for the project. It performs the following tasks -
        1. Preprocess the datasets
            a. Loading the datasets, filtering, selecting columns, renaming columns, column values conversion, concatenation
        2.
        3.
        4.
    """
    # 1. Preprocess the datasets
    datasets = Preprocess().main()

    # Visulaize the results
    # Compare the counts before and after preprocessing
    sujet_finance_senti = datasets["sujet_finance_sentiment_analysis"]
    sujet_finance_senti_new = datasets["sujet_finance_sentiment_analysis_new"]
    sujet_finance_yes_no = datasets["sujet_finance_yes_no_question"]
    sujet_finance_yes_no_new = datasets["sujet_finance_yes_no_question_new"]
    sujet_finance_ner_new = datasets["sujet_finance_ner_sentiment_analysis_new"]
    print("Sentimental Analysis :")
    print("\t Before:")
    print(sujet_finance_senti["train"].to_pandas()["label"].value_counts())
    print("\t After:")
    print(sujet_finance_senti_new["train"].to_pandas()["label"].value_counts())
    print("----------------------------------------------------------------")
    print("\nYes/No :")
    print("\t Before:")
    print(sujet_finance_yes_no["train"].to_pandas()["label"].value_counts())
    print("----------------------------------------------------------------")
    print("\t After:")
    print(sujet_finance_yes_no_new["train"].to_pandas()["label"].value_counts())
    print("\nNER Sentimental Analysis :")
    print(sujet_finance_ner_new["train"].to_pandas()["sentiment"].value_counts())


if __name__ == "__main__":
    main()
