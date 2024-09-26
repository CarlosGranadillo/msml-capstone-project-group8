"""
    This is the main module
"""
# Local Imports
from preprocess import Preprocess
from embeddings import Embeddings

# General Imports
import os
import warnings
warnings.filterwarnings("ignore")


def main(debug: bool = False, preprocess: bool = True, extract: bool = True):
    """
    This method is the starting point for the project. It performs the following tasks -
        1. Preprocess the datasets.
            -> Loading the datasets, filtering, null checks, drop duplicate rows, selecting required columns, renaming columns, datasets seggregation, column values conversion.
        2. Extract Embedding from the datasets using the LLM's.
        3. Run the base downstream model on the extracted embeddings.
        4.
    """
    # 1. Preprocess the datasets
    if preprocess:
        datasets = Preprocess(debug).preprocess()
    print(
        "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    )
    # 2. Extract the embeddings
    if extract:
        embeddings = Embeddings(debug).extract(datasets=datasets)

    save_path = "/Users/daniel22/Desktop/Masters_Arizona/Course_Modules/Capstone Project/code-base/embeddings"
    file_path = f"{save_path}/test.txt"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(file_path, 'w') as f:
        f.write("Your content here")

if __name__ == "__main__":
    main(debug=False, preprocess=False, extract=False)
