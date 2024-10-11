# LLMEmbed Finetuning on finance data classification task.
Repository containing all code and implementation files related to LLMEmbed.
* The tasks performed in this project are `sentiment analysis` and `yes no question`.
* All the results and processes are carried out seperately for both the tasks.


# Code setup
* Clone the repository from the url - https://github.com/CarlosGranadillo/msml-capstone-project-group8.git.
* Run the following command `python main.py` or `python3 main.py`.

# Code flow
`dataload` -> `preprocess` -> `embedding extraction (base llm's)` -> `fine tune llm's` -> `embedding extraction (finetuned llm's)` -> `run downstream model on both embeddings` -> `compare model resuts`

# Run steps
1. Set the following boolean values according to the tasks below as the `main` function parameters.
    ### Embeddings extraction using base models.
    * set `extract = True` to extract the embeddings

    ### Preprocessing the data and embeddings extraction.
    * save the preprocessed data in the local : `save_data_in_local = True` else `False`.

    * preprocess the data and extract the embeddings :  `read_data_from_local = False` to preprocess the data from hugging face and extract the embeddings.

    * extract emebeddings using base llm models : `use_finetuned_model = False`.

    ### Finetune the llm models.
    * set `fintune = True` to finetune the base models, additionally choose which model to fine tune by passing boolean values to `bert`, `roberta` and `llama2` parameters.

    ### Embeddings extraction using finetuned models.
    * set `extract = True` to extract the embeddings. 

    * read the preprocessed data from the local and extract the embeddings :  `read_data_from_local = True` else `False` to preprocess the data from hugging face and extract the embeddings.
    
    * extract emebeddings using finetuned llm models : `use_finetuned_model = True`.
