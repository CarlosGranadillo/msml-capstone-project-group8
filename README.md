# msml-capstone-project-group8
Repository containing all code and implementation files related to Group 8's capstone project for the MSML program at the University of Arizona

# code setup
1. clone the repository from the url - https://github.com/CarlosGranadillo/msml-capstone-project-group8.git.
2. run the following command `python main.py` or `python3 main.py`.
3. the tasks performed in this project are `sentiment analysis` and `yes no question`

# code flow
`[dataload]` -> `[preprocess]` -> `[embedding extraction (base llm's)]` -> `[fine tune llm's]` -> `[embedding extraction (finetuned llm's)]` -> `[run downstream model on both embeddings]` -> `[compare model resuts]`

# run steps
1. set the following boolean values according to the tasks below
    ## embeddings extraction using base models
    a. set `extract = True` to extract the embeddings

    ## preprocessing data while embeddings extraction
    a. save the preprocessed data in the local : `save_data_in_local = True` else `False`
    b. read the preprocessed data from the local and extract the embeddings :  `read_data_from_local = True` else `False` to preprocess the data from hugging face and extract the embeddings.
    c. extract emebeddings using base llm models : `use_finetuned_model = False` 

    ## finetune the llm models
    a. set `fintune = True` to finetune the base models, additionally choose which model to fine tune by passing boolean values to `bert`, `roberta` and `llama2` parameters.

    ## embeddings extraction using finetuned models
    a. set `extract = True` to extract the embeddings 
    b. b. read the preprocessed data from the local and extract the embeddings :  `read_data_from_local = True` else `False` to preprocess the data from hugging face and extract the embeddings.
    c. extract emebeddings using finetuned llm models : `use_finetuned_model = True`