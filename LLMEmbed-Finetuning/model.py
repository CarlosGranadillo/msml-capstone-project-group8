"""
    This module contains the LLM models, Classifier Neural Network that are used in the project.
"""
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from config import Config
from helpers import Helpers


class Model:
    """
    This class contains the LLM's models and the Neural network classifier
    """

    @classmethod
    def __init__(cls):
        """
        This method initializes the dictionaries to save the tokenizers and LLM's models.
        """
        cls.config = Config()
        cls.tokenizers = {}
        cls.helpers = Helpers()

    @classmethod
    def llm(cls, model_name: str) -> tuple:
        """
        This method returns the LLM model and its respective tokenizer as a tuple
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        return tokenizer, model

    @classmethod
    def extract(cls, datasets : dict) -> dict:
        """
        This method is the starting point to perform tokenization and extract emebeddings.
        """
        device = cls.helpers.check_if_gpu_available()

        # 1. Load the LLM models and perform tokenization and extract embeddings
        model_names = cls.config.get_llm_models()
        for model_name in model_names:
            model, tokenizer = cls.llm(model_name)
            # 2. Perform tokenization and extract emebeddings for all the datasets
            for dataset in datasets:
                cls.embeddings[f"{model_name}_{dataset}_embeds"] = dataset.map(
                    lambda x: {
                        f"{model_name}_embeds": cls.helpers.tokenize_and_extract_embeddings(
                            x,
                            tokenizer=tokenizer,
                            model=model,
                            column="text",
                            device=device,
                        )
                    },
                    batched=True,
                )
        return cls.embeddings