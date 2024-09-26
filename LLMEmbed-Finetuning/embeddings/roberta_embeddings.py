"""
    This module contains the class to extract embeddings using Roberta LLM.
"""
# Local Imports
from config import Config
from logger import Logger

# General Imports
import os
import torch
from transformers import RobertaTokenizer, RobertaModel
from tqdm import trange


class Roberta:
    """
    This class contains the methods to extract the embeddings using Bert
    """

    @classmethod
    def __init__(cls, enable_logging):
        """
        This method initialized the variables that are used in this class
        """
        cls.config = Config()
        cls.log = Logger()
        cls.model_name = "distilroberta-base"
        cls.device = cls.config.get_device()
        cls.enable_logging = enable_logging

        cls.log.log(
            message=f"\n[Started] - Loading the tokenizer for the {cls.model_name} LLM model from hugging face.",
            enable_logging=enable_logging,
        )
        cls.tokenizer = RobertaTokenizer.from_pretrained(cls.model_name)
        cls.log.log(
            message=f"[Completed] - Loading the tokenizer for the {cls.model_name} LLM model from hugging face.",
            enable_logging=enable_logging,
        )

        cls.log.log(
            message=f"\n[Started] - Loading the {cls.model_name} LLM model from hugging face.",
            enable_logging=enable_logging,
        )
        cls.model = RobertaModel.from_pretrained(cls.model_name).to(cls.device)
        cls.log.log(
            message=f"[Completed] - Loading the {cls.model_name} LLM model from hugging face.",
            enable_logging=enable_logging,
        )
        cls.max_length = 128
        cls.model.eval()

    @classmethod
    def extract_roberta_embeddings(
        cls, mode: str, device: str, sentences: list, labels: list, task: str
    ) -> dict:
        """
        This method performs the embeddings extractions using roberta.
        """
        cls.log.log(
            message=f"\n[Started] - Performing embeddings extraction using {cls.model_name}",
            enable_logging=cls.enable_logging,
        )
        path = f"roberta_embeddings/{task}/dataset_tensors/"
        sentences_reps = []
        step = 16  # Reduced batch size to save memory

        for idx in trange(0, len(sentences), step):
            idx_end = idx + step
            if idx_end > len(sentences):
                idx_end = len(sentences)
            sentences_batch = sentences[idx:idx_end]

            sentences_batch_encoding = cls.tokenizer(
                sentences_batch,
                return_tensors="pt",
                max_length=cls.max_length,
                padding="max_length",
                truncation=True,
            )
            sentences_batch_encoding = sentences_batch_encoding.to(device)

            with torch.no_grad():
                batch_outputs = cls.model(**sentences_batch_encoding)
                reps_batch = batch_outputs.last_hidden_state[:, 0, :]
            sentences_reps.append(reps_batch.cpu())

            # Clear cache and delete variables to free memory
            torch.cuda.empty_cache()
            del sentences_batch, sentences_batch_encoding, batch_outputs, reps_batch

        sentences_reps = torch.cat(sentences_reps)

        labels = torch.stack([torch.tensor(label) for label in labels])

        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(sentences_reps.to("cpu"), path + f"{mode}_sentences.pt")
        torch.save(labels, path + f"{mode}_labels.pt")

        cls.log.log(
            message=f"[Completed] - Performing embeddings extraction using {cls.model_name}",
            enable_logging=cls.enable_logging,
        )
        return sentences_reps
