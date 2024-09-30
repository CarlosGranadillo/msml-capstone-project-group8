"""
    This module contains the class to extract embeddings using Bert LLM
"""

# Local Imports
from config import Config
from logger import Logger
from helpers import Helpers

# General Imports
import os
import torch
from transformers import BertTokenizer, BertModel
from tqdm import trange


class Bert:
    """
    This class contains the methods to extract the embeddings using Bert
    """

    @classmethod
    def __init__(cls, enable_logging: bool):
        """
        This method initialized the variables that are used in this class
        """
        cls.config = Config()
        cls.log = Logger()
        cls.helpers = Helpers()
        cls.model_name = "bert-base-uncased"
        cls.device = cls.config.get_device()
        cls.enable_logging = enable_logging

        cls.log.log(
            message=f"\n[Started] - Loading the tokenizer for the {cls.model_name} LLM model from hugging face.",
            enable_logging=enable_logging,
        )
        cls.tokenizer = BertTokenizer.from_pretrained(cls.model_name)
        cls.log.log(
            message=f"[Completed] - Loading the tokenizer for the {cls.model_name} LLM model from hugging face.",
            enable_logging=enable_logging,
        )

        cls.log.log(
            message=f"\n[Started] - Loading the {cls.model_name} LLM model from hugging face.",
            enable_logging=enable_logging,
        )
        cls.model = BertModel.from_pretrained(cls.model_name).to(cls.device)
        cls.log.log(
            message=f"[Completed] - Loading the {cls.model_name} LLM model from hugging face.",
            enable_logging=enable_logging,
        )
        cls.max_length = 512
        cls.model.eval()

    @classmethod
    def extract_bert_embeddings(
        cls, mode: str, device: str, sentences: list, labels: list, task: str
    ) -> dict:
        """
        This method performs the embeddings extractions using bert.
        """
        cls.log.log(
            message=f"\n[Started] - Performing embeddings extraction using {cls.model_name} for {task} on {mode} data.",
            enable_logging=cls.enable_logging,
        )
        path = f"bert_embeddings/{task}/dataset_tensors/"
        sentences_reps = []
        step = 64  # Reduced batch size to save memory
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
                reps_batch = batch_outputs.pooler_output
            sentences_reps.append(reps_batch.cpu())

            # Clear cache after processing each batch to free up memory
            del sentences_batch_encoding, batch_outputs, reps_batch
            torch.cuda.empty_cache()

        sentences_reps = torch.cat(sentences_reps)

        for idx in range(len(labels)):
            labels[idx] = torch.tensor(labels[idx])
        labels = torch.stack(labels)

        cls.helpers.save_embeddings(
            sentences_embeds=sentences_reps, labels=labels, file_path=path, mode=mode
        )

        cls.log.log(
            message=f"[Completed] - Performing embeddings extraction using {cls.model_name} for {task} on {mode} data.",
            enable_logging=cls.enable_logging,
        )
        return sentences_reps
