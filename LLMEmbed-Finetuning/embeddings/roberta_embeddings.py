"""
    This module contains the class to extract embeddings using Roberta LLM.
"""

from config import Config

import os
import torch
from transformers import RobertaTokenizer, RobertaModel
from tqdm import trange


class Roberta:
    """
    This class contains the methods to extract the embeddings using Bert
    """

    @classmethod
    def __init__(cls):
        """
        This method initialized the variables that are used in this class
        """
        cls.config = Config()
        cls.model_name = "distilroberta-base"
        cls.device = cls.config.get_device()
        # Add a logging here
        cls.tokenizer = RobertaTokenizer.from_pretrained(cls.model_name)
        cls.model = RobertaModel.from_pretrained(cls.model_name).to(cls.device)
        cls.max_length = 128
        cls.model.eval()

    @classmethod
    def extract_roberta_embeddings(
        cls, mode: str, device: str, sentences: list, labels: list, task: str
    ) -> dict:
        """
        This method performs the embeddings extractions using roberta.
        """
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

        return sentences_reps
