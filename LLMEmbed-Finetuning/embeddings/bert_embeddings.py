"""
    This module contains the class to extract embeddings using Bert LLM
"""

from config import Config

import os
import torch
from transformers import BertTokenizer, BertModel
from tqdm import trange


class Bert:
    """
    This class contains the methods to extract the embeddings using Bert
    """

    @classmethod
    def __init__(cls):
        """
        This method initialized the variables that are used in this class
        """
        cls.config = Config()
        cls.model_name = "bert-base-uncased"  # Use a smaller model to save memory
        cls.device = cls.config.get_device()
        # Add a logging here
        cls.tokenizer = BertTokenizer.from_pretrained(cls.model_name)
        cls.model = BertModel.from_pretrained(cls.model_name).to(cls.device)
        cls.max_length = 512
        cls.model.eval()

    @classmethod
    def extract_bert_embeddings(
        cls, mode: str, device: str, sentences: list, labels: list, task: str
    ) -> dict:
        """
        This method performs the embeddings extractions.
        """
        path = f"bert_embeddings/{task}/dataset_tensors/"
        sents_reps = []
        step = 64  # Reduced batch size to save memory
        for idx in trange(0, len(sentences), step):
            idx_end = idx + step
            if idx_end > len(sentences):
                idx_end = len(sentences)
            sents_batch = sentences[idx:idx_end]

            sents_batch_encoding = cls.tokenizer(
                sents_batch,
                return_tensors="pt",
                max_length=cls.max_length,
                padding="max_length",
                truncation=True,
            )
            sents_batch_encoding = sents_batch_encoding.to(device)

            with torch.no_grad():
                batch_outputs = cls.model(**sents_batch_encoding)
                reps_batch = batch_outputs.pooler_output
            sents_reps.append(reps_batch.cpu())

            # Clear cache after processing each batch to free up memory
            del sents_batch_encoding, batch_outputs, reps_batch
            torch.cuda.empty_cache()

        sents_reps = torch.cat(sents_reps)

        for idx in range(len(labels)):
            labels[idx] = torch.tensor(labels[idx])
        labels = torch.stack(labels)

        # print(sents_reps.shape)
        # print(labels.shape)

        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(sents_reps.to("cpu"), path + f"{mode}_texts.pt")
        torch.save(labels, path + f"{mode}_labels.pt")

        return sents_reps
