"""
    This module contains the class to extract embeddings using Llama LLM.
"""

# Local Imports
from config import Config
from logger import Logger
from helpers import Helpers

# General Imports
import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from tqdm import trange
from huggingface_hub import login


class Llama2:
    """
    This class contains the methods to extract the embeddings using Bert
    """

    @classmethod
    def __init__(cls, enable_logging: bool):
        """
        This method initialized the variables that are used in this class
        """
        cls.config = Config()
        cls.helpers = Helpers()
        cls.model_name = "meta-llama/Llama-2-7b-hf"
        cls.login_token = cls.config.get_hugging_face_token()
        cls.max_length = 128
        cls.log = Logger()
        cls.device = cls.config.get_device()
        cls.enable_logging = enable_logging

        # Autheticate the model as it is a private gated repo
        cls.model_repo_login(enable_logging)
        cls.log.log(
            message=f"\n[Started] - Loading the tokenizer for the {cls.model_name} LLM model from hugging face.",
            enable_logging=enable_logging,
        )
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.log.log(
            message=f"[Completed] - Loading the tokenizer for the {cls.model_name} LLM model from hugging face.",
            enable_logging=enable_logging,
        )
        cls.tokenizer.pad_token = "[PAD]"
        cls.tokenizer.padding_side = "right"
        cls.config_kwargs = {
            "trust_remote_code": True,
            "cache_dir": None,
            "revision": "main",
            "output_hidden_states": True,
        }
        cls.model_config = AutoConfig.from_pretrained(
            cls.model_name, **cls.config_kwargs
        )
        cls.log.log(
            message=f"\n[Started] - Loading the {cls.model_name} LLM model from hugging face.",
            enable_logging=enable_logging,
        )
        cls.model = AutoModelForCausalLM.from_pretrained(
            cls.model_name,
            config=cls.model_config,
            device_map=cls.device,
            torch_dtype=torch.float16,
        )
        cls.log.log(
            message=f"[Completed] - Loading the {cls.model_name} LLM model from hugging face.",
            enable_logging=enable_logging,
        )
        cls.model.eval()

    @classmethod
    def model_repo_login(cls, enable_logging: bool):
        """
        The meta-llama/Llama-2-7b-hf is a private gate repo, hence we need to get access to it and authenticate using a token.
        please refer this url to get the access - https://huggingface.co/meta-llama/Llama-2-7b-hf.
        After getting the access please generate a token and authenticate the model.
        """
        cls.log.log(
            message=f"\n[Started] - Performing authentication using a hugging face token for {cls.model_name}",
            enable_logging=enable_logging,
        )
        login(cls.login_token)
        cls.log.log(
            message=f"[Completed] - Performing authentication using a hugging face token for {cls.model_name}",
            enable_logging=enable_logging,
        )

    @classmethod
    def extract_llama2_embeddings(
        cls, mode: str, device: str, sentences: list, labels: list, task: str
    ) -> dict:
        """
        This method performs the embeddings extractions using LLama2.
        """
        cls.log.log(
            message=f"\n[Started] - Performing embeddings extraction using {cls.model_name} for {task} on {mode} data.",
            enable_logging=cls.enable_logging,
        )
        path = f"llama2_embeddings/{task}/dataset_tensors/"
        step = 1
        sentences_reps = []
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

                reps_batch_5_layers = []
                for layer in range(-1, -6, -1):
                    reps_batch_5_layers.append(
                        torch.mean(batch_outputs.hidden_states[layer], axis=1)
                    )
                reps_batch_5_layers = torch.stack(reps_batch_5_layers, axis=1)

            sentences_reps.append(reps_batch_5_layers.cpu())

            # Clear CUDA cache and delete unused variables
            torch.cuda.empty_cache()
            del (
                sentences_batch,
                sentences_batch_encoding,
                batch_outputs,
                reps_batch_5_layers,
            )

        sentences_reps = torch.cat(sentences_reps)
        labels = torch.stack([torch.tensor(label) for label in labels])

        cls.helpers.save_embeddings(
            sentences_embeds=sentences_reps, labels=labels, file_path=path, mode=mode
        )

        cls.log.log(
            message=f"[Completed] - Performing embeddings extraction using {cls.model_name} for {task} on {mode} data.",
            enable_logging=cls.enable_logging,
        )
        return sentences_reps
