"""
    This module contains the models used in the LLMEmbed.
"""

# General Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import RobertaTokenizer, RobertaModel
from huggingface_hub import login

# Local Imports
from config import Config
from logger import Logger


class DownstreamModel(nn.Module):
    """
    This class contains the neural network that performs the classification tasks.
    """

    def __init__(self, class_num, SIGMA):
        super(DownstreamModel, self).__init__()
        self.SIGMA = SIGMA
        self.compress_layers = nn.ModuleList()
        for _ in range(5):
            layers = []
            layers.append(nn.Linear(4096, 1024))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            self.compress_layers.append(nn.Sequential(*layers))

        self.fc1 = nn.Linear(4145, 1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, class_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_l, input_b, input_r):
        """
        This method will apply the co-occurence pooling.
        """
        batch_size = input_l.shape[0]
        split_tensors = torch.split(input_l, 1, dim=1)

        input = []
        for i, split_tensor in enumerate(split_tensors):
            split_tensor = split_tensor.reshape(batch_size, -1)
            input.append(self.compress_layers[i](split_tensor))

        # Pad BERT and RoBERTa embeddings to match Llama2 (1024)
        input_b = F.pad(input_b, (0, 1024 - input_b.shape[1]))
        input_r = F.pad(input_r, (0, 1024 - input_r.shape[1]))

        input.append(input_b)
        input.append(input_r)

        input = torch.stack(input, dim=1)
        input_T = input.transpose(1, 2)
        input_P = torch.matmul(input, input_T)
        input_P = input_P.reshape(batch_size, -1)
        input_P = 2 * F.sigmoid(self.SIGMA * input_P) - 1

        a = torch.mean(input_l, dim=1)
        input = torch.cat([input_P, a], dim=1)

        output = self.fc1(input)
        output = self.relu1(output)
        output = self.dropout1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.dropout2(output)
        output = self.fc3(output)
        output = self.softmax(output)

        return output


class LLM:
    """
    This class contains the LLM models used for the embedding extraction.
    """

    @classmethod
    def __init__(cls, enable_logging: bool):
        cls.config = Config()
        cls.log = Logger()
        cls.enable_logging = enable_logging
        cls.device = cls.config.get_device()
        cls.login_token = cls.config.get_hugging_face_token()
        cls.bert_model_name = "google-bert/bert-large-uncased"
        cls.llama2_model_name = "meta-llama/Llama-2-7b-chat-hf"
        cls.roberta_model_name = "FacebookAI/roberta-large"

    @classmethod
    def model_repo_login(cls):
        """
        The meta-llama/Llama-2-7b-hf is a private gate repo, hence we need to get access to it and authenticate using a token.
        please refer this url to get the access - https://huggingface.co/meta-llama/Llama-2-7b-hf.
        After getting the access please generate a token and authenticate the model.
        """
        cls.log.log(
            message=f"\n[Started] - Performing authentication using a hugging face token for {cls.llama2_model_name}",
            enable_logging=cls.enable_logging,
        )
        login(cls.login_token)
        cls.log.log(
            message=f"[Completed] - Performing authentication using a hugging face token for {cls.llama2_model_name}",
            enable_logging=cls.enable_logging,
        )

    @classmethod
    def get_bert(cls) -> tuple:
        """
        This method returns the tokenizer and model for BERT.
        """
        tokenizer = BertTokenizer.from_pretrained(cls.bert_model_name)
        model = BertModel.from_pretrained(cls.bert_model_name).to(cls.device)
        return tokenizer, model

    @classmethod
    def get_llama2(cls) -> tuple:
        """
        This method returns the tokenizer and model for Llama2.
        """
        # Autheticate the model as it is a private gated repo
        cls.model_repo_login()

        tokenizer = AutoTokenizer.from_pretrained(cls.llama2_model_name)
        tokenizer.pad_token = "[PAD]"
        tokenizer.padding_side = "right"
        config_kwargs = {
            "trust_remote_code": True,
            "cache_dir": None,
            "revision": "main",
            "output_hidden_states": True,
        }
        model_config = AutoConfig.from_pretrained(
            cls.llama2_model_name, **config_kwargs
        )
        model = AutoModelForCausalLM.from_pretrained(
            cls.llama2_model_name,
            config=model_config,
            device_map=cls.device,
            torch_dtype=torch.float16,
        )

        return tokenizer, model

    @classmethod
    def get_roberta(cls) -> tuple:
        """
        This method returns the tokenizer and model for Roberta.
        """
        tokenizer = RobertaTokenizer.from_pretrained(cls.roberta_model_name)
        model = RobertaModel.from_pretrained(cls.roberta_model_name).to(cls.device)
        return tokenizer, model
