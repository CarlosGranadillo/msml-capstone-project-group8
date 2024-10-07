"""
This module contains the class that executes all the finetuning process.
"""

# Local Imports
from logger import Logger
from .llama2_finetuning import Llama2FineTune


class FineTune:
    """
    This class contains the methods to finetune the LLM models.

    """

    @classmethod
    def __init__(cls, enable_logging):
        cls.log = Logger()
        cls.enable_logging = enable_logging
        cls.llama2_finetune = Llama2FineTune(enable_logging=enable_logging)

    @classmethod
    def finetune(cls):
        cls.log.log(
            message="\n[Started] - Finetuning of Llama2 model on sentiment analysis",
            enable_logging=cls.enable_logging,
        )
        res = cls.llama2_finetune.finetune_llama2_sentiment_analysis()

        cls.log.log(
            message="[Completed] - Finetuning of Llama2 model on sentiment analysis",
            enable_logging=cls.enable_logging,
        )
