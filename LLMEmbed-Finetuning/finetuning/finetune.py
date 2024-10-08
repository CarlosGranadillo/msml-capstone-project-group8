"""
This module contains the class that executes all the finetuning process.
"""

# Local Imports
from logger import Logger
from config import Config
from .llama2_finetuning import Llama2FineTune


class FineTune:
    """
    This class contains the methods to finetune the LLM models.

    """

    @classmethod
    def __init__(cls, enable_logging):
        cls.log = Logger()
        cls.config = Config()
        cls.enable_logging = enable_logging
        cls.llama2_finetune = Llama2FineTune(enable_logging=enable_logging)

    @classmethod
    def finetune(cls):
        """
        This method executes the fine tuning of Llama2 model on sentiment analysis and yes no question.
        """
        tasks = cls.config.get_selected_task_types()
        for task in tasks:
            cls.log.log(
                message=f"\n[Started] - Finetuning of Llama2 model on {task}.",
                enable_logging=cls.enable_logging,
            )
            cls.llama2_finetune.finetune_llama2(task=task)

            cls.log.log(
                message=f"[Completed] - Finetuning of Llama2 model on {task}",
                enable_logging=cls.enable_logging,
            )
