"""
    This module contains the class to finetune the Llama2 LLM on sentiment analysis and yes no question tasks.
"""

# General Imports
import os
import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# Local Imports
from config import Config
from helpers import Helpers
from logger import Logger
from models.model import LLM


class Llama2FineTune:
    """
    This class contains the methods to perform Llama2 finetuning.
    """

    @classmethod
    def __init__(cls, enable_logging: bool):
        """
        This method initializes the variables and other instances.
        """
        cls.config = Config()
        cls.helpers = Helpers()
        cls.log = Logger()
        cls.llm = LLM(enable_logging=enable_logging)

        cls.enable_logging = enable_logging
        cls.data_path_sentiment_analysis = "/home/exouser/Desktop/msml-group8-code-base/data/sentiment_analysis_fine_tuning"

    @classmethod
    def create_llama2_prompt_sentiment_analysis(cls, row):
        """
        This method creates the prompt structure to fine tune Llama2 model for sentiment analysis.
        """
        row[
            "text"
        ] = f"""
        <s>
        [INST]
        <SYS>
        System Prompt :
        Which sentiment best describes the previous financial statement from the following options: negative, neutral, positive, bearish, bullish?
        </SYS>
        User Prompt :
        {row['text']}
        [/INST]
        Answer:
        {row['label']}
        </s>
        """
        return row

    @classmethod
    def set_parameters(cls):
        """
        This method sets the  parameters used for finetuning.
        """
        # LoRA parameters
        cls.lora_r = 64  # LoRA attention dimension
        cls.lora_alpha = 16  # Alpha parameter for LoRA scaling
        cls.lora_dropout = 0.1  # Dropout probability for LoRA layers

        # bitsandbytes parameters
        cls.use_4bit = True  # Activate 4-bit precision base model loading
        cls.bnb_4bit_compute_dtype = "float16"  # Compute dtype for 4-bit base models
        cls.bnb_4bit_quant_type = "nf4"  # Quantization type (fp4 or nf4)
        cls.use_nested_quant = False  # Activate nested quantization for 4-bit base models (double quantization)

        # TrainingArguments parameters
        cls.output_dir = "./results"  # Output directory where the model predictions and checkpoints will be stored
        cls.num_train_epochs = 1  # Number of training epochs

        # Enable fp16/bf16 training (set bf16 to True with an A100)
        cls.fp16 = False
        cls.bf16 = True
        cls.per_device_train_batch_size = 4  # Batch size per GPU for training
        cls.per_device_eval_batch_size = 4  # Batch size per GPU for evaluation
        cls.gradient_accumulation_steps = (
            1  # Number of update steps to accumulate the gradients for
        )
        cls.gradient_checkpointing = True  # Enable gradient checkpointing
        cls.max_grad_norm = 0.3  # Maximum gradient normal (gradient clipping)
        cls.learning_rate = 2e-4  # Initial learning rate (AdamW optimizer)
        cls.weight_decay = (
            0.001  # Weight decay to apply to all layers except bias/LayerNorm weights
        )
        cls.optim = "paged_adamw_32bit"  # Optimizer to use
        cls.lr_scheduler_type = "cosine"  # Learning rate schedule
        cls.max_steps = -1  # Number of training steps (overrides num_train_epochs)
        cls.warmup_ratio = (
            0.03  # Ratio of steps for a linear warmup (from 0 to learning rate)
        )
        cls.group_by_length = True  # Group sequences into batches with same length, Saves memory and speeds up training considerably
        cls.save_steps = 0  # Save checkpoint every X updates steps
        cls.logging_steps = 25  # Log every X updates steps

        # SFT parameters
        cls.max_seq_length = None  # Maximum sequence length to use
        cls.packing = False  # Pack multiple short examples in the same input sequence to increase efficiency
        cls.device_map = {"": 0}  # Load the entire model on the GPU 0

        cls.compute_dtype = getattr(torch, cls.bnb_4bit_compute_dtype)
        cls.bnb_config = BitsAndBytesConfig(
            load_in_4bit=cls.use_4bit,
            bnb_4bit_quant_type=cls.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=cls.compute_dtype,
            bnb_4bit_use_double_quant=cls.use_nested_quant,
        )

        # LORA and PEFT Config
        peft_config = LoraConfig(
            lora_alpha=cls.lora_alpha,
            lora_dropout=cls.lora_dropout,
            r=cls.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Set training parameters
        cls.training_arguments = TrainingArguments(
            output_dir=cls.output_dir,
            num_train_epochs=cls.num_train_epochs,
            per_device_train_batch_size=cls.per_device_train_batch_size,
            gradient_accumulation_steps=cls.gradient_accumulation_steps,
            optim=cls.optim,
            save_steps=cls.save_steps,
            logging_steps=cls.logging_steps,
            learning_rate=cls.learning_rate,
            weight_decay=cls.weight_decay,
            fp16=cls.fp16,
            bf16=cls.bf16,
            max_grad_norm=cls.max_grad_norm,
            max_steps=cls.max_steps,
            warmup_ratio=cls.warmup_ratio,
            group_by_length=cls.group_by_length,
            lr_scheduler_type=cls.lr_scheduler_type,
            report_to="tensorboard",
        )

    @classmethod
    def finetune_llama2_sentiment_analysis(cls):
        """
        This method performs the fine tuning for the Llama2 model.
        """
        sentiment_analysis_dataset = load_from_disk(cls.data_path_sentiment_analysis)
        # Convert integer labels to string labels for the prompt
        sentiment_mapping = {
            label_num: label_text
            for label_text, label_num in cls.config.get_sentiment_mapping().items()
        }
        sentiment_analysis_dataset = sentiment_analysis_dataset.map(
            cls.helpers.replace_int_with_string,
            fn_kwargs={"mapping": sentiment_mapping, "column_to_modify": "label"},
        )
        cls.log.log(
            message="\n[Started] - Conversion to Llama2 model template.",
            enable_logging=cls.enable_logging,
        )
        sent_analysis_prompt_dataset = sentiment_analysis_dataset.map(
            cls.create_llama2_prompt_sentiment_analysis
        )
        cls.log.log(
            message="[Completed] - Conversion to Llama2 model template.",
            enable_logging=cls.enable_logging,
        )
        cls.set_lora_parameters()

        # Check GPU compatibility with bfloat16
        if cls.compute_dtype == torch.float16 and cls.use_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16: accelerate training with bf16=True")
                print("=" * 80)

        model, tokenizer = cls.llm.get_llama2(use_finetuned_model=False, task=None)
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        trainer = SFTTrainer(
            model=model,
            train_dataset=sent_analysis_prompt_dataset,
            peft_config=cls.peft_config,
            dataset_text_field="text",
            max_seq_length=cls.max_seq_length,
            tokenizer=tokenizer,
            args=cls.training_arguments,
            packing=cls.packing,
        )
        trainer.train()

        new_model = "Llama-2-7b-chat-finetune-finance-sentiment"
        cls.helpers.save_finetuned_model(model_name=new_model)

