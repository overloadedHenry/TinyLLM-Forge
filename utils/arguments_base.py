from dataclasses import dataclass, field, fields
from typing import Optional, List
import os

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


@dataclass
class LoraArguments:
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA adapters."}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA rank."}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha."}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout."}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "out_proj", "up_proj", "down_proj"],
        metadata={"help": "Target modules for LoRA."}
    )
    lora_weights_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to LoRA weights."}
    )
    lora_bias: Optional[str] = field(
        default="none",
        metadata={"help": "LoRA bias configuration."}
    )

@dataclass
class DataArguments:
    train_file: str = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: str = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    max_len: int = field(
        default=512
    ),
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    tokenized_path: str = field(
        default='~/.cache/huggingface/datasets',
        metadata={"help": "Path to save the tokenized datasets."},
    )


@dataclass
class KDModelArguments:
    student_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    
    teacher_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


