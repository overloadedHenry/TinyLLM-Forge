from dataclasses import dataclass, field, fields

import os

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
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


