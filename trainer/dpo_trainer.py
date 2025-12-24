import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.hf_argparser import HfArgumentParser

from utils.arguments_base import ModelArguments, DataArguments
from utils.safe_save_models import safe_save_for_hf_trainer



if __name__ == "__main__":
    
    parser = HfArgumentParser((ModelArguments, DPOConfig, DataArguments))
    model_args, training_args, data_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    
    dataset = load_dataset("json", data_files=data_args.train_file)

    trainer = DPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        args=training_args
    )

    trainer.train()

    if training_args.resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    safe_save_for_hf_trainer(trainer, training_args.output_dir)
    with training_args.main_process_first():
        tokenizer.save_pretrained(training_args.output_dir)