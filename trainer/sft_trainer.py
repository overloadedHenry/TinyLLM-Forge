import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from transformers.hf_argparser import HfArgumentParser
from transformers.trainer_utils import is_main_process
from datasets import load_dataset
from utils.arguments_base import ModelArguments, DataArguments
from data.preprocess import SFTDataProcessor
from utils.safe_save_models import safe_save_for_hf_trainer


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_args_into_dataclasses()
    print(training_args)
    exit()
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)

    dataset = load_dataset('json', data_files=data_args.train_file)
    processor = SFTDataProcessor(tokenizer, max_len=data_args.max_len)
    
    with training_args.main_process_first():
        sft_dataset = dataset.map(processor, batched=True, batch_size=1000, num_proc=16, 
                                       load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0), 
                                       remove_columns=dataset["train"].column_names)
        if is_main_process(training_args.local_rank):
            print(f"SFT Dataset example:{tokenizer.decode(sft_dataset['train'][0]['input_ids'])}")
            # if training_args.should_save:
            #     sft_dataset.save_to_disk(data_args.tokenized_path)

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding='longest', max_length=data_args.max_len)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=sft_dataset["train"],
        data_collator=data_collator
    )

    if training_args.resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    safe_save_for_hf_trainer(trainer, training_args.output_dir)
    with training_args.main_process_first():
        tokenizer.save_pretrained(training_args.output_dir)
    

