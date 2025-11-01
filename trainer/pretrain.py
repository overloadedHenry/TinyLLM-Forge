import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import Trainer, TrainingArguments, HfArgumentParser, AutoTokenizer, AutoModelForCausalLM, AutoConfig, DataCollatorForLanguageModeling
from transformers.hf_argparser import HfArgumentParser
from transformers.trainer_utils import is_main_process
from datasets import load_dataset
from torch.optim import AdamW

from utils.arguments_base import ModelArguments, DataArguments
from data.preprocess import PretrainDataProcessor

from utils.safe_save_models import safe_save_for_hf_trainer
from utils.get_custom_opt import get_num_training_steps, get_cosine_schedule_with_lower_bound

if __name__ == "__main__":

    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path)

    dataset = load_dataset("json", data_files=data_args.train_file)

    processor = PretrainDataProcessor(tokenizer, max_len=data_args.max_len)
    
    with training_args.main_process_first():
        pretrain_dataset = dataset.map(processor, batched=True, batch_size=1000, num_proc=16, 
                                       load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0), 
                                       remove_columns=dataset["train"].column_names)
        
        if is_main_process(training_args.local_rank):
            print(f"Dataset example:{tokenizer.decode(pretrain_dataset['train'][0]['input_ids'])}")
        
            if training_args.should_save:
                    pretrain_dataset.save_to_disk(data_args.tokenized_path)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = AutoModelForCausalLM.from_config(model_config)

    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

    num_train_steps = get_num_training_steps(
        num_samples=len(pretrain_dataset["train"]),
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        num_epochs=training_args.num_train_epochs,
        world_size=int(os.environ.get("WORLD_SIZE", 8)),
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    )

    lr_scheduler = get_cosine_schedule_with_lower_bound(optimizer, num_train_steps)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=pretrain_dataset["train"],
        optimizers=(optimizer, lr_scheduler),
        data_collator=data_collator
    )

    if training_args.resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()

    safe_save_for_hf_trainer(trainer, training_args.output_dir)

    with training_args.main_process_first():
        tokenizer.save_pretrained(training_args.output_dir)