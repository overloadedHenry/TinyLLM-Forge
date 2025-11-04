import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from transformers.hf_argparser import HfArgumentParser
from transformers.trainer_utils import is_main_process
from peft import get_peft_model, LoraConfig, TaskType, AutoPeftModelForCausalLM
from datasets import load_dataset
from utils.arguments_base import ModelArguments, DataArguments, LoraArguments
from data.preprocess import SFTDataProcessor
from utils.safe_save_models import safe_save_for_hf_trainer


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments))
    model_args, training_args, data_args, lora_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)

    dataset = load_dataset('json', data_files=data_args.train_file)
    processor = SFTDataProcessor(tokenizer, max_len=data_args.max_len)
    
    with training_args.main_process_first():
        sft_dataset = dataset.map(processor, batched=True, batch_size=1000, num_proc=16, 
                                       load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0), 
                                       remove_columns=dataset["train"].column_names)
        if is_main_process(training_args.local_rank):
            print(f"SFT Dataset example:{tokenizer.decode(sft_dataset['train'][0]['input_ids'])}")


    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    if lora_args.use_lora and training_args.do_train:
        if lora_args.lora_weights_path is not None:
            model = AutoPeftModelForCausalLM.from_pretrained(model, lora_args.lora_weights_path)
        else:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_args.lora_r,
                inference_mode=False,
                lora_alpha=lora_args.lora_alpha,
                target_modules=lora_args.lora_target_modules,
                lora_dropout=lora_args.lora_dropout,
                bias=lora_args.lora_bias,
            )
            model = get_peft_model(model, peft_config)

        model.print_trainable_parameters()

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
    

