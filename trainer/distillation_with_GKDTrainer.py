from trl import GKDTrainer, GKDConfig
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from transformers import HfArgumentParser
from transformers.trainer_utils import is_main_process
from datasets import load_dataset, Dataset
from utils.arguments_base import DataArguments, ModelArguments
from utils.safe_save_models import safe_save_for_hf_trainer
from data.preprocess import ProcessorForChatML

    
if __name__ == "__main__":

    
    parser = HfArgumentParser((ModelArguments, GKDConfig, DataArguments))
    model_args, training_args, data_args = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
    
    dataset = load_dataset("json", data_files=data_args.train_file)
    processor = ProcessorForChatML()
    
    with training_args.main_process_first(desc="dataset map pre-processing"):
        sft_dataset = dataset.map(processor, batched=True, batch_size=1000, num_proc=16, 
                                       load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0), 
                                       remove_columns=dataset["train"].column_names)
        if is_main_process(training_args.local_rank):
            # if training_args.should_save:
            #     sft_dataset.save_to_disk(data_args.tokenized_path)
            print(f"sft_dataset: {sft_dataset['train'][0]}")

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    teacher_model = AutoModelForCausalLM.from_pretrained(training_args.teacher_model_name_or_path)
    
    trainer = GKDTrainer(
        model=model,
        teacher_model=teacher_model,
        train_dataset=sft_dataset["train"],
        args=training_args,
        processing_class=tokenizer
    )

    if training_args.resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()

    safe_save_for_hf_trainer(trainer, training_args.output_dir)

    with training_args.main_process_first():
        tokenizer.save_pretrained(training_args.output_dir)
    