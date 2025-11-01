import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from transformers import HfArgumentParser
from transformers.trainer_utils import is_main_process
from datasets import load_dataset
from utils.arguments_base import KDModelArguments, DataArguments
from utils.safe_save_models import safe_save_for_hf_trainer
import torch
import torch.nn.functional as F
from data.preprocess import SFTDataProcessor


def compute_forward_kl_loss(student_logits, teacher_logits, labels=None, temperature=1.0, reduction="batchmean"):
    
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    shifted_student_log_probs = student_log_probs[..., :-1, :]
    shifted_teacher_log_probs = teacher_log_probs[..., :-1, :]

    kd_loss = F.kl_div(shifted_student_log_probs, shifted_teacher_log_probs, reduction="none", log_target=True)

    shifted_labels = labels[..., 1:] if labels is not None else None
    if labels is not None:
        mask = shifted_labels != -100
        kd_loss = kd_loss[mask]

    # Apply reduction

    return kd_loss.sum() / mask.sum() if labels is not None else kd_loss.sum() / (kd_loss.size(0) * kd_loss.size(1))

#     return kl_loss
class KDTrainer(Trainer):
    def __init__(
        self,
        model,
        teacher_model,
        args,
        if_use_entropy=False,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        data_collator=None,
        compute_metrics=None,
        callbacks=None,
        temperature=1.0,
        alpha=0.5
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            model_init=model_init,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks
        )
        self.teacher_model = self.teacher_init(teacher_model)
        self.temperature = temperature
        self.alpha = alpha
        self.if_use_entropy = if_use_entropy

    def teacher_init(self, teacher_model):
        unwraped_teacher_model = self.accelerator.unwrap_model(teacher_model)

        teacher_model = self.accelerator.prepare_model(unwraped_teacher_model)

        return teacher_model


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        outputs = model(**inputs)
        
        self.teacher_model.eval()

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)

        loss = outputs.loss
        logits = outputs.logits
        teacher_logits = teacher_outputs.logits

        labels = inputs.labels

        if self.if_use_entropy:
            loss = loss + self.alpha * compute_forward_kl_loss(logits, teacher_logits, labels, self.temperature)

        return (loss, outputs) if return_outputs else loss

if __name__ == "__main__":
    
    parser = HfArgumentParser((KDModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained(model_args.student_model_name_or_path, use_fast=True)

    dataset = load_dataset("json", data_files=data_args.train_file)
    processor = SFTDataProcessor(tokenizer, max_len=data_args.max_len)
    
    with training_args.main_process_first(desc="dataset map pre-processing"):
        sft_dataset = dataset.map(processor, batched=True, batch_size=1000, num_proc=16, 
                                       load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0), 
                                       remove_columns=dataset["train"].column_names)
        if is_main_process(training_args.local_rank):
            print(f"SFT Dataset example:{tokenizer.decode(sft_dataset['train'][0]['input_ids'])}")
            
            # if training_args.should_save:
            #     sft_dataset.save_to_disk(data_args.tokenized_path)
        
    model = AutoModelForCausalLM.from_pretrained(model_args.student_model_name_or_path)
    teacher_model = AutoModelForCausalLM.from_pretrained(model_args.teacher_model_name_or_path)
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding='longest', max_length=data_args.max_len)

    trainer = KDTrainer(
        model=model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=sft_dataset["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        if_use_entropy=True,
        temperature=2.0,
        alpha=0.5
    )
    if training_args.resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()

    safe_save_for_hf_trainer(trainer, training_args.output_dir)

    with training_args.main_process_first():
        tokenizer.save_pretrained(training_args.output_dir)