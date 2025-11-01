student_model_name_or_path=<path_to_your_student_model>
teacher_model_name_or_path=<path_to_your_teacher_model>
train_data=<path_to_your_training_data>
output_dir=./checkpoints/distillation_checkpoints
deepspeed_config=./config/zero-3.yaml

torchrun --nproc_per_node 1 trainer/distillation_with_GKDTrainer.py \
    --max_len 512 \
    --overwrite_cache False \
    --model_name_or_path $student_model_name_or_path \
    --teacher_model_name_or_path $teacher_model_name_or_path \
    --beta 0 \
    --train_file $train_data \
    --output_dir $output_dir \
    --do_train False \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --learning_rate 5e-4 \
    --num_train_epochs 1 \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --report_to "none" \
    --logging_steps 1 \
    --run_name "minimind-pretrain" \
    --dataloader_num_workers 16 \
    --bf16 True