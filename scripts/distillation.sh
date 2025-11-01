student_model_name_or_path=<path_to_your_student_model>
teacher_model_name_or_path=<path_to_your_teacher_model>
train_data=<path_to_your_training_data>
output_dir=./checkpoints/distillation_checkpoints
deepspeed_config=./config/zero-3.yaml


torchrun --nproc_per_node 2 trainer/distillation.py \
    --max_len 512 \
    --overwrite_cache False \
    --student_model_name_or_path $student_model_name_or_path \
    --teacher_model_name_or_path $teacher_model_name_or_path \
    --train_file $train_data \
    --output_dir $output_dir \
    --do_train True \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-4 \
    --num_train_epochs 3 \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --report_to "none" \
    --logging_steps 1 \
    --run_name "minimind-pretrain" \
    --dataloader_num_workers 16 \
    --bf16 True
