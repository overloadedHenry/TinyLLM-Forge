model_name_or_path=<path_to_your_model>
train_data=<path_to_your_training_data>
output_dir=./checkpoints/pretrain_checkpoints
deepspeed_config=./config/zero-3.yaml

torchrun --nproc_per_node 1 trainer/sft_trainer.py \
    --max_len 512 \
    --overwrite_cache False \
    --model_name_or_path $model_name_or_path \
    --train_file $train_data \
    --output_dir $output_dir \
    --do_train True \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --lr_scheduler_type "cosine" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --logging_steps 1 \
    --report_to "swanlab" \
    --run_name "minimind-sft" \
    --dataloader_num_workers 16 \
    --bf16 True
