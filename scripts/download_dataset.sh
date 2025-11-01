### Download dataset
modelscope download --dataset 'gongjy/minimind_dataset' sft_1024.jsonl --local_dir ./minimind_dataset

hf download --repo-type dataset gongjy/minimind_dataset --local-dir ./minimind_dataset