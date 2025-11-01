import torch

def check_torch_backend():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.npu.is_available():
        return "npu"
    else:
        return "cpu"
    

def safe_save_for_hf_trainer(trainer, output_dir):
    
    backend = check_torch_backend()

    if trainer.deepspeed:
        if backend == "cuda":
            torch.cuda.synchronize()
        elif backend == "npu":
            import torch_npu
            torch_npu.npu.synchronize()
        trainer.save_model(output_dir)
        return
    
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, cpu_state_dict)
